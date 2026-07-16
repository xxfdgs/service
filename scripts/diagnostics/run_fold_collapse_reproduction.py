#!/usr/bin/env python3
"""Deterministically reproduce a GraphGPS fold with non-invasive diagnostics.

This runner reimplements the existing ``double`` loop for property_num=4.  It
does not alter the model forward, loss, optimizer, or scheduler mathematics.
The only additions are detached observations after the normal forward/backward
calls and checkpoints written to a new diagnostic directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import contextlib
import json
import math
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import graphgps  # noqa: F401,E402
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from graphgps.determinism import configure_determinism  # noqa: E402
from graphgps.lrx_add.compute_loss_multi4 import compute_loss_multi4  # noqa: E402
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402
from torch_geometric.graphgym.optim import OptimizerConfig, create_optimizer, create_scheduler  # noqa: E402


TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]


def tensor_stats(value: torch.Tensor) -> dict[str, float | int | bool]:
    value = value.detach().float().reshape(value.shape[0], -1) if value.ndim else value.detach().float().view(1, 1)
    finite = torch.isfinite(value)
    safe = value[finite]
    if safe.numel() == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "norm": math.nan,
                "sample_distance": math.nan, "zero_fraction": math.nan, "saturated_fraction": math.nan,
                "nan_count": int(torch.isnan(value).sum()), "inf_count": int(torch.isinf(value).sum())}
    sample_distance = math.nan
    if value.ndim == 2 and value.shape[0] > 1 and finite.all():
        sample_distance = float(torch.pdist(value).mean())
    return {
        "mean": float(safe.mean()), "std": float(safe.std(unbiased=False)), "min": float(safe.min()),
        "max": float(safe.max()), "norm": float(torch.linalg.vector_norm(safe)), "sample_distance": sample_distance,
        "zero_fraction": float((safe == 0).float().mean()), "saturated_fraction": float((safe.abs() >= 10.0).float().mean()),
        "nan_count": int(torch.isnan(value).sum()), "inf_count": int(torch.isinf(value).sum()),
    }


def extract_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = extract_tensor(item)
            if found is not None:
                return found
    if hasattr(value, "graph_feature") and torch.is_tensor(value.graph_feature):
        return value.graph_feature
    return None


class DynamicRecorder:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.active: dict[str, object] | None = None
        self.activation_rows: list[dict[str, object]] = []
        self.fusion_rows: list[dict[str, object]] = []
        self.anomaly_rows: list[dict[str, object]] = []
        self.handles = []
        self.modules = {
            "graph_encoder": model.gnn,
            "ratio_encoder": model.ratio_encoder,
            "fusion_input": model.FC_layers[0],
            "fusion_hidden": model.FC_layers[1],
            "main_head": model.FC_layers[2],
            "direct_head": model.FC_layers_2mlp[0],
            "middle_head": model.FC_layers_midle_mlp[0],
            "branch_weight_mlp": model.branch_weight_mlp,
            "additive_delta_head": model.additive_delta_head,
        }
        for name, module in self.modules.items():
            if name == "fusion_input":
                self.handles.append(module.register_forward_pre_hook(self._hook(name, pre=True)))
            else:
                self.handles.append(module.register_forward_hook(self._hook(name, pre=False)))

    def _hook(self, name: str, pre: bool):
        def callback(_module, inputs, output=None):
            if self.active is None:
                return
            value = inputs[0] if pre else output
            tensor = extract_tensor(value)
            if tensor is not None:
                self.capture(name, tensor)
        return callback

    def start(self, epoch: int, split: str, batch_index: int) -> None:
        self.active = {"epoch": epoch, "split": split, "batch_index": batch_index, "calls": {}}

    def capture(self, module: str, tensor: torch.Tensor) -> None:
        if self.active is None:
            return
        calls = self.active["calls"]
        call_index = int(calls.get(module, 0))
        calls[module] = call_index + 1
        row = {"epoch": self.active["epoch"], "split": self.active["split"], "batch_index": self.active["batch_index"],
               "module": module, "call_index": call_index, **tensor_stats(tensor)}
        self.activation_rows.append(row)
        self.anomaly_rows.append({k: row[k] for k in ("epoch", "split", "batch_index", "module", "call_index", "nan_count", "inf_count")})
        if module == "branch_weight_mlp":
            logits = tensor.detach().reshape(-1, 4, 3)
            weights = (torch.full_like(logits, 1.0 / 3.0)
                       if cfg.diagnostic_uniform_fusion else torch.softmax(logits, dim=-1))
            entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=-1)
            for target_index, target in enumerate(TARGETS):
                for branch_index, branch in enumerate(("main", "direct", "middle")):
                    values = weights[:, target_index, branch_index]
                    self.fusion_rows.append({
                        "epoch": self.active["epoch"], "split": self.active["split"], "batch_index": self.active["batch_index"],
                        "target": target, "branch": branch, "weight_mean": float(values.mean()),
                        "weight_std": float(values.std(unbiased=False)), "weight_min": float(values.min()), "weight_max": float(values.max()),
                        "entropy_mean": float(entropy[:, target_index].mean()), "entropy_min": float(entropy[:, target_index].min()),
                    })

    def capture_inputs(self, batches) -> None:
        if self.active is None:
            return
        ratio = torch.cat([batch.ratio.reshape(-1, 1).float() for batch in batches], dim=1)
        mordred = torch.cat([batch.mordred_feat.reshape(batch.num_graphs, -1).float() for batch in batches], dim=1)
        self.capture("ratio_formula_input", ratio)
        self.capture("descriptor_input_direct", mordred)

    def end(self) -> None:
        self.active = None

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def module_norm(module: torch.nn.Module, gradients: bool = False) -> float:
    parts = []
    for parameter in module.parameters():
        value = parameter.grad if gradients else parameter.detach()
        if value is not None:
            parts.append(value.detach().float().reshape(-1))
    return float(torch.linalg.vector_norm(torch.cat(parts))) if parts else math.nan


def module_state(model: torch.nn.Module, modules: dict[str, torch.nn.Module]) -> dict[str, torch.Tensor]:
    state = {}
    for name, module in modules.items():
        values = [parameter.detach().float().reshape(-1).cpu().clone() for parameter in module.parameters()]
        state[name] = torch.cat(values) if values else torch.empty(0)
    return state


def prepare_batches(items, split: str, device: torch.device):
    suffixes = ("", "_2", "_3", "_4", "_5")
    for batch, suffix in zip(items, suffixes):
        batch.split = split + suffix
        batch.to(device)
    return items


def property_losses(prediction: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.mean(torch.abs(prediction[index::4] - labels[index::4])) for index in range(4)])


def target_metrics(prediction: np.ndarray, labels: np.ndarray, split: str, epoch: int, total_loss: float, lr: float,
                   early_counter: int, best_epoch: int | None, best_loss: float | None) -> list[dict[str, object]]:
    rows = []
    for index, target in enumerate(TARGETS):
        y, p = labels[:, index] * 100.0, prediction[:, index] * 100.0
        rows.append({
            "epoch": epoch, "split": split, "target": target, "n_valid": int(np.isfinite(y).sum()),
            "target_mean": float(np.mean(y)), "target_std": float(np.std(y, ddof=1)),
            "prediction_mean": float(np.mean(p)), "prediction_std": float(np.std(p, ddof=1)),
            "std_ratio": float(np.std(p, ddof=1) / np.std(y, ddof=1)) if np.std(y, ddof=1) else math.nan,
            "mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p)),
            "spearman": float(spearmanr(y, p).statistic), "total_loss_normalized": total_loss,
            "lr": lr, "early_stopping_counter": early_counter, "best_epoch_candidate": best_epoch,
            "best_validation_loss_candidate": best_loss,
        })
    return rows


def evaluate(model, loader_groups, split: str, device: torch.device, recorder: DynamicRecorder | None, epoch: int):
    model.eval()
    predictions, labels, source_indices = [], [], []
    losses = []
    property_loss_rows = []
    with torch.no_grad():
        for batch_index, batches in enumerate(zip(*[group[{"train": 0, "val": 1, "test": 2}[split]] for group in loader_groups])):
            batches = prepare_batches(list(batches), split, device)
            if recorder is not None and batch_index == 0:
                recorder.start(epoch, split, batch_index)
                recorder.capture_inputs(batches)
            pred, label = model(*batches)
            if recorder is not None and batch_index == 0:
                recorder.capture("final_prediction", pred.reshape(-1, 4))
                recorder.end()
            losses.append(float(property_losses(pred, label).sum().detach().cpu()))
            property_loss_rows.append(property_losses(pred, label).detach().cpu().numpy())
            predictions.append(pred.detach().cpu().reshape(-1, 4).numpy())
            labels.append(label.detach().cpu().reshape(-1, 4).numpy())
            source_indices.append(batches[0].sample_uid.detach().cpu().numpy().reshape(-1))
    prediction = np.vstack(predictions)
    label = np.vstack(labels)
    # GraphGym's logger weights each batch loss by flattened target count.  The
    # whole-split mean below is algebraically identical and avoids a smaller
    # final batch biasing validation checkpoint selection.
    all_property_losses = np.mean(np.abs(prediction - label), axis=0)
    return (prediction, label, np.concatenate(source_indices),
            float(all_property_losses.sum()), all_property_losses)


def copy_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def append_frame(frame: pd.DataFrame, path: Path) -> None:
    """Append one completed observation unit without creating empty CSVs.

    The diagnostic runs are deliberately resumable because a GPU job can be
    interrupted between epochs.  Persisting each completed epoch keeps the
    diagnostic history aligned with ``resume_state.pt``; these writes do not
    participate in the model computation.
    """
    if frame.empty:
        return
    frame.to_csv(path, mode="a" if path.exists() else "w", header=not path.exists(), index=False)


def build_optimizer_scheduler(model: torch.nn.Module):
    optimizer = create_optimizer(model.parameters(), OptimizerConfig(
        optimizer=cfg.optim.optimizer, base_lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay, momentum=cfg.optim.momentum,
    ))
    scheduler = create_scheduler(optimizer, ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler, steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs, train_mode=cfg.train.mode, eval_period=cfg.train.eval_period,
    ))
    return optimizer, scheduler


def make_prediction_frame(prediction: np.ndarray, labels: np.ndarray, source_indices: np.ndarray, manifest: pd.DataFrame,
                          split: str, epoch: int, checkpoint: str) -> pd.DataFrame:
    mapping = manifest.loc[manifest.split == split, ["sample_id", "original_row_index"]].copy()
    lookup = dict(zip(mapping.original_row_index.astype(int), mapping.sample_id.astype(str)))
    rows = []
    for index, source_index in enumerate(source_indices):
        for target_index, target in enumerate(TARGETS):
            rows.append({"sample_id": lookup[int(source_index)], "source_index": int(source_index), "split": split,
                         "target": target, "epoch": epoch, "checkpoint": checkpoint,
                         "y_true": float(labels[index, target_index] * 100.0), "y_pred": float(prediction[index, target_index] * 100.0)})
    return pd.DataFrame(rows)


def collapse_timeline(epoch_metrics: pd.DataFrame, activation: pd.DataFrame, fusion: pd.DataFrame,
                      gradients: pd.DataFrame, parameters: pd.DataFrame, anomalies: pd.DataFrame,
                      initial_lr: float) -> pd.DataFrame:
    events: list[dict[str, object]] = []
    def first_row(condition: pd.Series, frame: pd.DataFrame, event: str, module: str, evidence: str):
        selected = frame.loc[condition]
        if selected.empty:
            events.append({"event": event, "first_epoch": np.nan, "first_batch": np.nan, "module": module, "observed": False, "evidence": evidence})
        else:
            row = selected.sort_values(["epoch", "batch_index"] if "batch_index" in selected else ["epoch"]).iloc[0]
            events.append({"event": event, "first_epoch": int(row.epoch), "first_batch": int(row.get("batch_index", -1)), "module": module, "observed": True, "evidence": evidence})
    first_row(epoch_metrics.std_ratio < 0.10, epoch_metrics, "prediction_std_below_target_10pct", "prediction", "prediction std / target std < 0.10")
    for target, group in epoch_metrics.groupby("target"):
        values = group.loc[group.split == "train"].sort_values("epoch")
        decreasing = values.prediction_std.diff().lt(0).rolling(5).sum().eq(5)
        first_row(decreasing, values, f"five_epoch_prediction_std_decline:{target}", "prediction", "five consecutive declines")
    first_row(activation['std'] < 1e-4, activation, "branch_embedding_std_below_1e-4", "activation", "captured module std < 1e-4")
    first_row(fusion.weight_mean > .98, fusion, "softmax_branch_weight_above_0.98", "fusion", "mean softmax weight > .98")
    first_row(fusion.entropy_mean < .05, fusion, "fusion_entropy_near_zero", "fusion", "mean branch entropy < .05")
    first_row(gradients.grad_norm < 1e-8, gradients, "gradient_norm_below_1e-8", "gradient", "module grad norm < 1e-8")
    first_row(gradients.grad_norm > 1e3, gradients, "gradient_norm_above_1e3", "gradient", "module grad norm > 1e3")
    first_row(parameters.parameter_delta_norm < 1e-10, parameters, "parameter_norm_unchanged", "parameter", "post-step parameter delta norm < 1e-10")
    first_row(epoch_metrics.lr < initial_lr * .01, epoch_metrics, "learning_rate_below_initial_1pct", "scheduler", "lr < initial 1%")
    first_row(anomalies.nan_count.gt(0) | anomalies.inf_count.gt(0), anomalies, "nan_or_inf", "activation", "captured activation NaN/Inf")
    return pd.DataFrame(events)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--stop-after-epochs", type=int, default=None,
                        help="Diagnostic execution limit that preserves the config's scheduler horizon.")
    parser.add_argument("--scheduler", choices=("original", "none"), default="original")
    parser.add_argument("--early-stopping", choices=("original", "disabled"), default="original")
    parser.add_argument("--uniform-fusion", action="store_true", help="Diagnostic only: replace learned branch softmax with 1/3 weights.")
    parser.add_argument("--resume", action="store_true", help="Resume a prior chunk from resume_state.pt.")
    parser.add_argument("--chunk-epochs", type=int, default=None, help="Maximum epochs for this invocation; state is saved after every epoch.")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite diagnostic run directory: {run_dir}")
    for directory in (run_dir, run_dir / "checkpoints", run_dir / "cache"):
        directory.mkdir(parents=True, exist_ok=True)

    set_cfg_gps(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(args.config.resolve()), opts=[]))
    cfg.seed = int(cfg.seed)
    cfg.run_dir = str(run_dir)
    cfg.out_dir = str(run_dir)
    cfg.dataset.dir = str(run_dir / "cache")
    cfg.dataset.cache_tag = f"fold-collapse-{args.fold}-{run_dir.name}"
    # A fresh diagnostic run must build an isolated cache.  A resumed run must
    # never delete that cache before loading it; the cache contents are
    # immutable inputs to the already-saved model/optimizer/RNG state.
    cfg.dataset.cache_refresh = not args.resume
    if args.max_epochs is not None:
        cfg.optim.max_epoch = args.max_epochs
    execution_max_epoch = int(args.stop_after_epochs or cfg.optim.max_epoch)
    if execution_max_epoch <= 0 or execution_max_epoch > int(cfg.optim.max_epoch):
        raise ValueError("--stop-after-epochs must be in [1, optim.max_epoch]")
    if args.scheduler == "none":
        cfg.optim.scheduler = "none"
    if args.early_stopping == "disabled":
        cfg.train.early_stop_patience = execution_max_epoch + 1
    cfg.diagnostic_uniform_fusion = bool(args.uniform_fusion)
    configure_determinism(cfg.seed, bool(cfg.train.deterministic))
    if not args.resume:
        shutil.copy2(args.config, run_dir / "source_config.yaml")
        (run_dir / "run_settings.json").write_text(json.dumps({
            "fold": args.fold, "seed": int(cfg.seed), "max_epoch": int(cfg.optim.max_epoch),
            "execution_max_epoch": execution_max_epoch,
            "scheduler": cfg.optim.scheduler, "early_stopping": args.early_stopping,
            "uniform_fusion": bool(args.uniform_fusion),
            "source_config": str(args.config.resolve()), "mathematics_changed": bool(args.uniform_fusion),
        }, indent=2) + "\n")
    elif not (run_dir / "resume_state.pt").is_file():
        raise FileNotFoundError(f"Missing resume_state.pt in {run_dir}")

    # Cache construction is verbose but irrelevant to epoch progress.  Keep it
    # in the run directory so the interactive session remains available for
    # deterministic per-epoch progress records.
    with (run_dir / "cache_build.log").open("w") as cache_log:
        with contextlib.redirect_stdout(cache_log), contextlib.redirect_stderr(cache_log):
            loader_groups = create_loader_5()
    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model = create_model_gps().to(device)
    core_model = model.model
    optimizer, scheduler = build_optimizer_scheduler(model)
    recorder = DynamicRecorder(core_model)
    parameter_modules = recorder.modules
    manifest = pd.read_csv(cfg.train.manifest_path, dtype={"sample_id": str})
    epoch_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    head_rows: list[dict[str, object]] = []
    best_loss = math.inf
    best_epoch: int | None = None
    best_state = None
    best_val_prediction = None
    early_counter = 0
    early_reference = math.inf
    start_epoch = 0
    last_result = None
    initial_lr = float(optimizer.param_groups[0]['lr'])

    if args.resume:
        resume_state = torch.load(run_dir / "resume_state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(resume_state["model_state"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        best_loss = float(resume_state["best_loss"])
        best_epoch = resume_state["best_epoch"]
        best_state = resume_state["best_state"]
        best_val_prediction = resume_state["best_val_prediction"]
        early_counter = int(resume_state["early_counter"])
        early_reference = float(resume_state["early_reference"])
        start_epoch = int(resume_state["next_epoch"])
        restore_rng_state(resume_state["rng_state"])

    completed = False
    for epoch in range(start_epoch, execution_max_epoch):
        # Record list boundaries so only observations from this epoch are
        # appended below.  ``DynamicRecorder`` deliberately retains prior
        # rows for the completed-run timeline calculation.
        activation_start = len(recorder.activation_rows)
        fusion_start = len(recorder.fusion_rows)
        anomaly_start = len(recorder.anomaly_rows)
        gradient_start = len(gradient_rows)
        parameter_start = len(parameter_rows)
        head_start = len(head_rows)
        model.train()
        optimizer.zero_grad()
        train_predictions, train_labels, train_sources = [], [], []
        train_property_losses = []
        train_total_losses = []
        first_step_done = False
        for batch_index, batches in enumerate(zip(*[group[0] for group in loader_groups])):
            batches = prepare_batches(list(batches), "train", device)
            observe = batch_index == 0
            if observe:
                recorder.start(epoch, "train", batch_index)
                recorder.capture_inputs(batches)
                before = module_state(model, parameter_modules)
            pred, label = model(*batches)
            if observe:
                recorder.capture("final_prediction", pred.reshape(-1, 4))
            losses = property_losses(pred, label)
            loss = losses.sum()
            loss.backward()
            if observe:
                for name, module in parameter_modules.items():
                    gradient_rows.append({"epoch": epoch, "batch_index": batch_index, "module": name, "grad_norm": module_norm(module, gradients=True)})
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if observe:
                after = module_state(model, parameter_modules)
                for name, module in parameter_modules.items():
                    parameter_rows.append({"epoch": epoch, "batch_index": batch_index, "module": name,
                                           "parameter_norm": module_norm(module), "parameter_delta_norm": float(torch.linalg.vector_norm(after[name] - before[name]))})
                recorder.end()
            train_total_losses.append(float(loss.detach().cpu()))
            train_property_losses.append(losses.detach().cpu().numpy())
            train_predictions.append(pred.detach().cpu().reshape(-1, 4).numpy())
            train_labels.append(label.detach().cpu().reshape(-1, 4).numpy())
            train_sources.append(batches[0].sample_uid.detach().cpu().numpy().reshape(-1))
            first_step_done = True
        if not first_step_done:
            raise RuntimeError("Training loader was empty")
        train_pred, train_label, train_source = np.vstack(train_predictions), np.vstack(train_labels), np.concatenate(train_sources)
        train_loss_by_target = np.mean(np.abs(train_pred - train_label), axis=0)
        train_loss = float(train_loss_by_target.sum())
        val_pred, val_label, val_source, val_loss, val_loss_by_target = evaluate(model, loader_groups, "val", device, recorder, epoch)
        lr = float(scheduler.get_last_lr()[0])
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy_state(model)
            best_val_prediction = (val_pred.copy(), val_label.copy(), val_source.copy())
            torch.save({"model_state": best_state, "epoch": epoch, "validation_loss": val_loss, "seed": int(cfg.seed), "fold": args.fold}, run_dir / "checkpoints" / f"best_candidate_epoch_{epoch}.pt")
        if early_reference == math.inf:
            early_counter = 0
            early_reference = val_loss
        elif val_loss < early_reference - float(cfg.train.early_stop_min_delta):
            early_reference = val_loss
            early_counter = 0
        else:
            early_counter += 1
        epoch_metric_rows = (
            target_metrics(train_pred, train_label, "train", epoch, train_loss, lr, early_counter, best_epoch, best_loss)
            + target_metrics(val_pred, val_label, "val", epoch, val_loss, lr, early_counter, best_epoch, best_loss)
        )
        epoch_rows.extend(epoch_metric_rows)
        epoch_loss_rows = []
        for split, losses in (("train", train_loss_by_target), ("val", val_loss_by_target)):
            for target, value in zip(TARGETS, losses):
                row = {"epoch": epoch, "split": split, "target": target, "loss_normalized": float(value), "total_loss_normalized": train_loss if split == "train" else val_loss}
                loss_rows.append(row)
                epoch_loss_rows.append(row)
        epoch_head_rows = [
            {"epoch": epoch, "split": "train", "target": target, "prediction_mean": float(train_pred[:, i].mean() * 100), "prediction_std": float(train_pred[:, i].std(ddof=1) * 100), "output_bias": float(core_model.FC_layers[2].bias.detach().cpu()[i] * 100), "head_weight_norm": float(torch.linalg.vector_norm(core_model.FC_layers[2].weight.detach().cpu()[i]))}
            for i, target in enumerate(TARGETS)
        ] + [
            {"epoch": epoch, "split": "val", "target": target, "prediction_mean": float(val_pred[:, i].mean() * 100), "prediction_std": float(val_pred[:, i].std(ddof=1) * 100), "output_bias": float(core_model.FC_layers[2].bias.detach().cpu()[i] * 100), "head_weight_norm": float(torch.linalg.vector_norm(core_model.FC_layers[2].weight.detach().cpu()[i]))}
            for i, target in enumerate(TARGETS)
        ]
        head_rows.extend(epoch_head_rows)
        last_result = (train_pred, train_label, train_source, val_pred, val_label, val_source, train_loss, val_loss)
        print(json.dumps({
            "progress": "epoch", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "best_epoch": best_epoch, "best_validation_loss": best_loss, "lr": lr,
            "early_stopping_counter": early_counter,
        }), flush=True)
        scheduler.step()
        # Persist the completed epoch before its resumable state.  This is
        # intentionally outside autograd/model code, so output durability
        # cannot alter the original training mathematics.
        append_frame(pd.DataFrame(epoch_metric_rows), run_dir / "epoch_metrics.csv")
        append_frame(
            pd.DataFrame(epoch_metric_rows).loc[:, ["epoch", "split", "target", "prediction_std", "target_std", "std_ratio", "prediction_mean"]],
            run_dir / "prediction_std_by_epoch.csv",
        )
        append_frame(pd.DataFrame(epoch_loss_rows), run_dir / "target_loss_by_epoch.csv")
        append_frame(pd.DataFrame(recorder.activation_rows[activation_start:]), run_dir / "branch_activation_stats.csv")
        append_frame(pd.DataFrame(recorder.fusion_rows[fusion_start:]), run_dir / "fusion_weight_history.csv")
        append_frame(pd.DataFrame(gradient_rows[gradient_start:]), run_dir / "gradient_norm_history.csv")
        append_frame(pd.DataFrame(parameter_rows[parameter_start:]), run_dir / "parameter_norm_history.csv")
        append_frame(pd.DataFrame(head_rows[head_start:]), run_dir / "head_output_history.csv")
        append_frame(pd.DataFrame(recorder.anomaly_rows[anomaly_start:]), run_dir / "numerical_anomalies.csv")
        torch.save({
            "next_epoch": epoch + 1, "model_state": copy_state(model), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "best_loss": best_loss, "best_epoch": best_epoch,
            "best_state": best_state, "best_val_prediction": best_val_prediction, "early_counter": early_counter,
            "early_reference": early_reference, "rng_state": rng_state(),
        }, run_dir / "resume_state.pt")
        if early_counter >= int(cfg.train.early_stop_patience) or epoch == execution_max_epoch - 1:
            completed = True
            break
        if args.chunk_epochs is not None and epoch - start_epoch + 1 >= args.chunk_epochs:
            break

    if best_state is None or best_val_prediction is None or last_result is None:
        raise RuntimeError("No best checkpoint candidate was produced")
    if not completed:
        recorder.close()
        print(json.dumps({"progress": "chunk_complete", "next_epoch": epoch + 1, "best_epoch": best_epoch, "best_validation_loss": best_loss}), flush=True)
        return

    last_epoch = epoch
    checkpoint_path = run_dir / "checkpoints" / "selected_best.pt"
    torch.save({"model_state": best_state, "epoch": best_epoch, "validation_loss": best_loss, "seed": int(cfg.seed), "fold": args.fold}, checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model_state"], strict=True)
    final_frames = []
    reload_diffs = []
    for split in ("train", "val", "test"):
        pred, label, source, loss, _ = evaluate(model, loader_groups, split, device, None, last_epoch)
        final_frames.append(make_prediction_frame(pred, label, source, manifest, split, int(best_epoch), str(checkpoint_path)))
        if split == "val":
            reference = best_val_prediction[0]
            reload_diffs.append(float(np.max(np.abs(pred - reference))))
    final_predictions = pd.concat(final_frames, ignore_index=True)
    final_predictions.to_csv(run_dir / "best_predictions.csv", index=False)
    epoch_metrics = pd.read_csv(run_dir / "epoch_metrics.csv")
    activation = pd.read_csv(run_dir / "branch_activation_stats.csv")
    fusion = pd.read_csv(run_dir / "fusion_weight_history.csv")
    gradients = pd.read_csv(run_dir / "gradient_norm_history.csv")
    parameters = pd.read_csv(run_dir / "parameter_norm_history.csv")
    anomalies = pd.read_csv(run_dir / "numerical_anomalies.csv")
    timeline = collapse_timeline(epoch_metrics, activation, fusion, gradients, parameters, anomalies, initial_lr)
    timeline.to_csv(run_dir / "collapse_timeline.csv", index=False)
    summary = {
        "fold": args.fold, "seed": int(cfg.seed), "last_epoch": int(last_epoch), "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss), "initial_lr": initial_lr,
        "last_lr_before_stop": float(epoch_metrics.lr.max()), "reload_best_val_max_abs_difference": max(reload_diffs, default=math.nan),
        "early_stopping_counter_at_stop": int(early_counter), "scheduler": str(cfg.optim.scheduler),
        "early_stopping": args.early_stopping, "has_nan_or_inf": bool((anomalies.nan_count.gt(0) | anomalies.inf_count.gt(0)).any()),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    recorder.close()
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
