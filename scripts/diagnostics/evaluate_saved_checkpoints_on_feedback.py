#!/usr/bin/env python3
"""Evaluate saved O12 and multi-task baseline checkpoints on labelled feedback.

No feedback label contributes to a model input, checkpoint choice, ensemble
weight, calibration, or parameter update.  Labels are accessed only after a
forward pass to calculate reporting metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphgps  # noqa: F401,E402
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402

from scripts.diagnostics.predict_o12_o22_feedback_ensemble import (  # noqa: E402
    MORDRED_11, SMILES_COLUMNS, build_feedback_mordred_lookup,
)
from scripts.diagnostics.run_five_component_multitask_baseline import (  # noqa: E402
    FiveComponentBaseline, FormulationDataset, TARGET_GROUPS as BASELINE_TARGET_GROUPS,
    collate_formulations, predict as predict_baseline,
)


CORE_TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]
NORM_TARGETS = ["Norm_before", "Norm_after"]
ALL_TARGETS = CORE_TARGETS + NORM_TARGETS


@dataclass(frozen=True)
class CheckpointSpec:
    family: str
    model: str
    target_group: str
    targets: tuple[str, ...]
    split_seed: int
    run_dir: Path
    checkpoint: Path
    config: Path | None
    settings: dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_feedback_manifest(frame: pd.DataFrame, output: Path) -> None:
    if frame.ID.isna().any() or frame.ID.astype(str).duplicated().any():
        raise ValueError("Feedback ID must be present and unique.")
    if len(frame) < 3:
        raise ValueError("Feedback inference requires at least three rows.")
    split = np.full(len(frame), "test", dtype=object)
    split[:2] = ("train", "val")
    pd.DataFrame({"ID": frame.ID.astype(str), "split": split,
                  "split_order": np.arange(len(frame), dtype=int)}).to_csv(output, index=False)


def metric_values(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if len(truth) < 2 or not np.std(truth) or not np.std(prediction):
        pearson = spearman = r2 = math.nan
    else:
        pearson = float(pearsonr(truth, prediction).statistic)
        spearman = float(spearmanr(truth, prediction).statistic)
        r2 = float(r2_score(truth, prediction))
    return {"n": int(len(truth)), "mae": float(mean_absolute_error(truth, prediction)),
            "rmse": float(mean_squared_error(truth, prediction) ** .5), "r2": r2,
            "pearson": pearson, "spearman": spearman}


def graphgps_context(config_path: Path, feedback_path: Path, manifest_path: Path,
                     cache_dir: Path, mordred_lookup: Path):
    set_cfg_gps(cfg)
    cfg.set_new_allowed(True)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(config_path.resolve()), opts=[]))
    cfg.read_csv = str(feedback_path.resolve())
    cfg.dataset.dir = str(cache_dir.resolve())
    cfg.dataset.cache_tag = f"feedback_checkpoint_eval_{config_path.parent.name}"
    cfg.dataset.cache_refresh = True
    cfg.dataset.diagnostic_split_path = str(manifest_path.resolve())
    cfg.dataset.diagnostic_id_column = "ID"
    cfg.dataset.diagnostic_manifest_id_column = "ID"
    cfg.mordred_feature_path = str(mordred_lookup.resolve())
    if not str(cfg.component_vocab_source).strip():
        raise RuntimeError(f"Missing original component vocabulary source: {config_path}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "cache_build.log").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            loaders = create_loader_5()
    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    return create_model_gps().to(device), loaders, device


def predict_graphgps(model: torch.nn.Module, loaders, device: torch.device,
                     targets: tuple[str, ...], scale: float, rows: int) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for loader_index in range(3):
            for batches in zip(*[group[loader_index] for group in loaders]):
                for suffix, batch in zip(("", "_2", "_3", "_4", "_5"), batches):
                    batch.split = "feedback" + suffix
                    batch.to(device)
                output, _ = model(*batches)
                values = output.detach().cpu().reshape(-1, len(targets)).numpy() * scale
                source = batches[0].sample_uid.detach().cpu().numpy().reshape(-1)
                result.extend((int(index), value) for index, value in zip(source, values))
    result.sort(key=lambda item: item[0])
    if [index for index, _ in result] != list(range(rows)):
        raise RuntimeError("GraphGPS feedback predictions do not align with source rows.")
    return np.vstack([value for _, value in result])


def graphgps_specs(root: Path, family: str, target_group: str, targets: tuple[str, ...],
                   single_target: str | None = None) -> list[CheckpointSpec]:
    specs = []
    for split_seed in range(100, 110):
        if single_target is None:
            run_dir = root / f"O12_split{split_seed}"
        else:
            run_dir = root / f"O12_{single_target}_split{split_seed}"
        checkpoint = run_dir / "checkpoints" / "selected_best.pt"
        config = run_dir / "effective_config.yaml"
        settings_path = run_dir / "run_settings.json"
        if not all(path.is_file() for path in (checkpoint, config, settings_path)):
            raise FileNotFoundError(f"Missing completed checkpoint artifacts in {run_dir}")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if single_target is not None and settings.get("single_target") != single_target:
            raise RuntimeError(f"Single-task target mismatch: {run_dir}")
        specs.append(CheckpointSpec(family, "O12", target_group, targets, split_seed,
                                    run_dir, checkpoint, config, settings))
    return specs


def baseline_specs(root: Path) -> list[CheckpointSpec]:
    specs = []
    for model in ("GCN", "GIN", "MPNN", "Transformer", "MLP"):
        for target_group, (targets, _) in BASELINE_TARGET_GROUPS.items():
            for split_seed in range(100, 110):
                run_dir = root / f"{model}_{target_group}_split{split_seed}"
                checkpoint, settings_path = run_dir / "selected_best.pt", run_dir / "run_settings.json"
                if not all(path.is_file() for path in (checkpoint, settings_path)):
                    raise FileNotFoundError(f"Missing completed checkpoint artifacts in {run_dir}")
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
                specs.append(CheckpointSpec("multitask_baseline", model, target_group, tuple(targets),
                                            split_seed, run_dir, checkpoint, None, settings))
    return specs


def record(spec: CheckpointSpec, prediction: np.ndarray, feedback: pd.DataFrame,
           prediction_rows: list[dict], metric_rows: list[dict]) -> None:
    for target_index, target in enumerate(spec.targets):
        truth = feedback[target].to_numpy(float)
        values = prediction[:, target_index]
        metric_rows.append({"family": spec.family, "model": spec.model,
                            "target_group": spec.target_group, "target": target,
                            "split_seed": spec.split_seed, "run": spec.run_dir.name,
                            "checkpoint": str(spec.checkpoint), **metric_values(truth, values)})
        prediction_rows.extend({"family": spec.family, "model": spec.model,
                                "target_group": spec.target_group, "target": target,
                                "split_seed": spec.split_seed, "run": spec.run_dir.name,
                                "checkpoint": str(spec.checkpoint), "sample_id": str(sample_id),
                                "y_true": float(true_value), "y_pred": float(predicted_value)}
                               for sample_id, true_value, predicted_value in zip(feedback.ID, truth, values))


def evaluate_graphgps(spec_groups: list[list[CheckpointSpec]], feedback: pd.DataFrame,
                      feedback_path: Path, manifest_path: Path, mordred_lookup: Path,
                      cache_root: Path, prediction_rows: list[dict], metric_rows: list[dict]) -> None:
    for group_index, specs in enumerate(spec_groups):
        if not specs:
            continue
        scale = 100.0 if specs[0].targets[0] in CORE_TARGETS else 1.0
        model, loaders, device = graphgps_context(specs[0].config, feedback_path, manifest_path,
                                                   cache_root / f"graphgps_group_{group_index}", mordred_lookup)
        for spec in specs:
            checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            prediction = predict_graphgps(model, loaders, device, spec.targets, scale, len(feedback))
            record(spec, prediction, feedback, prediction_rows, metric_rows)


def evaluate_baselines(specs: list[CheckpointSpec], feedback: pd.DataFrame,
                       prediction_rows: list[dict], metric_rows: list[dict]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grouped: dict[tuple[str, str], list[CheckpointSpec]] = {}
    for spec in specs:
        grouped.setdefault((spec.model, spec.target_group), []).append(spec)
    for (model_name, target_group), group_specs in grouped.items():
        targets, scale = BASELINE_TARGET_GROUPS[target_group]
        dataset = FormulationDataset(feedback, list(range(len(feedback))), targets, scale)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_formulations)
        settings = group_specs[0].settings
        model = FiveComponentBaseline(model_name, len(targets), int(settings["hidden_dim"]),
                                      int(settings["layers"]), float(settings["dropout"])).to(device)
        for spec in group_specs:
            checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            prediction, _, source = predict_baseline(model, loader, device)
            order = np.argsort(source)
            if source[order].tolist() != list(range(len(feedback))):
                raise RuntimeError(f"Baseline feedback predictions misalign: {spec.run_dir}")
            record(spec, prediction[order] * scale, feedback, prediction_rows, metric_rows)


def write_summaries(output: Path, prediction_rows: list[dict], metric_rows: list[dict]) -> None:
    prediction = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    prediction.to_csv(output / "feedback_predictions_by_checkpoint.csv", index=False)
    metrics.to_csv(output / "feedback_metrics_by_checkpoint.csv", index=False)
    target_summary = metrics.groupby(["family", "model", "target_group", "target"], as_index=False).agg(
        checkpoints=("run", "count"), mean_mae=("mae", "mean"), std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"), mean_r2=("r2", "mean"), std_r2=("r2", "std"),
        mean_pearson=("pearson", "mean"), std_pearson=("pearson", "std"),
        mean_spearman=("spearman", "mean"), std_spearman=("spearman", "std"),
    )
    target_summary.to_csv(output / "feedback_metrics_target_average.csv", index=False)
    macro_per_checkpoint = metrics.groupby(["family", "model", "target_group", "split_seed", "run"], as_index=False).agg(
        mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"),
        mean_pearson=("pearson", "mean"), mean_spearman=("spearman", "mean"))
    macro_per_checkpoint.to_csv(output / "feedback_metrics_macro_by_checkpoint.csv", index=False)
    macro_per_checkpoint.groupby(["family", "model", "target_group"], as_index=False).agg(
        checkpoints=("run", "count"), mean_mae=("mean_mae", "mean"), std_mae=("mean_mae", "std"),
        mean_r2=("mean_r2", "mean"), std_r2=("mean_r2", "std"),
        mean_pearson=("mean_pearson", "mean"), mean_spearman=("mean_spearman", "mean"),
    ).to_csv(output / "feedback_metrics_macro_average.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-csv", type=Path,
                        default=ROOT / "datasets_lrx/raw/feedback/20260703_validation.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/input_graphgps_optimization/feedback_checkpoint_benchmark_seed100_109")
    parser.add_argument("--mordred-metadata", type=Path,
                        default=ROOT / "results/input_graphgps_optimization/features/mordred11_train_standardized.json")
    parser.add_argument("--families", nargs="+", choices=("multitask_o12", "single_task_o12", "multitask_baseline"),
                        default=("multitask_o12", "single_task_o12", "multitask_baseline"))
    args = parser.parse_args()
    feedback_path, output = args.feedback_csv.resolve(), args.output_dir.resolve()
    feedback = pd.read_csv(feedback_path)
    required = {"ID", *SMILES_COLUMNS, *ALL_TARGETS}
    if missing := required.difference(feedback.columns):
        raise ValueError(f"Feedback CSV misses required columns: {sorted(missing)}")
    if feedback[ALL_TARGETS].isna().any().any():
        raise ValueError("Feedback labels must be complete for this metric report.")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "feedback_loader_manifest.csv"
    make_feedback_manifest(feedback, manifest_path)
    mordred_lookup = output / "mordred11_feedback_standardized.csv"
    mordred_summary = build_feedback_mordred_lookup(feedback, args.mordred_metadata.resolve(), mordred_lookup)
    graph_groups: list[list[CheckpointSpec]] = []
    if "multitask_o12" in args.families:
        graph_groups.extend([
            graphgps_specs(ROOT / "results/input_graphgps_optimization/five_split_runs", "multitask_o12", "core4", tuple(CORE_TARGETS)),
            graphgps_specs(ROOT / "results/input_graphgps_optimization/norm2_five_split_runs", "multitask_o12", "norm2", tuple(NORM_TARGETS)),
        ])
    if "single_task_o12" in args.families:
        single_root = ROOT / "results/input_graphgps_optimization/single_task_o12_six_targets"
        graph_groups.extend(graphgps_specs(single_root, "single_task_o12", "single", (target,), target)
                            for target in ALL_TARGETS)
    baseline = (baseline_specs(ROOT / "results/input_graphgps_optimization/multitask_baselines_seed100_109")
                if "multitask_baseline" in args.families else [])
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    evaluate_graphgps(graph_groups, feedback, feedback_path, manifest_path, mordred_lookup,
                      output / "cache", prediction_rows, metric_rows)
    if baseline:
        evaluate_baselines(baseline, feedback, prediction_rows, metric_rows)
    write_summaries(output, prediction_rows, metric_rows)
    (output / "provenance.json").write_text(json.dumps({
        "feedback_csv": str(feedback_path), "feedback_sha256": sha256(feedback_path),
        "feedback_rows": int(len(feedback)), "mordred_features": MORDRED_11,
        "mordred_lookup": str(mordred_lookup), "mordred_lookup_sha256": sha256(mordred_lookup),
        "mordred_summary": mordred_summary,
        "families": list(args.families), "checkpoint_counts": {
            "graphgps": int(sum(len(group) for group in graph_groups)), "baseline": int(len(baseline))},
        "feedback_labels_used_only_for": "post-prediction metrics",
        "feedback_calibration_or_finetuning": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(pd.read_csv(output / "feedback_metrics_macro_average.csv").to_string(index=False))


if __name__ == "__main__":
    main()
