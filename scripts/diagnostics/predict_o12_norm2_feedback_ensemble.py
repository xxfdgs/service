#!/usr/bin/env python3
"""Infer one frozen O12 target-group ensemble on feedback tables.

Predictions are the unweighted mean of O12_<target-group>_split100 through
split109.  Core4 predictions are restored from the loader's historical /100
label scale before metrics and plots are written.
When labels are present, they are replaced by zeros in the loader-only input
and used only after inference for metrics and true-vs-predicted scatter plots.
Unlabelled tables produce predictions and per-checkpoint uncertainty only.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphgps  # noqa: F401,E402
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402

from scripts.diagnostics.predict_o12_o22_feedback_ensemble import (  # noqa: E402
    MORDRED_11,
    SMILES_COLUMNS,
    build_feedback_mordred_lookup,
)


TARGET_GROUPS = {
    "core4": (
        ["EE_before", "EE_after", "Aerosolization_Efficiency",
         "mRNA_Recovery_Efficiency"],
        [100.0, 100.0, 100.0, 100.0],
    ),
    "norm2": (["Norm_before", "Norm_after"], [1.0, 1.0]),
}
ALL_LABELS = [
    "EE_before", "EE_after", "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency", "Norm_before", "Norm_after",
]
REQUIRED_INPUT_COLUMNS = {"ID", *SMILES_COLUMNS, "mol%_IL", "mol%_HL", "mol%_Chol",
                          "mol%_PEG", "mol%_Fifth"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_feedback(frame: pd.DataFrame, source: Path, output: Path,
                   targets: list[str]) -> tuple[pd.DataFrame, Path, bool]:
    missing = REQUIRED_INPUT_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} misses required columns: {sorted(missing)}")
    original = frame.copy()
    original["ID"] = original["ID"].astype(str)
    if original.ID.isna().any() or original.ID.duplicated().any() or len(original) < 3:
        raise ValueError(f"{source} requires at least three unique non-null IDs.")
    label_columns = set(targets).intersection(original.columns)
    if label_columns and label_columns != set(targets):
        raise ValueError(f"{source} has only part of the requested labels: {sorted(label_columns)}")
    has_labels = label_columns == set(targets)
    if has_labels and original[targets].isna().any().any():
        raise ValueError(f"{source} has missing requested target labels.")
    staged = original.copy()
    for label in ALL_LABELS:
        staged[label] = 0.0
    staged_path = output / "feedback_model_input_labels_zeroed.csv"
    staged.to_csv(staged_path, index=False)
    return original, staged_path, has_labels


def write_manifest(frame: pd.DataFrame, output: Path) -> Path:
    split = np.full(len(frame), "test", dtype=object)
    split[0], split[1] = "train", "val"  # Loader contract; all rows are inferred.
    manifest = output / "feedback_loader_manifest.csv"
    pd.DataFrame({"ID": frame.ID.astype(str), "split": split,
                  "split_order": np.arange(len(frame), dtype=int)}).to_csv(manifest, index=False)
    return manifest


def build_context(config_path: Path, model_input: Path, manifest: Path,
                  cache_dir: Path, mordred_lookup: Path, targets: list[str]):
    set_cfg_gps(cfg)
    cfg.set_new_allowed(True)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(config_path.resolve()), opts=[]))
    if cfg.model.type != "OneHotEmbedGPS" or int(cfg.property_num) != len(targets):
        raise RuntimeError(
            f"Expected a {len(targets)}-target OneHotEmbedGPS config: {config_path}"
        )
    if not str(cfg.component_vocab_source).strip():
        raise RuntimeError(f"Checkpoint config misses the original vocabulary source: {config_path}")
    cfg.read_csv = str(model_input.resolve())
    cfg.dataset.dir = str(cache_dir.resolve())
    # The effective checkpoint configuration determines the model width,
    # GraphGPS depth, positional encoding, and component vocabulary.  Do not
    # encode one particular learning-rate or hidden-size experiment here.
    cfg.dataset.cache_tag = f"o12_feedback_{config_path.parent.name}"
    cfg.dataset.cache_refresh = True
    cfg.dataset.diagnostic_split_path = str(manifest.resolve())
    cfg.dataset.diagnostic_id_column = "ID"
    cfg.dataset.diagnostic_manifest_id_column = "ID"
    cfg.mordred_feature_path = str(mordred_lookup.resolve())
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "cache_build.log").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            loaders = create_loader_5()
    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    return create_model_gps().to(device), loaders, device


def predict(model: torch.nn.Module, loaders, device: torch.device, rows: int,
            target_count: int) -> np.ndarray:
    predicted: list[tuple[int, np.ndarray]] = []
    model.eval()
    with torch.no_grad():
        for loader_index in range(3):
            for batches in zip(*[group[loader_index] for group in loaders]):
                for suffix, batch in zip(("", "_2", "_3", "_4", "_5"), batches):
                    batch.split = "feedback" + suffix
                    batch.to(device)
                output, _ = model(*batches)
                values = output.detach().cpu().reshape(-1, target_count).numpy()
                source = batches[0].sample_uid.detach().cpu().numpy().reshape(-1)
                predicted.extend((int(index), value) for index, value in zip(source, values))
    predicted.sort(key=lambda item: item[0])
    if [index for index, _ in predicted] != list(range(rows)):
        raise RuntimeError("Predicted rows do not align with feedback source rows.")
    return np.vstack([value for _, value in predicted])


def restore_prediction(prediction: np.ndarray, settings: dict[str, object],
                       targets: list[str], expected_scales: list[float]) -> np.ndarray:
    """Convert model-space output back to the CSV's original target units."""
    scales = settings.get("target_scales")
    if not isinstance(scales, list) or len(scales) != len(targets):
        raise RuntimeError(f"Checkpoint misses target_scales for {targets}: {scales!r}")
    scales = np.asarray(scales, dtype=float)
    if not np.allclose(scales, np.asarray(expected_scales, dtype=float)):
        raise RuntimeError(
            f"Unexpected target scales {scales.tolist()}; expected {expected_scales}"
        )
    restored = np.asarray(prediction, dtype=float).copy()
    scaler = settings.get("target_scaler", {"type": "identity"})
    if not isinstance(scaler, dict):
        raise RuntimeError("Checkpoint target_scaler metadata is invalid")
    if scaler.get("type", "identity") == "zscore":
        mean, std = np.asarray(scaler.get("mean"), dtype=float), np.asarray(scaler.get("std"), dtype=float)
        if len(mean) != len(targets) or len(std) != len(targets) or np.any(std <= 0):
            raise RuntimeError(f"Invalid target z-score metadata: {scaler}")
        restored = restored * std + mean
    elif scaler.get("type", "identity") != "identity":
        raise RuntimeError(f"Unsupported target scaler: {scaler.get('type')!r}")
    transform = settings.get("target_transform", "identity")
    if transform == "log1p":
        restored = np.maximum(np.expm1(restored), 0.0)
    elif transform != "identity":
        raise RuntimeError(f"Unsupported target transform: {transform!r}")
    return restored * scales


def metrics(frame: pd.DataFrame, prediction: np.ndarray, targets: list[str]) -> pd.DataFrame:
    rows = []
    for index, target in enumerate(targets):
        truth = frame[target].to_numpy(float)
        values = prediction[:, index]
        rows.append({"target": target, "n": len(frame),
                     "mae": mean_absolute_error(truth, values),
                     "rmse": mean_squared_error(truth, values) ** .5,
                     "r2": r2_score(truth, values) if np.std(truth) else np.nan})
    return pd.DataFrame(rows)


def scatter_plot(frame: pd.DataFrame, prediction: np.ndarray, metric: pd.DataFrame,
                 output: Path, target_group: str, targets: list[str]) -> None:
    """Write one true-vs-predicted plot for every target independently."""
    plots_dir = output / "scatter_by_target"
    plots_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for index, target in enumerate(targets):
        figure, axis = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
        values = prediction[:, index]
        truth = frame[target].to_numpy(float)
        lower = min(float(truth.min()), float(values.min()))
        upper = max(float(truth.max()), float(values.max()))
        padding = max((upper - lower) * .06, .1)
        limits = (lower - padding, upper + padding)
        summary = metric.loc[metric.target.eq(target)].iloc[0]
        axis.scatter(truth, values, s=34, alpha=.82, color=colors[index % len(colors)],
                     edgecolor="#222", linewidth=.35)
        axis.plot(limits, limits, "--", color="#d62728", linewidth=1.35, label="y = x")
        axis.set(xlabel="True value", ylabel="Predicted value", xlim=limits, ylim=limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=.25)
        axis.legend(loc="upper left", fontsize=8)
        axis.set_title(f"O12 {target_group}: {target}\nMAE = {summary.mae:.3f}, R² = {summary.r2:.3f}")
        figure.savefig(plots_dir / f"{target}_true_vs_pred.png", dpi=180, bbox_inches="tight")
        figure.savefig(plots_dir / f"{target}_true_vs_pred.pdf", bbox_inches="tight")
        plt.close(figure)


def run_dataset(source: Path, model_root: Path, output_root: Path, metadata: Path,
                target_group: str) -> dict[str, object]:
    targets, expected_scales = TARGET_GROUPS[target_group]
    output = output_root / source.stem
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, dtype={"ID": str})
    original, staged_path, has_labels = stage_feedback(frame, source, output, targets)
    manifest = write_manifest(original, output)
    lookup = output / "mordred11_feedback_standardized.csv"
    mordred_summary = build_feedback_mordred_lookup(original, metadata, lookup)
    specs = []
    expected_signature: dict[str, object] | None = None
    for seed in range(100, 110):
        run_dir = model_root / f"O12_{target_group}_split{seed}"
        checkpoint, config, settings_path = run_dir / "checkpoints/selected_best.pt", run_dir / "effective_config.yaml", run_dir / "run_settings.json"
        if not all(path.is_file() for path in (checkpoint, config, settings_path)):
            raise FileNotFoundError(f"Incomplete {target_group} checkpoint: {run_dir}")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if settings.get("loss_targets") != targets:
            raise RuntimeError(f"Checkpoint target group does not match {target_group}: {run_dir}")
        # Arbitrary widths and vocabulary sizes are supported: the first
        # checkpoint's effective_config.yaml constructs the model.  Only the
        # ten members of a single averaged ensemble must share a shape.
        signature = {
            key: settings.get(key)
            for key in (
                "model_type", "gps_layers", "graph_hidden_dim", "gnn_inner_dim",
                "rwse_dim", "component_vocab_sizes", "component_vocab_strict",
                "fusion_type", "head_type", "fusion_hidden_dim", "head_hidden_dim",
            )
        }
        if expected_signature is None:
            expected_signature = signature
        elif signature != expected_signature:
            raise RuntimeError(
                "All ten checkpoints in one ensemble must have the same model shape and "
                f"vocabulary metadata. Expected {expected_signature}, found {signature} "
                f"in {run_dir}. Run different model sizes as separate --model-root values."
            )
        specs.append((seed, checkpoint, config, settings))
    model, loaders, device = build_context(
        specs[0][2], staged_path, manifest, output / "cache", lookup, targets
    )
    all_predictions, checkpoint_metrics, long_rows = [], [], []
    for seed, checkpoint_path, _, settings in specs:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        prediction = restore_prediction(
            predict(model, loaders, device, len(original), len(targets)),
            settings, targets, expected_scales,
        )
        all_predictions.append(prediction)
        if has_labels:
            per_checkpoint = metrics(original, prediction, targets)
            per_checkpoint.insert(0, "split_seed", seed)
            per_checkpoint.insert(1, "checkpoint", str(checkpoint_path.resolve()))
            checkpoint_metrics.append(per_checkpoint)
        for target_index, target in enumerate(targets):
            truth = (original[target].to_numpy(float)
                     if has_labels else np.full(len(original), np.nan))
            long_rows.extend({"sample_id": sample_id, "split_seed": seed, "target": target,
                              "y_true": float(actual), "y_pred": float(value)}
                             for sample_id, actual, value in zip(
                                 original.ID, truth, prediction[:, target_index]))
    stacked = np.stack(all_predictions)
    mean, std = stacked.mean(axis=0), stacked.std(axis=0, ddof=0)
    # The older feedback CSV carries unrelated historical ``pred_*_average``
    # columns (and Excel-export ``Unnamed:`` columns).  Do not mix them with
    # this frozen ensemble's clearly named output columns.
    ensemble = original.loc[:, [column for column in original.columns
                                if not column.startswith("pred_")
                                and not column.startswith("Unnamed:")]].copy()
    for index, target in enumerate(targets):
        ensemble[f"pred_{target}_mean"] = mean[:, index]
        ensemble[f"pred_{target}_std_10models"] = std[:, index]
    ensemble.to_csv(output / "ensemble_mean_predictions.csv", index=False)
    pd.DataFrame(long_rows).to_csv(output / "predictions_by_checkpoint.csv", index=False)
    if has_labels:
        ensemble_metric = metrics(original, mean, targets)
        ensemble_metric.to_csv(output / "metrics_ensemble.csv", index=False)
        all_checkpoint_metrics = pd.concat(checkpoint_metrics, ignore_index=True)
        all_checkpoint_metrics.to_csv(output / "metrics_by_checkpoint.csv", index=False)
        per_target_dir = output / "metrics_by_target"
        per_target_dir.mkdir(parents=True, exist_ok=True)
        for target in targets:
            ensemble_metric.loc[ensemble_metric["target"].eq(target)].to_csv(
                per_target_dir / f"{target}_ensemble.csv", index=False
            )
            all_checkpoint_metrics.loc[all_checkpoint_metrics["target"].eq(target)].to_csv(
                per_target_dir / f"{target}_by_checkpoint.csv", index=False
            )
        scatter_plot(original, mean, ensemble_metric, output, target_group, targets)
    (output / "provenance.json").write_text(json.dumps({
        "feedback_csv": str(source.resolve()), "feedback_sha256": sha256(source), "rows": len(original),
        "target_group": target_group, "targets": targets,
        "target_scales": expected_scales,
        "ensemble": f"unweighted arithmetic mean over O12_{target_group}_split100...109",
        "has_ground_truth_labels": has_labels,
        "label_use": ("labels replaced by zero in model input and used only for metrics/plots"
                      if has_labels else "no labels supplied; prediction-only output"),
        "mordred_features": MORDRED_11, "mordred_lookup": str(lookup.resolve()),
        "mordred_lookup_sha256": sha256(lookup), "mordred_summary": mordred_summary,
        "model_signature": expected_signature,
        "checkpoints": [{"split_seed": seed, "path": str(path.resolve()), "sha256": sha256(path)}
                        for seed, path, _, _ in specs],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"dataset": source.name, "rows": len(original),
               "has_ground_truth_labels": has_labels, "output": str(output)}
    if has_labels:
        summary.update({f"{row.target}_mae": row.mae
                        for row in ensemble_metric.itertuples(index=False)})
        summary.update({f"{row.target}_r2": row.r2
                        for row in ensemble_metric.itertuples(index=False)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path,
                        default=ROOT / "results/input_graphgps_optimization/o12_multitask_seed100_109_lr01")
    parser.add_argument("--target-group", choices=tuple(TARGET_GROUPS), default="norm2",
                        help="Checkpoint target group to load.")
    parser.add_argument("--feedback-files", type=Path, nargs="+", default=[
        ROOT / "datasets_lrx/raw/feedback/20260703_validation.csv",
        ROOT / "datasets_lrx/raw/feedback/new_validation.csv",
    ])
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/input_graphgps_optimization/o12_multitask_seed100_109_lr01/feedback_norm2_ensemble")
    parser.add_argument("--mordred-metadata", type=Path,
                        default=ROOT / "results/input_graphgps_optimization/features/mordred11_train_standardized.json")
    args = parser.parse_args()
    if not args.mordred_metadata.is_file():
        raise FileNotFoundError(f"Missing Mordred scaler metadata: {args.mordred_metadata}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for source in args.feedback_files:
        if not source.is_file():
            raise FileNotFoundError(f"Missing feedback CSV: {source}")
        summary.append(run_dataset(source.resolve(), args.model_root.resolve(), args.output_root.resolve(),
                                   args.mordred_metadata.resolve(), args.target_group))
    pd.DataFrame(summary).to_csv(args.output_root / "run_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
