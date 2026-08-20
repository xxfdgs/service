#!/usr/bin/env python3
"""Validate and summarize frozen selected-checkpoint predictions on matching val sets."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASELINE_PATTERN = re.compile(r"^(GCN|GIN|MPNN|Transformer|MLP)_(core4|norm2)_split(10[0-9])$")
SINGLE_PATTERN = re.compile(r"^O12_(.+)_split(10[0-9])$")


def correlation(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or not np.std(truth) or not np.std(prediction):
        return math.nan
    return float(function(truth, prediction).statistic)


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    truth, prediction = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
    return {
        "n": int(len(frame)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** .5),
        "r2": float(r2_score(truth, prediction)) if np.std(truth) else math.nan,
        "pearson": correlation(pearsonr, truth, prediction),
        "spearman": correlation(spearmanr, truth, prediction),
    }


def expected_validation_ids(settings: dict, manifest_root: Path) -> tuple[set[str], Path]:
    manifest_path = Path(settings["split_manifest"]).resolve()
    if not manifest_path.is_file():
        local_path = manifest_root / manifest_path.name
        if not local_path.is_file():
            raise FileNotFoundError(
                f"Neither recorded nor local manifest exists: {manifest_path}, {local_path}")
        manifest_path = local_path.resolve()
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    expected = set(manifest.loc[manifest.split.eq("val"), "sample_id"].astype(str))
    if not expected:
        raise RuntimeError(f"No validation IDs in {manifest_path}")
    return expected, manifest_path


def collect_run(run_dir: Path, family: str, model: str, target_group: str,
                split_seed: int, manifest_root: Path, detail: list[dict],
                inventory: list[dict]) -> None:
    predictions_path = run_dir / "predictions.csv"
    summary_path = run_dir / "summary.json"
    settings_path = run_dir / "run_settings.json"
    common = {"family": family, "model": model, "target_group": target_group,
              "split_seed": split_seed, "run": run_dir.name}
    if not all(path.is_file() for path in (predictions_path, summary_path, settings_path)):
        inventory.append({**common, "status": "incomplete"})
        return
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_ids, manifest_path = expected_validation_ids(settings, manifest_root)
    predictions = pd.read_csv(predictions_path, dtype={"sample_id": str})
    validation = predictions.loc[predictions.split.eq("val")].copy()
    if validation.empty or validation.duplicated(["sample_id", "target"]).any():
        raise RuntimeError(f"Invalid validation predictions: {predictions_path}")
    checkpoint_values = validation.checkpoint.dropna().astype(str).unique()
    if len(checkpoint_values) != 1:
        raise RuntimeError(f"Validation rows do not reference one checkpoint: {predictions_path}")
    local_candidates = [Path(checkpoint_values[0]), run_dir / "selected_best.pt",
                        run_dir / "checkpoints" / "selected_best.pt"]
    selected_checkpoint = next((path.resolve() for path in local_candidates if path.is_file()), None)
    if selected_checkpoint is None:
        raise RuntimeError(f"No local selected checkpoint exists for {predictions_path}")
    targets = validation.target.drop_duplicates().tolist()
    for target, part in validation.groupby("target", sort=False):
        actual_ids = set(part.sample_id.astype(str))
        if actual_ids != expected_ids:
            raise RuntimeError(f"Validation IDs differ from manifest for {run_dir} / {target}")
        detail.append({**common, "target": target, "best_epoch": int(summary["best_epoch"]),
                       "selected_checkpoint": str(selected_checkpoint),
                       "recorded_checkpoint": checkpoint_values[0],
                       "split_manifest": str(manifest_path), **metric_values(part)})
    inventory.append({**common, "status": "completed", "targets": len(targets),
                      "n_validation": len(expected_ids), "best_epoch": int(summary["best_epoch"]),
                      "selected_checkpoint": str(selected_checkpoint),
                      "recorded_checkpoint": checkpoint_values[0],
                      "split_manifest": str(manifest_path)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--single-task-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    detail, inventory = [], []
    for run_dir in sorted(path for path in args.baseline_root.resolve().iterdir() if path.is_dir()):
        match = BASELINE_PATTERN.fullmatch(run_dir.name)
        if match:
            model, target_group, split_seed = match.groups()
            collect_run(run_dir, "multitask_baseline", model, target_group,
                        int(split_seed), args.manifest_root.resolve(), detail, inventory)
    for run_dir in sorted(path for path in args.single_task_root.resolve().iterdir() if path.is_dir()):
        match = SINGLE_PATTERN.fullmatch(run_dir.name)
        if match:
            target, split_seed = match.groups()
            collect_run(run_dir, "single_task_o12", "O12", target,
                        int(split_seed), args.manifest_root.resolve(), detail, inventory)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    detail_frame = pd.DataFrame(detail).sort_values(
        ["family", "model", "target_group", "split_seed", "target"])
    inventory_frame = pd.DataFrame(inventory).sort_values(
        ["family", "model", "target_group", "split_seed"])
    detail_frame.to_csv(output / "validation_metrics_by_checkpoint_target.csv", index=False)
    inventory_frame.to_csv(output / "validation_inference_inventory.csv", index=False)

    checkpoint_macro = detail_frame.groupby(
        ["family", "model", "target_group", "split_seed", "run", "best_epoch",
         "selected_checkpoint", "split_manifest"], as_index=False).agg(
        targets=("target", "count"), n_per_target=("n", "first"),
        mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"),
        mean_pearson=("pearson", "mean"), mean_spearman=("spearman", "mean"))
    checkpoint_macro.to_csv(output / "validation_mae_by_checkpoint.csv", index=False)

    checkpoint_macro.groupby(["family", "model", "target_group"], as_index=False).agg(
        completed_seeds=("split_seed", "nunique"), mean_validation_mae=("mean_mae", "mean"),
        std_validation_mae=("mean_mae", "std"), min_validation_mae=("mean_mae", "min"),
        max_validation_mae=("mean_mae", "max"), mean_validation_r2=("mean_r2", "mean"),
        std_validation_r2=("mean_r2", "std"), mean_validation_pearson=("mean_pearson", "mean"),
        mean_validation_spearman=("mean_spearman", "mean"),
    ).to_csv(output / "validation_mae_10seed_average.csv", index=False)

    detail_frame.groupby(["family", "model", "target_group", "target"], as_index=False).agg(
        completed_seeds=("split_seed", "nunique"), mean_validation_mae=("mae", "mean"),
        std_validation_mae=("mae", "std"), mean_validation_rmse=("rmse", "mean"),
        mean_validation_r2=("r2", "mean"), std_validation_r2=("r2", "std"),
        mean_validation_pearson=("pearson", "mean"),
        mean_validation_spearman=("spearman", "mean"),
    ).to_csv(output / "validation_metrics_target_10seed_average.csv", index=False)


if __name__ == "__main__":
    main()
