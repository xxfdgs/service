#!/usr/bin/env python3
"""Aggregate corresponding-test metrics for completed multi-task baselines."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RUN_PATTERN = re.compile(r"^(GCN|GIN|MPNN|Transformer|MLP)_(core4|norm2)_split(\d+)$")


def correlation(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or not np.std(truth) or not np.std(prediction):
        return float("nan")
    return float(function(truth, prediction).statistic)


def values(frame: pd.DataFrame) -> dict[str, float]:
    truth, prediction = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
    return {"n": len(frame), "mae": float(mean_absolute_error(truth, prediction)),
            "rmse": float(mean_squared_error(truth, prediction) ** .5), "r2": float(r2_score(truth, prediction)),
            "pearson": correlation(pearsonr, truth, prediction),
            "spearman": correlation(spearmanr, truth, prediction)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    details, inventory, validation_selection = [], [], []
    for run_dir in sorted(path for path in arguments.runs_root.resolve().iterdir() if path.is_dir()):
        match = RUN_PATTERN.fullmatch(run_dir.name)
        if not match:
            continue
        model, target_group, split_seed = match.groups()
        common = {"run": run_dir.name, "model": model, "target_group": target_group,
                  "split_seed": int(split_seed)}
        prediction_path, summary_path, settings_path = (run_dir / "predictions.csv", run_dir / "summary.json",
                                                         run_dir / "run_settings.json")
        if not all(path.is_file() for path in (prediction_path, summary_path, settings_path)):
            inventory.append({**common, "status": "incomplete"})
            continue
        prediction = pd.read_csv(prediction_path)
        test = prediction.loc[prediction.split.eq("test")].copy()
        if test.empty or test.duplicated(["sample_id", "target"]).any():
            raise RuntimeError(f"Invalid test predictions: {prediction_path}")
        summary, settings = json.loads(summary_path.read_text()), json.loads(settings_path.read_text())
        for target, part in test.groupby("target", sort=False):
            details.append({**common, "target": target, "best_epoch": int(summary["best_epoch"]),
                            "selected_checkpoint": summary["selected_checkpoint"], **values(part)})
        selection = {**common, "best_epoch": int(summary["best_epoch"]),
                     "best_validation_loss_normalized": float(summary["best_validation_loss_normalized"]),
                     "parameter_count": int(summary["parameter_count"]),
                     "split_manifest": settings["split_manifest"],
                     "manifest_sha256": settings["manifest_sha256"],
                     "selected_checkpoint": summary["selected_checkpoint"]}
        validation_selection.append(selection)
        inventory.append({**common, "status": "completed", **selection})
    detail = pd.DataFrame(details).sort_values(["model", "target_group", "split_seed", "target"])
    inventory_frame = pd.DataFrame(inventory).sort_values(["model", "target_group", "split_seed"])
    detail.to_csv(output / "baseline_test_metrics_by_target.csv", index=False)
    inventory_frame.to_csv(output / "baseline_inventory.csv", index=False)
    pd.DataFrame(validation_selection).sort_values(["model", "target_group", "split_seed"]).to_csv(
        output / "baseline_validation_selection.csv", index=False)
    if detail.empty:
        return
    summary = detail.groupby(["model", "target_group", "target"], as_index=False).agg(
        completed_splits=("run", "count"), mean_test_mae=("mae", "mean"), std_test_mae=("mae", "std"),
        mean_test_rmse=("rmse", "mean"), std_test_rmse=("rmse", "std"),
        mean_test_r2=("r2", "mean"), std_test_r2=("r2", "std"),
        mean_test_pearson=("pearson", "mean"), std_test_pearson=("pearson", "std"),
        mean_test_spearman=("spearman", "mean"), std_test_spearman=("spearman", "std"),
    ).sort_values(["target_group", "target", "model"])
    summary.to_csv(output / "baseline_test_metrics_target_average.csv", index=False)
    macro = detail.groupby(["model", "target_group", "split_seed"], as_index=False).agg(
        mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"),
        mean_pearson=("pearson", "mean"), mean_spearman=("spearman", "mean"))
    macro.to_csv(output / "baseline_test_metrics_macro_by_split.csv", index=False)
    macro.groupby(["model", "target_group"], as_index=False).agg(
        completed_splits=("split_seed", "count"), mean_test_mae=("mean_mae", "mean"),
        std_test_mae=("mean_mae", "std"), mean_test_r2=("mean_r2", "mean"),
        std_test_r2=("mean_r2", "std"), mean_test_pearson=("mean_pearson", "mean"),
        mean_test_spearman=("mean_spearman", "mean"),
    ).to_csv(output / "baseline_test_metrics_macro_average.csv", index=False)


if __name__ == "__main__":
    main()
