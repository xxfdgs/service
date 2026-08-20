#!/usr/bin/env python3
"""Export corresponding-test metrics for every completed selected checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def correlation(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.std(truth) == 0 or np.std(prediction) == 0:
        return float("nan")
    return float(function(truth, prediction).statistic)


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame.y_true.to_numpy(float)
    prediction = frame.y_pred.to_numpy(float)
    return {
        "n": len(frame),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
        "pearson": correlation(pearsonr, truth, prediction),
        "spearman": correlation(spearmanr, truth, prediction),
    }


def split_seed(name: str) -> int | None:
    match = re.search(r"_split(\d+)$", name)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    details, macro, inventory = [], [], []
    # Accept the historical ``O12_split100`` naming and target-qualified
    # one-output runs such as ``O12_EE_before_split100``.  The model prefix
    # remains the aggregation key, while ``target`` comes from predictions.
    run_pattern = re.compile(r"^(O12|O22)(?:_[A-Za-z0-9_]+)?_split\d+$")
    for run_dir in sorted(path for path in arguments.runs_root.resolve().iterdir()
                          if path.is_dir() and run_pattern.fullmatch(path.name)):
        prediction_path = run_dir / "predictions.csv"
        summary_path = run_dir / "summary.json"
        settings_path = run_dir / "run_settings.json"
        checkpoint_path = run_dir / "checkpoints" / "selected_best.pt"
        model = run_dir.name.split("_", 1)[0]
        common = {"run": run_dir.name, "model": model, "split_seed": split_seed(run_dir.name)}
        if not all(path.is_file() for path in (prediction_path, summary_path, settings_path, checkpoint_path)):
            inventory.append({**common, "status": "incomplete"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        prediction = pd.read_csv(prediction_path)
        test = prediction.loc[prediction.split == "test"].copy()
        if test.empty or test.duplicated(["sample_id", "target"]).any():
            raise RuntimeError(f"Invalid test predictions in {prediction_path}")
        recorded_checkpoints = test.checkpoint.dropna().astype(str).unique()
        if len(recorded_checkpoints) != 1:
            raise RuntimeError(f"Expected one selected checkpoint in {prediction_path}")
        metadata = {
            **common,
            "candidate": summary["candidate"],
            "best_epoch": int(summary["best_epoch"]),
            "selected_checkpoint": recorded_checkpoints[0],
            "split_manifest": settings.get("split_manifest", ""),
            "model_seed": settings.get("seed"),
        }
        target_rows = []
        for target, target_frame in test.groupby("target", sort=False):
            row = {**metadata, "target": target, **metric_values(target_frame)}
            details.append(row)
            target_rows.append(row)
        target_metrics = pd.DataFrame(target_rows)
        macro.append({**metadata, "targets": len(target_metrics),
                      "n_per_target": int(target_metrics.n.iloc[0]),
                      "mean_mae": float(target_metrics.mae.mean()),
                      "mean_rmse": float(target_metrics.rmse.mean()),
                      "mean_r2": float(target_metrics.r2.mean()),
                      "mean_pearson": float(target_metrics.pearson.mean()),
                      "mean_spearman": float(target_metrics.spearman.mean())})
        inventory.append({**common, "status": "completed", "best_epoch": int(summary["best_epoch"])})

    detail_frame = pd.DataFrame(details).sort_values(["model", "split_seed", "target"])
    macro_frame = pd.DataFrame(macro).sort_values(["model", "split_seed"])
    detail_frame.to_csv(output / "selected_checkpoint_test_metrics_by_target.csv", index=False)
    macro_frame.to_csv(output / "selected_checkpoint_test_metrics_macro.csv", index=False)
    pd.DataFrame(inventory).sort_values(["model", "split_seed"]).to_csv(
        output / "selected_checkpoint_inventory.csv", index=False)
    if not macro_frame.empty:
        macro_frame.groupby("model", as_index=False).agg(
            checkpoints=("run", "count"), mean_test_mae=("mean_mae", "mean"),
            std_test_mae=("mean_mae", "std"), mean_test_r2=("mean_r2", "mean"),
            std_test_r2=("mean_r2", "std"),
        ).to_csv(output / "selected_checkpoint_test_metrics_model_average.csv", index=False)
    if not detail_frame.empty:
        detail_frame.groupby(["model", "target"], as_index=False).agg(
            checkpoints=("run", "count"), mean_test_mae=("mae", "mean"),
            std_test_mae=("mae", "std"), mean_test_rmse=("rmse", "mean"),
            std_test_rmse=("rmse", "std"), mean_test_r2=("r2", "mean"),
            std_test_r2=("r2", "std"), mean_test_pearson=("pearson", "mean"),
            mean_test_spearman=("spearman", "mean"),
        ).to_csv(output / "selected_checkpoint_test_metrics_target_average.csv", index=False)


if __name__ == "__main__":
    main()
