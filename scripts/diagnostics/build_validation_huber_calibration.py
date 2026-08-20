#!/usr/bin/env python3
"""Fit a fixed Huber calibration head on validation predictions only.

The input predictions must already contain train/val/test rows from one
frozen GraphGPS checkpoint.  Four independent affine heads are fit on the
validation split only and then applied unchanged to every split.  In
particular, test labels are loaded only after fitting has completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation(function, y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 2 or np.std(y) == 0 or np.std(prediction) == 0:
        return float("nan")
    return float(function(y, prediction).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    source = arguments.predictions.resolve()
    output = arguments.output_dir.resolve()
    table = pd.read_csv(source)
    required = {"sample_id", "split", "target", "y_true", "y_pred"}
    if missing := required - set(table.columns):
        raise RuntimeError(f"Predictions miss required columns: {sorted(missing)}")
    if set(table.split) != {"train", "val", "test"}:
        raise RuntimeError("Calibration requires exactly train/val/test prediction rows")

    calibrated, coefficients, metrics = [], [], []
    targets = list(dict.fromkeys(table.target))
    # This loop intentionally fits before it touches target's test rows.
    for target in targets:
        validation = table.loc[(table.split == "val") & (table.target == target)].copy()
        if len(validation) < 2:
            raise RuntimeError(f"Validation contains fewer than two {target} predictions")
        head = HuberRegressor().fit(validation[["y_pred"]], validation.y_true)
        coefficients.append({"target": target, "coefficient": float(head.coef_[0]),
                             "intercept": float(head.intercept_), "fit_split": "val",
                             "fit_samples": int(len(validation))})
        for split in ("train", "val", "test"):
            part = table.loc[(table.split == split) & (table.target == target)].copy()
            part["y_pred_uncalibrated"] = part.y_pred
            part["y_pred"] = head.predict(part[["y_pred"]])
            calibrated.append(part)
            y, prediction = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
            metrics.append({
                "split": split, "target": target, "n": int(len(part)),
                "mae": float(mean_absolute_error(y, prediction)),
                "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
                "r2": float(r2_score(y, prediction)) if len(y) > 1 and np.std(y) else float("nan"),
                "pearson": correlation(pearsonr, y, prediction),
                "spearman": correlation(spearmanr, y, prediction),
            })
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.concat(calibrated, ignore_index=True)
    metrics_frame = pd.DataFrame(metrics)
    prediction_frame.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(coefficients).to_csv(output / "coefficients.csv", index=False)
    metrics_frame.to_csv(output / "metrics.csv", index=False)
    metrics_frame.groupby("split", as_index=False)[["mae", "rmse", "r2", "pearson", "spearman"]].mean().rename(
        columns={"mae": "mean_mae", "rmse": "mean_rmse", "r2": "mean_r2",
                 "pearson": "mean_pearson", "spearman": "mean_spearman"}
    ).to_csv(output / "metrics_summary.csv", index=False)
    (output / "protocol.json").write_text(json.dumps({
        "base_predictions": str(source), "base_predictions_sha256": sha256(source),
        "calibration": "independent fixed HuberRegressor affine heads",
        "fit_split": "val", "test_read_after_fit_only": True, "feedback_read": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
