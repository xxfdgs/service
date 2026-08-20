#!/usr/bin/env python3
"""Evaluate and plot six-target O12 ensemble predictions on labelled data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGETS = [
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
    "Norm_before",
    "Norm_after",
]
DISPLAY_NAMES = {
    "EE_before": "EE before",
    "EE_after": "EE after",
    "Aerosolization_Efficiency": "Aerosolization efficiency",
    "mRNA_Recovery_Efficiency": "mRNA recovery efficiency",
    "Norm_before": "Norm before",
    "Norm_after": "Norm after",
}
COLORS = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b", "#b279a2", "#e45756"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Merged all-six ensemble_mean_predictions.csv file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="o12_10seed_feedback")
    args = parser.parse_args()

    table = pd.read_csv(args.predictions, dtype={"ID": str})
    required = {"ID"}
    required.update(TARGETS)
    required.update(f"pred_{target}_mean" for target in TARGETS)
    required.update(f"pred_{target}_std_10models" for target in TARGETS)
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")

    rows = []
    for target in TARGETS:
        truth = pd.to_numeric(table[target], errors="coerce").to_numpy(float)
        prediction = pd.to_numeric(table[f"pred_{target}_mean"], errors="coerce").to_numpy(float)
        uncertainty = pd.to_numeric(
            table[f"pred_{target}_std_10models"], errors="coerce").to_numpy(float)
        valid = np.isfinite(truth) & np.isfinite(prediction) & np.isfinite(uncertainty)
        if not valid.any():
            raise ValueError(f"{target} has no finite labelled predictions.")
        rows.append({
            "target": target,
            "n": int(valid.sum()),
            "mae": float(mean_absolute_error(truth[valid], prediction[valid])),
            "rmse": float(mean_squared_error(truth[valid], prediction[valid]) ** .5),
            "r2": float(r2_score(truth[valid], prediction[valid])) if valid.sum() > 1 else np.nan,
            "mean_model_std": float(uncertainty[valid].mean()),
        })
    metrics = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / f"{args.stem}_metrics.csv", index=False)

    figure, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for axis, target, color in zip(axes.flat, TARGETS, COLORS):
        truth = pd.to_numeric(table[target], errors="coerce").to_numpy(float)
        prediction = pd.to_numeric(table[f"pred_{target}_mean"], errors="coerce").to_numpy(float)
        uncertainty = pd.to_numeric(
            table[f"pred_{target}_std_10models"], errors="coerce").to_numpy(float)
        valid = np.isfinite(truth) & np.isfinite(prediction) & np.isfinite(uncertainty)
        truth, prediction, uncertainty = truth[valid], prediction[valid], uncertainty[valid]
        lower, upper = float(min(truth.min(), prediction.min())), float(max(truth.max(), prediction.max()))
        padding = max((upper - lower) * .06, .05)
        limits = (lower - padding, upper + padding)
        metric = metrics.loc[metrics.target.eq(target)].iloc[0]
        axis.errorbar(truth, prediction, yerr=uncertainty, fmt="none", ecolor=color,
                      alpha=.28, linewidth=.8, capsize=1.5, zorder=1,
                      label="± 1 model std")
        axis.scatter(truth, prediction, s=42, alpha=.84, color=color,
                     edgecolor="#222222", linewidth=.35, zorder=2)
        axis.plot(limits, limits, color="#d62728", linestyle="--", linewidth=1.4,
                  label="y = x")
        axis.set(xlim=limits, ylim=limits, xlabel="True value", ylabel="O12 10-seed mean")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{DISPLAY_NAMES[target]}\nMAE = {metric.mae:.3f}; R² = {metric.r2:.3f}")
        axis.grid(alpha=.25)
        axis.legend(loc="upper left", fontsize=7)
    figure.suptitle("O12 ten-seed ensemble: labelled feedback evaluation", fontsize=16, y=.98)
    figure.tight_layout(rect=(0, 0, 1, .95))
    for suffix in ("png", "pdf"):
        figure.savefig(args.output_dir / f"{args.stem}_true_vs_pred.{suffix}",
                       dpi=180 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
