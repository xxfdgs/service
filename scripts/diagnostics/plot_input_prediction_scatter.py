#!/usr/bin/env python3
"""Plot per-property true-versus-predicted values from an input-only CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


TARGETS = [
    "EE_before", "EE_after", "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
]
DISPLAY_NAMES = ["EE before", "EE after", "Aerosolization", "mRNA recovery"]
COLORS = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="input_prediction")
    args = parser.parse_args()

    if "feedback" in str(args.predictions).lower():
        raise ValueError("This plotting utility is restricted to input-only prediction files.")
    table = pd.read_csv(args.predictions)
    required = {"sample_id", "split", "target", "y_true", "y_pred"}
    if missing := required - set(table.columns):
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    values = table.loc[table.split.eq(args.split)].copy()
    rows = []
    for target in TARGETS:
        target_values = values.loc[values.target.eq(target)]
        if target_values.empty or target_values.sample_id.duplicated().any():
            raise ValueError(f"Expected one or more unique {args.split} rows for {target}.")
        truth = target_values.y_true.to_numpy(float)
        prediction = target_values.y_pred.to_numpy(float)
        rows.append({"target": target, "n": len(target_values),
                     "mae": float(mean_absolute_error(truth, prediction)),
                     "r2": float(r2_score(truth, prediction))})
    metrics = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / f"{args.stem}_{args.split}_metrics.csv", index=False)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 9.2))
    for axis, target, title, color in zip(axes.flat, TARGETS, DISPLAY_NAMES, COLORS):
        target_values = values.loc[values.target.eq(target)]
        truth = target_values.y_true.to_numpy(float)
        prediction = target_values.y_pred.to_numpy(float)
        lower, upper = float(min(truth.min(), prediction.min())), float(max(truth.max(), prediction.max()))
        padding = max((upper - lower) * .05, 1.0)
        limits = (lower - padding, upper + padding)
        metric = metrics.loc[metrics.target.eq(target)].iloc[0]
        axis.scatter(truth, prediction, s=35, alpha=.78, color=color,
                     edgecolor="#222222", linewidth=.35)
        axis.plot(limits, limits, color="#d62728", linestyle="--", linewidth=1.5,
                  label="y = x")
        axis.set(xlim=limits, ylim=limits, xlabel="True value", ylabel="Predicted value")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{title}\nMAE = {metric.mae:.3f}, R² = {metric.r2:.3f}")
        axis.grid(alpha=.25)
        axis.legend(loc="upper left", fontsize=8)
    figure.suptitle(f"Input-only {args.split} predictions", fontsize=15, y=.98)
    figure.tight_layout(rect=(0, 0, 1, .96))
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"{args.stem}_{args.split}_true_vs_pred.{extension}",
                       dpi=180 if extension == "png" else None, bbox_inches="tight")
    plt.close(figure)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
