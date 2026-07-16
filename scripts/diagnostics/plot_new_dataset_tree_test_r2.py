#!/usr/bin/env python3
"""Recompute and plot ExtraTrees test or feedback prediction diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
]
DISPLAY_NAMES = ["EE_before", "EE_after", "Aerosolization", "mRNA Recovery"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
    )
    parser.add_argument("--evaluation-set", choices=("test", "feedback"), default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/new_dataset_benchmark_20260713/figures",
    )
    args = parser.parse_args()

    evaluation = args.evaluation_set
    prediction_path = args.predictions or ROOT / "results/new_dataset_benchmark_20260713" / (
        "tabular_baseline_test_predictions.csv" if evaluation == "test" else "tabular_baseline_feedback_predictions.csv"
    )
    split_column = "split" if evaluation == "test" else "evaluation_set"
    expected_samples = 70 if evaluation == "test" else 97
    predictions = pd.read_csv(prediction_path)
    required = {"sample_id", split_column, "target", "model", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    selected = predictions.loc[
        predictions[split_column].eq(evaluation) & predictions["model"].eq("ExtraTrees")
    ].copy()

    rows = []
    for target in TARGETS:
        values = selected.loc[selected["target"].eq(target)].copy()
        if len(values) != expected_samples or values["sample_id"].duplicated().any():
            raise ValueError(
                f"Expected {expected_samples} unique {evaluation} predictions for {target}; found {len(values)}"
            )
        truth, prediction = values["y_true"].to_numpy(float), values["y_pred"].to_numpy(float)
        residual_ss = float(np.square(truth - prediction).sum())
        total_ss = float(np.square(truth - truth.mean()).sum())
        calculated_r2 = float(r2_score(truth, prediction))
        if not np.isclose(calculated_r2, 1.0 - residual_ss / total_ss):
            raise AssertionError(f"R² identity failed for {target}")
        rows.append({
            "target": target,
            "n_evaluation": len(values),
            "evaluation_target_mean": float(truth.mean()),
            "residual_sum_of_squares": residual_ss,
            "total_sum_of_squares": total_ss,
            "mae": float(mean_absolute_error(truth, prediction)),
            "r2": calculated_r2,
        })
    breakdown = pd.DataFrame(rows)
    macro_r2 = float(breakdown["r2"].mean())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    breakdown.to_csv(args.output_dir / f"extratrees_{evaluation}_r2_breakdown.csv", index=False)

    colors = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b"]
    fig, axis = plt.subplots(figsize=(9.5, 5.4))
    bars = axis.bar(DISPLAY_NAMES, breakdown["r2"], color=colors, edgecolor="#333333", linewidth=0.7)
    axis.axhline(0.0, color="#555555", linewidth=1.0)
    axis.axhline(macro_r2, color="#d62728", linestyle="--", linewidth=1.8,
                 label=f"macro mean = {macro_r2:.4f}")
    axis.set_ylim(min(-0.05, breakdown["r2"].min() - .05), max(.65, breakdown["r2"].max() + .08))
    axis.set_ylabel(f"{evaluation.capitalize()} R²")
    axis.set_title(f"ExtraTrees: per-target {evaluation} R² and macro mean (n = {expected_samples} per target)")
    axis.grid(axis="y", alpha=.25)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, breakdown["r2"]):
        axis.text(bar.get_x() + bar.get_width() / 2, value + .012, f"{value:.4f}",
                  ha="center", va="bottom", fontsize=10, fontweight="bold")
    axis.legend(loc="lower right")
    axis.text(.01, -.20, "R²_target = 1 − Σ(y − ŷ)² / Σ(y − ȳ_test)²; macro R² = mean of the four target R² values.",
              transform=axis.transAxes, fontsize=9, color="#333333")
    fig.tight_layout()
    fig.savefig(args.output_dir / f"extratrees_{evaluation}_r2_macro.png", dpi=180, bbox_inches="tight")
    fig.savefig(args.output_dir / f"extratrees_{evaluation}_r2_macro.pdf", bbox_inches="tight")
    plt.close(fig)

    # One panel per property: measured values on x, predictions on y.  The
    # diagonal is a reference only and is not a fitted regression line.
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2))
    for axis, target, display_name, color in zip(axes.flat, TARGETS, DISPLAY_NAMES, colors):
        values = selected.loc[selected["target"].eq(target)]
        truth = values["y_true"].to_numpy(float)
        prediction = values["y_pred"].to_numpy(float)
        lower = float(min(truth.min(), prediction.min()))
        upper = float(max(truth.max(), prediction.max()))
        padding = max((upper - lower) * .05, 1.0)
        limits = (lower - padding, upper + padding)
        summary = breakdown.loc[breakdown["target"].eq(target)].iloc[0]
        axis.scatter(truth, prediction, s=35, alpha=.78, color=color,
                     edgecolor="#222222", linewidth=.35)
        axis.plot(limits, limits, color="#d62728", linestyle="--", linewidth=1.5,
                  label="y = x (perfect prediction)")
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("True value")
        axis.set_ylabel("Predicted value")
        axis.set_title(f"{display_name}\nMAE = {summary.mae:.3f}, R² = {summary.r2:.3f}")
        axis.grid(alpha=.25)
        axis.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"ExtraTrees {evaluation} predictions (n = {expected_samples} per property)", fontsize=15, y=.98)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(args.output_dir / f"extratrees_{evaluation}_true_vs_pred.png", dpi=180, bbox_inches="tight")
    fig.savefig(args.output_dir / f"extratrees_{evaluation}_true_vs_pred.pdf", bbox_inches="tight")
    plt.close(fig)
    print(breakdown.to_string(index=False))
    print(f"macro_r2={macro_r2:.12f}")


if __name__ == "__main__":
    main()
