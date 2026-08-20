#!/usr/bin/env python3
"""Create a fixed post-selection external report for the Stage-9 candidate.

This is intentionally a reporting-only program: model selection must already
have been completed using the locked Fifth-identity OOD proxy.  It combines the
saved ten-checkpoint P0/P1 external predictions with one selected Stage-9
candidate ensemble, then writes descriptive metrics, a point-level audit, and
three figures.  It never trains, refits, or chooses an ensemble weight.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASELINE_COLUMNS = {
    "P0_random": "P0_random_strict_no_mordred",
    "P1_PT_D": "P1_PT_D_strict_no_mordred",
}


def safe_spearman(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 2 or not np.std(y) or not np.std(prediction):
        return math.nan
    return float(spearmanr(y, prediction).statistic)


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = prediction - y
    true_high, predicted_high = y > 1.0, prediction > 1.0
    tp = int(np.sum(true_high & predicted_high))
    fp = int(np.sum(~true_high & predicted_high))
    fn = int(np.sum(true_high & ~predicted_high))
    under = error < 0
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, prediction)) if len(y) else math.nan,
        "rmse": float(mean_squared_error(y, prediction) ** 0.5) if len(y) else math.nan,
        "median_ae": float(np.median(np.abs(error))) if len(y) else math.nan,
        "mean_signed_error": float(np.mean(error)) if len(y) else math.nan,
        "underprediction_mae": float(np.mean(np.abs(error[under]))) if np.any(under) else 0.0,
        "r2": float(r2_score(y, prediction)) if len(y) > 1 and np.std(y) else math.nan,
        "spearman": safe_spearman(y, prediction),
        "prediction_mean": float(np.mean(prediction)) if len(y) else math.nan,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_gt1": tp / (tp + fp) if tp + fp else math.nan,
        "recall_gt1": tp / (tp + fn) if tp + fn else math.nan,
        "f2_gt1": 5 * tp / (5 * tp + 4 * fn + fp) if 5 * tp + 4 * fn + fp else math.nan,
    }


def model_prefix(frame: pd.DataFrame) -> str:
    columns = [column for column in frame if column.endswith("_ensemble_mean")]
    if len(columns) != 1:
        raise ValueError("Candidate file must contain exactly one ensemble-mean column.")
    return columns[0].removesuffix("_ensemble_mean")


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small numeric table without requiring the optional tabulate package."""
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for index, values in frame.iterrows():
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append("" if not np.isfinite(value) else f"{value:.4f}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, divider, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--candidate-label", default="H30_PTD_weighted_huber")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_predictions, dtype={"ID": str})
    candidate = pd.read_csv(args.candidate_predictions, dtype={"ID": str})
    required = {"ID", "y_true", "Fifth_class", "Fifth"}
    for label, frame in (("baseline", baseline), ("candidate", candidate)):
        missing = required.difference(frame.columns)
        if missing or frame.ID.duplicated().any():
            raise ValueError(f"{label} prediction table is invalid; missing={sorted(missing)}")
    if baseline.ID.tolist() != candidate.ID.tolist():
        raise ValueError("Baseline and candidate external IDs/orders differ.")
    if not np.allclose(baseline.y_true, candidate.y_true, equal_nan=False):
        raise ValueError("Baseline and candidate external labels differ.")

    candidate_prefix = model_prefix(candidate)
    model_columns = {
        label: prefix for label, prefix in BASELINE_COLUMNS.items()
    }
    model_columns[args.candidate_label] = candidate_prefix
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    point = baseline[["ID", "y_true", "Fifth_class", "Fifth"]].copy()
    for label, prefix in model_columns.items():
        origin = candidate if label == args.candidate_label else baseline
        point[f"{label}_prediction"] = origin[f"{prefix}_ensemble_mean"].to_numpy(float)
        point[f"{label}_ensemble_std"] = origin[f"{prefix}_ensemble_std"].to_numpy(float)
        point[f"{label}_signed_error"] = point[f"{label}_prediction"] - point.y_true
        point[f"{label}_absolute_error"] = point[f"{label}_signed_error"].abs()
        point[f"{label}_false_negative_gt1"] = (
            point.y_true.gt(1.0) & point[f"{label}_prediction"].le(1.0)
        )
    point["candidate_improved_vs_p1"] = (
        point[f"{args.candidate_label}_absolute_error"] < point["P1_PT_D_absolute_error"]
    )
    point["double_gt1"] = point.Fifth_class.eq("double") & point.y_true.gt(1.0)
    point = point.sort_values(["double_gt1", "y_true", "ID"], ascending=[False, False, True])
    point.to_csv(output / "new_validation_point_audit.csv", index=False)

    subset_masks = {
        "all": np.ones(len(point), dtype=bool),
        "single": point.Fifth_class.eq("single").to_numpy(),
        "double": point.Fifth_class.eq("double").to_numpy(),
        "double_gt1": point.double_gt1.to_numpy(),
    }
    rows = []
    for label in model_columns:
        for subset, mask in subset_masks.items():
            rows.append({"model": label, "subset": subset, **metrics(
                point.loc[mask, "y_true"].to_numpy(float),
                point.loc[mask, f"{label}_prediction"].to_numpy(float),
            )})
    metric_frame = pd.DataFrame(rows)
    metric_frame.to_csv(output / "new_validation_metrics_by_subset.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = point.y_true.to_numpy(float)
    values = np.concatenate([y] + [point[f"{name}_prediction"].to_numpy(float) for name in model_columns])
    lo, hi = min(0.0, float(values.min())), float(values.max())
    pad = max(0.08 * (hi - lo), 0.05); lo -= pad; hi += pad
    fig, axes = plt.subplots(1, len(model_columns), figsize=(5.4 * len(model_columns), 5.0), squeeze=False)
    for axis, label in zip(axes[0], model_columns):
        prediction = point[f"{label}_prediction"].to_numpy(float)
        single, double = point.Fifth_class.eq("single"), point.Fifth_class.eq("double")
        axis.scatter(y[single], prediction[single], s=45, alpha=.8, label="single")
        axis.scatter(y[double], prediction[double], s=52, alpha=.8, marker="^", label="double")
        fn = point[f"{label}_false_negative_gt1"].to_numpy()
        axis.scatter(y[fn], prediction[fn], s=110, facecolors="none", edgecolors="crimson", linewidths=1.5, label="double >1 FN")
        axis.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
        axis.axvline(1.0, color="grey", linestyle=":", linewidth=1)
        axis.axhline(1.0, color="grey", linestyle=":", linewidth=1)
        axis.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="True Norm_before", ylabel="Prediction", title=label)
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "true_vs_prediction_by_class.png", dpi=220); plt.close(fig)

    # The focused panel is deliberately double-only: it makes every true-high
    # prediction below 1 visible without the much larger single population
    # obscuring the decision-boundary failure mode.
    fig, axes = plt.subplots(1, len(model_columns), figsize=(5.4 * len(model_columns), 5.0), squeeze=False)
    double = point.Fifth_class.eq("double").to_numpy()
    for axis, label in zip(axes[0], model_columns):
        prediction = point[f"{label}_prediction"].to_numpy(float)
        high = double & (y > 1.0)
        low = double & ~high
        axis.scatter(y[low], prediction[low], s=50, alpha=.78, label="double, true ≤1")
        axis.scatter(y[high], prediction[high], s=58, alpha=.84, marker="^", label="double, true >1")
        fn = point[f"{label}_false_negative_gt1"].to_numpy()
        axis.scatter(y[fn], prediction[fn], s=125, facecolors="none", edgecolors="crimson", linewidths=1.7, label="FN")
        axis.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
        axis.axvline(1.0, color="grey", linestyle=":", linewidth=1)
        axis.axhline(1.0, color="grey", linestyle=":", linewidth=1)
        axis.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="True Norm_before", ylabel="Prediction", title=label)
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "double_true_vs_prediction_false_negatives.png", dpi=220); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.0, 5.2))
    for label in model_columns:
        axis.scatter(y, point[f"{label}_signed_error"], alpha=.75, s=42, label=label)
    axis.axhline(0, color="black", linewidth=1); axis.axvline(1, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="True Norm_before", ylabel="Prediction − true", title="Error versus true value")
    axis.grid(alpha=.2); axis.legend(); fig.tight_layout(); fig.savefig(output / "signed_error_vs_true.png", dpi=220); plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    for label in model_columns:
        axis.scatter(point[f"{label}_ensemble_std"], point[f"{label}_absolute_error"], alpha=.75, s=42, label=label)
    axis.set(xlabel="Across-seed prediction std", ylabel="Absolute error", title="Ensemble dispersion versus error")
    axis.grid(alpha=.2); axis.legend(); fig.tight_layout(); fig.savefig(output / "absolute_error_vs_ensemble_std.png", dpi=220); plt.close(fig)

    high = metric_frame.loc[metric_frame.subset.eq("double_gt1")].set_index("model")
    p1_high, candidate_high = high.loc["P1_PT_D"], high.loc[args.candidate_label]
    p1_double = metric_frame.loc[
        (metric_frame.model.eq("P1_PT_D")) & (metric_frame.subset.eq("double"))
    ].iloc[0]
    candidate_double = metric_frame.loc[
        (metric_frame.model.eq(args.candidate_label)) & (metric_frame.subset.eq("double"))
    ].iloc[0]
    conclusion = (
        f"On the six external double>1 rows, {args.candidate_label} changed MAE "
        f"from {p1_high.mae:.3f} to {candidate_high.mae:.3f} and false negatives "
        f"from {int(p1_high.fn)} to {int(candidate_high.fn)}. On all double rows, "
        f"MAE changed from {p1_double.mae:.3f} to {candidate_double.mae:.3f} and "
        f"false positives changed from {int(p1_double.fp)} to {int(candidate_double.fp)}. "
        "This is a fixed external readout, not a selection criterion. If the primary "
        "high-value improvement is small while low-value false positives increase, do not "
        "promote the candidate as a general replacement despite an internal OOD gain."
    )
    report = [
        "# Stage-9 fixed external evaluation", "",
        "The H30 candidate was selected on the frozen internal Fifth-identity OOD proxy before this labelled table was scored. These external metrics are descriptive only; no checkpoint, weight, or objective was changed after reading them.", "",
        "## Double & Norm_before > 1", "",
        markdown_table(high.reset_index()[["model", "n", "mae", "rmse", "median_ae", "mean_signed_error", "underprediction_mae", "recall_gt1", "f2_gt1", "fn", "fp"]]), "",
        "## Scientific conclusion", "", conclusion, "",
        "## Files", "",
        "- `new_validation_metrics_by_subset.csv`: all/single/double/double_gt1 metrics.",
        "- `new_validation_point_audit.csv`: per-point P0/P1/candidate predictions, errors, FN flags, and ensemble standard deviations.",
        "- Figures: class-marked true-vs-prediction, double-only FN panel, signed error, and ensemble-dispersion diagnostics.",
    ]
    (output / "new_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "report_manifest.json").write_text(json.dumps({
        "selection_policy": "Internal frozen Fifth-identity OOD selection completed before external scoring.",
        "baseline_predictions": str(args.baseline_predictions.resolve()),
        "candidate_predictions": str(args.candidate_predictions.resolve()),
        "candidate_label": args.candidate_label,
        "models": list(model_columns),
        "ensemble": "Saved arithmetic mean across ten validation-selected checkpoints; no external reweighting.",
    }, indent=2) + "\n", encoding="utf-8")
    print(metric_frame.to_string(index=False))


if __name__ == "__main__":
    main()
