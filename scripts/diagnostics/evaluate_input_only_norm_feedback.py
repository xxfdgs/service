#!/usr/bin/env python3
"""Final external evaluation for a frozen input-only Norm regressor.

The 1.0 side-agreement statistics are computed only here, after training and
model selection have completed.  This script never trains or selects a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


TARGETS = ("Norm_before", "Norm_after")
ALL_TARGETS = (
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
    *TARGETS,
)
THRESHOLD = 1.0


def metric_row(target: str, model: str, truth: np.ndarray,
               prediction: np.ndarray) -> dict[str, object]:
    true_high = truth > THRESHOLD
    predicted_high = prediction > THRESHOLD
    return {
        "target": target,
        "model": model,
        "n": len(truth),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)),
        "threshold": THRESHOLD,
        "side_agreement_n": int((true_high == predicted_high).sum()),
        "side_agreement": float((true_high == predicted_high).mean()),
        "balanced_side_accuracy": float(
            balanced_accuracy_score(true_high, predicted_high)),
        "true_gt_count": int(true_high.sum()),
        "predicted_gt_count": int(predicted_high.sum()),
        "true_gt_predicted_gt": int((true_high & predicted_high).sum()),
        "true_le_predicted_le": int((~true_high & ~predicted_high).sum()),
        "gt_recall": float((true_high & predicted_high).sum() / true_high.sum())
        if true_high.any() else np.nan,
        "le_recall": float((~true_high & ~predicted_high).sum() / (~true_high).sum())
        if (~true_high).any() else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    improved = pd.read_csv(args.predictions, dtype={"ID": str})
    baseline = pd.read_csv(args.baseline, dtype={"ID": str})
    if improved.ID.duplicated().any() or baseline.ID.duplicated().any():
        raise ValueError("Prediction IDs must be unique.")
    required_improved = {
        "ID", *TARGETS,
        *[f"pred_{target}_input_only_mean" for target in TARGETS],
    }
    required_baseline = {
        "ID", *[f"pred_{target}_mean" for target in TARGETS],
    }
    if missing := required_improved.difference(improved.columns):
        raise ValueError(f"Improved predictions miss columns: {sorted(missing)}")
    if missing := required_baseline.difference(baseline.columns):
        raise ValueError(f"Baseline predictions miss columns: {sorted(missing)}")
    if set(improved.ID) != set(baseline.ID):
        raise ValueError("Improved and baseline IDs differ.")
    baseline = baseline.set_index("ID").loc[improved.ID].reset_index()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    per_row = improved[["ID", *TARGETS]].copy()
    for target in TARGETS:
        truth = improved[target].to_numpy(float)
        base_prediction = baseline[f"pred_{target}_mean"].to_numpy(float)
        new_prediction = improved[f"pred_{target}_input_only_mean"].to_numpy(float)
        rows.append(metric_row(target, "O12_10seed_baseline", truth, base_prediction))
        rows.append(metric_row(target, "input_only_log_rf_10seed", truth, new_prediction))
        per_row[f"baseline_{target}"] = base_prediction
        per_row[f"improved_{target}"] = new_prediction
        per_row[f"baseline_{target}_same_side"] = (
            (truth > THRESHOLD) == (base_prediction > THRESHOLD))
        per_row[f"improved_{target}_same_side"] = (
            (truth > THRESHOLD) == (new_prediction > THRESHOLD))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "new_validation_norm_metrics.csv", index=False)
    per_row.to_csv(output / "new_validation_norm_predictions_and_sides.csv", index=False)

    comparison_rows = []
    for target in TARGETS:
        base = metrics.loc[
            metrics.target.eq(target) & metrics.model.eq("O12_10seed_baseline")].iloc[0]
        new = metrics.loc[
            metrics.target.eq(target) & metrics.model.eq("input_only_log_rf_10seed")].iloc[0]
        comparison_rows.append({
            "target": target,
            "baseline_mae": base.mae,
            "improved_mae": new.mae,
            "mae_reduction": base.mae - new.mae,
            "mae_reduction_percent": 100.0 * (base.mae - new.mae) / base.mae,
            "baseline_side_agreement": base.side_agreement,
            "improved_side_agreement": new.side_agreement,
            "side_agreement_gain": new.side_agreement - base.side_agreement,
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "new_validation_norm_improvement.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 9.4))
    colors = {"O12_10seed_baseline": "#9c9c9c", "input_only_log_rf_10seed": "#4c78a8"}
    titles = {"Norm_before": "Norm before", "Norm_after": "Norm after"}
    for row_index, target in enumerate(TARGETS):
        truth = improved[target].to_numpy(float)
        for column_index, model_name in enumerate(
                ("O12_10seed_baseline", "input_only_log_rf_10seed")):
            axis = axes[row_index, column_index]
            prediction = (
                baseline[f"pred_{target}_mean"].to_numpy(float)
                if model_name == "O12_10seed_baseline"
                else improved[f"pred_{target}_input_only_mean"].to_numpy(float)
            )
            metric = metrics.loc[
                metrics.target.eq(target) & metrics.model.eq(model_name)].iloc[0]
            lower = float(min(truth.min(), prediction.min()))
            upper = float(max(truth.max(), prediction.max()))
            padding = max((upper - lower) * 0.06, 0.05)
            limits = (lower - padding, upper + padding)
            axis.scatter(
                truth, prediction, s=45, alpha=0.84, color=colors[model_name],
                edgecolor="#222222", linewidth=0.35)
            axis.plot(limits, limits, "--", color="#d62728", linewidth=1.4)
            axis.axvline(THRESHOLD, color="#f58518", linestyle=":", linewidth=1.2)
            axis.axhline(THRESHOLD, color="#f58518", linestyle=":", linewidth=1.2)
            axis.set(
                xlim=limits,
                ylim=limits,
                xlabel="True value",
                ylabel="Predicted value",
            )
            axis.set_aspect("equal", adjustable="box")
            model_title = "Baseline O12" if column_index == 0 else "Input-only log-RF"
            axis.set_title(
                f"{titles[target]} — {model_title}\n"
                f"MAE={metric.mae:.3f}; same side={metric.side_agreement_n}/"
                f"{metric.n} ({100 * metric.side_agreement:.1f}%)")
            axis.grid(alpha=0.25)
    figure.suptitle(
        "Frozen input-only Norm model: final new_validation evaluation",
        fontsize=15, y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("png", "pdf"):
        figure.savefig(
            output / f"new_validation_norm_baseline_vs_improved.{suffix}",
            dpi=180 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)

    hybrid = baseline.copy()
    for target in TARGETS:
        hybrid[f"pred_{target}_o12_baseline_mean"] = hybrid[f"pred_{target}_mean"]
        hybrid[f"pred_{target}_mean"] = improved[f"pred_{target}_input_only_mean"]
        hybrid[f"pred_{target}_std_10models"] = improved[
            f"pred_{target}_input_only_std_10models"]
    hybrid.to_csv(output / "improved_ensemble_mean_predictions_all6.csv", index=False)
    if all(
        target in hybrid and f"pred_{target}_mean" in hybrid
        for target in ALL_TARGETS
    ):
        all_six_rows = []
        for target in ALL_TARGETS:
            truth = hybrid[target].to_numpy(float)
            baseline_prediction = baseline[f"pred_{target}_mean"].to_numpy(float)
            improved_prediction = hybrid[f"pred_{target}_mean"].to_numpy(float)
            for name, prediction in (
                ("O12_10seed_baseline", baseline_prediction),
                ("hybrid_input_only_norm", improved_prediction),
            ):
                all_six_rows.append({
                    "target": target,
                    "model": name,
                    "n": len(truth),
                    "mae": float(mean_absolute_error(truth, prediction)),
                    "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
                    "r2": float(r2_score(truth, prediction)),
                })
        all_six = pd.DataFrame(all_six_rows)
        all_six.to_csv(output / "new_validation_all6_metrics.csv", index=False)
        (
            all_six.groupby("model", as_index=False)
            .agg(mean_six_target_mae=("mae", "mean"))
            .to_csv(output / "new_validation_all6_macro_mae.csv", index=False)
        )

    report = {
        "evaluation_only": True,
        "training_or_selection_performed": False,
        "threshold_used_only_for_final_external_evaluation": THRESHOLD,
        "rows": len(improved),
        "comparison": comparison.to_dict(orient="records"),
    }
    (output / "evaluation_protocol.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
