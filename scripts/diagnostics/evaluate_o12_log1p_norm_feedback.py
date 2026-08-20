#!/usr/bin/env python3
"""Final external evaluation of a frozen optimized O12 Norm ensemble.

This script does not train, calibrate, select, or alter a model.  The 1.0
same-side diagnostic is introduced only here, after all ten input-only
checkpoints and their predictions have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
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
DISPLAY_NAMES = {
    "Norm_before": "Norm before",
    "Norm_after": "Norm after",
}
THRESHOLD = 1.0
REQUIRED_SIDE_AGREEMENT = 0.80


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_row(target: str, model: str, truth: np.ndarray,
               prediction: np.ndarray) -> dict[str, object]:
    true_high = truth > THRESHOLD
    predicted_high = prediction > THRESHOLD
    agreements = true_high == predicted_high
    return {
        "target": target,
        "model": model,
        "n": int(len(truth)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)),
        "threshold": THRESHOLD,
        "side_agreement_n": int(agreements.sum()),
        "side_agreement": float(agreements.mean()),
        "side_agreement_percent": float(100.0 * agreements.mean()),
        "balanced_side_accuracy": float(
            balanced_accuracy_score(true_high, predicted_high)),
        "true_gt_count": int(true_high.sum()),
        "predicted_gt_count": int(predicted_high.sum()),
        "true_gt_predicted_gt": int((true_high & predicted_high).sum()),
        "true_le_predicted_le": int((~true_high & ~predicted_high).sum()),
        "gt_recall": (
            float((true_high & predicted_high).sum() / true_high.sum())
            if true_high.any() else np.nan
        ),
        "le_recall": (
            float((~true_high & ~predicted_high).sum() / (~true_high).sum())
            if (~true_high).any() else np.nan
        ),
        "strictly_above_80_percent": bool(
            agreements.mean() > REQUIRED_SIDE_AGREEMENT),
    }


def validate_prediction_provenance(
        path: Path) -> tuple[dict[str, object], str]:
    provenance_path = path.parent / "provenance_norm2.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"Missing prediction provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("labels_used_for_model_input") is not False:
        raise RuntimeError("Prediction provenance does not rule out label input.")
    if provenance.get("model_family") in {
        "O12_continuous_blend_10_paired_seeds",
        "O12_continuous_pair_blend_10_paired_seeds",
    }:
        paired_seeds = provenance.get("paired_split_seeds", [])
        if len(paired_seeds) != 10 or len(set(paired_seeds)) != 10:
            raise RuntimeError("Optimized blend does not cover ten paired seeds.")
        if provenance.get("labels_used_for_blending") is not False:
            raise RuntimeError("Prediction provenance does not rule out blend-label use.")
        model_name = provenance.get(
            "evaluation_model_name", "O12_continuous_blend_10seed")
    else:
        if provenance.get("target_transform") != "log1p":
            raise RuntimeError(
                "Predictions are neither a frozen log1p ensemble nor its "
                "input-validation-selected continuous blend.")
        checkpoints = provenance.get("checkpoints", [])
        if len(checkpoints) != 10:
            raise RuntimeError(
                f"Expected ten frozen checkpoints, found {len(checkpoints)}.")
        model_name = "O12_log1p_10seed"
    return provenance, str(model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Frozen O12 log1p ensemble_mean_predictions_norm2.csv.")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Frozen identity-target O12 ten-model prediction CSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    prediction_path = args.predictions.resolve()
    baseline_path = args.baseline.resolve()
    provenance, optimized_model_name = validate_prediction_provenance(
        prediction_path)
    optimized = pd.read_csv(prediction_path, dtype={"ID": str})
    baseline = pd.read_csv(baseline_path, dtype={"ID": str})
    required = {
        "ID", *TARGETS,
        *[f"pred_{target}_mean" for target in TARGETS],
        *[f"pred_{target}_std_10models" for target in TARGETS],
    }
    if missing := required.difference(optimized.columns):
        raise ValueError(
            f"Optimized predictions miss columns: {sorted(missing)}")
    baseline_required = {
        "ID", *[f"pred_{target}_mean" for target in TARGETS]}
    if missing := baseline_required.difference(baseline.columns):
        raise ValueError(f"Baseline predictions miss columns: {sorted(missing)}")
    if optimized["ID"].duplicated().any() or baseline["ID"].duplicated().any():
        raise ValueError("Prediction IDs must be unique.")
    if set(optimized["ID"]) != set(baseline["ID"]):
        raise ValueError("Optimized and baseline prediction IDs differ.")
    baseline = baseline.set_index("ID").loc[optimized["ID"]].reset_index()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    row_output = optimized[["ID", *TARGETS]].copy()
    for target in TARGETS:
        truth = pd.to_numeric(optimized[target], errors="coerce").to_numpy(float)
        old_prediction = pd.to_numeric(
            baseline[f"pred_{target}_mean"], errors="coerce").to_numpy(float)
        new_prediction = pd.to_numeric(
            optimized[f"pred_{target}_mean"], errors="coerce").to_numpy(float)
        if not (
            np.isfinite(truth).all()
            and np.isfinite(old_prediction).all()
            and np.isfinite(new_prediction).all()
        ):
            raise ValueError(f"{target} contains non-finite evaluation values.")
        metric_rows.append(
            metric_row(target, "O12_identity_10seed", truth, old_prediction))
        metric_rows.append(
            metric_row(target, optimized_model_name, truth, new_prediction))
        row_output[f"baseline_{target}"] = old_prediction
        row_output[f"optimized_{target}"] = new_prediction
        row_output[f"baseline_{target}_same_side"] = (
            (truth > THRESHOLD) == (old_prediction > THRESHOLD))
        row_output[f"optimized_{target}_same_side"] = (
            (truth > THRESHOLD) == (new_prediction > THRESHOLD))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "new_validation_norm_metrics.csv", index=False)
    row_output.to_csv(
        output / "new_validation_norm_predictions_and_sides.csv", index=False)
    comparison_rows = []
    for target in TARGETS:
        old = metrics.loc[
            metrics["target"].eq(target)
            & metrics["model"].eq("O12_identity_10seed")
        ].iloc[0]
        new = metrics.loc[
            metrics["target"].eq(target)
            & metrics["model"].eq(optimized_model_name)
        ].iloc[0]
        comparison_rows.append({
            "target": target,
            "baseline_mae": old["mae"],
            "optimized_mae": new["mae"],
            "mae_reduction": old["mae"] - new["mae"],
            "mae_reduction_percent": (
                100.0 * (old["mae"] - new["mae"]) / old["mae"]),
            "baseline_side_agreement": old["side_agreement"],
            "optimized_side_agreement": new["side_agreement"],
            "side_agreement_gain": (
                new["side_agreement"] - old["side_agreement"]),
            "strictly_above_80_percent": new["strictly_above_80_percent"],
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        output / "new_validation_norm_improvement.csv", index=False)

    colors = {
        "O12_identity_10seed": "#9c9c9c",
        optimized_model_name: "#4c78a8",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 9.4))
    for row_index, target in enumerate(TARGETS):
        truth = optimized[target].to_numpy(float)
        for column_index, model_name in enumerate(
                ("O12_identity_10seed", optimized_model_name)):
            axis = axes[row_index, column_index]
            prediction = (
                baseline[f"pred_{target}_mean"].to_numpy(float)
                if column_index == 0
                else optimized[f"pred_{target}_mean"].to_numpy(float)
            )
            metric = metrics.loc[
                metrics["target"].eq(target)
                & metrics["model"].eq(model_name)
            ].iloc[0]
            lower = float(min(truth.min(), prediction.min()))
            upper = float(max(truth.max(), prediction.max()))
            padding = max((upper - lower) * 0.06, 0.05)
            limits = (lower - padding, upper + padding)
            axis.scatter(
                truth, prediction, s=45, alpha=0.84,
                color=colors[model_name], edgecolor="#222222", linewidth=0.35)
            axis.plot(
                limits, limits, "--", color="#d62728", linewidth=1.4,
                label="y = x")
            axis.axvline(
                THRESHOLD, color="#f58518", linestyle=":", linewidth=1.2)
            axis.axhline(
                THRESHOLD, color="#f58518", linestyle=":", linewidth=1.2)
            axis.set(
                xlim=limits,
                ylim=limits,
                xlabel="True value",
                ylabel="Predicted value",
            )
            axis.set_aspect("equal", adjustable="box")
            model_title = (
                "Identity-target O12"
                if column_index == 0 else "Input-only optimized O12")
            axis.set_title(
                f"{DISPLAY_NAMES[target]} — {model_title}\n"
                f"MAE={metric['mae']:.3f}; same side="
                f"{metric['side_agreement_n']}/{metric['n']} "
                f"({metric['side_agreement_percent']:.1f}%)")
            axis.grid(alpha=0.25)
            axis.legend(loc="upper left", fontsize=8)
    figure.suptitle(
        "Frozen GraphGPS O12 ensembles: final new_validation evaluation",
        fontsize=15, y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("png", "pdf"):
        figure.savefig(
            output / f"new_validation_o12_baseline_vs_optimized.{suffix}",
            dpi=180 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)

    optimized_figure, optimized_axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, target, color in zip(
            optimized_axes, TARGETS, ("#4c78a8", "#e45756")):
        truth = optimized[target].to_numpy(float)
        prediction = optimized[f"pred_{target}_mean"].to_numpy(float)
        uncertainty = optimized[
            f"pred_{target}_std_10models"].to_numpy(float)
        metric = metrics.loc[
            metrics["target"].eq(target)
            & metrics["model"].eq(optimized_model_name)
        ].iloc[0]
        lower = float(min(truth.min(), prediction.min()))
        upper = float(max(truth.max(), prediction.max()))
        padding = max((upper - lower) * 0.06, 0.05)
        limits = (lower - padding, upper + padding)
        axis.errorbar(
            truth, prediction, yerr=uncertainty, fmt="none", ecolor=color,
            alpha=0.24, linewidth=0.8, capsize=1.5)
        axis.scatter(
            truth, prediction, s=46, alpha=0.86, color=color,
            edgecolor="#222222", linewidth=0.35)
        axis.plot(limits, limits, "--", color="#d62728", linewidth=1.4)
        axis.axvline(
            THRESHOLD, color="#f58518", linestyle=":", linewidth=1.2)
        axis.axhline(
            THRESHOLD, color="#f58518", linestyle=":", linewidth=1.2)
        axis.set(
            xlim=limits,
            ylim=limits,
            xlabel="True value",
            ylabel="Optimized O12 ten-seed mean",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"{DISPLAY_NAMES[target]}\nMAE={metric['mae']:.3f}; "
            f"same side={metric['side_agreement_percent']:.1f}%")
        axis.grid(alpha=0.25)
    optimized_figure.suptitle(
        "Optimized GraphGPS O12: Norm property scatter plots",
        fontsize=15, y=1.01)
    optimized_figure.tight_layout()
    for suffix in ("png", "pdf"):
        optimized_figure.savefig(
            output / f"new_validation_o12_optimized_norm_scatter.{suffix}",
            dpi=180 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(optimized_figure)

    optimized_metrics = metrics.loc[
        metrics["model"].eq(optimized_model_name)]
    report = {
        "evaluation_only": True,
        "training_or_selection_performed": False,
        "threshold_used_only_for_final_external_evaluation": THRESHOLD,
        "strict_requirement": (
            f"each target side agreement > {REQUIRED_SIDE_AGREEMENT:.2f}"),
        "all_targets_meet_requirement": bool(
            optimized_metrics["strictly_above_80_percent"].all()),
        "rows": int(len(optimized)),
        "optimized_model": optimized_model_name,
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "baseline": str(baseline_path),
        "baseline_sha256": sha256(baseline_path),
        "prediction_provenance": provenance,
    }
    (output / "evaluation_protocol.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(metrics.to_string(index=False))
    print()
    print(comparison.to_string(index=False))
    print(
        "\nAll optimized targets strictly above 80%:",
        report["all_targets_meet_requirement"])


if __name__ == "__main__":
    main()
