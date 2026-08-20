#!/usr/bin/env python3
"""Average seed100-109 feedback predictions by model and create comparison plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency",
           "mRNA_Recovery_Efficiency", "Norm_before", "Norm_after"]
MODEL_LABELS = {
    ("multitask_o12", "O12"): "O12_multitask",
    ("single_task_o12", "O12"): "O12_single_task",
    ("multitask_baseline", "GCN"): "GCN_multitask",
    ("multitask_baseline", "GIN"): "GIN_multitask",
    ("multitask_baseline", "MPNN"): "MPNN_multitask",
    ("multitask_baseline", "Transformer"): "Transformer_multitask",
    ("multitask_baseline", "MLP"): "MLP_multitask",
}
MODEL_ORDER = list(MODEL_LABELS.values())
DISPLAY = {
    "EE_before": "EE before", "EE_after": "EE after",
    "Aerosolization_Efficiency": "Aerosolization efficiency",
    "mRNA_Recovery_Efficiency": "mRNA recovery efficiency",
    "Norm_before": "Norm before", "Norm_after": "Norm after",
}
COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6"]


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    truth, prediction = frame.y_true.to_numpy(float), frame.y_pred_mean.to_numpy(float)
    return {"n_feedback": len(frame), "mae": float(mean_absolute_error(truth, prediction)),
            "rmse": float(mean_squared_error(truth, prediction) ** .5),
            "r2": float(r2_score(truth, prediction)),
            "prediction_std_mean": float(frame.y_pred_std.mean())}


def plot_model(frame: pd.DataFrame, model: str, output: Path, color: str) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for axis, target in zip(axes.flat, TARGETS):
        part = frame.loc[(frame.model == model) & (frame.target == target)]
        truth, prediction = part.y_true.to_numpy(float), part.y_pred_mean.to_numpy(float)
        lower, upper = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
        margin = max(.05 * (upper - lower), .05)
        limits = (lower - margin, upper + margin)
        summary = metrics(part)
        axis.scatter(truth, prediction, s=28, alpha=.72, color=color,
                     edgecolors="#222222", linewidths=.25)
        axis.plot(limits, limits, "--", color="#d62728", linewidth=1.2)
        axis.set(xlim=limits, ylim=limits, xlabel="True value", ylabel="Mean prediction")
        axis.grid(alpha=.22)
        axis.set_title(f"{DISPLAY[target]}\nMAE={summary['mae']:.3f}; R²={summary['r2']:.3f}")
    figure.suptitle(f"Feedback: {model} (mean of seed100–109 checkpoints)", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, .96))
    figure.savefig(output / f"{model}_feedback_scatter.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / f"{model}_feedback_scatter.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--feedback-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(arguments.predictions)
    required = {"family", "model", "target", "split_seed", "sample_id", "y_true", "y_pred"}
    if missing := required.difference(raw.columns):
        raise ValueError(f"Prediction CSV misses columns: {sorted(missing)}")
    raw["model_label"] = [MODEL_LABELS.get((family, model))
                          for family, model in zip(raw.family, raw.model)]
    if raw.model_label.isna().any():
        raise ValueError("Unexpected model family/name in prediction input.")
    raw = raw.loc[raw.target.isin(TARGETS)].copy()
    counts = raw.groupby(["model_label", "target", "sample_id"], as_index=False).agg(
        checkpoints=("split_seed", "nunique"), true_values=("y_true", "nunique"))
    if not counts.checkpoints.eq(10).all() or not counts.true_values.eq(1).all():
        raise RuntimeError("Each model/target/feedback ID must have ten checkpoints and one truth value.")
    mean = raw.groupby(["model_label", "target", "sample_id"], as_index=False).agg(
        y_true=("y_true", "first"), y_pred_mean=("y_pred", "mean"), y_pred_std=("y_pred", "std"),
        checkpoints=("split_seed", "nunique"))
    mean = mean.rename(columns={"model_label": "model", "sample_id": "feedback_id"})
    feedback = pd.read_csv(arguments.feedback_csv, dtype={"ID": str})
    if feedback.ID.duplicated().any():
        raise ValueError("Feedback CSV IDs must be unique.")
    order = {identifier: index for index, identifier in enumerate(feedback.ID.astype(str))}
    if set(mean.feedback_id.astype(str)) != set(order):
        raise ValueError("Prediction IDs do not match the feedback CSV.")
    mean["feedback_id"] = mean.feedback_id.astype(str)
    mean["feedback_order"] = mean.feedback_id.map(order)
    mean["model_order"] = mean.model.map({model: index for index, model in enumerate(MODEL_ORDER)})
    mean["target_order"] = mean.target.map({target: index for index, target in enumerate(TARGETS)})
    mean = mean.sort_values(["feedback_order", "target_order", "model_order"]).drop(
        columns=["feedback_order", "model_order", "target_order"])
    mean.to_csv(output / "feedback_model_mean_predictions_long.csv", index=False)

    truth = mean[["feedback_id", "target", "y_true"]].drop_duplicates(
        ["feedback_id", "target"]
    ).pivot(index="feedback_id", columns="target", values="y_true").reindex(columns=TARGETS)
    truth.columns = [f"true__{target}" for target in truth.columns]
    prediction = mean.pivot(index="feedback_id", columns=["model", "target"], values="y_pred_mean")
    prediction = prediction.reindex(columns=pd.MultiIndex.from_product([MODEL_ORDER, TARGETS]))
    prediction.columns = [f"{model}__{target}" for model, target in prediction.columns]
    wide = pd.concat([truth, prediction], axis=1).reindex(feedback.ID.astype(str))
    wide.index.name = "feedback_id"
    wide.reset_index().to_csv(output / "feedback_id_model_mean_predictions.csv", index=False)

    metric_rows = []
    for (model, target), part in mean.groupby(["model", "target"], sort=False):
        metric_rows.append({"model": model, "target": target, **metrics(part)})
    metric = pd.DataFrame(metric_rows)
    metric["model_order"] = metric.model.map({model: index for index, model in enumerate(MODEL_ORDER)})
    metric["target_order"] = metric.target.map({target: index for index, target in enumerate(TARGETS)})
    metric.sort_values(["model_order", "target_order"]).drop(columns=["model_order", "target_order"]).to_csv(
        output / "feedback_model_mean_prediction_metrics.csv", index=False)

    plot_dir = output / "scatter_plots"
    plot_dir.mkdir(exist_ok=True)
    for model, color in zip(MODEL_ORDER, COLORS):
        plot_model(mean, model, plot_dir, color)
    (output / "provenance.json").write_text(json.dumps({
        "source_predictions": str(arguments.predictions.resolve()),
        "feedback_csv": str(arguments.feedback_csv.resolve()),
        "models": MODEL_ORDER, "targets": TARGETS, "seed_splits": list(range(100, 110)),
        "aggregation": "arithmetic mean and sample standard deviation of ten checkpoint predictions",
        "wide_csv_layout": "one row per feedback ID; true and mean-prediction columns for every model/target",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
