#!/usr/bin/env python3
"""Score saved O12 ten-model ensemble predictions on a labelled CSV.

The script never loads or changes checkpoints.  It scores the arithmetic-mean
prediction and every individual checkpoint prediction already written by
``predict_o12_10seed_ensemble.py``.  Labels are used only here, after model
inference, to compute metrics and true-vs-predicted scatter plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET_GROUPS = {
    "core4": [
        "EE_before", "EE_after", "Aerosolization_Efficiency",
        "mRNA_Recovery_Efficiency",
    ],
    "norm2": ["Norm_before", "Norm_after"],
}
ALL_TARGETS = [target for targets in TARGET_GROUPS.values() for target in targets]


def target_key(target_group: str, single_target: str | None) -> str:
    if single_target is None:
        return target_group
    return f"single_{single_target.lower()}"


def correlation(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.std(truth) == 0 or np.std(prediction) == 0:
        return float("nan")
    return float(function(truth, prediction).statistic)


def metric_frame(frame: pd.DataFrame, targets: list[str], prediction_columns: dict[str, str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        truth = frame[target].to_numpy(dtype=float)
        prediction = frame[prediction_columns[target]].to_numpy(dtype=float)
        rows.append({
            "target": target,
            "n": len(frame),
            "mae": float(mean_absolute_error(truth, prediction)),
            "rmse": float(mean_squared_error(truth, prediction) ** .5),
            "r2": float(r2_score(truth, prediction)),
            "pearson": correlation(pearsonr, truth, prediction),
            "spearman": correlation(spearmanr, truth, prediction),
        })
    return pd.DataFrame(rows)


def scatter(frame: pd.DataFrame, targets: list[str], metric: pd.DataFrame, output: Path,
            target_group: str, model_label: str) -> None:
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    per_target = output / "scatter_by_target"
    per_target.mkdir(parents=True, exist_ok=True)
    for index, target in enumerate(targets):
        truth = frame[target].to_numpy(float)
        prediction = frame[f"pred_{target}_mean"].to_numpy(float)
        lower, upper = min(float(truth.min()), float(prediction.min())), max(float(truth.max()), float(prediction.max()))
        padding = max((upper - lower) * .06, .1)
        limits = (lower - padding, upper + padding)
        result = metric.loc[metric.target.eq(target)].iloc[0]
        figure, axis = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
        # Fifth_class is a dataset field used only for visual stratification;
        # neither model inference nor the metric calculation above uses it.
        # Keep an explicit fallback for non-feedback tables without the field.
        classes = (frame["Fifth_class"].fillna("other").astype(str).str.strip().str.lower()
                   if "Fifth_class" in frame else pd.Series("other", index=frame.index))
        marker_spec = (("single", "o", "#4c78a8"), ("double", "s", "#f58518"))
        plotted = pd.Series(False, index=frame.index)
        for label, marker, color in marker_spec:
            selected = classes.eq(label)
            if selected.any():
                axis.scatter(truth[selected], prediction[selected], s=38, alpha=.84,
                             marker=marker, color=color, edgecolor="#222", linewidth=.35,
                             label=f"Fifth_class = {label}")
                plotted |= selected
        if (~plotted).any():
            axis.scatter(truth[~plotted], prediction[~plotted], s=42, alpha=.84,
                         marker="X", color="#777777", edgecolor="#222", linewidth=.3,
                         label="Fifth_class = other/missing")
        axis.plot(limits, limits, "--", color="#d62728", linewidth=1.35, label="y = x")
        axis.set(xlabel="True value", ylabel="Predicted value", xlim=limits, ylim=limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=.25)
        axis.legend(loc="upper left", fontsize=8)
        axis.set_title(f"{model_label}: {target}\nMAE = {result.mae:.3f}, R² = {result.r2:.3f}")
        figure.savefig(per_target / f"{target}_true_vs_pred.png", dpi=180, bbox_inches="tight")
        figure.savefig(per_target / f"{target}_true_vs_pred.pdf", bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-group", choices=tuple(TARGET_GROUPS), required=True)
    parser.add_argument("--single-target", choices=ALL_TARGETS, default=None,
                        help="Score a one-output model; uses single_<target> prediction files.")
    parser.add_argument("--model-label", default="O12 input-only 700",
                        help="Text written in the per-target scatter-plot titles.")
    args = parser.parse_args()
    targets = [args.single_target] if args.single_target is not None else TARGET_GROUPS[args.target_group]
    key = target_key(args.target_group, args.single_target)
    labels = pd.read_csv(args.labels_csv, dtype={"ID": str})
    predictions = pd.read_csv(
        args.prediction_dir / f"ensemble_mean_predictions_{key}.csv", dtype={"ID": str}
    )
    long_predictions = pd.read_csv(
        args.prediction_dir / f"predictions_by_model_long_{key}.csv", dtype={"ID": str}
    )
    required = {"ID", *targets}
    if missing := required.difference(labels.columns):
        raise ValueError(f"Labels CSV misses columns: {sorted(missing)}")
    if labels.ID.duplicated().any() or predictions.ID.duplicated().any():
        raise ValueError("Labels and ensemble predictions must have unique IDs.")
    mean_columns = {target: f"pred_{target}_mean" for target in targets}
    if missing := set(mean_columns.values()).difference(predictions.columns):
        raise ValueError(f"Ensemble prediction columns missing: {sorted(missing)}")
    if set(labels.ID) != set(predictions.ID):
        raise ValueError("Labels and predictions do not contain the same IDs.")
    label_columns = ["ID", *targets]
    if "Fifth_class" in labels:
        label_columns.append("Fifth_class")
    merged = labels[label_columns].merge(
        predictions[["ID", *mean_columns.values()]], on="ID", how="left", validate="one_to_one"
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "ensemble_mean_predictions_with_labels.csv", index=False)
    ensemble_metric = metric_frame(merged, targets, mean_columns)
    ensemble_metric.to_csv(output / "metrics_ensemble.csv", index=False)

    checkpoint_rows = []
    for seed in sorted(long_predictions.split_seed.unique()):
        part = long_predictions.loc[long_predictions.split_seed.eq(seed)]
        wide = part.pivot(index="ID", columns="target", values="prediction").reset_index()
        checkpoint_columns = {target: f"checkpoint_pred_{target}" for target in targets}
        wide = wide.rename(columns=checkpoint_columns)
        wide = labels[["ID", *targets]].merge(wide, on="ID", how="left", validate="one_to_one")
        seed_metric = metric_frame(wide, targets, checkpoint_columns)
        seed_metric.insert(0, "split_seed", int(seed))
        checkpoint_rows.append(seed_metric)
    pd.concat(checkpoint_rows, ignore_index=True).to_csv(output / "metrics_by_checkpoint.csv", index=False)
    scatter(merged, targets, ensemble_metric, output, args.target_group, args.model_label)
    print(ensemble_metric.to_string(index=False))


if __name__ == "__main__":
    main()
