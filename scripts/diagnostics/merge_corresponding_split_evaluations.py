#!/usr/bin/env python3
"""Merge short-batch outputs from evaluate_o12_10seed_corresponding_splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def write_full_source_ensemble(parts_root: Path, output: Path, seeds: list[int], source: pd.DataFrame) -> None:
    """Average existing per-seed full-source predictions and stratify plots."""
    classes = source[["ID", "Fifth_class"]].copy() if "Fifth_class" in source else None
    for group in ("core4", "norm2"):
        tables = []
        for seed in seeds:
            path = parts_root / f"seed{seed}" / "full_training_data_ensemble" / group / "ensemble_mean_predictions.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing full-source prediction: {path}")
            tables.append(pd.read_csv(path, dtype={"sample_id": str}))
        first = tables[0]
        target_names = [name.removeprefix("y_true_") for name in first if name.startswith("y_true_")]
        table = first[["source_index", "sample_id", *[f"y_true_{name}" for name in target_names]]].copy()
        if classes is not None:
            table = table.merge(classes.rename(columns={"ID": "sample_id"}), on="sample_id", how="left", validate="one_to_one")
        out = output / "full_training_data_ensemble" / group
        plot_dir = out / "scatter_by_target"; plot_dir.mkdir(parents=True, exist_ok=True)
        metric_rows = []
        for target in target_names:
            prediction = np.stack([frame[f"pred_{target}_mean"].to_numpy(float) for frame in tables]).mean(axis=0)
            truth = table[f"y_true_{target}"].to_numpy(float)
            table[f"pred_{target}_mean"] = prediction
            table[f"pred_{target}_std_10models"] = np.stack(
                [frame[f"pred_{target}_mean"].to_numpy(float) for frame in tables]).std(axis=0, ddof=0)
            metric_rows.append({"target_group": group, "target": target, "n": len(table),
                                "mae": float(mean_absolute_error(truth, prediction)),
                                "r2": float(r2_score(truth, prediction))})
            lower, upper = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
            padding = max((upper - lower) * .06, .1); limits = (lower - padding, upper + padding)
            figure, axis = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
            labels = table.Fifth_class.fillna("other").astype(str).str.strip().str.lower() if "Fifth_class" in table else pd.Series("other", index=table.index)
            for label, marker, color in (("single", "o", "#4c78a8"), ("double", "s", "#f58518")):
                selected = labels.eq(label)
                if selected.any():
                    axis.scatter(truth[selected], prediction[selected], s=28, alpha=.78, marker=marker,
                                 color=color, edgecolor="#222", linewidth=.3, label=f"Fifth_class = {label}")
            other = ~labels.isin(("single", "double"))
            if other.any():
                axis.scatter(truth[other], prediction[other], s=35, alpha=.78, marker="X", color="#777",
                             edgecolor="#222", linewidth=.3, label="Fifth_class = other/missing")
            axis.plot(limits, limits, "--", color="#d62728", linewidth=1.35, label="y = x")
            axis.set(xlabel="True value", ylabel="Mean prediction", xlim=limits, ylim=limits)
            axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.25); axis.legend(loc="upper left", fontsize=8)
            row = metric_rows[-1]
            axis.set_title(f"O13-C no-Mordred, full training-source data: {target}\nMAE = {row['mae']:.3f}, R² = {row['r2']:.3f}")
            figure.savefig(plot_dir / f"{target}_true_vs_pred.png", dpi=180, bbox_inches="tight")
            figure.savefig(plot_dir / f"{target}_true_vs_pred.pdf", bbox_inches="tight")
            plt.close(figure)
        table.to_csv(out / "ensemble_mean_predictions.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(out / "metrics_ensemble.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--target-groups", nargs="+", default=["core4", "norm2"])
    parser.add_argument("--input-csv", type=Path,
                        help="Optional source CSV. When supplied, also creates class-stratified full-source ensemble plots.")
    args = parser.parse_args()
    metrics, predictions = [], []
    for seed in args.seeds:
        part = args.parts_root / f"seed{seed}"
        metric_path = part / "metrics_by_checkpoint_target.csv"
        prediction_path = part / "predictions_by_checkpoint.csv"
        if not metric_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"Missing completed evaluation part for seed {seed}: {part}")
        metric = pd.read_csv(metric_path)
        prediction = pd.read_csv(prediction_path, dtype={"sample_id": str})
        if set(metric.split_seed) != {seed} or set(prediction.split_seed) != {seed}:
            raise RuntimeError(f"Evaluation part contains unexpected split seeds: {part}")
        metrics.append(metric)
        predictions.append(prediction)
    metrics = pd.concat(metrics, ignore_index=True).sort_values(
        ["target_group", "split", "split_seed", "target"])
    predictions = pd.concat(predictions, ignore_index=True).sort_values(
        ["target_group", "split", "split_seed", "target", "source_index"])
    expected = set(args.seeds)
    for group in args.target_groups:
        found = set(metrics.loc[metrics.target_group.eq(group), "split_seed"])
        if found != expected:
            raise RuntimeError(f"{group} has incomplete seed coverage: found {sorted(found)}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics_by_checkpoint_target.csv", index=False)
    predictions.to_csv(output / "predictions_by_checkpoint.csv", index=False)
    target_summary = metrics.groupby(["target_group", "split", "target"], as_index=False).agg(
        checkpoints=("split_seed", "nunique"),
        mean_mae=("mae", "mean"), variance_mae=("mae", "var"), std_mae=("mae", "std"),
        mean_r2=("r2", "mean"), variance_r2=("r2", "var"), std_r2=("r2", "std"),
        mean_rmse=("rmse", "mean"), variance_rmse=("rmse", "var"),
    ).sort_values(["split", "target_group", "target"])
    target_summary.to_csv(output / "metrics_target_10seed_mean_variance.csv", index=False)
    macro = metrics.groupby(["target_group", "split", "split_seed"], as_index=False).agg(
        targets=("target", "count"), mean_mae=("mae", "mean"),
        mean_r2=("r2", "mean"), mean_rmse=("rmse", "mean"))
    macro.to_csv(output / "metrics_macro_by_checkpoint.csv", index=False)
    macro_summary = macro.groupby(["target_group", "split"], as_index=False).agg(
        checkpoints=("split_seed", "nunique"),
        mean_mae=("mean_mae", "mean"), variance_mae=("mean_mae", "var"), std_mae=("mean_mae", "std"),
        mean_r2=("mean_r2", "mean"), variance_r2=("mean_r2", "var"), std_r2=("mean_r2", "std"),
    ).sort_values(["split", "target_group"])
    macro_summary.to_csv(output / "metrics_macro_10seed_mean_variance.csv", index=False)
    (output / "provenance.json").write_text(json.dumps({
        "parts_root": str(args.parts_root.resolve()), "seeds": args.seeds,
        "target_groups": args.target_groups,
        "method": "Concatenated independent frozen single-seed evaluations; no checkpoint was altered.",
    }, indent=2) + "\n", encoding="utf-8")
    if args.input_csv:
        write_full_source_ensemble(args.parts_root.resolve(), output, args.seeds,
                                   pd.read_csv(args.input_csv, dtype={"ID": str}))
    print(target_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
