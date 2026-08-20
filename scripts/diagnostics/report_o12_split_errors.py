#!/usr/bin/env python3
"""Report validation/test prediction errors across O12 split checkpoints.

The error bars in the MAE figure are the sample standard deviation across the
available split seeds; they are not standard errors or confidence intervals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def read_metrics(runs_root: Path, target_group: str, seeds: range) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        prediction_path = runs_root / f"O12_{target_group}_split{seed}" / "predictions.csv"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing prediction file: {prediction_path}")
        predictions = pd.read_csv(prediction_path)
        required = {"split", "target", "y_true", "y_pred"}
        missing = required.difference(predictions.columns)
        if missing:
            raise ValueError(f"{prediction_path} misses columns: {sorted(missing)}")
        for split in ("val", "test"):
            for target, part in predictions.loc[predictions["split"].eq(split)].groupby("target"):
                if part.empty:
                    continue
                truth, values = part["y_true"].to_numpy(float), part["y_pred"].to_numpy(float)
                rows.append({
                    "target_group": target_group,
                    "split_seed": seed,
                    "split": split,
                    "target": target,
                    "n": len(part),
                    "mae": mean_absolute_error(truth, values),
                    "rmse": mean_squared_error(truth, values) ** 0.5,
                    "r2": r2_score(truth, values) if np.std(truth) else np.nan,
                })
    if not rows:
        raise RuntimeError("No validation or test predictions were found.")
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["target_group", "split", "target"], as_index=False)
        .agg(
            completed_seeds=("split_seed", "nunique"),
            samples_per_seed=("n", "first"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
        )
    )


def plot_mae(summary: pd.DataFrame, output: Path) -> None:
    splits = [split for split in ("val", "test") if split in set(summary["split"])]
    targets = list(summary["target"].drop_duplicates())
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    x = np.arange(len(targets), dtype=float)
    width = 0.34 if len(splits) == 2 else 0.56
    colors = {"val": "#4c78a8", "test": "#f58518"}
    for index, split in enumerate(splits):
        part = summary.loc[summary["split"].eq(split)].set_index("target").reindex(targets)
        positions = x + (index - (len(splits) - 1) / 2) * width
        bars = axis.bar(
            positions, part["mean_mae"], width=width,
            yerr=part["std_mae"], capsize=5, color=colors.get(split),
            edgecolor="#222", linewidth=.5, label=f"{split}: mean ± SD (n=10)",
        )
        for bar, mean, std in zip(bars, part["mean_mae"], part["std_mae"]):
            axis.annotate(
                f"{mean:.3f}±{std:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height() + std),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8,
            )
    axis.set(
        xticks=x, xticklabels=targets, ylabel="MAE", xlabel="Target property",
        title="O12 validation/test prediction error across split seeds",
    )
    axis.grid(axis="y", alpha=.25)
    axis.legend()
    figure.savefig(output / "mae_mean_std_errorbar.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / "mae_mean_std_errorbar.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--target-group", default="norm2")
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-end", type=int, default=109)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be no smaller than --seed-start")
    runs_root = args.runs_root.resolve()
    output = (args.output_dir or runs_root / "validation_test_error_report").resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics(runs_root, args.target_group, range(args.seed_start, args.seed_end + 1))
    summary = summarize(metrics)
    metrics.to_csv(output / "prediction_error_by_seed.csv", index=False)
    summary.to_csv(output / "prediction_error_mean_std.csv", index=False)
    plot_mae(summary, output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
