#!/usr/bin/env python3
"""Summarize and plot test-set predictions from completed model repeat runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


TARGETS = (
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
)
MODEL_LABELS = {
    "O12": "O12",
    "O22": "O22",
    "ensemble_huber": "O12_O22_ensemble_huber",
}


def correlation(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.std(truth) == 0 or np.std(prediction) == 0:
        return float("nan")
    return float(function(truth, prediction).statistic)


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame.y_true.to_numpy(float)
    prediction = frame.y_pred.to_numpy(float)
    return {
        "n": len(frame),
        "mae": float(mean_absolute_error(truth, prediction)),
        "r2": float(r2_score(truth, prediction)),
        "pearson": correlation(pearsonr, truth, prediction),
        "spearman": correlation(spearmanr, truth, prediction),
    }


def scatter_plot(frame: pd.DataFrame, model: str, repeat: int, seed: int, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, target in zip(axes.flat, TARGETS):
        target_frame = frame.loc[frame.target == target]
        result = metrics(target_frame)
        truth = target_frame.y_true.to_numpy(float)
        prediction = target_frame.y_pred.to_numpy(float)
        minimum = min(truth.min(), prediction.min())
        maximum = max(truth.max(), prediction.max())
        margin = max(1.0, (maximum - minimum) * .04)
        line = np.array([minimum - margin, maximum + margin])
        axis.scatter(truth, prediction, s=24, alpha=.78, color="#2e75b6", edgecolors="none")
        axis.plot(line, line, color="#d94841", linewidth=1.5, label="y = x")
        axis.set_xlim(line[0], line[1])
        axis.set_ylim(line[0], line[1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("True value")
        axis.set_ylabel("Predicted value")
        axis.set_title(f"{target}\nMAE={result['mae']:.3f}, R²={result['r2']:.3f}")
        axis.grid(alpha=.22)
    figure.suptitle(f"{model} test predictions — repeat {repeat:02d}, training seed {seed}", fontsize=14)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="repeat10_o12_o22 directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_LABELS), required=True,
                        help="Repeat subdirectory to evaluate.")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = arguments.output_dir.resolve()
    model_label = MODEL_LABELS[arguments.model]
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    macro_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for repeat_dir in sorted((root / "repeats").glob("repeat_*_seed_*")):
        tokens = repeat_dir.name.split("_")
        repeat, seed = int(tokens[1]), int(tokens[-1])
        prediction_path = repeat_dir / arguments.model / "predictions.csv"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        prediction = pd.read_csv(prediction_path)
        test = prediction.loc[prediction.split == "test"].copy()
        if set(test.target) != set(TARGETS):
            raise RuntimeError(f"Unexpected test targets in {prediction_path}")
        test["repeat"] = repeat
        test["seed"] = seed
        prediction_rows.append(test)
        for target, target_frame in test.groupby("target", sort=False):
            rows.append({"repeat": repeat, "seed": seed, "split": "test", "target": target,
                         **metrics(target_frame)})
        metric_frame = pd.DataFrame(rows).query("repeat == @repeat")
        macro_rows.append({"repeat": repeat, "seed": seed, "split": "test",
                           "mean_mae": float(metric_frame.mae.mean()),
                           "mean_r2": float(metric_frame.r2.mean()),
                           "mean_pearson": float(metric_frame.pearson.mean()),
                           "mean_spearman": float(metric_frame.spearman.mean())})
        scatter_plot(test, model_label, repeat, seed,
                     output / f"{model_label}_repeat{repeat:02d}_seed{seed}_test_scatter")

    per_target = pd.DataFrame(rows).sort_values(["repeat", "target"])
    per_repeat = pd.DataFrame(macro_rows).sort_values("repeat")
    average_target = per_target.groupby("target", as_index=False).agg(
        repeats=("repeat", "nunique"), mean_mae=("mae", "mean"), std_mae=("mae", "std"),
        mean_r2=("r2", "mean"), std_r2=("r2", "std"),
        mean_pearson=("pearson", "mean"), mean_spearman=("spearman", "mean"),
    )
    average_macro = pd.DataFrame([{
        "repeats": len(per_repeat), "mean_mae": float(per_repeat.mean_mae.mean()),
        "std_mae": float(per_repeat.mean_mae.std(ddof=1)),
        "mean_r2": float(per_repeat.mean_r2.mean()),
        "std_r2": float(per_repeat.mean_r2.std(ddof=1)),
        "mean_pearson": float(per_repeat.mean_pearson.mean()),
        "mean_spearman": float(per_repeat.mean_spearman.mean()),
    }])
    prefix = f"{model_label}_repeat10_test"
    pd.concat(prediction_rows, ignore_index=True).to_csv(output / f"{prefix}_predictions.csv", index=False)
    per_target.to_csv(output / f"{prefix}_metrics_by_target.csv", index=False)
    per_repeat.to_csv(output / f"{prefix}_metrics_by_seed.csv", index=False)
    average_target.to_csv(output / f"{prefix}_average_by_target.csv", index=False)
    average_macro.to_csv(output / f"{prefix}_average_macro.csv", index=False)
    print(average_macro.to_csv(index=False).strip())


if __name__ == "__main__":
    main()
