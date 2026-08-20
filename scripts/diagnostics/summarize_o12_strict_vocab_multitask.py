#!/usr/bin/env python3
"""Summarize frozen O12 validation and test predictions for one vocabulary setup.

The vocabulary sizes and unknown-row policy are read from the saved run
metadata.  They may be either the historical [3, 4, 3, 4] setup or a strict
[2, 3, 2, 3] setup, but all ten checkpoints in one summary must match.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET_GROUPS = {
    "core4": [
        "EE_before",
        "EE_after",
        "Aerosolization_Efficiency",
        "mRNA_Recovery_Efficiency",
    ],
    "norm2": ["Norm_before", "Norm_after"],
}


def correlation(first: np.ndarray, second: np.ndarray, method: str) -> float:
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(pd.Series(first).corr(pd.Series(second), method=method))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    arguments = parser.parse_args()
    runs_root = arguments.runs_root.resolve()
    rows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    expected_vocab_sizes: list[int] | None = None
    expected_vocab_strict: bool | None = None

    for target_group, targets in TARGET_GROUPS.items():
        for split_seed in range(100, 110):
            run_dir = runs_root / f"O12_{target_group}_split{split_seed}"
            settings_path = run_dir / "run_settings.json"
            predictions_path = run_dir / "predictions.csv"
            checkpoint_path = run_dir / "checkpoints" / "selected_best.pt"
            if not settings_path.is_file() or not predictions_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(f"Incomplete strict-vocabulary run: {run_dir}")
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            vocabulary_sizes = settings.get("component_vocab_sizes")
            vocabulary_strict = settings.get("component_vocab_strict")
            if (not isinstance(vocabulary_sizes, list)
                    or len(vocabulary_sizes) != 4
                    or not all(isinstance(size, int) and size > 0 for size in vocabulary_sizes)
                    or not isinstance(vocabulary_strict, bool)):
                raise RuntimeError(
                    f"Incomplete component-vocabulary metadata in {run_dir}: "
                    f"sizes={vocabulary_sizes!r}, strict={vocabulary_strict!r}"
                )
            if expected_vocab_sizes is None:
                expected_vocab_sizes = vocabulary_sizes
                expected_vocab_strict = vocabulary_strict
            elif (vocabulary_sizes != expected_vocab_sizes
                  or vocabulary_strict != expected_vocab_strict):
                raise RuntimeError(
                    "Cannot summarize checkpoints with different component vocabularies: "
                    f"expected sizes={expected_vocab_sizes}, strict={expected_vocab_strict}; "
                    f"found sizes={vocabulary_sizes}, strict={vocabulary_strict} in {run_dir}"
                )
            predictions = pd.read_csv(predictions_path)
            inventory.append(
                {
                    "target_group": target_group,
                    "split_seed": split_seed,
                    "run_dir": str(run_dir),
                    "checkpoint": str(checkpoint_path),
                    "best_epoch": int(predictions["epoch"].iloc[0]),
                    "component_vocab_sizes": json.dumps(vocabulary_sizes),
                    "component_vocab_strict": vocabulary_strict,
                }
            )
            for split in ("val", "test"):
                for target in targets:
                    part = predictions.loc[
                        predictions["split"].eq(split) & predictions["target"].eq(target)
                    ]
                    if len(part) == 0:
                        raise RuntimeError(
                            f"Expected at least one {split}/{target} row in {run_dir}, got none"
                        )
                    y_true = part["y_true"].to_numpy(float)
                    y_pred = part["y_pred"].to_numpy(float)
                    rows.append(
                        {
                            "target_group": target_group,
                            "split_seed": split_seed,
                            "split": split,
                            "target": target,
                            "n": len(part),
                            "mae": mean_absolute_error(y_true, y_pred),
                            "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
                            "r2": r2_score(y_true, y_pred),
                            "pearson": correlation(y_true, y_pred, "pearson"),
                            "spearman": correlation(y_true, y_pred, "spearman"),
                        }
                    )

    metrics = pd.DataFrame(rows)
    target_average = (
        metrics.groupby(["target_group", "split", "target"], as_index=False)
        .agg(
            completed_seeds=("split_seed", "nunique"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_pearson=("pearson", "mean"),
            mean_spearman=("spearman", "mean"),
        )
    )
    macro = (
        metrics.groupby(["target_group", "split", "split_seed"], as_index=False)[
            ["mae", "rmse", "r2", "pearson", "spearman"]
        ]
        .mean()
        .groupby(["target_group", "split"], as_index=False)
        .agg(
            completed_seeds=("split_seed", "nunique"),
            mean_macro_mae=("mae", "mean"),
            std_macro_mae=("mae", "std"),
            mean_macro_r2=("r2", "mean"),
            std_macro_r2=("r2", "std"),
            mean_macro_pearson=("pearson", "mean"),
            mean_macro_spearman=("spearman", "mean"),
        )
    )
    pd.DataFrame(inventory).to_csv(runs_root / "checkpoint_inventory.csv", index=False)
    metrics.to_csv(runs_root / "validation_test_metrics_by_seed_target.csv", index=False)
    target_average.to_csv(runs_root / "validation_test_metrics_target_average.csv", index=False)
    macro.to_csv(runs_root / "validation_test_metrics_macro_average.csv", index=False)
    per_target_dir = runs_root / "validation_test_metrics_by_target"
    per_target_dir.mkdir(parents=True, exist_ok=True)
    for target in sorted(metrics["target"].unique()):
        seed_metrics = metrics.loc[metrics["target"].eq(target)].copy()
        average_metrics = target_average.loc[target_average["target"].eq(target)].copy()
        seed_metrics.to_csv(per_target_dir / f"{target}_by_seed.csv", index=False)
        average_metrics.to_csv(per_target_dir / f"{target}_average.csv", index=False)
    print(target_average.to_string(index=False))
    print()
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
