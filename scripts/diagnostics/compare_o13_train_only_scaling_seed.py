#!/usr/bin/env python3
"""Compare the strict train-only-scaled O12 and O13-C checkpoints for one seed.

This intentionally reports a *single* frozen Fifth-identity-OOD split.  It
does not relabel that split as a random-split result and it does not create a
meaningless across-seed standard deviation for one checkpoint.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


KEYS = ["target_group", "split_seed", "split", "target"]


def correlation(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.std(truth) == 0 or np.std(prediction) == 0:
        return math.nan
    return float(function(truth, prediction).statistic)


def metrics(predictions: pd.DataFrame, source: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {*KEYS, "sample_id", "y_true", "y_pred"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"{label} predictions miss columns: {sorted(missing)}")
    test = predictions.loc[predictions.split.eq("test")].copy()
    rows = []
    for key, frame in test.groupby(KEYS, sort=True):
        truth, prediction = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
        true_std = float(np.std(truth, ddof=0))
        prediction_std = float(np.std(prediction, ddof=0))
        rows.append({**dict(zip(KEYS, key)), "n": len(frame),
                     "mae": float(mean_absolute_error(truth, prediction)),
                     "r2": float(r2_score(truth, prediction)) if true_std else math.nan,
                     "pearson": correlation(pearsonr, truth, prediction),
                     "spearman": correlation(spearmanr, truth, prediction),
                     "prediction_std": prediction_std, "true_std": true_std,
                     "prediction_std_to_true_std": prediction_std / true_std if true_std else math.nan})
    joined = test.merge(source[["ID", "Fifth_class"]].rename(columns={"ID": "sample_id"}),
                        on="sample_id", how="left", validate="many_to_one")
    if joined.Fifth_class.isna().any():
        raise RuntimeError(f"{label} includes sample IDs absent from locked source input")
    class_rows = []
    joined["fifth_class"] = joined.Fifth_class.astype(str).str.strip().str.lower()
    for key, frame in joined.loc[joined.fifth_class.isin(["single", "double"])].groupby(KEYS + ["fifth_class"], sort=True):
        truth, prediction = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
        class_rows.append({**dict(zip(KEYS + ["fifth_class"], key)), "n": len(frame),
                           "spearman": correlation(spearmanr, truth, prediction),
                           "true_mean": float(np.mean(truth)), "predicted_mean": float(np.mean(prediction))})
    return pd.DataFrame(rows), pd.DataFrame(class_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--o12-predictions", type=Path, required=True)
    parser.add_argument("--o13c-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.input_csv, dtype={"ID": str})
    if source.ID.duplicated().any() or "Fifth_class" not in source:
        raise ValueError("Input CSV must have unique ID and Fifth_class columns")
    o12, o12_class = metrics(pd.read_csv(args.o12_predictions, dtype={"sample_id": str}), source, "O12")
    o13c, o13c_class = metrics(pd.read_csv(args.o13c_predictions, dtype={"sample_id": str}), source, "O13-C")
    if set(map(tuple, o12[KEYS].to_numpy())) != set(map(tuple, o13c[KEYS].to_numpy())):
        raise RuntimeError("O12/O13-C test target keys differ")
    paired = o12.merge(o13c, on=KEYS, suffixes=("_o12", "_o13c"), validate="one_to_one")
    for name in ("mae", "r2", "pearson", "spearman", "prediction_std", "true_std", "prediction_std_to_true_std"):
        paired[f"delta_o13c_minus_o12_{name}"] = paired[f"{name}_o13c"] - paired[f"{name}_o12"]
    class_keys = KEYS + ["fifth_class"]
    paired_class = o12_class.merge(o13c_class, on=class_keys, suffixes=("_o12", "_o13c"), validate="one_to_one")
    paired_class["delta_o13c_minus_o12_spearman"] = paired_class.spearman_o13c - paired_class.spearman_o12
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "o12_vs_o13c_seed_paired_test_metrics.csv", index=False)
    paired_class.to_csv(output / "o12_vs_o13c_seed_paired_test_by_fifth_class.csv", index=False)
    lines = ["# Part A: strict train-only descriptor scaling, one Fifth-OOD split", "",
             "All deltas are O13-C mean-pooling minus O12 add-pooling.  This is one seed, so no cross-seed standard deviation is reported.", "",
             "| target | O12 MAE | O13-C MAE | ΔMAE | O12 R² | O13-C R² | ΔR² | ΔSpearman | Δ(σpred/σtrue) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in paired.itertuples(index=False):
        lines.append(f"| {row.target} | {row.mae_o12:.4f} | {row.mae_o13c:.4f} | {row.delta_o13c_minus_o12_mae:+.4f} | {row.r2_o12:.4f} | {row.r2_o13c:.4f} | {row.delta_o13c_minus_o12_r2:+.4f} | {row.delta_o13c_minus_o12_spearman:+.4f} | {row.delta_o13c_minus_o12_prediction_std_to_true_std:+.4f} |")
    lines += ["", "## Fifth-OOD within-class Spearman", "", "| target | class | O12 | O13-C | Δ |", "|---|---|---:|---:|---:|"]
    for row in paired_class.itertuples(index=False):
        lines.append(f"| {row.target} | {row.fifth_class} | {row.spearman_o12:.4f} | {row.spearman_o13c:.4f} | {row.delta_o13c_minus_o12_spearman:+.4f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
