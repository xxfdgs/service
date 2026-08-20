#!/usr/bin/env python3
"""Seed-paired Fifth-OOD comparison: O13-E vs strict-scaled O13-C."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


KEYS = ["target_group", "split_seed", "split", "target"]
METRICS = ("mae", "r2", "pearson", "spearman", "prediction_std", "true_std", "prediction_std_to_true_std")


def corr(function, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return math.nan
    return float(function(y_true, y_pred).statistic)


def per_seed(metrics_path: Path, predictions_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path, dtype={"sample_id": str})
    required = {*KEYS, "mae", "r2", "pearson", "spearman"}
    if missing := required.difference(metrics.columns):
        raise ValueError(f"Metric table misses {sorted(missing)}")
    rows = []
    for values, frame in predictions.groupby(KEYS, sort=False):
        true_std = float(np.std(frame.y_true, ddof=0))
        rows.append({**dict(zip(KEYS, values)), "prediction_std": float(np.std(frame.y_pred, ddof=0)),
                     "true_std": true_std,
                     "prediction_std_to_true_std": float(np.std(frame.y_pred, ddof=0) / true_std)
                     if true_std else math.nan})
    merged = metrics[KEYS + ["mae", "r2", "pearson", "spearman"]].merge(
        pd.DataFrame(rows), on=KEYS, validate="one_to_one")
    return merged, predictions


def class_metrics(predictions: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    fifth_class = source[["ID", "Fifth_class"]].rename(columns={"ID": "sample_id"}).copy()
    fifth_class["fifth_class"] = fifth_class.Fifth_class.astype(str).str.strip().str.lower()
    joined = predictions.loc[predictions.split.eq("test")].merge(
        fifth_class[["sample_id", "fifth_class"]], on="sample_id", how="left", validate="many_to_one")
    if joined.fifth_class.isna().any():
        raise RuntimeError("Prediction IDs are not present in the locked source CSV")
    rows = []
    for values, frame in joined.loc[joined.fifth_class.isin(["single", "double"])].groupby(KEYS + ["fifth_class"]):
        y_true, y_pred = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
        rows.append({**dict(zip(KEYS + ["fifth_class"], values)), "n": len(frame),
                     "mae": float(mean_absolute_error(y_true, y_pred)),
                     "r2": float(r2_score(y_true, y_pred)) if np.std(y_true) else math.nan,
                     "pearson": corr(pearsonr, y_true, y_pred), "spearman": corr(spearmanr, y_true, y_pred)})
    return pd.DataFrame(rows)


def summarize(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, target), frame in paired.groupby(["split", "target"], sort=True):
        row = {"split": split, "target": target, "seeds": len(frame)}
        for metric in METRICS:
            base, candidate = frame[f"{metric}_o13c"], frame[f"{metric}_o13e"]
            delta = candidate - base
            row.update({f"o13c_{metric}_mean": float(base.mean()), f"o13c_{metric}_std": float(base.std(ddof=1)),
                        f"o13e_{metric}_mean": float(candidate.mean()), f"o13e_{metric}_std": float(candidate.std(ddof=1)),
                        f"paired_delta_o13e_minus_o13c_{metric}_mean": float(delta.mean()),
                        f"paired_delta_o13e_minus_o13c_{metric}_std": float(delta.std(ddof=1))})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--o13c-metrics", type=Path, required=True)
    parser.add_argument("--o13c-predictions", type=Path, required=True)
    parser.add_argument("--o13e-metrics", type=Path, required=True)
    parser.add_argument("--o13e-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.input_csv, dtype={"ID": str})
    if source.ID.duplicated().any() or "Fifth_class" not in source:
        raise ValueError("Locked source CSV must have unique ID and Fifth_class")
    o13c, o13c_pred = per_seed(args.o13c_metrics, args.o13c_predictions)
    o13e, o13e_pred = per_seed(args.o13e_metrics, args.o13e_predictions)
    if set(map(tuple, o13c[KEYS].to_numpy())) != set(map(tuple, o13e[KEYS].to_numpy())):
        raise RuntimeError("O13-C/E metrics have unequal seed/split/target keys")
    paired = o13c.merge(o13e, on=KEYS, suffixes=("_o13c", "_o13e"), validate="one_to_one")
    for metric in METRICS:
        paired[f"delta_o13e_minus_o13c_{metric}"] = paired[f"{metric}_o13e"] - paired[f"{metric}_o13c"]
    c_class, e_class = class_metrics(o13c_pred, source), class_metrics(o13e_pred, source)
    class_keys = KEYS + ["fifth_class"]
    if set(map(tuple, c_class[class_keys].to_numpy())) != set(map(tuple, e_class[class_keys].to_numpy())):
        raise RuntimeError("O13-C/E within-class metric keys differ")
    paired_class = c_class.merge(e_class, on=class_keys, suffixes=("_o13c", "_o13e"), validate="one_to_one")
    for metric in ("mae", "r2", "pearson", "spearman"):
        paired_class[f"delta_o13e_minus_o13c_{metric}"] = paired_class[f"{metric}_o13e"] - paired_class[f"{metric}_o13c"]
    class_summary = paired_class.groupby(["target", "fifth_class"], as_index=False).agg(
        seeds=("split_seed", "nunique"), o13c_spearman_mean=("spearman_o13c", "mean"),
        o13c_spearman_std=("spearman_o13c", "std"), o13e_spearman_mean=("spearman_o13e", "mean"),
        o13e_spearman_std=("spearman_o13e", "std"),
        paired_delta_spearman_mean=("delta_o13e_minus_o13c_spearman", "mean"),
        paired_delta_spearman_std=("delta_o13e_minus_o13c_spearman", "std"))
    summary = summarize(paired)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "o13e_vs_o13c_paired_per_seed_metrics.csv", index=False)
    summary.to_csv(output / "o13e_vs_o13c_paired_summary.csv", index=False)
    paired_class.to_csv(output / "o13e_vs_o13c_paired_within_class_per_seed.csv", index=False)
    class_summary.to_csv(output / "o13e_vs_o13c_within_class_spearman_summary.csv", index=False)
    test = summary.loc[summary.split.eq("test")]
    better_mae = int(test.paired_delta_o13e_minus_o13c_mae_mean.lt(0).sum())
    better_r2 = int(test.paired_delta_o13e_minus_o13c_r2_mean.gt(0).sum())
    better_rank = int(test.paired_delta_o13e_minus_o13c_spearman_mean.gt(0).sum())
    lines = ["# O13-E strict train-only-scaling paired Fifth-OOD diagnostic", "",
             "All deltas are O13-E minus strict-scaling O13-C; ΔMAE<0 and ΔR²/ΔSpearman>0 improve.", "",
             "| target | O13-C MAE | O13-E MAE | ΔMAE | ΔR² | ΔPearson | ΔSpearman | Δ(σpred/σtrue) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in test.itertuples(index=False):
        lines.append(f"| {row.target} | {row.o13c_mae_mean:.4f} ± {row.o13c_mae_std:.4f} | {row.o13e_mae_mean:.4f} ± {row.o13e_mae_std:.4f} | {row.paired_delta_o13e_minus_o13c_mae_mean:+.4f} | {row.paired_delta_o13e_minus_o13c_r2_mean:+.4f} | {row.paired_delta_o13e_minus_o13c_pearson_mean:+.4f} | {row.paired_delta_o13e_minus_o13c_spearman_mean:+.4f} | {row.paired_delta_o13e_minus_o13c_prediction_std_to_true_std_mean:+.4f} |")
    lines += ["", "## Within-class test Spearman", "", "| target | Fifth_class | O13-C | O13-E | Δ |", "|---|---|---:|---:|---:|"]
    for row in class_summary.itertuples(index=False):
        lines.append(f"| {row.target} | {row.fifth_class} | {row.o13c_spearman_mean:.4f} ± {row.o13c_spearman_std:.4f} | {row.o13e_spearman_mean:.4f} ± {row.o13e_spearman_std:.4f} | {row.paired_delta_spearman_mean:+.4f} |")
    lines += ["", "## Aggregate direction", "", f"- Test MAE improves on {better_mae}/6 targets; R² improves on {better_r2}/6; Spearman improves on {better_rank}/6.",
              "- Final acceptance/rejection must use these paired tables and seed-level stability, not one holdout split."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
