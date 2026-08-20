#!/usr/bin/env python3
"""
Summarize Stage-5 P0/P1/P2 selected-best test predictions.

Reads each run's predictions.csv and selects:
    split == test
    target == Norm_before
    checkpoint == selected_best when the column exists

Outputs per-run and paired summaries with:
    MAE, RMSE, R2, Pearson, Spearman
    threshold metrics at y > 1:
        precision, recall, F1, F2, TP/TN/FP/FN

This is an evaluation script only; it never influences checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


LABELS = ("P0_random", "P1_PT_D", "P2_PT_DF")


def safe_corr(fn, y, p):
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        return math.nan
    return float(fn(y, p).statistic)


def threshold_metrics(y, p, threshold=1.0):
    true_pos = y > threshold
    pred_pos = p > threshold

    tp = int(np.sum(true_pos & pred_pos))
    tn = int(np.sum(~true_pos & ~pred_pos))
    fp = int(np.sum(~true_pos & pred_pos))
    fn = int(np.sum(true_pos & ~pred_pos))

    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan

    def fbeta(beta):
        if not np.isfinite(precision) or not np.isfinite(recall):
            return math.nan
        denom = beta * beta * precision + recall
        return (
            (1 + beta * beta) * precision * recall / denom
            if denom > 0
            else 0.0
        )

    return {
        "precision_gt1": precision,
        "recall_gt1": recall,
        "f1_gt1": fbeta(1.0),
        "f2_gt1": fbeta(2.0),
        "tp_gt1": tp,
        "tn_gt1": tn,
        "fp_gt1": fp,
        "fn_gt1": fn,
    }


def load_selected_test(path: Path):
    df = pd.read_csv(path)

    required = {"split", "target", "y_true", "y_pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df = df.loc[
        df["split"].astype(str).eq("test")
        & df["target"].astype(str).eq("Norm_before")
    ].copy()

    if "checkpoint" in df.columns:
        values = set(df["checkpoint"].dropna().astype(str))
        if "selected_best" in values:
            df = df.loc[df["checkpoint"].astype(str).eq("selected_best")]
        elif len(values) > 1:
            raise ValueError(
                f"{path}: multiple test checkpoint labels without selected_best: {sorted(values)}"
            )

    # One selected prediction per sample.
    if "sample_id" in df.columns and df["sample_id"].duplicated().any():
        duplicated = df.loc[df["sample_id"].duplicated(False), "sample_id"].tolist()[:10]
        raise ValueError(f"{path}: duplicated selected test samples: {duplicated}")

    if df.empty:
        raise ValueError(f"{path}: no selected test Norm_before predictions")

    return df


def metrics(df):
    y = df["y_true"].to_numpy(dtype=float)
    p = df["y_pred"].to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]

    out = {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 and np.std(y) else math.nan,
        "pearson": safe_corr(pearsonr, y, p),
        "spearman": safe_corr(spearmanr, y, p),
    }
    out.update(threshold_metrics(y, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage5-root",
        type=Path,
        default=Path("results/fifth_pretraining/stage5_downstream_transfer"),
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        type=int,
        default=[100, 101, 102],
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = ap.parse_args()

    root = args.stage5_root.resolve()
    outdir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "analysis"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for label in LABELS:
        for split_seed in args.splits:
            run_dir = root / label / f"split{split_seed}"
            pred_path = run_dir / "predictions.csv"
            init_path = run_dir / "comp5_initialization.json"

            if not pred_path.is_file():
                raise FileNotFoundError(pred_path)
            if not init_path.is_file():
                raise FileNotFoundError(init_path)

            init = json.loads(init_path.read_text())
            if init.get("label") != label:
                raise ValueError(
                    f"{init_path}: label {init.get('label')!r} != {label!r}"
                )

            row = {
                "label": label,
                "split_seed": split_seed,
                **metrics(load_selected_test(pred_path)),
                "init_mode": init.get("mode"),
                "checkpoint_sha256": init.get("checkpoint_sha256"),
            }
            rows.append(row)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(outdir / "stage5_per_run_metrics.csv", index=False)

    metric_cols = [
        "mae", "rmse", "r2", "pearson", "spearman",
        "precision_gt1", "recall_gt1", "f1_gt1", "f2_gt1",
        "tp_gt1", "tn_gt1", "fp_gt1", "fn_gt1",
    ]

    summary_rows = []
    for label, group in per_run.groupby("label", sort=False):
        row = {"label": label, "splits": int(len(group))}
        for col in metric_cols:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else math.nan
            row[f"{col}_median"] = float(vals.median())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "stage5_group_summary.csv", index=False)

    baseline = per_run.loc[per_run["label"].eq("P0_random")].set_index("split_seed")
    paired_rows = []

    # Sign convention: positive improvement is always better.
    for label in ("P1_PT_D", "P2_PT_DF"):
        candidate = per_run.loc[per_run["label"].eq(label)].set_index("split_seed")
        if set(candidate.index) != set(baseline.index):
            raise ValueError(f"{label}: split seeds do not pair with P0")

        for split_seed in sorted(baseline.index):
            b = baseline.loc[split_seed]
            c = candidate.loc[split_seed]
            paired_rows.append({
                "label": label,
                "split_seed": int(split_seed),
                "improvement_mae": float(b["mae"] - c["mae"]),
                "improvement_rmse": float(b["rmse"] - c["rmse"]),
                "improvement_r2": float(c["r2"] - b["r2"]),
                "improvement_pearson": float(c["pearson"] - b["pearson"]),
                "improvement_spearman": float(c["spearman"] - b["spearman"]),
                "improvement_recall_gt1": float(c["recall_gt1"] - b["recall_gt1"]),
                "improvement_f2_gt1": float(c["f2_gt1"] - b["f2_gt1"]),
                "fn_reduction_gt1": float(b["fn_gt1"] - c["fn_gt1"]),
                "fp_reduction_gt1": float(b["fp_gt1"] - c["fp_gt1"]),
            })

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(outdir / "stage5_paired_improvements.csv", index=False)

    print("=" * 84)
    print("STAGE 5 P0 / P1 / P2 SCREENING SUMMARY")
    print("=" * 84)
    show = [
        "label",
        "mae_mean",
        "r2_mean",
        "spearman_mean",
        "recall_gt1_mean",
        "f2_gt1_mean",
        "fn_gt1_mean",
        "fp_gt1_mean",
    ]
    print(summary[show].to_string(index=False))
    print()
    print("Paired improvement means vs P0 (positive = better):")
    print(
        paired.groupby("label")[
            [
                "improvement_mae",
                "improvement_r2",
                "improvement_spearman",
                "improvement_recall_gt1",
                "improvement_f2_gt1",
                "fn_reduction_gt1",
            ]
        ].mean().to_string()
    )
    print()
    print(f"Results written to: {outdir}")


if __name__ == "__main__":
    main()
