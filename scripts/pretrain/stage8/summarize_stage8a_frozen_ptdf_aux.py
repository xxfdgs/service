#!/usr/bin/env python3
"""Compare Stage-8 frozen PT-DF auxiliary branch to Stage-5 baselines."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


MODELS = (
    "P0_random",
    "P1_PT_D",
    "P2_PT_DF",
    "P3_PT_DF_FrozenAux",
)


def selected_test(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"split", "target", "y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")

    frame = frame.loc[
        frame["split"].astype(str).eq("test")
        & frame["target"].astype(str).eq("Norm_before")
    ].copy()

    if "checkpoint" in frame.columns:
        labels = set(frame["checkpoint"].dropna().astype(str))
        selected = [x for x in labels if "selected_best.pt" in x]
        if selected:
            frame = frame.loc[
                frame["checkpoint"].astype(str).isin(selected)
            ].copy()
        elif len(labels) > 1:
            raise ValueError(f"{path}: ambiguous checkpoints {sorted(labels)}")

    if frame.empty:
        raise ValueError(f"{path}: no selected test predictions")
    return frame


def metrics(frame: pd.DataFrame) -> dict:
    y = frame["y_true"].to_numpy(float)
    p = frame["y_pred"].to_numpy(float)

    true_pos = y > 1.0
    pred_pos = p > 1.0
    tp = int(np.sum(true_pos & pred_pos))
    tn = int(np.sum(~true_pos & ~pred_pos))
    fp = int(np.sum(~true_pos & pred_pos))
    fn = int(np.sum(true_pos & ~pred_pos))

    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    if np.isfinite(precision) and np.isfinite(recall):
        denom = 4 * precision + recall
        f2 = 5 * precision * recall / denom if denom > 0 else 0.0
    else:
        f2 = math.nan

    spearman = (
        float(spearmanr(y, p).statistic)
        if len(y) > 1 and np.std(y) and np.std(p)
        else math.nan
    )

    return {
        "n": len(y),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)) if len(y) > 1 and np.std(y) else math.nan,
        "spearman": spearman,
        "precision_gt1": precision,
        "recall_gt1": recall,
        "f2_gt1": f2,
        "tp_gt1": tp,
        "fn_gt1": fn,
        "fp_gt1": fp,
        "tn_gt1": tn,
        "prediction_mean": float(np.mean(p)),
        "prediction_std": float(np.std(p, ddof=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/fifth_pretraining/stage5_downstream_transfer"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        default=[100, 101, 102],
    )
    args = parser.parse_args()

    rows = []
    for model in MODELS:
        for split in args.splits:
            path = args.root / model / f"split{split}" / "predictions.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append({
                "model": model,
                "split_seed": split,
                **metrics(selected_test(path)),
            })

    df = pd.DataFrame(rows)
    output = args.root / "analysis_stage8a"
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "stage8a_per_run_metrics.csv", index=False)

    cols = [
        "mae", "r2", "spearman", "precision_gt1", "recall_gt1", "f2_gt1",
        "fn_gt1", "fp_gt1", "prediction_mean", "prediction_std",
    ]

    summaries = []
    for model in MODELS:
        group = df.loc[df.model.eq(model)]
        row = {"model": model, "splits": len(group)}
        for col in cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1))
        summaries.append(row)

    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "stage8a_group_summary.csv", index=False)

    baseline = df.loc[df.model.eq("P0_random")].set_index("split_seed")
    p3 = df.loc[df.model.eq("P3_PT_DF_FrozenAux")].set_index("split_seed")

    paired = []
    for split in sorted(p3.index):
        b, c = baseline.loc[split], p3.loc[split]
        paired.append({
            "split_seed": split,
            "improvement_mae": b.mae - c.mae,
            "improvement_r2": c.r2 - b.r2,
            "improvement_spearman": c.spearman - b.spearman,
            "improvement_recall_gt1": c.recall_gt1 - b.recall_gt1,
            "improvement_f2_gt1": c.f2_gt1 - b.f2_gt1,
            "fn_reduction_gt1": b.fn_gt1 - c.fn_gt1,
            "fp_reduction_gt1": b.fp_gt1 - c.fp_gt1,
        })

    paired = pd.DataFrame(paired)
    paired.to_csv(output / "stage8a_p3_vs_p0_paired.csv", index=False)

    display = [
        "model", "mae_mean", "r2_mean", "spearman_mean",
        "precision_gt1_mean", "recall_gt1_mean", "f2_gt1_mean",
        "fn_gt1_mean", "fp_gt1_mean",
        "prediction_mean_mean", "prediction_std_mean",
    ]

    print("=" * 116)
    print("STAGE 8A — FROZEN PT-DF AUXILIARY BRANCH SCREENING")
    print("=" * 116)
    print(summary[display].to_string(index=False))
    print()
    print("P3 vs P0 paired improvements (positive = P3 better):")
    print(paired.to_string(index=False))
    print()
    print("Mean paired improvements:")
    print(paired.drop(columns="split_seed").mean().to_string())
    print()
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
