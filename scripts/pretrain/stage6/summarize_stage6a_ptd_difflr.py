#!/usr/bin/env python3
"""Compare Stage-5 P0/P1 with Stage-6A PT-D differential-LR on Fifth-OOD."""

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
    "P1_PT_D_diffLR1e4",
)


def selected_test_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"split", "target", "y_true", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    frame = frame.loc[
        frame["split"].astype(str).eq("test")
        & frame["target"].astype(str).eq("Norm_before")
    ].copy()

    if "checkpoint" in frame.columns:
        labels = set(frame["checkpoint"].dropna().astype(str))
        selected = [label for label in labels if "selected_best.pt" in label]
        if selected:
            frame = frame.loc[frame["checkpoint"].astype(str).isin(selected)]
        elif len(labels) > 1:
            raise ValueError(
                f"{path} has multiple test checkpoint labels: {sorted(labels)}"
            )

    if frame.empty:
        raise ValueError(f"{path}: no test Norm_before predictions")
    if "sample_id" in frame.columns and frame["sample_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate selected test sample_id values")
    return frame


def safe_spearman(y, p):
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        return math.nan
    return float(spearmanr(y, p).statistic)


def threshold_metrics(y, p, threshold=1.0):
    positive = y > threshold
    predicted = p > threshold
    tp = int(np.sum(positive & predicted))
    tn = int(np.sum(~positive & ~predicted))
    fp = int(np.sum(~positive & predicted))
    fn = int(np.sum(positive & ~predicted))
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan

    if np.isfinite(precision) and np.isfinite(recall):
        denom = 4 * precision + recall
        f2 = 5 * precision * recall / denom if denom > 0 else 0.0
    else:
        f2 = math.nan

    return {
        "precision_gt1": precision,
        "recall_gt1": recall,
        "f2_gt1": f2,
        "tp_gt1": tp,
        "tn_gt1": tn,
        "fp_gt1": fp,
        "fn_gt1": fn,
    }


def metrics(frame):
    y = frame["y_true"].to_numpy(dtype=float)
    p = frame["y_pred"].to_numpy(dtype=float)
    result = {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)) if len(y) > 1 and np.std(y) else math.nan,
        "spearman": safe_spearman(y, p),
        "target_mean": float(np.mean(y)),
        "prediction_mean": float(np.mean(p)),
        "target_std": float(np.std(y, ddof=0)),
        "prediction_std": float(np.std(p, ddof=0)),
    }
    result.update(threshold_metrics(y, p))
    return result


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "analysis_stage6a"
    )
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in MODELS:
        for split_seed in args.splits:
            run_dir = root / model / f"split{split_seed}"
            pred_path = run_dir / "predictions.csv"
            if not pred_path.is_file():
                raise FileNotFoundError(pred_path)

            row = {
                "model": model,
                "split_seed": int(split_seed),
                **metrics(selected_test_predictions(pred_path)),
            }

            opt_meta = run_dir / "optimizer_parameter_groups.json"
            if opt_meta.is_file():
                import json
                meta = json.loads(opt_meta.read_text())
                row["rest_lr"] = meta.get("rest_lr")
                row["comp5_lr"] = meta.get("comp5_lr")
            else:
                row["rest_lr"] = 0.001
                row["comp5_lr"] = 0.001

            rows.append(row)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(output / "stage6a_per_run_metrics.csv", index=False)

    metric_cols = [
        "mae", "r2", "spearman",
        "precision_gt1", "recall_gt1", "f2_gt1",
        "fn_gt1", "fp_gt1",
        "prediction_mean", "prediction_std",
    ]

    summary_rows = []
    for model, group in per_run.groupby("model", sort=False):
        row = {"model": model, "splits": len(group)}
        for column in metric_cols:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else math.nan
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "stage6a_group_summary.csv", index=False)

    baseline = per_run.loc[
        per_run["model"].eq("P1_PT_D")
    ].set_index("split_seed")
    diff = per_run.loc[
        per_run["model"].eq("P1_PT_D_diffLR1e4")
    ].set_index("split_seed")

    if set(baseline.index) != set(diff.index):
        raise ValueError("Full-FT and diff-LR PT-D splits do not pair exactly.")

    paired_rows = []
    for split_seed in sorted(baseline.index):
        b = baseline.loc[split_seed]
        d = diff.loc[split_seed]
        paired_rows.append({
            "split_seed": int(split_seed),
            # Positive = differential LR is better.
            "improvement_mae": float(b["mae"] - d["mae"]),
            "improvement_r2": float(d["r2"] - b["r2"]),
            "improvement_spearman": float(d["spearman"] - b["spearman"]),
            "improvement_recall_gt1": float(
                d["recall_gt1"] - b["recall_gt1"]
            ),
            "improvement_f2_gt1": float(d["f2_gt1"] - b["f2_gt1"]),
            "fn_reduction_gt1": float(b["fn_gt1"] - d["fn_gt1"]),
            "fp_reduction_gt1": float(b["fp_gt1"] - d["fp_gt1"]),
        })

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "stage6a_paired_vs_ptd_fullft.csv", index=False)

    print("=" * 100)
    print("STAGE 6A — PT-D DIFFERENTIAL LR SCREENING")
    print("=" * 100)
    columns = [
        "model",
        "mae_mean",
        "r2_mean",
        "spearman_mean",
        "recall_gt1_mean",
        "f2_gt1_mean",
        "fn_gt1_mean",
        "fp_gt1_mean",
        "prediction_mean_mean",
        "prediction_std_mean",
    ]
    print(summary[columns].to_string(index=False))

    print()
    print("Paired diff-LR improvement vs PT-D full FT (positive = better):")
    print(
        paired[
            [
                "split_seed",
                "improvement_mae",
                "improvement_r2",
                "improvement_spearman",
                "improvement_recall_gt1",
                "improvement_f2_gt1",
                "fn_reduction_gt1",
                "fp_reduction_gt1",
            ]
        ].to_string(index=False)
    )
    print()
    print("Mean paired improvement:")
    print(
        paired.drop(columns=["split_seed"]).mean().to_string()
    )
    print()
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
