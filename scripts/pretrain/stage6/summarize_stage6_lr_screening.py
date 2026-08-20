#!/usr/bin/env python3
"""Summarize Stage-6 PT-D differential-LR screening on the same Fifth-OOD splits.

Compared models
---------------
P0_random
P1_PT_D
P1_PT_D_diffLR1e4
P1_PT_D_diffLR3e4

Primary metrics
---------------
MAE, R2, Spearman, Precision/Recall/F2 at Norm_before > 1,
FN/FP, prediction mean/std.

Outputs
-------
stage6_lr_per_run_metrics.csv
stage6_lr_group_summary.csv
stage6_lr_paired_vs_ptd_fullft.csv
stage6_lr_paired_3e4_vs_1e4.csv
stage6_lr_win_counts_vs_ptd_fullft.csv

Positive paired "improvement_*" values always mean the row model is better
than the comparator:
- MAE: comparator - candidate
- R2/Spearman/Recall/F2: candidate - comparator
- FN/FP reduction: comparator - candidate
"""

from __future__ import annotations

import argparse
import json
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
    "P1_PT_D_diffLR3e4",
)

FULL_FT_MODEL = "P1_PT_D"
DIFF_1E4_MODEL = "P1_PT_D_diffLR1e4"
DIFF_3E4_MODEL = "P1_PT_D_diffLR3e4"


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

    # Prefer selected_best checkpoint rows when predictions.csv contains
    # predictions from more than one checkpoint.
    if "checkpoint" in frame.columns:
        labels = set(frame["checkpoint"].dropna().astype(str))
        selected = [
            label for label in labels
            if "selected_best.pt" in label
        ]
        if selected:
            frame = frame.loc[
                frame["checkpoint"].astype(str).isin(selected)
            ].copy()
        elif len(labels) > 1:
            raise ValueError(
                f"{path} has multiple test checkpoint labels but none can be "
                f"identified as selected_best.pt: {sorted(labels)}"
            )

    if frame.empty:
        raise ValueError(f"{path}: no selected test Norm_before predictions")

    if "sample_id" in frame.columns and frame["sample_id"].duplicated().any():
        duplicated = (
            frame.loc[frame["sample_id"].duplicated(keep=False), "sample_id"]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"{path}: duplicate selected test sample_id values: "
            f"{duplicated[:20]}"
        )

    return frame.reset_index(drop=True)


def safe_spearman(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        return math.nan
    return float(spearmanr(y, p).statistic)


def threshold_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float = 1.0,
) -> dict[str, float | int]:
    positive = y > threshold
    predicted = p > threshold

    tp = int(np.sum(positive & predicted))
    tn = int(np.sum(~positive & ~predicted))
    fp = int(np.sum(~positive & predicted))
    fn = int(np.sum(positive & ~predicted))

    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan

    if np.isfinite(precision) and np.isfinite(recall):
        denominator = 4.0 * precision + recall
        f2 = (
            5.0 * precision * recall / denominator
            if denominator > 0
            else 0.0
        )
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


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame["y_true"].to_numpy(dtype=float)
    p = frame["y_pred"].to_numpy(dtype=float)

    result = {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "r2": (
            float(r2_score(y, p))
            if len(y) > 1 and np.std(y) > 0
            else math.nan
        ),
        "spearman": safe_spearman(y, p),
        "target_mean": float(np.mean(y)),
        "prediction_mean": float(np.mean(p)),
        "target_std": float(np.std(y, ddof=0)),
        "prediction_std": float(np.std(p, ddof=0)),
    }
    result.update(threshold_metrics(y, p))
    return result


def read_lr_metadata(run_dir: Path, model: str) -> tuple[float, float]:
    metadata_path = run_dir / "optimizer_parameter_groups.json"

    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            float(metadata.get("rest_lr", 0.001)),
            float(metadata.get("comp5_lr", 0.001)),
        )

    # Historical Stage-5 runs did not necessarily have this file.
    if model in {"P0_random", "P1_PT_D"}:
        return 0.001, 0.001

    raise FileNotFoundError(
        f"{metadata_path} is required for differential-LR model {model}"
    )


def paired_improvements(
    per_run: pd.DataFrame,
    candidate_model: str,
    comparator_model: str,
) -> pd.DataFrame:
    candidate = (
        per_run.loc[per_run["model"].eq(candidate_model)]
        .set_index("split_seed")
        .sort_index()
    )
    comparator = (
        per_run.loc[per_run["model"].eq(comparator_model)]
        .set_index("split_seed")
        .sort_index()
    )

    if set(candidate.index) != set(comparator.index):
        raise ValueError(
            f"Split mismatch: {candidate_model} has {sorted(candidate.index)}, "
            f"{comparator_model} has {sorted(comparator.index)}"
        )

    rows = []
    for split_seed in sorted(candidate.index):
        c = candidate.loc[split_seed]
        b = comparator.loc[split_seed]

        rows.append({
            "candidate_model": candidate_model,
            "comparator_model": comparator_model,
            "split_seed": int(split_seed),
            # Positive always means candidate better.
            "improvement_mae": float(b["mae"] - c["mae"]),
            "improvement_r2": float(c["r2"] - b["r2"]),
            "improvement_spearman": float(
                c["spearman"] - b["spearman"]
            ),
            "improvement_precision_gt1": float(
                c["precision_gt1"] - b["precision_gt1"]
            ),
            "improvement_recall_gt1": float(
                c["recall_gt1"] - b["recall_gt1"]
            ),
            "improvement_f2_gt1": float(
                c["f2_gt1"] - b["f2_gt1"]
            ),
            "fn_reduction_gt1": float(
                b["fn_gt1"] - c["fn_gt1"]
            ),
            "fp_reduction_gt1": float(
                b["fp_gt1"] - c["fp_gt1"]
            ),
            "prediction_std_change": float(
                c["prediction_std"] - b["prediction_std"]
            ),
            "prediction_mean_change": float(
                c["prediction_mean"] - b["prediction_mean"]
            ),
        })

    return pd.DataFrame(rows)


def summarize_groups(per_run: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "mae",
        "r2",
        "spearman",
        "precision_gt1",
        "recall_gt1",
        "f2_gt1",
        "tp_gt1",
        "tn_gt1",
        "fn_gt1",
        "fp_gt1",
        "prediction_mean",
        "prediction_std",
        "target_mean",
        "target_std",
    ]

    rows = []
    for model in MODELS:
        group = per_run.loc[per_run["model"].eq(model)]
        if group.empty:
            raise ValueError(f"No rows for model {model}")

        row = {
            "model": model,
            "splits": int(len(group)),
            "rest_lr": float(group["rest_lr"].iloc[0]),
            "comp5_lr": float(group["comp5_lr"].iloc[0]),
        }

        for column in metric_cols:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else math.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_win_counts(
    paired_vs_full: pd.DataFrame,
) -> pd.DataFrame:
    improvement_columns = [
        "improvement_mae",
        "improvement_r2",
        "improvement_spearman",
        "improvement_precision_gt1",
        "improvement_recall_gt1",
        "improvement_f2_gt1",
        "fn_reduction_gt1",
        "fp_reduction_gt1",
    ]

    rows = []
    for candidate, group in paired_vs_full.groupby(
        "candidate_model",
        sort=False,
    ):
        for column in improvement_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            finite = values[np.isfinite(values)]
            rows.append({
                "candidate_model": candidate,
                "comparator_model": FULL_FT_MODEL,
                "metric": column,
                "n_valid": int(len(finite)),
                "wins": int((finite > 0).sum()),
                "ties": int((finite == 0).sum()),
                "losses": int((finite < 0).sum()),
                "mean_improvement": (
                    float(finite.mean()) if len(finite) else math.nan
                ),
                "median_improvement": (
                    float(finite.median()) if len(finite) else math.nan
                ),
            })

    return pd.DataFrame(rows)


def print_group_summary(summary: pd.DataFrame) -> None:
    columns = [
        "model",
        "rest_lr",
        "comp5_lr",
        "mae_mean",
        "r2_mean",
        "spearman_mean",
        "precision_gt1_mean",
        "recall_gt1_mean",
        "f2_gt1_mean",
        "fn_gt1_mean",
        "fp_gt1_mean",
        "prediction_mean_mean",
        "prediction_std_mean",
    ]

    print("=" * 124)
    print("STAGE 6 — PT-D DIFFERENTIAL LR SCREENING")
    print("=" * 124)
    print(summary[columns].to_string(index=False))


def print_paired(title: str, paired: pd.DataFrame) -> None:
    columns = [
        "candidate_model",
        "comparator_model",
        "split_seed",
        "improvement_mae",
        "improvement_r2",
        "improvement_spearman",
        "improvement_recall_gt1",
        "improvement_f2_gt1",
        "fn_reduction_gt1",
        "fp_reduction_gt1",
        "prediction_std_change",
    ]

    print()
    print(title)
    print("(positive improvement/reduction = candidate better)")
    print(paired[columns].to_string(index=False))

    numeric = [
        column for column in columns
        if column not in {
            "candidate_model",
            "comparator_model",
            "split_seed",
        }
    ]
    print()
    print("Mean paired change:")
    print(paired[numeric].mean(numeric_only=True).to_string())

    print()
    print("Median paired change:")
    print(paired[numeric].median(numeric_only=True).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "results/fifth_pretraining/stage5_downstream_transfer"
        ),
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
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "analysis_stage6_lr"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in MODELS:
        for split_seed in args.splits:
            run_dir = root / model / f"split{split_seed}"
            prediction_path = run_dir / "predictions.csv"

            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)

            rest_lr, comp5_lr = read_lr_metadata(run_dir, model)

            rows.append({
                "model": model,
                "split_seed": int(split_seed),
                "rest_lr": rest_lr,
                "comp5_lr": comp5_lr,
                **metrics(selected_test_predictions(prediction_path)),
            })

    per_run = pd.DataFrame(rows)
    summary = summarize_groups(per_run)

    paired_frames = []
    for candidate in (
        DIFF_1E4_MODEL,
        DIFF_3E4_MODEL,
    ):
        paired_frames.append(
            paired_improvements(
                per_run,
                candidate_model=candidate,
                comparator_model=FULL_FT_MODEL,
            )
        )
    paired_vs_full = pd.concat(paired_frames, ignore_index=True)

    paired_3e4_vs_1e4 = paired_improvements(
        per_run,
        candidate_model=DIFF_3E4_MODEL,
        comparator_model=DIFF_1E4_MODEL,
    )

    win_counts = build_win_counts(paired_vs_full)

    per_run_path = output_dir / "stage6_lr_per_run_metrics.csv"
    summary_path = output_dir / "stage6_lr_group_summary.csv"
    paired_full_path = (
        output_dir / "stage6_lr_paired_vs_ptd_fullft.csv"
    )
    paired_lr_path = (
        output_dir / "stage6_lr_paired_3e4_vs_1e4.csv"
    )
    win_path = (
        output_dir / "stage6_lr_win_counts_vs_ptd_fullft.csv"
    )

    per_run.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired_vs_full.to_csv(paired_full_path, index=False)
    paired_3e4_vs_1e4.to_csv(paired_lr_path, index=False)
    win_counts.to_csv(win_path, index=False)

    print_group_summary(summary)

    print_paired(
        "Paired differential-LR improvement vs PT-D full fine-tuning:",
        paired_vs_full,
    )

    print_paired(
        "Direct paired comparison: Comp5 LR=3e-4 vs Comp5 LR=1e-4:",
        paired_3e4_vs_1e4,
    )

    print()
    print("Wins / ties / losses vs PT-D full FT:")
    print(win_counts.to_string(index=False))

    print()
    print("Outputs:")
    for path in (
        per_run_path,
        summary_path,
        paired_full_path,
        paired_lr_path,
        win_path,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
