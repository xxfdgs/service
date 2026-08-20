#!/usr/bin/env python3
"""Summarize the O13-C no-Mordred Fifth-identity-OOD ablation.

The comparison is strictly seed-paired against the frozen two-layer O13-C
mean-pooling reference.  Both runs use the same Fifth-identity manifests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["target_group", "split_seed", "split", "target"]
METRICS = ("mae", "r2", "pearson", "spearman", "prediction_std_to_true_std")


def add_dispersion(metrics: pd.DataFrame, predictions: pd.DataFrame, label: str) -> pd.DataFrame:
    required_metrics = {*KEYS, "mae", "r2", "pearson", "spearman"}
    required_predictions = {*KEYS, "y_true", "y_pred"}
    if missing := required_metrics.difference(metrics.columns):
        raise ValueError(f"{label} metrics miss columns: {sorted(missing)}")
    if missing := required_predictions.difference(predictions.columns):
        raise ValueError(f"{label} predictions miss columns: {sorted(missing)}")
    rows = []
    for keys, frame in predictions.groupby(KEYS, sort=True):
        true_std = float(frame.y_true.std(ddof=0))
        prediction_std = float(frame.y_pred.std(ddof=0))
        if true_std <= 0:
            raise RuntimeError(f"{label} has zero true standard deviation: {keys}")
        rows.append({**dict(zip(KEYS, keys)), "prediction_std": prediction_std,
                     "true_std": true_std,
                     "prediction_std_to_true_std": prediction_std / true_std})
    dispersion = pd.DataFrame(rows)
    compact = metrics[KEYS + ["mae", "r2", "pearson", "spearman"]]
    if compact.duplicated(KEYS).any() or dispersion.duplicated(KEYS).any():
        raise RuntimeError(f"{label} has duplicate split/seed/target rows")
    return compact.merge(dispersion, on=KEYS, how="inner", validate="one_to_one")


def summary(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    for (target_group, split, target), part in frame.groupby(
            ["target_group", "split", "target"], sort=True):
        row = {"target_group": target_group, "split": split, "target": target,
               "seeds": int(part.split_seed.nunique())}
        for metric in METRICS:
            row[f"{prefix}_{metric}_mean"] = float(part[metric].mean())
            row[f"{prefix}_{metric}_std"] = float(part[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    if set(map(tuple, reference[KEYS].to_numpy())) != set(map(tuple, candidate[KEYS].to_numpy())):
        raise RuntimeError("Reference and no-Mordred runs do not share identical split/seed/target keys.")
    paired = reference.merge(candidate, on=KEYS, suffixes=("_o13c", "_no_mordred"),
                             validate="one_to_one")
    rows = []
    for values, part in paired.groupby(["target_group", "split", "target"], sort=True):
        row = {"target_group": values[0], "split": values[1], "target": values[2],
               "seeds": int(len(part))}
        for metric in METRICS:
            baseline = part[f"{metric}_o13c"]
            ablation = part[f"{metric}_no_mordred"]
            delta = ablation - baseline
            row.update({
                f"o13c_{metric}_mean": float(baseline.mean()),
                f"o13c_{metric}_std": float(baseline.std(ddof=1)),
                f"no_mordred_{metric}_mean": float(ablation.mean()),
                f"no_mordred_{metric}_std": float(ablation.std(ddof=1)),
                f"paired_delta_no_mordred_minus_o13c_{metric}_mean": float(delta.mean()),
                f"paired_delta_no_mordred_minus_o13c_{metric}_std": float(delta.std(ddof=1)),
            })
        rows.append(row)
    return paired, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--no-mordred-metrics", type=Path, required=True)
    parser.add_argument("--no-mordred-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reference = add_dispersion(pd.read_csv(args.reference_metrics),
                               pd.read_csv(args.reference_predictions), "O13-C reference")
    no_mordred = add_dispersion(pd.read_csv(args.no_mordred_metrics),
                                pd.read_csv(args.no_mordred_predictions), "O13-C no-Mordred")
    paired, comparison = paired_summary(reference, no_mordred)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference.to_csv(output / "o13c_reference_per_seed_metrics.csv", index=False)
    no_mordred.to_csv(output / "no_mordred_per_seed_metrics.csv", index=False)
    summary(no_mordred, "no_mordred").to_csv(output / "no_mordred_10seed_summary.csv", index=False)
    paired.to_csv(output / "paired_per_seed_o13c_vs_no_mordred.csv", index=False)
    comparison.to_csv(output / "o13c_vs_no_mordred_paired_summary.csv", index=False)

    test = comparison.loc[comparison.split.eq("test")]
    lines = [
        "# O13-C no-Mordred OOD ablation", "",
        "Both models use mean pooling, two GraphGPS layers, and component auxiliary features. "
        "The only feature difference is Mordred11: enabled in O13-C and disabled in this ablation.", "",
        "Values are mean ± sample standard deviation across the seed-paired Fifth-identity OOD splits 100–109. "
        "Deltas are no-Mordred minus O13-C; negative ΔMAE and positive ΔR²/ΔSpearman indicate improvement.", "",
        "## Fifth-identity OOD test", "",
        "| target | O13-C MAE | no-Mordred MAE | ΔMAE | ΔR² | ΔPearson | ΔSpearman | Δ(σpred/σtrue) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.o13c_mae_mean:.3f} ± {row.o13c_mae_std:.3f} "
            f"| {row.no_mordred_mae_mean:.3f} ± {row.no_mordred_mae_std:.3f} "
            f"| {row.paired_delta_no_mordred_minus_o13c_mae_mean:+.3f} "
            f"| {row.paired_delta_no_mordred_minus_o13c_r2_mean:+.3f} "
            f"| {row.paired_delta_no_mordred_minus_o13c_pearson_mean:+.3f} "
            f"| {row.paired_delta_no_mordred_minus_o13c_spearman_mean:+.3f} "
            f"| {row.paired_delta_no_mordred_minus_o13c_prediction_std_to_true_std_mean:+.3f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
