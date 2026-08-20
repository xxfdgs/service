#!/usr/bin/env python3
"""Compare frozen P1 and Stage-9 audits using matched Fifth-OOD splits only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("mae", "rmse", "median_ae", "mean_signed_error", "underprediction_mae",
           "recall_gt1", "f2_gt1", "fn", "fp", "r2", "spearman", "prediction_mean")
LOWER_IS_BETTER = {"mae", "rmse", "median_ae", "underprediction_mae", "fn", "fp"}
DEFAULT_SEEDS = (100, 101, 102)


def read_metrics(directory: Path) -> pd.DataFrame:
    path = directory / "p1_ptd_internal_ood_metrics_per_split.csv"
    frame = pd.read_csv(path)
    if frame.duplicated(["seed", "subset"]).any():
        raise ValueError(f"Audit is not unique by seed/subset: {path}")
    return frame


def summary_row(label: str, frame: pd.DataFrame, seeds: tuple[int, ...]) -> dict:
    frame = frame.loc[frame.seed.isin(seeds)]
    high = frame.loc[frame.subset.eq("double_gt1")]
    double = frame.loc[frame.subset.eq("double")]
    if len(high) != len(seeds):
        raise ValueError(f"{label}: expected {len(seeds)} completed splits, got {len(high)}")
    result = {"variant": label, "completed_splits": int(len(high))}
    for prefix, values in (("double_gt1", high), ("double", double)):
        for metric in METRICS:
            value = pd.to_numeric(values[metric], errors="coerce")
            result[f"{prefix}_{metric}_mean"] = float(value.mean())
            result[f"{prefix}_{metric}_std"] = float(value.std(ddof=1)) if value.notna().sum() > 1 else np.nan
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], metavar="LABEL=AUDIT_DIR",
                        help="May be passed multiple times.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Matched completed frozen OOD splits (default: 100 101 102).")
    args = parser.parse_args()
    candidates = {}
    for value in args.candidate:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"--candidate must be LABEL=AUDIT_DIR, got {value!r}")
        candidates[label] = read_metrics(Path(path))
    if not candidates:
        raise ValueError("At least one --candidate is required.")
    baseline = read_metrics(args.baseline_audit)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(args.seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must be unique.")
    rows = [summary_row("P1_PT_D_MAE_baseline", baseline, seeds)]
    paired_rows = []
    base_high = baseline.loc[baseline.subset.eq("double_gt1")].set_index("seed")
    base_double = baseline.loc[baseline.subset.eq("double")].set_index("seed")
    for label, frame in candidates.items():
        rows.append(summary_row(label, frame, seeds))
        for subset, base in (("double_gt1", base_high), ("double", base_double)):
            candidate = frame.loc[frame.subset.eq(subset)].set_index("seed")
            common = base.index.intersection(candidate.index).sort_values()
            common = common.intersection(pd.Index(seeds)).sort_values()
            if len(common) != len(seeds):
                raise ValueError(f"{label}/{subset}: splits do not match the requested baseline seeds")
            for metric in METRICS:
                delta = candidate.loc[common, metric].astype(float) - base.loc[common, metric].astype(float)
                improved = delta < 0 if metric in LOWER_IS_BETTER else delta > 0
                paired_rows.append({
                    "variant": label, "subset": subset, "metric": metric,
                    "definition": "candidate minus P1_PT_D_MAE_baseline",
                    "matched_splits": int(len(common)), "mean_delta": float(delta.mean()),
                    "std_delta": float(delta.std(ddof=1)), "median_delta": float(delta.median()),
                    "candidate_better_splits": int(improved.sum()),
                })
    pd.DataFrame(rows).to_csv(output / "stage9_screen_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(output / "stage9_screen_paired_deltas.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
