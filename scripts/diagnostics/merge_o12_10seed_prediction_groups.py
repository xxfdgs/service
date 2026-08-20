#!/usr/bin/env python3
"""Merge the core4 and norm2 ten-model ensemble outputs into six-target CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CORE_TARGETS = [
    "EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency",
]
NORM_TARGETS = ["Norm_before", "Norm_after"]


def group_file(dataset_dir: Path, group: str, prefix: str) -> Path:
    preferred = dataset_dir / f"{prefix}_{group}.csv"
    # Backward-compatible support for the core4 outputs made before the
    # target-group suffix was introduced.
    fallback = dataset_dir / f"{prefix}.csv"
    if preferred.is_file():
        return preferred
    if group == "core4" and fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Missing {group} result in {dataset_dir}: {preferred}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    rows = []
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        core_path = group_file(dataset_dir, "core4", "ensemble_mean_predictions")
        norm_path = group_file(dataset_dir, "norm2", "ensemble_mean_predictions")
        core = pd.read_csv(core_path, dtype={"ID": str})
        norm = pd.read_csv(norm_path, dtype={"ID": str})
        if core.ID.duplicated().any() or norm.ID.duplicated().any() or set(core.ID) != set(norm.ID):
            raise RuntimeError(f"Core/norm ID mismatch in {dataset_dir}")
        norm_columns = [
            f"pred_{target}_{suffix}" for target in NORM_TARGETS
            for suffix in ("mean", "std_10models")
        ]
        if missing := set(norm_columns).difference(norm.columns):
            raise RuntimeError(f"Norm output misses columns in {dataset_dir}: {sorted(missing)}")
        merged = core.merge(norm[["ID", *norm_columns]], on="ID", how="left", validate="one_to_one")
        if merged[norm_columns].isna().any().any():
            raise RuntimeError(f"Missing merged norm predictions in {dataset_dir}")
        # This is the primary easy-to-consume result; retain the explicit all6
        # filename too, so prior core-only output names are not ambiguous.
        merged.to_csv(dataset_dir / "ensemble_mean_predictions_all6.csv", index=False)
        merged.to_csv(dataset_dir / "ensemble_mean_predictions.csv", index=False)

        summaries = []
        long_frames = []
        for group, targets in (("core4", CORE_TARGETS), ("norm2", NORM_TARGETS)):
            summary_path = group_file(dataset_dir, group, "ensemble_prediction_summary")
            long_path = group_file(dataset_dir, group, "predictions_by_model_long")
            summary = pd.read_csv(summary_path)
            long = pd.read_csv(long_path, dtype={"ID": str})
            if "target_group" not in summary:
                summary.insert(0, "target_group", group)
            if "target_group" not in long:
                long.insert(1, "target_group", group)
            if set(long.target) != set(targets):
                raise RuntimeError(f"Unexpected {group} targets in {dataset_dir}")
            summaries.append(summary)
            long_frames.append(long)
        pd.concat(summaries, ignore_index=True).to_csv(
            dataset_dir / "ensemble_prediction_summary_all6.csv", index=False)
        pd.concat(long_frames, ignore_index=True).to_csv(
            dataset_dir / "predictions_by_model_long_all6.csv", index=False)
        rows.append({"dataset": dataset_dir.name, "rows": len(merged), "targets": 6,
                     "output": str(dataset_dir / "ensemble_mean_predictions.csv")})
    pd.DataFrame(rows).to_csv(root / "run_summary_all6.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
