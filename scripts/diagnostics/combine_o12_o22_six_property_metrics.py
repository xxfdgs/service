#!/usr/bin/env python3
"""Combine matched O12/O22 test metrics for all six input properties.

The first four percentage properties and the two Norm properties share the
same 36 saved split manifests.  This script verifies that correspondence
before writing one per-checkpoint table and one mean ± standard-deviation
summary table.  It intentionally does not compute a six-property MAE macro:
the Norm labels and percentage labels have different physical scales.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CORE_TARGETS = [
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
]
NORM_TARGETS = ["Norm_before", "Norm_after"]
METRICS = ["mae", "rmse", "r2", "pearson", "spearman"]


def load_metrics(path: Path, targets: list[str], group: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"run", "model", "split_seed", "split_manifest", "target", *METRICS}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    table = table.loc[table.target.isin(targets)].copy()
    if table.duplicated(["model", "split_seed", "target"]).any():
        raise ValueError(f"Duplicate model/split/target rows in {path}")
    table["property_group"] = group
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-metrics", type=Path, required=True)
    parser.add_argument("--norm-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    core = load_metrics(args.core_metrics, CORE_TARGETS, "core_percentage")
    norm = load_metrics(args.norm_metrics, NORM_TARGETS, "norm_original_unit")
    core_splits = core[["model", "split_seed", "split_manifest"]].drop_duplicates()
    norm_splits = norm[["model", "split_seed", "split_manifest"]].drop_duplicates()
    paired = core_splits.merge(norm_splits, on=["model", "split_seed"], how="outer",
                               suffixes=("_core", "_norm"), indicator=True)
    if not (paired["_merge"] == "both").all():
        raise ValueError("Core and Norm metrics do not contain the same model/split pairs.")
    if not (paired.split_manifest_core == paired.split_manifest_norm).all():
        raise ValueError("A Core and Norm run use different split manifests.")

    combined = pd.concat([core, norm], ignore_index=True)
    order = {target: index for index, target in enumerate(CORE_TARGETS + NORM_TARGETS)}
    combined["target_order"] = combined.target.map(order)
    combined = combined.sort_values(["model", "split_seed", "target_order"]).drop(
        columns="target_order")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "o12_o22_six_property_test_metrics_by_checkpoint.csv", index=False)

    summary = combined.groupby(["model", "property_group", "target"], as_index=False).agg(
        checkpoints=("run", "count"),
        mean_test_mae=("mae", "mean"), std_test_mae=("mae", "std"),
        mean_test_rmse=("rmse", "mean"), std_test_rmse=("rmse", "std"),
        mean_test_r2=("r2", "mean"), std_test_r2=("r2", "std"),
        mean_test_pearson=("pearson", "mean"), std_test_pearson=("pearson", "std"),
        mean_test_spearman=("spearman", "mean"), std_test_spearman=("spearman", "std"),
    )
    summary["target_order"] = summary.target.map(order)
    summary.sort_values(["model", "target_order"]).drop(columns="target_order").to_csv(
        output / "o12_o22_six_property_test_metrics_summary.csv", index=False)


if __name__ == "__main__":
    main()
