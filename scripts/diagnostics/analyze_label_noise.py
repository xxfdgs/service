#!/usr/bin/env python3
"""Measure repeated-formulation variability and likely label-noise ceilings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import add_common_arguments, discover_schema, load_frames  # noqa: E402


def _group_target_statistics(
    frame: pd.DataFrame, key_column: str, group_type: str, target: str,
    total_standard_deviation: float,
) -> list[dict[str, object]]:
    """Return per-group target spread records for groups with repeated observations."""
    records: list[dict[str, object]] = []
    for group_key, group_frame in frame.groupby(key_column, dropna=False):
        values = group_frame[target].dropna().astype(float)
        if len(values) < 2:
            continue
        median_value = float(values.median())
        target_range = float(values.max() - values.min())
        records.append({
            "group_type": group_type,
            "group_key": str(group_key),
            "target": target,
            "repeat_count": int(len(values)),
            "unique_ratio_count": int(group_frame["formula_ratio_key"].nunique()),
            "sample_ids": "|".join(group_frame["diagnostic_sample_id"].astype(str)),
            "target_mean": float(values.mean()),
            "target_std": float(values.std(ddof=1)),
            "target_range": target_range,
            "target_mad": float(np.median(np.abs(values.to_numpy() - median_value))),
            "range_overall_target_std": float(target_range / total_standard_deviation)
            if total_standard_deviation > 0 else np.nan,
            "std_overall_target_std": float(values.std(ddof=1) / total_standard_deviation)
            if total_standard_deviation > 0 else np.nan,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = discover_schema(arguments.train_csv, arguments.feedback_csv)
    train_frame, _ = load_frames(schema)

    group_definitions = [
        ("exact_formula_ratio", "formula_ratio_key"),
        ("same_components_ratio_varied", "formula_identity_key"),
        ("same_fifth_component", "fifth_component_key"),
    ]
    all_records: list[dict[str, object]] = []
    for target in schema.targets:
        overall_standard_deviation = float(train_frame[target].std(ddof=1))
        for group_type, key_column in group_definitions:
            records = _group_target_statistics(
                train_frame, key_column, group_type, target, overall_standard_deviation
            )
            if group_type == "same_components_ratio_varied":
                records = [
                    record for record in records if int(record["unique_ratio_count"]) > 1
                ]
            all_records.extend(records)
    duplicates = pd.DataFrame(all_records)
    duplicates.to_csv(output_dir / "duplicate_formulations.csv", index=False)
    if duplicates.empty:
        summary = pd.DataFrame(columns=["group_type", "target", "group_count"])
        high_noise = duplicates
    else:
        summary = duplicates.groupby(["group_type", "target"], as_index=False).agg(
            group_count=("group_key", "nunique"),
            repeated_observations=("repeat_count", "sum"),
            median_group_std=("target_std", "median"),
            median_group_range=("target_range", "median"),
            max_group_range=("target_range", "max"),
            median_range_overall_target_std=("range_overall_target_std", "median"),
        )
        high_noise = duplicates.sort_values(
            ["range_overall_target_std", "target_range"], ascending=False
        ).head(50)
    summary.to_csv(output_dir / "label_noise_summary.csv", index=False)
    high_noise.to_csv(output_dir / "high_noise_groups.csv", index=False)
    print(f"Repeated-group records: {len(duplicates)}")
    if not summary.empty:
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
