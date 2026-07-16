#!/usr/bin/env python3
"""Audit exact formulation duplicates without deleting or modifying source records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_common import (  # noqa: E402
    add_stage2_arguments, load_training_frame, record_execution, stage2_output,
)


def _difference_record(group_key: str, column: str, values: pd.Series) -> dict[str, object]:
    """Record distinct raw values in one repeated formulation group."""
    unique_values = values.fillna("<missing>").astype(str).drop_duplicates().tolist()
    return {
        "formula_ratio_key": group_key,
        "column": column,
        "unique_value_count": len(unique_values),
        "values": " | ".join(unique_values[:20]),
        "changed": len(unique_values) > 1,
    }


def _target_stats(values: pd.Series, global_std: float) -> dict[str, float]:
    """Calculate robust within-group target statistics."""
    numeric_values = values.dropna().astype(float)
    if numeric_values.empty:
        return {key: np.nan for key in (
            "mean", "median", "std", "mad", "range", "coefficient_of_variation",
            "range_overall_std", "std_overall_std",
        )}
    median_value = float(numeric_values.median())
    std_value = float(numeric_values.std(ddof=1)) if len(numeric_values) > 1 else 0.0
    range_value = float(numeric_values.max() - numeric_values.min())
    mad_value = float(np.median(np.abs(numeric_values.to_numpy() - median_value)))
    mean_value = float(numeric_values.mean())
    return {
        "mean": mean_value,
        "median": median_value,
        "std": std_value,
        "mad": mad_value,
        "range": range_value,
        "coefficient_of_variation": std_value / abs(mean_value) if mean_value else np.nan,
        "range_overall_std": range_value / global_std if global_std else np.nan,
        "std_overall_std": std_value / global_std if global_std else np.nan,
    }


def _classify_group(
    group_frame: pd.DataFrame, condition_columns: list[str], targets: list[str],
    global_stds: dict[str, float],
) -> tuple[str, list[str], list[str]]:
    """Assign an evidence-bound duplicate class; never silently discard uncertainty."""
    changed_conditions = [
        column for column in condition_columns
        if group_frame[column].fillna("<missing>").astype(str).nunique() > 1
    ]
    ratio_columns = [column for column in group_frame if column.startswith("mol%_")]
    ratio_total = group_frame[ratio_columns].sum(axis=1, min_count=len(ratio_columns))
    invalid_ratio = ratio_total.notna() & ~np.isclose(ratio_total, 100.0, atol=1e-3)
    invalid_target = any(
        ((group_frame[target] < 0) | (group_frame[target] > 100)).fillna(False).any()
        for target in targets
    )
    large_label_conflict = any(
        group_frame[target].dropna().max() - group_frame[target].dropna().min() > global_stds[target]
        for target in targets if group_frame[target].notna().any()
    )
    reasons: list[str] = []
    if changed_conditions:
        reasons.append("non_target_raw_columns_differ")
        return "hidden_condition_difference", changed_conditions, reasons
    if invalid_ratio.any():
        reasons.append("ratio_total_not_100")
    if invalid_target:
        reasons.append("target_outside_0_100")
    if large_label_conflict:
        reasons.append("label_range_exceeds_global_std")
    if reasons:
        return "likely_record_error", changed_conditions, reasons
    if len(group_frame) > 1:
        return "true_replicate", changed_conditions, ["all_available_conditions_match"]
    return "unresolved", changed_conditions, ["insufficient_repeat_count"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_stage2_arguments(parser)
    arguments = parser.parse_args()
    output_dir = stage2_output(arguments.output_dir)
    data_audit_dir = output_dir / "data_audit"
    data_audit_dir.mkdir(parents=True, exist_ok=True)
    schema, train_frame, _ = load_training_frame(arguments.train_csv, arguments.feedback_csv)
    targets = schema.targets
    global_stds = {target: float(train_frame[target].std(ddof=1)) for target in targets}
    outcome_columns = set(targets + ["Norm_before", "Norm_after"])
    identifier_columns = {schema.id_column, "sample_id", "diagnostic_sample_id", "raw_index"}
    # All non-target raw fields are compared and recorded. Identifiers and
    # derived outcomes do not by themselves indicate a hidden experimental condition.
    raw_columns = [column for column in pd.read_csv(schema.train_path, nrows=1).columns]
    compared_columns = [column for column in raw_columns if column not in targets]
    condition_columns = [
        column for column in raw_columns
        if column not in identifier_columns and column not in outcome_columns
    ]

    classifications: list[dict[str, object]] = []
    differences: list[dict[str, object]] = []
    statistics: list[dict[str, object]] = []
    suspected_errors: list[dict[str, object]] = []
    row_classification: dict[str, dict[str, object]] = {}
    for group_key, group_frame in train_frame.groupby("formula_ratio_key", dropna=False):
        if len(group_frame) < 2:
            continue
        group_class, changed_conditions, reasons = _classify_group(
            group_frame, condition_columns, targets, global_stds
        )
        classification = {
            "formula_ratio_key": str(group_key),
            "group_class": group_class,
            "group_size": int(len(group_frame)),
            "sample_ids": "|".join(group_frame["sample_id"].astype(str)),
            "raw_indices": "|".join(group_frame["raw_index"].astype(str)),
            "changed_condition_columns": "|".join(changed_conditions),
            "classification_reasons": "|".join(reasons),
        }
        classifications.append(classification)
        for column in compared_columns:
            differences.append(_difference_record(str(group_key), column, group_frame[column]))
        for target in targets:
            target_statistics = _target_stats(group_frame[target], global_stds[target])
            statistics.append({
                **classification, "target": target, **target_statistics,
            })
            median_value = target_statistics["median"]
            mad_value = target_statistics["mad"]
            for _, row in group_frame.iterrows():
                value = row[target]
                if pd.isna(value):
                    continue
                outside_mad = abs(float(value) - median_value) > 3 * mad_value if mad_value > 0 else abs(float(value) - median_value) > 0
                outside_global = abs(float(value) - median_value) > global_stds[target]
                if outside_mad or outside_global:
                    suspected_errors.append({
                        "formula_ratio_key": str(group_key), "group_class": group_class,
                        "sample_id": row["sample_id"], "raw_index": int(row["raw_index"]),
                        "target": target, "value": float(value), "group_median": median_value,
                        "group_mad": mad_value, "overall_target_std": global_stds[target],
                        "outside_median_plus_3mad": bool(outside_mad),
                        "difference_exceeds_overall_std": bool(outside_global),
                    })
        for sample_id in group_frame["sample_id"].astype(str):
            row_classification[sample_id] = classification

    classification_frame = pd.DataFrame(classifications)
    difference_frame = pd.DataFrame(differences)
    statistics_frame = pd.DataFrame(statistics)
    error_frame = pd.DataFrame(suspected_errors)
    classification_frame.to_csv(data_audit_dir / "duplicate_group_classification.csv", index=False)
    difference_frame.to_csv(data_audit_dir / "duplicate_column_differences.csv", index=False)
    error_frame.to_csv(data_audit_dir / "suspected_record_errors.csv", index=False)
    statistics_frame.to_csv(data_audit_dir / "replicate_statistics.csv", index=False)

    raw_records = train_frame.copy()
    raw_records["replicate_group_class"] = raw_records["sample_id"].map(
        lambda value: row_classification.get(str(value), {}).get("group_class", "singleton")
    )
    raw_records["replicate_group_size"] = raw_records["sample_id"].map(
        lambda value: row_classification.get(str(value), {}).get("group_size", 1)
    ).astype(int)
    raw_records.to_csv(data_audit_dir / "dataset_raw_records.csv", index=False)

    median_records: list[pd.Series] = []
    for group_key, group_frame in raw_records.groupby("formula_ratio_key", dropna=False, sort=False):
        group_class = group_frame["replicate_group_class"].iloc[0]
        if group_class == "true_replicate" and len(group_frame) > 1:
            median_row = group_frame.iloc[0].copy()
            median_row["source_sample_ids"] = "|".join(group_frame["sample_id"].astype(str))
            median_row["source_raw_indices"] = "|".join(group_frame["raw_index"].astype(str))
            median_row["replicate_count"] = len(group_frame)
            for target in targets:
                values = group_frame[target].dropna().astype(float)
                median_row[target] = values.median()
                median_row[f"replicate_std_{target}"] = values.std(ddof=1) if len(values) > 1 else 0.0
                median_row[f"replicate_mad_{target}"] = np.median(np.abs(values - values.median()))
            median_records.append(median_row)
        else:
            for _, row in group_frame.iterrows():
                copied = row.copy()
                copied["source_sample_ids"] = str(copied["sample_id"])
                copied["source_raw_indices"] = str(copied["raw_index"])
                copied["replicate_count"] = 1
                median_records.append(copied)
    replicate_median = pd.DataFrame(median_records).reset_index(drop=True)
    replicate_median.to_csv(data_audit_dir / "dataset_replicate_median.csv", index=False)

    weighted_records = raw_records.copy()
    true_group_sizes = weighted_records.loc[
        weighted_records["replicate_group_class"] == "true_replicate"
    ].groupby("formula_ratio_key").size()
    weighted_records["replicate_base_weight"] = weighted_records["formula_ratio_key"].map(
        lambda key: 1.0 / true_group_sizes[key] if key in true_group_sizes else 1.0
    )
    for target in targets:
        variance_by_group = weighted_records.loc[
            weighted_records["replicate_group_class"] == "true_replicate"
        ].groupby("formula_ratio_key")[target].var(ddof=1)
        variance_floor = max(float(train_frame[target].var(ddof=1)) * 0.01, 1e-6)
        raw_uncertainty = weighted_records["formula_ratio_key"].map(
            lambda key: 1.0 / (float(variance_by_group.get(key, 0.0)) + variance_floor)
            if key in true_group_sizes else 1.0
        )
        positive_values = raw_uncertainty[raw_uncertainty > 0]
        normalized = raw_uncertainty / positive_values.median()
        weighted_records[f"uncertainty_weight_{target}"] = normalized.clip(0.25, 4.0)
        weighted_records[f"sample_weight_{target}"] = (
            weighted_records["replicate_base_weight"] * weighted_records[f"uncertainty_weight_{target}"]
        ).clip(0.05, 4.0)
    weighted_records.to_csv(data_audit_dir / "dataset_replicate_weighted.csv", index=False)

    class_counts = classification_frame["group_class"].value_counts().to_dict() if not classification_frame.empty else {}
    report = [
        "# 重复配方审计", "",
        f"- 原始训练记录：{len(train_frame)}；exact_formula_ratio 重复组：{len(classification_frame)}。",
        f"- 分类计数：{class_counts}。",
        f"- 疑似异常 target-record 对：{len(error_frame)}；未自动删除任何记录。",
        "- `true_replicate` 仅在所有可用非结果条件字段一致时聚合；"
        "`hidden_condition_difference` 始终保留为独立记录。",
        "- `dataset_replicate_weighted.csv` 使用每目标的截断不确定性权重；"
        "权重仅供支持 sample_weight 的模型使用。",
    ]
    (data_audit_dir / "data_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    record_execution(output_dir, Path(__file__).name, details={
        "seed": arguments.seed, "n_jobs": arguments.n_jobs,
        "train_csv": str(schema.train_path), "exact_duplicate_groups": len(classification_frame),
    })
    print(f"Wrote duplicate audit to {data_audit_dir}")
    print(f"Classification counts: {class_counts}")


if __name__ == "__main__":
    main()
