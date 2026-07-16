#!/usr/bin/env python3
"""Build a non-destructive manual-review queue from duplicate and hard-error evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import add_normalized_keys, discover_schema


TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]


def error_wide(path: Path, model_name: str) -> pd.DataFrame:
    """Pivot per-target OOF absolute errors by sample ID when an artifact exists."""
    if not path.is_file():
        return pd.DataFrame(columns=["sample_id"])
    frame = pd.read_csv(path, dtype={"sample_id": str})
    if "model" in frame:
        frame = frame.loc[frame["model"] == model_name]
    if frame.empty:
        return pd.DataFrame(columns=["sample_id"])
    return frame.pivot_table(index="sample_id", columns="target", values="absolute_error", aggfunc="mean").add_prefix(
        f"{model_name}_oof_error_"
    ).reset_index()


def recommendation(group_class: object) -> str:
    """Return only one of the permitted manual review actions."""
    if group_class == "likely_record_error":
        return "verify_metadata"
    if group_class == "hidden_condition_difference":
        return "keep_as_independent_condition"
    if group_class == "true_replicate":
        return "confirm_true_replicate"
    return "unresolved"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generalization_stage3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    review_dir = output_dir / "data_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    schema = discover_schema()
    frame = pd.read_csv(schema.train_path).copy()
    frame["sample_id"] = frame[schema.id_column].astype(str)
    frame["source_raw_index"] = frame.index.astype(int)
    add_normalized_keys(frame, schema)
    classification = pd.read_csv(ROOT / "results/generalization_stage2/data_audit/duplicate_group_classification.csv")
    differences = pd.read_csv(ROOT / "results/generalization_stage2/data_audit/duplicate_column_differences.csv")
    duplicate = classification[["formula_ratio_key", "group_class", "classification_reasons"]].drop_duplicates()
    difference_text = differences.groupby("formula_ratio_key").apply(
        lambda group: "|".join(sorted(set(group.get("changed_column", pd.Series(dtype=str)).dropna().astype(str))))
    ).rename("difference_fields").reset_index()
    queue = frame.merge(duplicate, on="formula_ratio_key", how="left", validate="many_to_one").merge(
        difference_text, on="formula_ratio_key", how="left", validate="many_to_one")
    tree_errors = error_wide(output_dir / "tree_cv/pooled_oof_predictions.csv", "ExtraTrees")
    graph_errors = error_wide(output_dir / "graphgps_raw_cv/pooled_oof_predictions.csv", "GraphGPS_coarse_mordred_ensemble")
    queue = queue.merge(tree_errors, on="sample_id", how="left").merge(graph_errors, on="sample_id", how="left")
    grouped_labels = frame.groupby("formula_ratio_key")[TARGETS].apply(
        lambda group: json.dumps(group.to_dict(orient="records"), ensure_ascii=False)
    ).rename("other_group_labels").reset_index()
    queue = queue.merge(grouped_labels, on="formula_ratio_key", how="left")
    error_columns = [column for column in queue if column.startswith("ExtraTrees_oof_error_")]
    high_error = queue[error_columns].max(axis=1, skipna=True) if error_columns else pd.Series(0.0, index=queue.index)
    queue["suspected_problem_type"] = queue["group_class"].fillna("high_oof_error")
    queue["recommended_manual_action"] = queue["group_class"].map(recommendation)
    queue.loc[queue.group_class.isna() & high_error.notna() & (high_error > high_error.quantile(0.95)),
              "recommended_manual_action"] = "verify_target"
    queue = queue.loc[queue.group_class.notna() | (high_error > high_error.quantile(0.95))].copy()
    component_columns = [item for component in schema.components for item in (component["name_column"], component["smiles_column"], component["ratio_column"])]
    output_columns = ["sample_id", "formula_ratio_key", "group_class", "classification_reasons", "difference_fields",
                      *component_columns, *TARGETS, "other_group_labels", "suspected_problem_type",
                      "recommended_manual_action", "source_raw_index", *error_columns,
                      *[column for column in queue if column.startswith("GraphGPS_coarse_mordred_ensemble_oof_error_")]]
    queue[output_columns].sort_values(["group_class", "sample_id"], na_position="last").to_csv(
        review_dir / "manual_review_queue.csv", index=False)
    print(f"Wrote {review_dir / 'manual_review_queue.csv'}")


if __name__ == "__main__":
    main()
