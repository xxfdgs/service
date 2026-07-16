#!/usr/bin/env python3
"""Create reproducible random, group, and feedback-like diagnostic splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    COMPONENTS, add_common_arguments, build_feature_frames, discover_schema,
    load_frames, load_mordred_table, safe_json_dump,
)


def _random_labels(sample_count: int, seed: int) -> np.ndarray:
    """Mirror the current loader's two successive 90/10 random splits."""
    all_indices = np.arange(sample_count)
    train_validation, test_indices = train_test_split(
        all_indices, train_size=0.9, test_size=0.1, random_state=seed
    )
    train_indices, validation_indices = train_test_split(
        train_validation, train_size=0.9, test_size=0.1, random_state=seed
    )
    labels = np.full(sample_count, "unassigned", dtype=object)
    labels[train_indices] = "train"
    labels[validation_indices] = "val"
    labels[test_indices] = "test"
    return labels


def _group_labels(groups: pd.Series, seed: int) -> np.ndarray:
    """Assign groups to train/val/test without any group crossing a boundary."""
    group_values = groups.astype(str).to_numpy()
    all_indices = np.arange(len(group_values))
    outer_splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    train_validation_indices, test_indices = next(
        outer_splitter.split(all_indices, groups=group_values)
    )
    remaining_groups = group_values[train_validation_indices]
    inner_splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed + 1)
    train_relative, validation_relative = next(
        inner_splitter.split(train_validation_indices, groups=remaining_groups)
    )
    train_indices = train_validation_indices[train_relative]
    validation_indices = train_validation_indices[validation_relative]
    labels = np.full(len(group_values), "unassigned", dtype=object)
    labels[train_indices] = "train"
    labels[validation_indices] = "val"
    labels[test_indices] = "test"
    return labels


def _feedback_like_labels(
    train_features: pd.DataFrame, feedback_features: pd.DataFrame, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve training samples nearest to feedback as external-like val/test."""
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    imputed_train = imputer.fit_transform(train_features)
    imputed_feedback = imputer.transform(feedback_features)
    # Both imputation and scaling are fit only on candidate training data.
    scaler = StandardScaler().fit(imputed_train)
    scaled_train = scaler.transform(imputed_train)
    scaled_feedback = scaler.transform(imputed_feedback)
    minimum_distances = pairwise_distances(
        scaled_train, scaled_feedback, metric="euclidean", n_jobs=1
    ).min(axis=1)
    random_generator = np.random.default_rng(seed)
    tie_breaker = random_generator.random(len(minimum_distances)) * 1e-12
    closest_order = np.argsort(minimum_distances + tie_breaker)
    test_count = max(1, int(round(len(closest_order) * 0.10)))
    validation_count = max(1, int(round(len(closest_order) * 0.09)))
    labels = np.full(len(closest_order), "train", dtype=object)
    labels[closest_order[:test_count]] = "test"
    labels[closest_order[test_count:test_count + validation_count]] = "val"
    return labels, minimum_distances


def _split_quality(split_frame: pd.DataFrame, group_column: str | None) -> dict[str, object]:
    """Check sample and optional group leakage for a single persisted split."""
    labels = split_frame["split"]
    split_counts = labels.value_counts().to_dict()
    overlap_count = int(split_frame["diagnostic_sample_id"].duplicated().sum())
    group_crossing_count = 0
    crossing_examples: list[str] = []
    if group_column:
        split_counts_per_group = split_frame.groupby(group_column)["split"].nunique()
        crossing_groups = split_counts_per_group[split_counts_per_group > 1]
        group_crossing_count = int(len(crossing_groups))
        crossing_examples = [str(value) for value in crossing_groups.index[:5]]
    return {
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "sample_overlap_count": overlap_count,
        "group_crossing_count": group_crossing_count,
        "group_crossing_examples": crossing_examples,
    }


def _normalized_smiles_aliases(train_frame: pd.DataFrame) -> dict[str, int]:
    """Count raw SMILES spellings which collapse to the same canonical structure."""
    alias_counts: dict[str, int] = {}
    for position, (_, smiles_column, _) in enumerate(COMPONENTS, start=1):
        temporary = pd.DataFrame({
            "raw": train_frame[smiles_column].astype(str),
            "canonical": train_frame[f"component_{position}_key"].astype(str),
        })
        aliases = temporary.groupby("canonical")["raw"].nunique()
        alias_counts[f"component_{position}"] = int((aliases > 1).sum())
    return alias_counts


def _split_output_frame(
    train_frame: pd.DataFrame, labels: np.ndarray, feedback_distance: np.ndarray | None = None
) -> pd.DataFrame:
    output_columns = [
        "diagnostic_sample_id", "ID", "split", "fifth_component_key",
        "formula_identity_key", "formula_ratio_key",
    ]
    output_frame = train_frame.copy()
    output_frame["split"] = labels
    if feedback_distance is not None:
        output_frame["feedback_distance"] = feedback_distance
        output_columns.append("feedback_distance")
    return output_frame[output_columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-mordred-features", type=int, default=256)
    arguments = parser.parse_args()

    output_dir = arguments.output_dir.resolve()
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    schema = discover_schema(arguments.train_csv, arguments.feedback_csv)
    train_frame, feedback_frame = load_frames(schema)

    target_summary = {
        target: {
            "train_missing_fraction": float(train_frame[target].isna().mean()),
            "feedback_missing_fraction": float(feedback_frame[target].isna().mean()),
            "train_summary": train_frame[target].describe().to_dict(),
            "feedback_summary": feedback_frame[target].describe().to_dict(),
        }
        for target in schema.targets
    }
    schema_payload = {
        "actual_training_csv": str(schema.train_path),
        "feedback_csv": str(schema.feedback_path),
        "selected_best_config": str(schema.config_path),
        "targets": schema.targets,
        "sample_id_column": schema.id_column,
        "components": schema.components,
        "mordred_descriptor_caches": [str(path) for path in schema.descriptor_paths],
        "rdkit_descriptor_features": {
            "cache": None,
            "source": "computed lazily from each component SMILES by scripts/diagnostics/common.py",
            "features_per_component": [
                "mol_weight", "mol_logp", "heavy_atom_count", "ring_count",
                "h_bond_donor_count", "h_bond_acceptor_count",
            ],
        },
        "nominal_config_split": schema.nominal_split,
        "effective_current_loader_split": {"train": 0.81, "val": 0.09, "test": 0.10},
        "train_rows": int(len(train_frame)),
        "feedback_rows": int(len(feedback_frame)),
        "target_summary": target_summary,
    }
    safe_json_dump(schema_payload, output_dir / "data_schema.json")

    split_specs: list[tuple[str, np.ndarray, str | None, np.ndarray | None]] = [
        ("random_split", _random_labels(len(train_frame), arguments.seed), None, None),
        ("fifth_component_group_split", _group_labels(
            train_frame["fifth_component_key"], arguments.seed), "fifth_component_key", None),
        ("formula_identity_group_split", _group_labels(
            train_frame["formula_identity_key"], arguments.seed), "formula_identity_key", None),
        ("formula_ratio_group_split", _group_labels(
            train_frame["formula_ratio_key"], arguments.seed), "formula_ratio_key", None),
    ]

    mordred_frame = load_mordred_table(schema)
    train_features, _ = build_feature_frames(
        train_frame, schema, mordred_frame, arguments.max_mordred_features
    )
    feedback_features, _ = build_feature_frames(
        feedback_frame, schema, mordred_frame, arguments.max_mordred_features
    )
    feedback_labels, feedback_distances = _feedback_like_labels(
        train_features, feedback_features, arguments.seed
    )
    split_specs.append(("feedback_like_split", feedback_labels, None, feedback_distances))

    quality_payload: dict[str, object] = {
        "normalized_smiles_raw_alias_groups": _normalized_smiles_aliases(train_frame),
        "splits": {},
    }
    for split_name, labels, group_column, distances in split_specs:
        split_frame = _split_output_frame(train_frame, labels, distances)
        split_path = split_dir / f"{split_name}.csv"
        split_frame.to_csv(split_path, index=False)
        quality_payload["splits"][split_name] = _split_quality(split_frame, group_column)

    safe_json_dump(quality_payload, output_dir / "split_quality_checks.json")
    print(f"Wrote {len(split_specs)} split files to {split_dir}")
    for split_name, details in quality_payload["splits"].items():
        print(f"{split_name}: {details['split_counts']}, group leaks={details['group_crossing_count']}")


if __name__ == "__main__":
    main()
