"""Shared manifest, validation, and execution helpers for stage-two diagnostics."""

from __future__ import annotations

import json
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

from common import ROOT, DataSchema, discover_schema, load_frames, safe_json_dump


STAGE2_OUTPUT = ROOT / "results" / "generalization_stage2"
MANIFEST_COLUMNS = ["sample_id", "split", "group_id", "raw_index"]


def stage2_output(path: Path | None = None) -> Path:
    """Resolve and create the stage-two output directory."""
    resolved = (path or STAGE2_OUTPUT).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def add_stage2_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the required common stage-two command-line controls."""
    parser.add_argument("--output-dir", type=Path, default=STAGE2_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--feedback-csv", type=Path, default=None)


def load_training_frame(
    train_csv: Path | None = None, feedback_csv: Path | None = None,
) -> tuple[DataSchema, pd.DataFrame, pd.DataFrame]:
    """Load immutable raw frames with canonical component/formulation keys."""
    schema = discover_schema(train_csv, feedback_csv)
    train_frame, feedback_frame = load_frames(schema)
    train_frame["raw_index"] = train_frame.index.astype(int)
    feedback_frame["raw_index"] = feedback_frame.index.astype(int)
    train_frame["sample_id"] = train_frame["diagnostic_sample_id"].astype(str)
    feedback_frame["sample_id"] = feedback_frame["diagnostic_sample_id"].astype(str)
    return schema, train_frame, feedback_frame


def legacy_split_labels(sample_count: int, seed: int) -> np.ndarray:
    """Exactly reproduce the loader's two successive 90/10 train_test_split calls."""
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


def legacy_split_manifest(frame: pd.DataFrame, seed: int, group_column: str) -> pd.DataFrame:
    """Recreate legacy membership and the exact sklearn output order per split."""
    all_indices = np.arange(len(frame))
    train_validation, test_indices = train_test_split(
        all_indices, train_size=0.9, test_size=0.1, random_state=seed
    )
    train_indices, validation_indices = train_test_split(
        train_validation, train_size=0.9, test_size=0.1, random_state=seed
    )
    records: list[dict[str, Any]] = []
    for split_name, indices in (
        ("train", train_indices), ("val", validation_indices), ("test", test_indices),
    ):
        for split_order, row_position in enumerate(indices):
            row = frame.iloc[int(row_position)]
            records.append({
                "sample_id": str(row["sample_id"]),
                "split": split_name,
                "group_id": str(row[group_column]),
                "raw_index": int(row["raw_index"]),
                "split_order": split_order,
            })
    manifest = pd.DataFrame(records)
    validate_manifest(manifest, frame)
    return manifest


def manifest_from_labels(
    frame: pd.DataFrame, labels: np.ndarray | pd.Series, group_column: str,
) -> pd.DataFrame:
    """Return a self-contained, row-order-invariant explicit split manifest."""
    if len(frame) != len(labels):
        raise ValueError("Frame and split labels have different lengths.")
    if group_column not in frame:
        raise ValueError(f"Group column is absent: {group_column}")
    manifest = pd.DataFrame({
        "sample_id": frame["sample_id"].astype(str).to_numpy(),
        "split": np.asarray(labels, dtype=object),
        "group_id": frame[group_column].astype(str).to_numpy(),
        "raw_index": frame["raw_index"].astype(int).to_numpy(),
    })
    validate_manifest(manifest, frame)
    return manifest.sort_values("raw_index").reset_index(drop=True)


def validate_manifest(manifest: pd.DataFrame, frame: pd.DataFrame) -> None:
    """Assert the manifest can uniquely and completely map source samples."""
    missing_columns = set(MANIFEST_COLUMNS) - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest missing columns: {sorted(missing_columns)}")
    invalid_labels = set(manifest["split"].astype(str)) - {"train", "val", "test"}
    if invalid_labels:
        raise ValueError(f"Manifest has invalid split labels: {sorted(invalid_labels)}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Manifest has duplicate sample_id values.")
    if manifest["raw_index"].duplicated().any():
        raise ValueError("Manifest has duplicate raw_index values.")
    source = frame[["sample_id", "raw_index"]].copy()
    if source["sample_id"].duplicated().any() or source["raw_index"].duplicated().any():
        raise ValueError("Source frame does not have unique sample identifiers/indexes.")
    if len(manifest) != len(source):
        raise ValueError("Manifest row count differs from source frame.")
    if set(manifest["sample_id"]) != set(source["sample_id"]):
        raise ValueError("Manifest sample_id set differs from source frame.")
    mapping = manifest.merge(source, on="sample_id", suffixes=("_manifest", "_source"), validate="one_to_one")
    if not (mapping["raw_index_manifest"] == mapping["raw_index_source"]).all():
        raise ValueError("Manifest raw_index cannot uniquely map to source rows.")
    if any((manifest["split"] == split_name).sum() == 0 for split_name in ("train", "val", "test")):
        raise ValueError("Manifest contains an empty train, val, or test split.")
    cross_split_groups = manifest.groupby("group_id")["split"].nunique()
    # A group can intentionally cross only in legacy random manifests. Group
    # manifests call this validation after setting a unique per-row group.
    if (cross_split_groups > 1).any() and manifest.attrs.get("strict_group", False):
        raise ValueError("Strict group manifest leaks a group across splits.")


def group_cv_manifests(
    frame: pd.DataFrame, group_column: str, protocol: str, output_dir: Path,
    seed: int, n_splits: int = 5,
) -> list[Path]:
    """Create GroupKFold test folds and group-isolated validation partitions."""
    groups = frame[group_column].astype(str).to_numpy()
    unique_group_count = len(np.unique(groups))
    if unique_group_count < n_splits:
        raise ValueError(
            f"{protocol} has only {unique_group_count} groups, fewer than n_splits={n_splits}."
        )
    protocol_dir = output_dir / "manifests" / protocol
    protocol_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    splitter = GroupKFold(n_splits=n_splits)
    all_indices = np.arange(len(frame))
    for fold_index, (train_validation_indices, test_indices) in enumerate(
        splitter.split(all_indices, groups=groups)
    ):
        remaining_groups = groups[train_validation_indices]
        validation_splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.125, random_state=seed + fold_index
        )
        train_relative, validation_relative = next(
            validation_splitter.split(train_validation_indices, groups=remaining_groups)
        )
        train_indices = train_validation_indices[train_relative]
        validation_indices = train_validation_indices[validation_relative]
        labels = np.full(len(frame), "unassigned", dtype=object)
        labels[train_indices] = "train"
        labels[validation_indices] = "val"
        labels[test_indices] = "test"
        manifest = manifest_from_labels(frame, labels, group_column)
        manifest.attrs["strict_group"] = True
        validate_manifest(manifest, frame)
        path = protocol_dir / f"fold_{fold_index}.csv"
        manifest.to_csv(path, index=False)
        paths.append(path)
    return paths


def load_manifest_frame(frame: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    """Attach manifest split/group columns to a source frame with strict checks."""
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str, "group_id": str})
    validate_manifest(manifest, frame)
    merged = frame.merge(
        manifest, on=["sample_id", "raw_index"], how="inner", validate="one_to_one",
        suffixes=("", "_manifest"),
    )
    if len(merged) != len(frame):
        raise ValueError(f"Manifest did not map every source row: {manifest_path}")
    return merged


def split_indices(manifest_frame: pd.DataFrame) -> dict[str, pd.Index]:
    """Return original DataFrame indexes for each explicitly labelled split."""
    return {
        split_name: manifest_frame.index[manifest_frame["split"] == split_name]
        for split_name in ("train", "val", "test")
    }


def record_execution(
    output_dir: Path, script_name: str, arguments: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Append a reproducible command/config record without overwriting history."""
    path = output_dir / "execution_manifest.json"
    payload: list[dict[str, Any]] = []
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        payload = list(existing.get("executions", []))
    payload.append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": script_name,
        "command": [sys.executable, *sys.argv],
        "arguments": arguments or sys.argv[1:],
        "details": details or {},
    })
    safe_json_dump({"executions": payload}, path)
