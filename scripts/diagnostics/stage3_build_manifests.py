#!/usr/bin/env python3
"""Build hash-checked, nested GroupKFold manifests for stage-three evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import add_normalized_keys, discover_schema, safe_json_dump


PROTOCOLS = {
    "fifth_component_group_cv": "fifth_component_key",
    "formula_identity_group_cv": "formula_identity_key",
}


def file_hash(path: Path) -> str:
    """Return a SHA256 digest for a persisted manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_version_frame(version: str, schema_path: Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Load a data version and retain an immutable original-row identifier."""
    schema = discover_schema(schema_path)
    if version == "raw_records":
        path = schema.train_path
    elif version == "replicate_median":
        path = ROOT / "results/generalization_stage2/data_audit/dataset_replicate_median.csv"
    else:
        raise ValueError(f"Unsupported data version: {version}")
    frame = pd.read_csv(path).copy()
    if schema.id_column not in frame:
        raise ValueError(f"{path} lacks sample identifier column {schema.id_column!r}")
    frame["sample_id"] = frame[schema.id_column].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate sample_id values; cannot build an explicit manifest.")
    frame["original_row_index"] = np.arange(len(frame), dtype=int)
    add_normalized_keys(frame, schema)
    return frame, path


def build_manifest(frame: pd.DataFrame, group_column: str, protocol: str,
                   version: str, fold: int, train_validation: np.ndarray,
                   validation: np.ndarray, test: np.ndarray) -> pd.DataFrame:
    """Assemble an order-stable manifest with group and version provenance."""
    split = np.full(len(frame), "", dtype=object)
    split[train_validation] = "train"
    split[validation] = "val"
    split[test] = "test"
    manifest = pd.DataFrame({
        "sample_id": frame["sample_id"].astype(str),
        "split": split,
        "outer_fold": int(fold),
        "group_id": frame[group_column].astype(str),
        "original_row_index": frame["original_row_index"].astype(int),
        "data_version": version,
    }).sort_values("original_row_index", kind="stable").reset_index(drop=True)
    if (manifest["split"] == "").any() or manifest["sample_id"].duplicated().any():
        raise ValueError(f"{protocol}/{version}/fold_{fold} has invalid sample assignment.")
    if manifest.groupby("group_id")["split"].nunique().gt(1).any():
        raise ValueError(f"{protocol}/{version}/fold_{fold} leaks a group across partitions.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generalization_stage3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--train-csv", type=Path, default=None)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    manifests_root = output_dir / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    integrity_rows: list[dict[str, object]] = []
    report_rows: list[str] = ["# Stage 3 Manifest Report", ""]
    for version in ("raw_records", "replicate_median"):
        frame, source_path = load_version_frame(version, arguments.train_csv)
        for protocol, group_column in PROTOCOLS.items():
            groups = frame[group_column].astype(str).to_numpy()
            splitter = GroupKFold(n_splits=arguments.n_splits)
            outer_test_counts = np.zeros(len(frame), dtype=int)
            for fold, (outer_train_validation, outer_test) in enumerate(splitter.split(frame, groups=groups)):
                inner_groups = groups[outer_train_validation]
                inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=arguments.seed + fold)
                train_relative, validation_relative = next(inner.split(outer_train_validation, groups=inner_groups))
                outer_train = outer_train_validation[train_relative]
                outer_validation = outer_train_validation[validation_relative]
                manifest = build_manifest(frame, group_column, protocol, version, fold,
                                          outer_train, outer_validation, outer_test)
                output_path = manifests_root / protocol / version / f"fold_{fold}.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                manifest.to_csv(output_path, index=False)
                outer_test_counts[outer_test] += 1
                integrity_rows.append({
                    "protocol": protocol, "data_version": version, "fold": fold,
                    "path": str(output_path), "manifest_hash": file_hash(output_path),
                    "source_path": str(source_path), "n_total": len(manifest),
                    "n_train": int((manifest["split"] == "train").sum()),
                    "n_val": int((manifest["split"] == "val").sum()),
                    "n_test": int((manifest["split"] == "test").sum()),
                    "sample_id_unique": not manifest["sample_id"].duplicated().any(),
                    "group_leakage": bool(manifest.groupby("group_id")["split"].nunique().gt(1).any()),
                })
            if not np.all(outer_test_counts == 1):
                raise ValueError(f"{protocol}/{version} outer-test folds do not cover each sample exactly once.")
            report_rows.append(f"- `{protocol}/{version}`: {len(frame)} samples; all samples appear in outer test exactly once.")
    integrity = pd.DataFrame(integrity_rows)
    integrity.to_csv(manifests_root / "manifest_integrity.csv", index=False)
    safe_json_dump({"seed": arguments.seed, "n_splits": arguments.n_splits,
                    "protocols": list(PROTOCOLS), "versions": ["raw_records", "replicate_median"]},
                   manifests_root / "manifest_settings.json")
    (manifests_root / "manifest_report.md").write_text("\n".join(report_rows) + "\n", encoding="utf-8")
    print(f"Wrote {manifests_root}")


if __name__ == "__main__":
    main()
