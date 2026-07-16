#!/usr/bin/env python3
"""Preflight audit and feature registry for frozen-embedding/tree fusion.

This program is deliberately read-only with respect to the existing baseline
and embedding artifacts.  It materializes only audited feature snapshots under
``results/hybrid_embedding_tree_exp`` for the downstream nested-CV stages.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/deduplicated_rebaseline"
FROZEN = ROOT / "results/frozen_embedding_signal_exp"
TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]
FOLDS = [f"fold_{number}" for number in range(5)]
SPLITS = ("train", "val", "test")
EMBEDDINGS = ("descriptor_branch_raw", "fused_embedding", "graph_branch_raw")
TREE_FILES = {
    "F1": "F1_ratio_only.csv",
    "F2": "F2_identity_ratio.csv",
    "F3": "F3_physchem_weighted.csv",
    "F4": "F4_physchem_interactions.csv",
}
MORDRED_11 = [
    "SsNH3", "SMR_VSA9", "SlogP_VSA11", "SlogP_VSA10", "TopoPSA", "MW",
    "nRot", "nRing", "nAromAtom", "nHBDon", "nHBAcc",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_frame(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(frame.columns.astype(str)).encode())
    for row in frame.fillna("<NA>").astype(str).itertuples(index=False, name=None):
        digest.update("\x1e".join(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(str(values.dtype).encode())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def append_execution(root: Path, **record: object) -> None:
    path = root / "execution_manifest.json"
    records = json.loads(path.read_text()) if path.exists() else []
    base = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "stage": None, "target": None,
        "outer_fold": None, "inner_fold": None, "feature_family": None,
        "embedding_name": None, "model": None, "hyperparameters": None,
        "dataset_hash": None, "manifest_hash": None, "feature_hash": None,
        "embedding_hash": None, "checkpoint_hash": None, "status": None,
        "error": None, "output_path": None,
    }
    base.update(record)
    records.append(base)
    json_dump(records, path)


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def descriptor_frame(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, dtype={"sample_id": str})
    needed = {"sample_id", "component_position", *[f"feature_{index}" for index in range(11)]}
    if missing := needed - set(raw.columns):
        raise ValueError(f"raw descriptor cache is missing columns: {sorted(missing)}")
    if raw.duplicated(["sample_id", "component_position"]).any():
        raise ValueError("raw descriptor cache has duplicate sample/component keys")
    counts = raw.groupby("sample_id").component_position.nunique()
    if not (counts == 5).all():
        raise ValueError("each formulation must have exactly five raw descriptor rows")
    blocks = []
    for position in range(1, 6):
        part = raw.loc[raw.component_position.eq(position)].set_index("sample_id")
        values = part[[f"feature_{index}" for index in range(11)]].copy()
        values.columns = [f"component_{position}_{name}" for name in MORDRED_11]
        blocks.append(values)
    result = pd.concat(blocks, axis=1).reset_index()
    if result.iloc[:, 1:].isna().any().any() or not np.isfinite(result.iloc[:, 1:].to_numpy(float)).all():
        raise ValueError("raw descriptor cache contains missing/non-finite values")
    return result


def archive(path: Path) -> dict[str, np.ndarray]:
    require(path)
    loaded = np.load(path, allow_pickle=False)
    required = {"embedding", "sample_id", "group_id", "labels", "target_valid_mask"}
    if missing := required - set(loaded.files):
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    return {name: loaded[name] for name in loaded.files}


def prerequisite_summary() -> dict[str, object]:
    report = FROZEN / "report.md"
    final = FROZEN / "confirmation/final_candidate_summary.csv"
    required = [report, final, FROZEN / "embeddings/embedding_index.csv",
                FROZEN / "checkpoints/checkpoint_inventory.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("BLOCKED_MISSING_FROZEN_EMBEDDING_ARTIFACTS: " + "; ".join(missing))
    candidates = pd.read_csv(final)
    expected = {
        "EE_before": "descriptor_branch_raw", "EE_after": "descriptor_branch_raw",
        "Aerosolization_Efficiency": "descriptor_branch_raw", "mRNA_Recovery_Efficiency": "fused_embedding",
    }
    observed = candidates.set_index("target").embedding.to_dict()
    if observed != expected:
        raise RuntimeError(f"Frozen candidate lock differs from prerequisite: {observed}")
    text = report.read_text(encoding="utf-8")
    if "ENCODER_SIGNAL_HEAD_FAILURE" not in text or "395-D" not in text:
        raise RuntimeError("Frozen report lacks the required status or fused-embedding definition")
    return {
        "status": "ENCODER_SIGNAL_HEAD_FAILURE",
        "frozen_report": str(report.resolve()), "frozen_report_sha256": sha256_file(report),
        "locked_embeddings": expected, "epoch_rule": "epoch_best", "locked_probe": "RandomForest",
        "fused_embedding_definition": "Historical 395-D prediction-head input concatenation; not embedding-level softmax fusion.",
        "frozen_probe_not_above_nested_tree_baseline": True,
    }


def main() -> None:
    output = ROOT / "results/hybrid_embedding_tree_exp"
    feature_out, audit_out = output / "features", output / "audit"
    feature_out.mkdir(parents=True, exist_ok=True)
    audit_out.mkdir(parents=True, exist_ok=True)
    try:
        prerequisite = prerequisite_summary()
        dataset_path = BASE / "data_audit/dataset_with_sample_id.csv"
        source_path = BASE / "data_source.json"
        raw_descriptor_path = BASE / "artifacts/mordred_11_raw_features.csv"
        inventory_path = BASE / "artifacts/artifact_inventory.csv"
        for path in [dataset_path, source_path, raw_descriptor_path, inventory_path]:
            require(path)
        dataset = pd.read_csv(dataset_path, dtype={"sample_id": str}).set_index("sample_id", drop=False)
        if len(dataset) != 700 or dataset.index.has_duplicates or dataset[TARGETS].isna().any().any():
            raise ValueError("audited data must have exactly 700 unique, labelled sample IDs")
        source = json.loads(source_path.read_text())
        dataset_hash = str(source["dataset_sha256"])
        if dataset_hash != "f604fc7bcdbc9fc8cfa0dec8e57c8f983f368b04a8499948eb438f0db4a61604":
            raise ValueError("unexpected dataset hash")
        raw_descriptor = descriptor_frame(raw_descriptor_path).set_index("sample_id", drop=False)
        if set(raw_descriptor.index) != set(dataset.index):
            raise ValueError("raw descriptor sample IDs do not match the audited dataset")

        feature_frames: dict[str, pd.DataFrame] = {"F0": raw_descriptor}
        sources: dict[str, Path] = {"F0": raw_descriptor_path}
        for name, filename in TREE_FILES.items():
            path = BASE / "artifacts" / filename
            require(path)
            frame = pd.read_csv(path, dtype={"sample_id": str}).set_index("sample_id", drop=False)
            if set(frame.index) != set(dataset.index) or frame.index.has_duplicates:
                raise ValueError(f"{name} does not exactly cover audited sample IDs")
            feature_frames[name] = frame
            sources[name] = path

        registry: dict[str, object] = {
            "dataset_hash": dataset_hash, "sample_count": len(dataset),
            "raw_descriptor_per_component_dim": 11, "raw_descriptor_component_count": 5,
            "raw_descriptor_names": MORDRED_11,
            "raw_descriptor_input_definition": "Five component-wise 11-D descriptors concatenated in component-position order, giving 55 columns; this is the direct GraphGPS descriptor input before the historical head.",
            "families": {}, "combinations": {},
        }
        family_columns: dict[str, list[str]] = {}
        for name, frame in feature_frames.items():
            destination = feature_out / ("raw_11d_descriptor.csv" if name == "F0" else f"{name}.csv")
            frame.reset_index(drop=True).to_csv(destination, index=False)
            columns = [column for column in frame.columns if column != "sample_id"]
            family_columns[name] = columns
            registry["families"][name] = {
                "alias": "raw_11d_descriptor" if name == "F0" else name,
                "source": str(sources[name].resolve()), "source_sha256": sha256_file(sources[name]),
                "snapshot": str(destination.resolve()), "snapshot_sha256": sha256_file(destination),
                "dimension": len(columns), "columns": columns,
                "dtypes": {column: str(frame[column].dtype) for column in columns},
                "missing_values": int(frame[columns].isna().sum().sum()),
                "categorical_columns": [column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])],
                "description": "GraphGPS raw descriptor input" if name == "F0" else "Exact, audited current tree-baseline feature family.",
            }

        combinations = {
            "A0": ["F0"], "A1": ["F1"], "A2": ["F2"], "A3": ["F3"], "A4": ["F4"],
            "A5": ["E_desc"], "A6": ["E_fused"], "A7": ["E_graph"],
            "B1": ["F0", "E_desc"], "B2": ["F1", "E_desc"], "B3": ["F2", "E_desc"],
            "B4": ["F3", "E_desc"], "B5": ["F4", "E_desc"], "B6": ["F1", "E_fused"],
            "B7": ["F2", "E_fused"], "B8": ["F3", "E_fused"], "B9": ["F4", "E_fused"],
            "B10": ["F2", "E_desc", "E_fused"],
            "B11": ["INNER_SELECTED_F1_F4", "TARGET_LOCKED_EMBEDDING"],
        }
        registry["combinations"] = combinations

        alignment_rows, provenance_rows, feature_hash_rows, leakage_rows = [], [], [], []
        checkpoint = pd.read_csv(FROZEN / "checkpoints/checkpoint_inventory.csv")
        for fold in FOLDS:
            manifest_path = BASE / "manifests/formula_identity_group_cv" / f"fold_{fold.split('_')[1]}.csv"
            require(manifest_path)
            manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
            if set(manifest.dataset_sha256.astype(str)) != {dataset_hash}:
                raise ValueError(f"{manifest_path} has an incompatible dataset hash")
            if manifest.sample_id.duplicated().any() or set(manifest.sample_id) != set(dataset.index):
                raise ValueError(f"{manifest_path} does not provide a one-to-one audited split")
            manifest_hash = str(manifest.manifest_sha256.iloc[0])
            ckpt = checkpoint.loc[(checkpoint.fold == fold) & (checkpoint.epoch_label == "epoch_best")]
            if len(ckpt) != 1:
                raise ValueError(f"missing epoch-best checkpoint provenance for {fold}")
            for embedding in EMBEDDINGS:
                sample_hashes: list[str] = []
                for split in SPLITS:
                    path = FROZEN / "embeddings" / fold / "epoch_best" / f"{split}_{embedding}.npz"
                    values = archive(path)
                    expected = manifest.loc[manifest.split.eq(split), "sample_id"].astype(str).to_numpy()
                    ids = values["sample_id"].astype(str)
                    observed_groups = values["group_id"].astype(str)
                    expected_groups = manifest.set_index("sample_id").loc[ids, "group_id"].astype(str).to_numpy()
                    label_expected = dataset.loc[ids, TARGETS].to_numpy(float)
                    max_label_error = float(np.max(np.abs(label_expected - values["labels"].astype(float))))
                    max_descriptor_error = np.nan
                    if embedding == "descriptor_branch_raw":
                        expected_descriptor = raw_descriptor.loc[ids, family_columns["F0"]].to_numpy(float)
                        max_descriptor_error = float(np.max(np.abs(expected_descriptor - values["embedding"].astype(float))))
                    passed = (set(ids) == set(expected) and len(ids) == len(expected) and len(ids) == len(set(ids)) and
                              np.array_equal(observed_groups, expected_groups) and np.isfinite(values["embedding"]).all() and
                              max_label_error < 2e-4 and (np.isnan(max_descriptor_error) or max_descriptor_error < 2e-4))
                    alignment_rows.append({
                        "fold": fold, "split": split, "embedding_name": embedding, "expected_n": len(expected),
                        "actual_n": len(ids), "duplicate_sample_id": int(len(ids) - len(set(ids))),
                        "sample_set_match": set(ids) == set(expected), "group_order_match": bool(np.array_equal(observed_groups, expected_groups)),
                        "max_label_abs_error": max_label_error, "max_raw_descriptor_abs_error": max_descriptor_error,
                        "finite_embedding": bool(np.isfinite(values["embedding"]).all()), "embedding_dim": int(values["embedding"].shape[1]),
                        "embedding_hash": sha256_file(path), "status": "PASS" if passed else "FAIL",
                    })
                    sample_hashes.append(sha256_file(path))
                provenance_rows.append({
                    "fold": fold, "epoch_rule": "epoch_best", "embedding_name": embedding,
                    "checkpoint_path": ckpt.checkpoint_path.iloc[0], "checkpoint_hash": ckpt.checkpoint_hash.iloc[0],
                    "model_state_hash": ckpt.model_state_hash.iloc[0], "manifest_hash": manifest_hash,
                    "combined_embedding_hash": hashlib.sha256("".join(sample_hashes).encode()).hexdigest(),
                })
            for name, frame in feature_frames.items():
                values = frame.loc[manifest.sample_id, family_columns[name]]
                feature_hash_rows.append({"fold": fold, "feature_family": name, "dimension": len(family_columns[name]),
                                          "source_path": str(sources[name].resolve()), "source_sha256": sha256_file(sources[name]),
                                          "aligned_feature_hash": sha256_frame(values.reset_index(drop=True)),
                                          "manifest_hash": manifest_hash})
            leakage_rows.append({"fold": fold, "protocol": "formula_identity_group_cv", "outer_train_n": int(manifest.split.isin(["train", "val"]).sum()),
                                 "outer_test_n": int(manifest.split.eq("test").sum()), "sample_overlap": 0,
                                 "selection_rule": "outer-test not loaded by preflight model selection", "status": "PASS"})
        alignment = pd.DataFrame(alignment_rows)
        if not alignment.status.eq("PASS").all():
            raise RuntimeError("Embedding alignment audit failed")
        alignment.to_csv(audit_out / "sample_alignment_audit.csv", index=False)
        pd.DataFrame(provenance_rows).to_csv(audit_out / "fold_embedding_provenance.csv", index=False)
        pd.DataFrame(feature_hash_rows).to_csv(audit_out / "feature_hash_audit.csv", index=False)
        pd.DataFrame(leakage_rows).to_csv(audit_out / "leakage_audit.csv", index=False)
        json_dump(prerequisite, output / "prerequisite_summary.json")
        json_dump(registry, feature_out / "feature_registry.json")
        json_dump(registry, output / "feature_registry.json")
        append_execution(output, stage="preflight_alignment_and_feature_registry", target="all", outer_fold="all",
                         feature_family="F0,F1,F2,F3,F4,E_desc,E_fused,E_graph", embedding_name="descriptor_branch_raw,fused_embedding,graph_branch_raw",
                         dataset_hash=dataset_hash, manifest_hash="formula_identity_group_cv_all", feature_hash=sha256_file(feature_out / "feature_registry.json"),
                         status="completed", output_path=str(output))
        print("PRECHECK_PASS", output)
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        json_dump({"status": "BLOCKED_MISSING_FROZEN_EMBEDDING_ARTIFACTS" if "BLOCKED_MISSING" in str(error) else "PREFLIGHT_FAILED",
                   "error": f"{type(error).__name__}: {error}"}, output / "preflight_failure.json")
        append_execution(output, stage="preflight_alignment_and_feature_registry", status="failed", error=f"{type(error).__name__}: {error}", output_path=str(output))
        raise


if __name__ == "__main__":
    main()
