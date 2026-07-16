#!/usr/bin/env python3
"""Rebuild only new-dataset artifacts after a passing deduplication audit.

This program never reads old feature/cached graph/scaler/checkpoint artifacts.
It is intentionally gated on the audit status and keeps labels unscaled: every
modeling fold must fit its own label scaler from its outer-training partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from rdkit import Chem


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

from audit_deduplicated_dataset import COMPONENTS, TARGETS, append_execution, sha256_file, sha256_text  # noqa: E402
from stable_formulation import build_stable_feature_sets  # noqa: E402


# Fixed from the referenced 11-dimensional GraphGPS configuration.  These are
# descriptor definitions, not values, scalers, or a feedback-selected rerun.
MORDRED_11 = [
    "SsNH3", "SMR_VSA9", "SlogP_VSA11", "SlogP_VSA10", "TopoPSA", "MW",
    "nRot", "nRing", "nAromAtom", "nHBDon", "nHBAcc",
]


def json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def code_hash() -> str:
    files = [Path(__file__), SCRIPT_DIR / "stable_formulation.py", ROOT / "graph_feature.py"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def canonical_smiles(value: object) -> str:
    molecule = Chem.MolFromSmiles(str(value)) if not pd.isna(value) else None
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True)


def graph_summary(smiles: str, coarse_grain: bool) -> dict[str, object]:
    """Materialize a topology cache keyed by canonical SMILES, not row position."""
    from graph_feature import smiles2graph

    molecule = Chem.MolFromSmiles(smiles)
    graph = smiles2graph(molecule, coarse_grain, 6)
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    node_features = np.asarray(graph["node_feat"])
    edge_features = np.asarray(graph["edge_feat"])
    graph_hash = hashlib.sha256()
    for array in (edge_index, node_features, edge_features):
        graph_hash.update(np.ascontiguousarray(array).tobytes())
    return {
        "canonical_smiles": smiles, "nodes": int(graph["num_nodes"]), "edges": int(edge_index.shape[1]),
        "topology_hash": graph_hash.hexdigest(), "coarse_grain": coarse_grain,
    }


def component_graph_cache(frame: pd.DataFrame, output_path: Path, coarse_grain: bool) -> None:
    rows: list[dict[str, object]] = []
    for _, record in frame.iterrows():
        for position, (_, smiles_column, _) in enumerate(COMPONENTS, start=1):
            smiles = canonical_smiles(record[smiles_column])
            if not smiles and position != 5:
                raise ValueError(f"{record.sample_id} has invalid component {position} SMILES after a passing audit.")
            masked = not bool(smiles)
            rows.append({"sample_id": record.sample_id, "component_position": position, "masked_component": masked,
                         **graph_summary(smiles or "[Fr]", coarse_grain)})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def mordred_features(frame: pd.DataFrame, output_path: Path) -> None:
    """Compute raw 11D descriptors solely from structures in this dataset."""
    # Mordred 1.x still imports the NumPy 1.x alias ``product``.  Defining the
    # exact legacy alias locally preserves the descriptor implementation under
    # the project's NumPy 2 environment without changing descriptor values.
    if not hasattr(np, "product"):
        np.product = np.prod  # type: ignore[attr-defined]
    from mordred import Calculator, descriptors

    calculator = Calculator(descriptors, ignore_3D=True)
    available = {str(descriptor): descriptor for descriptor in calculator.descriptors}
    missing = [name for name in MORDRED_11 if name not in available]
    if missing:
        raise RuntimeError(f"Installed mordred does not expose required descriptors: {missing}")
    selected_calculator = Calculator([available[name] for name in MORDRED_11], ignore_3D=True)
    rows: list[dict[str, object]] = []
    for _, record in frame.iterrows():
        for position, (_, smiles_column, _) in enumerate(COMPONENTS, start=1):
            smiles = canonical_smiles(record[smiles_column])
            if not smiles and position != 5:
                raise ValueError(f"{record.sample_id} has invalid component {position} SMILES after a passing audit.")
            if not smiles:
                numeric = pd.Series(np.zeros(len(MORDRED_11), dtype=float), index=MORDRED_11)
            else:
                molecule = Chem.MolFromSmiles(smiles)
                values = selected_calculator(molecule).asdict()
                numeric = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)
            rows.append({"sample_id": record.sample_id, "component_position": position,
                         "canonical_smiles": smiles or "[Fr]", "masked_component": not bool(smiles),
                         **{f"feature_{index}": numeric.iloc[index] for index in range(len(MORDRED_11))}})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def mordred_lookup(raw_path: Path, output_path: Path) -> None:
    """Create the GraphGPS lookup table from this run's raw descriptor cache.

    The loader resolves descriptors by canonical SMILES rather than sample row.
    Keep one vector per structure, assert that repeated uses agree, and retain
    the `[Fr]` sentinel for the explicitly masked fifth component.  This is a
    format conversion of the freshly built cache, never a reuse of legacy
    descriptor files.
    """
    raw = pd.read_csv(raw_path)
    feature_columns = [f"feature_{index}" for index in range(len(MORDRED_11))]
    if raw["canonical_smiles"].isna().any() or raw[feature_columns].isna().any().any():
        raise ValueError("Mordred cache has missing lookup keys or descriptor values.")
    grouped = raw.groupby("canonical_smiles", dropna=False, sort=True)
    rows: list[dict[str, object]] = []
    for smiles, records in grouped:
        unique_vectors = records[feature_columns].drop_duplicates()
        if len(unique_vectors) != 1:
            raise ValueError(f"Inconsistent Mordred vectors for canonical SMILES {smiles!r}.")
        row = {"smiles": smiles}
        row.update(unique_vectors.iloc[0].to_dict())
        rows.append(row)
    lookup = pd.DataFrame(rows, columns=["smiles", *feature_columns])
    if lookup.smiles.duplicated().any() or lookup.empty:
        raise ValueError("Mordred lookup must contain one non-empty row per canonical SMILES.")
    lookup.to_csv(output_path, index=False)


def artifact_record(path: Path, name: str, dataset_hash: str, sample_hash: str, schema: object, version: str) -> dict[str, object]:
    return {
        "artifact": name, "path": str(path.resolve()), "sha256": sha256_file(path),
        "dataset_sha256": dataset_hash, "sample_id_hash": sample_hash, "feature_schema": json.dumps(schema, ensure_ascii=False),
        "generator_code_sha256": version, "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def require_passing_audit(output_dir: Path) -> tuple[dict[str, object], Path]:
    source_path = output_dir / "data_source.json"
    profile_path = output_dir / "data_audit" / "dataset_profile.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if source.get("audit_status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise RuntimeError(f"Artifact rebuild blocked: audit status is {source.get('audit_status')!r}.")
    dataset_path = output_dir / "data_audit" / "dataset_with_sample_id.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing audited dataset: {dataset_path}")
    if sha256_file(Path(source["dataset_path"])) != source["dataset_sha256"]:
        raise RuntimeError("Raw selected dataset changed since the audit; rerun the audit before rebuilding.")
    return profile, dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "deduplicated_rebaseline")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    try:
        profile, dataset_path = require_passing_audit(output_dir)
    except Exception as error:
        append_execution(output_dir, {
            "timestamp": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
            "dataset_path": None, "dataset_sha256": None, "protocol": None, "fold": None, "seed": None,
            "manifest_sha256": None, "feature_hash": None, "config_hash": None, "checkpoint": None,
            "status": "BLOCKED_AUDIT_GATE", "error": str(error), "output_path": str(output_dir / "artifacts"),
        })
        raise SystemExit(f"{error}")

    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(dataset_path, dtype={"sample_id": str})
    if frame.sample_id.duplicated().any() or frame.sample_id.isna().any():
        raise ValueError("Audited dataset must contain exactly one non-null sample_id per row.")
    sample_hash = sha256_text("\n".join(sorted(frame.sample_id)))
    if sample_hash != profile["sample_id_hash"]:
        raise RuntimeError("sample_id hash differs from the completed audit.")
    dataset_hash = profile["sha256"]
    version = code_hash()
    schema = SimpleNamespace(components=[{"name_column": name, "smiles_column": smiles, "ratio_column": ratio}
                                         for name, smiles, ratio in COMPONENTS])

    smiles_rows: list[dict[str, object]] = []
    for _, record in frame.iterrows():
        for position, (_, smiles_column, _) in enumerate(COMPONENTS, start=1):
            smiles_rows.append({"sample_id": record.sample_id, "component_position": position,
                                "canonical_smiles": canonical_smiles(record[smiles_column]) or "[Fr]"})
    canonical_path = artifact_dir / "canonical_smiles_mapping.csv"
    pd.DataFrame(smiles_rows).to_csv(canonical_path, index=False)
    regular_graph_path = artifact_dir / "graph_cache.csv"
    coarse_graph_path = artifact_dir / "coarse_grain_graph_cache.csv"
    component_graph_cache(frame, regular_graph_path, coarse_grain=False)
    component_graph_cache(frame, coarse_graph_path, coarse_grain=True)
    mordred_path = artifact_dir / "mordred_11_raw_features.csv"
    mordred_features(frame, mordred_path)
    mordred_lookup_path = artifact_dir / "mordred_11_lookup.csv"
    mordred_lookup(mordred_path, mordred_lookup_path)
    feature_sets, _, feature_schema = build_stable_feature_sets(frame, schema)
    feature_paths: list[tuple[str, Path]] = []
    for feature_name, values in feature_sets.items():
        path = artifact_dir / f"{feature_name}.csv"
        pd.concat([frame[["sample_id"]], values.reset_index(drop=True)], axis=1).to_csv(path, index=False)
        feature_paths.append((feature_name, path))
    labels_path = artifact_dir / "raw_labels_by_sample_id.csv"
    frame[["sample_id", *TARGETS]].to_csv(labels_path, index=False)
    feature_schema.update({"mordred_11_descriptors": MORDRED_11, "label_scaling": "not fitted here; fit only inside every outer-training fold"})
    json_dump(feature_schema, artifact_dir / "feature_schema.json")

    artifacts = [("canonical_smiles", canonical_path, {"columns": ["sample_id", "component_position", "canonical_smiles"]}),
                 ("graph", regular_graph_path, {"coarse_grain": False}),
                 ("coarse_grain_graph", coarse_graph_path, {"coarse_grain": True}),
                 ("mordred_11_raw", mordred_path, {"descriptors": MORDRED_11}),
                 ("mordred_11_lookup", mordred_lookup_path,
                  {"key": "canonical_smiles", "columns": ["smiles", *[f"feature_{index}" for index in range(len(MORDRED_11))]]}),
                 ("raw_labels", labels_path, {"targets": TARGETS}),
                 *((name, path, feature_schema) for name, path in feature_paths)]
    inventory = pd.DataFrame([artifact_record(path, name, dataset_hash, sample_hash, artifact_schema, version)
                              for name, path, artifact_schema in artifacts])
    inventory.to_csv(artifact_dir / "artifact_inventory.csv", index=False)
    integrity_rows: list[dict[str, object]] = []
    expected = set(frame.sample_id)
    for name, path, _ in artifacts:
        table = pd.read_csv(path, dtype={"sample_id": str})
        if name == "mordred_11_lookup":
            expected_smiles = set(pd.read_csv(mordred_path)["canonical_smiles"])
            actual_smiles = set(table["smiles"])
            integrity_rows.append({"artifact": name, "expected_sample_ids": len(expected),
                                   "present_sample_ids": None, "missing_sample_ids": None,
                                   "extra_sample_ids": None, "entries_per_sample": None,
                                   "valid_entry_count": bool(actual_smiles == expected_smiles and not table.smiles.duplicated().any()),
                                   "status": "PASS" if actual_smiles == expected_smiles and not table.smiles.duplicated().any() else "FAIL"})
            continue
        actual = set(table.sample_id)
        counts = table.groupby("sample_id").size()
        expected_count = 5 if name in {"canonical_smiles", "graph", "coarse_grain_graph", "mordred_11_raw"} else 1
        integrity_rows.append({"artifact": name, "expected_sample_ids": len(expected), "present_sample_ids": len(actual),
                               "missing_sample_ids": len(expected - actual), "extra_sample_ids": len(actual - expected),
                               "entries_per_sample": expected_count, "valid_entry_count": bool((counts == expected_count).all()),
                               "status": "PASS" if actual == expected and (counts == expected_count).all() else "FAIL"})
    integrity = pd.DataFrame(integrity_rows)
    integrity.to_csv(artifact_dir / "cache_integrity.csv", index=False)
    report = ["# Deduplicated Artifact Rebuild", "", f"- Dataset SHA256: `{dataset_hash}`", f"- sample_id hash: `{sample_hash}`",
              "- No old manifests, scalers, feature caches, graph caches, checkpoints, or OOF predictions were read.",
              "- Label values remain raw; every scaler must be fitted inside its outer training fold.",
              f"- Cache integrity status: {'PASS' if (integrity.status == 'PASS').all() else 'FAIL'}."]
    (artifact_dir / "cache_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    append_execution(output_dir, {
        "timestamp": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
        "dataset_path": str(dataset_path), "dataset_sha256": dataset_hash, "protocol": None, "fold": None, "seed": None,
        "manifest_sha256": None, "feature_hash": sha256_file(artifact_dir / "feature_schema.json"), "config_hash": None,
        "checkpoint": None, "status": "PASS" if (integrity.status == 'PASS').all() else "FAIL",
        "error": None, "output_path": str(artifact_dir),
    })
    if not (integrity.status == "PASS").all():
        raise SystemExit("Artifact integrity failed.")
    print(f"Wrote {artifact_dir}")


if __name__ == "__main__":
    main()
