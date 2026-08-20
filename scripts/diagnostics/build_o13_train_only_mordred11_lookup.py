#!/usr/bin/env python3
"""Build an O13 seed-specific 11-D Mordred lookup fitted on train rows only.

This intentionally reuses the frozen raw canonical-SMILES lookup.  It neither
recalculates Mordred descriptors nor reads labels, validation rows, test rows,
or external data.  Its component-occurrence and absence semantics mirror
``csv_pyg_five_multi.py``: a SMILES value is absent only when it is ``nan`` or
``[Fr]``; a present structure at ratio zero remains a real occurrence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


SMILES_COLUMNS = ("IL_SMILE", "HL_SMILE", "Chol_SMILE", "PEG_SMILE", "Fifth_SMILE")
DESCRIPTORS = ("SsNH3", "SMR_VSA9", "SlogP_VSA11", "SlogP_VSA10", "TopoPSA", "MW",
               "nRot", "nRing", "nAromAtom", "nHBDon", "nHBAcc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def loader_key(value: object) -> str | None:
    """Match loader absence semantics and canonicalization for descriptor lookup."""
    if str(value) in {"nan", "[Fr]"}:
        return None
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        # The loader turns invalid component strings into the [Fr] graph.
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-lookup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_csv, manifest_path, raw_path, output = (args.input_csv.resolve(), args.manifest.resolve(),
                                                   args.raw_lookup.resolve(), args.output.resolve())
    source = pd.read_csv(input_csv, dtype={"ID": str})
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    needed = {"ID", *SMILES_COLUMNS}
    if missing := needed.difference(source.columns):
        raise ValueError(f"Input misses required columns: {sorted(missing)}")
    if source.ID.duplicated().any() or manifest.sample_id.duplicated().any():
        raise ValueError("Input ID and manifest sample_id must both be unique.")
    if set(manifest.split) != {"train", "val", "test"} or len(manifest) != len(source):
        raise ValueError("Manifest must cover every source row once with train/val/test labels.")
    source_index = pd.Series(source.index.to_numpy(), index=source.ID)
    missing_ids = set(manifest.sample_id) - set(source.ID)
    if missing_ids:
        raise ValueError(f"Manifest IDs absent from input: {sorted(missing_ids)[:10]}")
    rows_by_split = {
        split: set(source_index.loc[manifest.loc[manifest.split.eq(split), "sample_id"]].astype(int))
        for split in ("train", "val", "test")
    }
    if rows_by_split["train"] & rows_by_split["val"] or rows_by_split["train"] & rows_by_split["test"]:
        raise RuntimeError("Scaler train rows overlap validation or test rows.")

    raw = pd.read_csv(raw_path, keep_default_na=False)
    feature_columns = [f"feature_{index}" for index in range(len(DESCRIPTORS))]
    if set(raw.columns) != {"smiles", *feature_columns} or raw.smiles.duplicated().any():
        raise ValueError("Raw lookup must have unique smiles and exactly the frozen 11 feature columns.")
    raw = raw.copy()
    raw["canonical_smiles"] = raw.smiles.map(loader_key)
    # The raw [Fr] row becomes None and deliberately remains unqueryable by
    # the loader, preserving current zero-vector absent-component behaviour.
    lookup = raw.loc[raw.canonical_smiles.notna()].set_index("canonical_smiles")[feature_columns]
    if lookup.index.duplicated().any():
        raise ValueError("Raw lookup canonicalization created duplicate molecular keys.")

    train = source.iloc[sorted(rows_by_split["train"])]
    occurrence_rows = []
    for source_index_value, record in train.iterrows():
        for component_position, column in enumerate(SMILES_COLUMNS, start=1):
            key = loader_key(record[column])
            if key is not None:
                occurrence_rows.append({"source_index": int(source_index_value),
                                        "component_position": component_position,
                                        "canonical_smiles": key})
    occurrences = pd.DataFrame(occurrence_rows)
    missing_lookup = sorted(set(occurrences.canonical_smiles) - set(lookup.index))
    if missing_lookup:
        raise ValueError(f"Raw lookup misses train component structures: {missing_lookup[:10]}")
    fit_values = lookup.loc[occurrences.canonical_smiles, feature_columns].to_numpy(float)
    means = fit_values.mean(axis=0)
    stds = fit_values.std(axis=0, ddof=0)
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds < 1e-12).any():
        raise ValueError("Train-only descriptor scaler has a non-finite or zero-variance feature.")
    transformed = (lookup[feature_columns].to_numpy(float) - means) / stds
    standardized = pd.DataFrame({"smiles": lookup.index, **{
        name: transformed[:, index] for index, name in enumerate(feature_columns)}})
    output.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(output, index=False)
    payload = {
        "input_csv": str(input_csv), "input_sha256": sha256(input_csv),
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "raw_lookup": str(raw_path), "raw_lookup_sha256": sha256(raw_path),
        "descriptor_names": list(DESCRIPTORS), "feature_columns": feature_columns,
        "means": means.tolist(), "stds": stds.tolist(),
        "scaler_fit_policy": (
            "Fit population mean/std only on non-absent component occurrences from manifest split=train. "
            "Absent means raw SMILES nan/[Fr]/invalid exactly as the O12 loader; present structures at ratio=0 remain occurrences. "
            "The [Fr] lookup row is excluded so loader fallback remains an all-zero descriptor vector."
        ),
        "scaler_fit_rows": len(rows_by_split["train"]),
        "scaler_fit_component_occurrences": len(occurrences),
        "split_row_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "leakage_check": {
            "fit_rows_intersect_val_rows": len(rows_by_split["train"] & rows_by_split["val"]),
            "fit_rows_intersect_test_rows": len(rows_by_split["train"] & rows_by_split["test"]),
            "status": "PASS",
        },
        "standardized_lookup_rows": len(standardized), "feedback_read": False, "targets_read": False,
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "fit_rows": payload["scaler_fit_rows"],
                      "fit_occurrences": payload["scaler_fit_component_occurrences"],
                      "leakage_check": payload["leakage_check"]}, indent=2))


if __name__ == "__main__":
    main()
