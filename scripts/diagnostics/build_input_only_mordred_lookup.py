#!/usr/bin/env python3
"""Fit an 11-D Mordred lookup scaler on the fixed input training split only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


SMILES_COLUMNS = ["IL_SMILE", "HL_SMILE", "Chol_SMILE", "PEG_SMILE", "Fifth_SMILE"]


def canonical(value: object) -> str:
    text = str(value)
    if text in {"", "nan", "None", "[Fr]"}:
        return ""
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise ValueError(f"Cannot canonicalize component SMILES: {text}")
    return Chem.MolToSmiles(molecule, canonical=True)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-lookup", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    input_csv = arguments.input_csv.resolve()
    manifest_path = arguments.manifest.resolve()
    lookup_path = arguments.raw_lookup.resolve()
    output = arguments.output.resolve()
    frame = pd.read_csv(input_csv)
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    if len(frame) != len(manifest) or set(manifest.split) != {"train", "val", "test"}:
        raise RuntimeError("Input/manifest integrity check failed")
    if any(column not in frame for column in SMILES_COLUMNS):
        raise RuntimeError("Input lacks a required five-component SMILES column")
    lookup = pd.read_csv(lookup_path)
    features = [column for column in lookup if column.startswith("feature_")]
    if len(features) != 11:
        raise RuntimeError(f"Expected raw 11-D lookup, found {len(features)} columns")
    lookup = lookup.copy()
    lookup["canonical"] = lookup.smiles.map(canonical)
    if lookup.canonical.duplicated().any():
        raise RuntimeError("Raw lookup has duplicate canonical molecules")
    lookup = lookup.set_index("canonical")
    train_indices = manifest.loc[manifest.split.eq("train"), "original_row_index"].astype(int).to_numpy()
    train = frame.iloc[train_indices]
    train_keys = [canonical(value) for column in SMILES_COLUMNS for value in train[column] if canonical(value)]
    missing = sorted(set(train_keys) - set(lookup.index))
    if missing:
        raise RuntimeError(f"Raw lookup misses {len(missing)} input-training components")
    training_values = lookup.loc[train_keys, features].to_numpy(float)
    center = training_values.mean(axis=0)
    scale = training_values.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    transformed = lookup[features].to_numpy(float)
    transformed = (transformed - center) / scale
    out = pd.DataFrame({"smiles": lookup.index})
    for index, name in enumerate(features):
        out[name] = transformed[:, index]
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    stats = {
        "input_csv": str(input_csv), "input_sha256": digest(input_csv),
        "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
        "raw_lookup": str(lookup_path), "raw_lookup_sha256": digest(lookup_path),
        "fit_rows": int(len(train)), "fit_component_occurrences": int(len(train_keys)),
        "feature_count": len(features), "feedback_read": False,
        "means": center.tolist(), "stds": scale.tolist(),
    }
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "unique_molecules": len(out), **{key: stats[key] for key in ("fit_rows", "fit_component_occurrences", "feature_count", "feedback_read")}}))


if __name__ == "__main__":
    main()
