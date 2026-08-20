#!/usr/bin/env python3
"""Build the label-free raw descriptor lookup required by O13-E.

The frozen raw Mordred11 table is reused as-is; only the twelve documented
RDKit graph descriptors are calculated here.  This script reads neither
targets, manifests, nor external validation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

from graphgps.lrx_add.fifth_mechanistic_descriptors import (
    DESCRIPTOR_DEFINITIONS,
    MECHANISTIC_DESCRIPTOR_NAMES,
    descriptor_vector,
)


MORDRED_NAMES = (
    "SsNH3", "SMR_VSA9", "SlogP_VSA11", "SlogP_VSA10", "TopoPSA", "MW",
    "nRot", "nRing", "nAromAtom", "nHBDon", "nHBAcc",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    if str(value) in {"nan", "[Fr]"}:
        return "[Fr]"
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return "[Fr]"
    return Chem.MolToSmiles(molecule, canonical=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--raw-mordred-lookup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_csv, mordred_path, output = (args.input_csv.resolve(), args.raw_mordred_lookup.resolve(),
                                        args.output.resolve())
    source = pd.read_csv(input_csv, dtype={"ID": str})
    if "Fifth_SMILE" not in source:
        raise ValueError("Input CSV has no Fifth_SMILE column")
    raw_mordred = pd.read_csv(mordred_path, keep_default_na=False)
    feature_columns = [f"feature_{index}" for index in range(len(MORDRED_NAMES))]
    if set(raw_mordred.columns) != {"smiles", *feature_columns}:
        raise ValueError("Raw Mordred lookup must be the frozen 11-feature table")
    raw_mordred["canonical_smiles"] = raw_mordred.smiles.map(canonical)
    mordred_map = raw_mordred.loc[raw_mordred.canonical_smiles.ne("[Fr]")].set_index(
        "canonical_smiles")[feature_columns]
    if mordred_map.index.duplicated().any():
        raise RuntimeError("Raw Mordred lookup has duplicate canonical molecular keys")

    fifth_keys = sorted({canonical(value) for value in source.Fifth_SMILE})
    rows = []
    missing = []
    for key in fifth_keys:
        if key == "[Fr]":
            mordred = [0.0] * len(MORDRED_NAMES)
            mechanism = [0.0] * len(MECHANISTIC_DESCRIPTOR_NAMES)
        else:
            if key not in mordred_map.index:
                missing.append(key)
                continue
            mordred = mordred_map.loc[key, feature_columns].to_numpy(float).tolist()
            mechanism = descriptor_vector(key).astype(float).tolist()
        rows.append({"canonical_smiles": key,
                     **dict(zip(MORDRED_NAMES, mordred)),
                     **dict(zip(MECHANISTIC_DESCRIPTOR_NAMES, mechanism))})
    if missing:
        raise ValueError(f"Frozen raw Mordred lookup misses Fifth structures: {missing[:10]}")
    table = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    metadata = {
        "input_csv": str(input_csv), "input_sha256": sha256(input_csv),
        "raw_mordred_lookup": str(mordred_path), "raw_mordred_lookup_sha256": sha256(mordred_path),
        "output": str(output), "raw_mordred_descriptor_names": list(MORDRED_NAMES),
        "mechanistic_descriptor_names": list(MECHANISTIC_DESCRIPTOR_NAMES),
        "mechanistic_descriptor_definitions": DESCRIPTOR_DEFINITIONS,
        "rows": len(table), "absent_key": "[Fr]",
        "absent_raw_descriptor_policy": "All raw Mordred11 and O13-E mechanism descriptors are 0 for [Fr]/absent Fifth.",
        "label_or_external_data_read": False,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(table),
                      "mechanistic_descriptors": len(MECHANISTIC_DESCRIPTOR_NAMES)}, indent=2))


if __name__ == "__main__":
    main()
