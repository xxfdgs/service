#!/usr/bin/env python3
"""Build the label-free, structure-derived raw O13-F semantic lookup."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

from graphgps.lrx_add.fifth_semantic_features import SEMANTIC_DEFINITIONS, semantic_features


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    if pd.isna(value) or str(value) in {"nan", "[Fr]"}:
        return "[Fr]"
    molecule = Chem.MolFromSmiles(str(value))
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else "[Fr]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.input_csv, dtype={"ID": str})
    if "Fifth_SMILE" not in source:
        raise ValueError("Input CSV must contain Fifth_SMILE")
    rows = []
    for smiles in sorted({canonical(value) for value in source.Fifth_SMILE}):
        result = semantic_features(smiles)
        rows.append({"canonical_smiles": smiles, **result.as_row()})
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    warnings = table.parse_warnings.fillna("").astype(str)
    numeric = table.select_dtypes(include="number")
    audit = {
        "version": "O13-F semantic-v1",
        "input_csv": str(args.input_csv.resolve()), "input_sha256": sha256(args.input_csv),
        "canonical_smiles_sha256": hashlib.sha256("\n".join(table.canonical_smiles).encode()).hexdigest(),
        "feature_definitions": SEMANTIC_DEFINITIONS,
        "rows": int(len(table)),
        "family_assignment_counts": table.family_type.value_counts(dropna=False).to_dict(),
        "parse_status_counts": table.parse_status.value_counts(dropna=False).to_dict(),
        "structures_with_warnings": int(warnings.ne("").sum()),
        "unknown_uc_amino_acid_count": int(table.UC_amino_acid_type.eq("UNK").sum()),
        "feature_unique_value_counts": {column: int(table[column].nunique(dropna=False)) for column in table.columns},
        "numeric_nonzero_fraction": {column: float((numeric[column] != 0).mean()) for column in numeric},
        "targets_read": False, "external_data_read": False,
        "absent_policy": "[Fr] is family_type=other with every numeric semantic feature zero.",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "audit": str(args.audit),
                      "family_assignment_counts": audit["family_assignment_counts"],
                      "parse_status_counts": audit["parse_status_counts"]}, indent=2))


if __name__ == "__main__":
    main()
