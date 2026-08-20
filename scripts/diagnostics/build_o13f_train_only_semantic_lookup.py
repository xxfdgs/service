#!/usr/bin/env python3
"""Fit O13-F numeric scaling and categorical vocabularies on train rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from graphgps.lrx_add.fifth_semantic_features import SEMANTIC_NUMERIC_NAMES

CATEGORICAL = ("family_type", "UC_amino_acid_type")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(value: object) -> str | None:
    if pd.isna(value) or str(value) in {"nan", "[Fr]"}:
        return None
    molecule = Chem.MolFromSmiles(str(value))
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-lookup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, manifest, raw = (pd.read_csv(args.input_csv, dtype={"ID": str}),
                             pd.read_csv(args.manifest, dtype={"sample_id": str}),
                             pd.read_csv(args.raw_lookup))
    if source.ID.duplicated().any() or manifest.sample_id.duplicated().any():
        raise ValueError("Input IDs and manifest IDs must be unique")
    if set(manifest.split) != {"train", "val", "test"} or len(manifest) != len(source):
        raise ValueError("Manifest must exactly partition the input rows")
    required = {"canonical_smiles", *SEMANTIC_NUMERIC_NAMES, *CATEGORICAL}
    if missing := required.difference(raw.columns):
        raise ValueError(f"Raw semantic lookup misses columns: {sorted(missing)}")
    if raw.canonical_smiles.duplicated().any():
        raise ValueError("Raw semantic lookup has duplicate canonical keys")
    lookup = raw.set_index("canonical_smiles")
    index_by_id = pd.Series(source.index.to_numpy(), index=source.ID)
    rows = {split: index_by_id.loc[manifest.loc[manifest.split.eq(split), "sample_id"]].to_numpy(int)
            for split in ("train", "val", "test")}
    if set(rows["train"]) & set(rows["val"]) or set(rows["train"]) & set(rows["test"]):
        raise RuntimeError("Scaler-fit train rows overlap validation/test rows")
    train_occurrences = pd.DataFrame({"source_index": rows["train"],
                                      "canonical_smiles": [key(source.iloc[index].Fifth_SMILE)
                                                           for index in rows["train"]]})
    train_occurrences = train_occurrences.dropna(subset=["canonical_smiles"])
    if missing := set(train_occurrences.canonical_smiles) - set(lookup.index):
        raise ValueError(f"Raw semantic lookup misses train Fifth structures: {sorted(missing)[:5]}")
    train_values = lookup.loc[train_occurrences.canonical_smiles, list(SEMANTIC_NUMERIC_NAMES)].to_numpy(float)
    mean, raw_std = train_values.mean(axis=0), train_values.std(axis=0, ddof=0)
    std = np.where(raw_std < 1e-12, 1.0, raw_std)
    vocabularies = {}
    for column in CATEGORICAL:
        values = sorted(set(lookup.loc[train_occurrences.canonical_smiles, column].astype(str)))
        vocabularies[column] = {"__UNK__": 0, **{value: index + 1 for index, value in enumerate(values)}}
    present = lookup.loc[lookup.index != "[Fr]"]
    numeric = (present[list(SEMANTIC_NUMERIC_NAMES)].to_numpy(float) - mean) / std
    categorical_parts = []
    layout = []
    for column in CATEGORICAL:
        vocabulary = vocabularies[column]
        encoded = np.zeros((len(present), len(vocabulary)), dtype=float)
        for row_index, value in enumerate(present[column].astype(str)):
            encoded[row_index, vocabulary.get(value, 0)] = 1.0
        categorical_parts.append(encoded)
        layout.extend([f"{column}={value}" for value in vocabulary])
    matrix = np.concatenate([numeric, *categorical_parts], axis=1)
    standardized = pd.DataFrame({"smiles": present.index, **{
        f"feature_{index}": matrix[:, index] for index in range(matrix.shape[1])}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(args.output, index=False)
    unknown_counts = {}
    for split, indices in rows.items():
        keys = [key(source.iloc[index].Fifth_SMILE) for index in indices]
        values = lookup.loc[[value for value in keys if value is not None], list(CATEGORICAL)]
        unknown_counts[split] = {column: int((~values[column].astype(str).isin(vocabularies[column])).sum())
                                 for column in CATEGORICAL}
    metadata = {
        "input_csv": str(args.input_csv.resolve()), "input_sha256": sha256(args.input_csv),
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
        "raw_lookup": str(args.raw_lookup.resolve()), "raw_lookup_sha256": sha256(args.raw_lookup),
        "numeric_feature_names": list(SEMANTIC_NUMERIC_NAMES), "numeric_mean": mean.tolist(),
        "numeric_population_std": raw_std.tolist(), "numeric_effective_std": std.tolist(),
        "categorical_vocabularies": vocabularies,
        "feature_layout": [*SEMANTIC_NUMERIC_NAMES, *layout], "feature_dim": int(matrix.shape[1]),
        "unknown_category_counts_by_split": unknown_counts,
        "scaler_fit_policy": "Numeric mean/population std and category vocabulary use only present Fifth occurrences in manifest split=train; val/test are transform-only.",
        "scaler_fit_rows": int(len(rows["train"])), "scaler_fit_fifth_occurrences": int(len(train_occurrences)),
        "leakage_check": {"fit_rows_intersect_val_rows": 0, "fit_rows_intersect_test_rows": 0, "status": "PASS"},
        "absent_policy": "[Fr] omitted from lookup; loader fallback is an all-zero semantic vector.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "feature_dim": metadata["feature_dim"],
                      "leakage_check": metadata["leakage_check"], "unknown": unknown_counts}, indent=2))


if __name__ == "__main__":
    main()
