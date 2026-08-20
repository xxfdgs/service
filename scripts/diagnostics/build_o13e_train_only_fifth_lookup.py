#!/usr/bin/env python3
"""Fit and audit seed-specific O13-E fifth descriptors on train rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from graphgps.lrx_add.fifth_mechanistic_descriptors import MECHANISTIC_DESCRIPTOR_NAMES


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


def loader_key(value: object) -> str | None:
    if str(value) in {"nan", "[Fr]"}:
        return None
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def correlation_table(values: pd.DataFrame) -> pd.DataFrame:
    names = list(values.columns)
    rows = []
    for left in names:
        for right in names:
            if np.std(values[left].to_numpy(float), ddof=0) == 0 or np.std(values[right].to_numpy(float), ddof=0) == 0:
                value = np.nan
            else:
                value = values[left].corr(values[right])
            rows.append({"left": left, "right": right,
                         "pearson": value})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-fifth-lookup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    input_csv, manifest_path, raw_path, output = (args.input_csv.resolve(), args.manifest.resolve(),
                                                   args.raw_fifth_lookup.resolve(), args.output.resolve())
    source = pd.read_csv(input_csv, dtype={"ID": str})
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    if source.ID.duplicated().any() or manifest.sample_id.duplicated().any():
        raise ValueError("Input IDs and manifest sample IDs must be unique")
    if set(manifest.split) != {"train", "val", "test"} or len(manifest) != len(source):
        raise ValueError("Manifest must exactly cover the source rows with train/val/test")
    index_by_id = pd.Series(source.index.to_numpy(), index=source.ID)
    if not set(manifest.sample_id).issubset(set(index_by_id.index)):
        raise ValueError("Manifest references IDs absent from source input")
    row_sets = {
        split: set(index_by_id.loc[manifest.loc[manifest.split.eq(split), "sample_id"]].astype(int))
        for split in ("train", "val", "test")
    }
    if row_sets["train"] & row_sets["val"] or row_sets["train"] & row_sets["test"]:
        raise RuntimeError("Train scaler-fit rows overlap validation/test rows")

    descriptor_names = list(MECHANISTIC_DESCRIPTOR_NAMES)
    raw = pd.read_csv(raw_path)
    expected = {"canonical_smiles", *MORDRED_NAMES, *descriptor_names}
    if set(raw.columns) != expected or raw.canonical_smiles.duplicated().any():
        raise ValueError("Raw O13-E lookup schema or canonical keys are invalid")
    lookup = raw.set_index("canonical_smiles")
    train = source.iloc[sorted(row_sets["train"])]
    occurrences = pd.DataFrame({
        "source_index": train.index.to_numpy(int),
        "canonical_smiles": [loader_key(value) for value in train.Fifth_SMILE],
    })
    occurrences = occurrences.loc[occurrences.canonical_smiles.notna()].copy()
    missing = sorted(set(occurrences.canonical_smiles) - set(lookup.index))
    if missing:
        raise ValueError(f"Raw O13-E lookup misses present training Fifth structures: {missing[:10]}")
    fit_values = lookup.loc[occurrences.canonical_smiles, descriptor_names].to_numpy(float)
    means, raw_stds = fit_values.mean(axis=0), fit_values.std(axis=0, ddof=0)
    if not np.isfinite(means).all() or not np.isfinite(raw_stds).all():
        raise RuntimeError("O13-E train-only descriptor scaler contains non-finite values")
    # A semantically well-defined descriptor can legitimately be constant in
    # a particular training fold (for example, no tertiary amine). Keep the
    # fixed 12-D architecture and make its training values exactly zero. The
    # effective denominator 1 only avoids division by zero for held-out novel
    # structures; the raw population std=0 remains explicitly recorded.
    zero_variance = [name for name, std in zip(descriptor_names, raw_stds) if std < 1e-12]
    stds = np.where(raw_stds < 1e-12, 1.0, raw_stds)

    # [Fr] exists in the raw table for auditability but is deliberately omitted
    # from standardized lookup. Loader fallback then preserves the all-zero
    # absent-component vector, exactly matching O12's absence semantics.
    present_lookup = lookup.loc[lookup.index != "[Fr]"]
    transformed = (present_lookup[descriptor_names].to_numpy(float) - means) / stds
    standardized = pd.DataFrame({"smiles": present_lookup.index, **{
        f"feature_{index}": transformed[:, index] for index in range(len(descriptor_names))
    }})
    output.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(output, index=False)

    # Redundancy is evaluated only in actual train Fifth occurrences so all
    # 23 columns share identical rows. It is an audit, not feature selection.
    audit_values = lookup.loc[occurrences.canonical_smiles, [*MORDRED_NAMES, *descriptor_names]].copy()
    audit_values.index = occurrences.source_index.to_numpy()
    feature_summary = pd.DataFrame({
        "descriptor": audit_values.columns,
        "unique_value_count": [audit_values[name].nunique(dropna=False) for name in audit_values],
        "variance_population": [float(np.var(audit_values[name], ddof=0)) for name in audit_values],
        "nonzero_fraction": [float((audit_values[name] != 0).mean()) for name in audit_values],
    })
    exact_duplicates = []
    for index, left in enumerate(audit_values.columns):
        for right in audit_values.columns[index + 1:]:
            if np.array_equal(audit_values[left].to_numpy(), audit_values[right].to_numpy()):
                exact_duplicates.append({"left": left, "right": right})
    audit_dir = args.audit_dir.resolve(); audit_dir.mkdir(parents=True, exist_ok=True)
    audit_values.to_csv(audit_dir / "train_fifth_descriptor_values_raw.csv", index_label="source_index")
    feature_summary.to_csv(audit_dir / "train_fifth_descriptor_redundancy_summary.csv", index=False)
    correlation_table(audit_values).to_csv(audit_dir / "train_fifth_descriptor_pairwise_pearson.csv", index=False)
    pd.DataFrame(exact_duplicates, columns=["left", "right"]).to_csv(
        audit_dir / "train_fifth_descriptor_exact_duplicates.csv", index=False)

    payload = {
        "input_csv": str(input_csv), "input_sha256": sha256(input_csv),
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "raw_fifth_lookup": str(raw_path), "raw_fifth_lookup_sha256": sha256(raw_path),
        "descriptor_names": descriptor_names, "means": means.tolist(),
        "population_stds": raw_stds.tolist(), "effective_stds": stds.tolist(),
        "zero_variance_train_descriptors": zero_variance,
        "scaler_fit_policy": (
            "Fit population mean/std only on present Fifth_SMILE occurrences in manifest split=train. "
            "Fifth ratio is intentionally not used for occurrence filtering, matching loader descriptor lookup semantics. "
            "[Fr]/nan/invalid is absent and omitted from standardized lookup so loader fallback is zero."
        ),
        "scaler_fit_rows": len(row_sets["train"]), "scaler_fit_fifth_occurrences": len(occurrences),
        "split_row_counts": {split: len(rows) for split, rows in row_sets.items()},
        "leakage_check": {"fit_rows_intersect_val_rows": len(row_sets["train"] & row_sets["val"]),
                          "fit_rows_intersect_test_rows": len(row_sets["train"] & row_sets["test"]), "status": "PASS"},
        "redundancy_audit_dir": str(audit_dir), "exact_duplicate_pairs": exact_duplicates,
        "targets_read": False, "external_data_read": False,
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "fit_rows": len(row_sets["train"]),
                      "fit_fifth_occurrences": len(occurrences), "leakage_check": payload["leakage_check"],
                      "zero_variance_train_descriptors": zero_variance,
                      "exact_duplicates": exact_duplicates}, indent=2))


if __name__ == "__main__":
    main()
