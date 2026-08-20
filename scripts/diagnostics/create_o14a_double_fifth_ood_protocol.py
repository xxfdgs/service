#!/usr/bin/env python3
"""Derive double-only manifests from the frozen full Fifth-identity-OOD folds.

This is deliberately a *filter*, not a re-split: for every seed, each double
row retains the train/val/test assignment of the corresponding frozen 700-row
manifest.  Thus Full and Double experiments hold out the same Fifth identities
where that identity occurs in the double domain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


NORM_TARGETS = ("Norm_before", "Norm_after")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_fifth(value: object) -> str:
    text = str(value).strip()
    if pd.isna(value) or text in {"", "nan", "[Fr]"}:
        return "[Fr]"
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise ValueError(f"Invalid Fifth_SMILE: {value!r}")
    return Chem.MolToSmiles(molecule, canonical=True)


def target_audit(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result = {}
    for target in NORM_TARGETS:
        values = pd.to_numeric(frame[target], errors="coerce").dropna()
        result[target] = {
            "count": int(len(values)),
            "high_gt1_count": int((values > 1.0).sum()),
            "low_or_equal_1_count": int((values <= 1.0).sum()),
            "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
            "median": float(values.median()) if len(values) else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--full-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(100, 110)))
    args = parser.parse_args()

    source_path = args.input_csv.resolve()
    manifest_dir = args.full_manifest_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(source_path, dtype={"ID": str})
    required = {"ID", "Fifth_SMILE", "Fifth", "Fifth_class", *NORM_TARGETS}
    if missing := required.difference(source.columns):
        raise ValueError(f"Source misses required columns: {sorted(missing)}")
    if source["ID"].isna().any() or source["ID"].duplicated().any():
        raise ValueError("Source ID values must be complete and unique.")

    source["_fifth_class"] = source["Fifth_class"].fillna("").astype(str).str.strip().str.lower()
    double = source.loc[source["_fifth_class"].eq("double")].copy().reset_index(
        names="full_original_row_index")
    if double.empty:
        raise ValueError("No double rows were found in the supplied source.")
    double["fifth_identity"] = double["Fifth_SMILE"].map(canonical_fifth)
    if (double["fifth_identity"] == "[Fr]").any():
        raise ValueError("Double-domain input unexpectedly contains absent Fifth ([Fr]).")
    double["original_row_index"] = np.arange(len(double), dtype=np.int64)
    input_output = output / "o14a_double_input.csv"
    double.drop(columns=["_fifth_class"]).to_csv(input_output, index=False)
    id_to_local = dict(zip(double["ID"].astype(str), double["original_row_index"], strict=True))

    inventory = []
    for seed in args.seeds:
        full_path = manifest_dir / f"fifth_identity_manifest_seed{seed}.csv"
        if not full_path.is_file():
            raise FileNotFoundError(f"Missing frozen full manifest: {full_path}")
        full = pd.read_csv(full_path, dtype={"sample_id": str})
        required_manifest = {"sample_id", "split", "original_row_index"}
        if missing := required_manifest.difference(full.columns):
            raise ValueError(f"{full_path} misses {sorted(missing)}")
        if len(full) != len(source) or full.sample_id.duplicated().any():
            raise ValueError(f"{full_path} is not a complete unique source manifest.")
        derived = full.loc[full.sample_id.astype(str).isin(id_to_local)].copy()
        if len(derived) != len(double):
            raise RuntimeError(f"Seed {seed} double filtering lost source rows.")
        derived["full_original_row_index"] = derived["original_row_index"].astype(int)
        derived["original_row_index"] = derived.sample_id.astype(str).map(id_to_local).astype(int)
        source_by_id = double.set_index("ID")
        derived["fifth_identity"] = derived.sample_id.astype(str).map(source_by_id["fifth_identity"])
        derived["Fifth"] = derived.sample_id.astype(str).map(source_by_id["Fifth"])
        derived["Fifth_class"] = "double"
        derived["split_order"] = derived.groupby("split", sort=False).cumcount()
        if not set(derived["split"]).issubset({"train", "val", "test"}):
            raise ValueError(f"Seed {seed} contains an invalid split label.")
        if not all((derived["split"] == name).any() for name in ("train", "val", "test")):
            raise ValueError(f"Seed {seed} has an empty double train/val/test partition.")
        identities = {
            split: set(derived.loc[derived.split.eq(split), "fifth_identity"])
            for split in ("train", "val", "test")
        }
        overlaps = {
            "train_val": sorted(identities["train"] & identities["val"]),
            "train_test": sorted(identities["train"] & identities["test"]),
            "val_test": sorted(identities["val"] & identities["test"]),
        }
        if any(overlaps.values()):
            raise RuntimeError(f"Seed {seed} violates Fifth-identity OOD: {overlaps}")
        output_manifest = output / f"double_fifth_identity_manifest_seed{seed}.csv"
        derived[["sample_id", "split", "original_row_index", "full_original_row_index",
                 "fifth_identity", "Fifth", "Fifth_class", "split_order"]].to_csv(
                     output_manifest, index=False)
        split_audits = {}
        for split in ("train", "val", "test"):
            ids = derived.loc[derived.split.eq(split), "sample_id"].astype(str)
            subset = source_by_id.loc[ids]
            split_audits[split] = {
                "rows": int(len(subset)),
                "unique_fifth_identities": int(subset["fifth_identity"].nunique()),
                "all_fifth_class_double": bool(subset["Fifth_class"].astype(str).str.lower().eq("double").all()),
                "targets": target_audit(subset),
            }
        inventory.append({
            "seed": int(seed), "full_manifest": str(full_path),
            "full_manifest_sha256": file_sha256(full_path),
            "derived_manifest": str(output_manifest),
            "derived_manifest_sha256": file_sha256(output_manifest),
            "identity_leakage_pass": True, "overlap_counts": {key: len(value) for key, value in overlaps.items()},
            "splits": split_audits,
        })

    protocol = {
        "protocol": "O14-A double-domain Fifth-identity OOD",
        "source_csv": str(source_path), "source_sha256": file_sha256(source_path),
        "double_input_csv": str(input_output), "double_input_sha256": file_sha256(input_output),
        "selection_rule": "Fifth_class normalized exactly to 'double'",
        "split_rule": "filter each frozen full Fifth-identity-OOD manifest; never re-split rows",
        "invariant": "double train/val/test canonical Fifth identities are pairwise disjoint",
        "seeds": [int(seed) for seed in args.seeds], "per_seed": inventory,
    }
    (output / "o14a_double_fifth_ood_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{
        "seed": item["seed"], "train_rows": item["splits"]["train"]["rows"],
        "val_rows": item["splits"]["val"]["rows"], "test_rows": item["splits"]["test"]["rows"],
        "train_fifth": item["splits"]["train"]["unique_fifth_identities"],
        "val_fifth": item["splits"]["val"]["unique_fifth_identities"],
        "test_fifth": item["splits"]["test"]["unique_fifth_identities"],
        "identity_leakage_pass": item["identity_leakage_pass"],
    } for item in inventory]).to_csv(output / "o14a_double_fifth_ood_inventory.csv", index=False)
    print(f"Created {len(inventory)} matched double-only OOD manifests in {output}")


if __name__ == "__main__":
    main()
