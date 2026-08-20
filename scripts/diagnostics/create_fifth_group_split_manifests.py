#!/usr/bin/env python3
"""Create input-only splits with disjoint fifth-component identities.

These manifests test molecular generalization instead of allowing an exact
fifth-component SMILES to occur in train and validation/test simultaneously.
No target value is used to construct a split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_smiles(value: object) -> str:
    if pd.isna(value) or str(value).strip() in {"", "nan", "[Fr]"}:
        return "[Fr]"
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        raise ValueError(f"Invalid fifth-component SMILES: {value!r}")
    return Chem.MolToSmiles(molecule, canonical=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--holdout-group-fraction", type=float, default=0.10)
    args = parser.parse_args()

    source = args.input_csv.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(source, dtype={"ID": str})
    required = {"ID", "Fifth_SMILE"}
    if missing := required.difference(data.columns):
        raise ValueError(f"Input CSV misses columns: {sorted(missing)}")
    if data["ID"].isna().any() or data["ID"].duplicated().any():
        raise ValueError("Input IDs must be complete and unique.")
    groups = data["Fifth_SMILE"].map(canonical_smiles)
    if groups.nunique() < 10:
        raise ValueError("Too few fifth-component identities for grouped splits.")

    inventory = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        outer = GroupShuffleSplit(
            n_splits=1,
            test_size=args.holdout_group_fraction,
            random_state=seed,
        )
        train_val_index, test_index = next(
            outer.split(data, groups=groups))
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=args.holdout_group_fraction / (
                1.0 - args.holdout_group_fraction),
            random_state=seed + 10_000,
        )
        train_relative, val_relative = next(
            inner.split(
                data.iloc[train_val_index],
                groups=groups.iloc[train_val_index],
            ))
        train_index = train_val_index[train_relative]
        val_index = train_val_index[val_relative]
        split_by_index = pd.Series(index=data.index, dtype=object)
        split_by_index.iloc[train_index] = "train"
        split_by_index.iloc[val_index] = "val"
        split_by_index.iloc[test_index] = "test"
        if split_by_index.isna().any():
            raise RuntimeError(f"Split seed {seed} did not cover every input row.")
        group_sets = {
            split: set(groups.loc[split_by_index.eq(split)])
            for split in ("train", "val", "test")
        }
        if any(
            group_sets[left].intersection(group_sets[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise RuntimeError(f"Fifth-component leakage in split seed {seed}.")

        manifest = pd.DataFrame({
            "sample_id": data["ID"],
            "split": split_by_index,
            "original_row_index": data.index,
        })
        manifest["split_order"] = manifest.groupby(
            "split", sort=False).cumcount()
        manifest_path = output / f"fifth_group_manifest_seed{seed}.csv"
        manifest.to_csv(manifest_path, index=False)
        counts = manifest["split"].value_counts()
        inventory.append({
            "seed": seed,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "train_rows": int(counts["train"]),
            "val_rows": int(counts["val"]),
            "test_rows": int(counts["test"]),
            "train_fifth_groups": len(group_sets["train"]),
            "val_fifth_groups": len(group_sets["val"]),
            "test_fifth_groups": len(group_sets["test"]),
            "group_overlap_count": 0,
        })

    inventory_path = output / "fifth_group_manifest_inventory.csv"
    pd.DataFrame(inventory).to_csv(inventory_path, index=False)
    protocol = {
        "source": str(source),
        "source_sha256": sha256(source),
        "rows": int(len(data)),
        "unique_fifth_component_smiles": int(groups.nunique()),
        "first_seed": args.first_seed,
        "seeds": args.seeds,
        "holdout_group_fraction_per_validation_and_test": (
            args.holdout_group_fraction),
        "split_feature": "canonical Fifth_SMILE identity",
        "target_columns_read_for_splitting": [],
        "external_feedback_read": False,
        "threshold_or_side_criterion_used": False,
        "invariant": "train, validation, and test fifth-component identities are disjoint",
    }
    (output / "fifth_group_split_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(pd.DataFrame(inventory).to_string(index=False))


if __name__ == "__main__":
    main()
