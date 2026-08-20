#!/usr/bin/env python3
"""
Create deterministic row-level random train/validation/test splits.

Unlike fifth-group OOD splitting, this script randomly shuffles individual
samples. The same Fifth_SMILE identity may therefore appear in train, val,
and test.

Default:
    train = 80%
    val   = 10%
    test  = 10%

Seeds:
    100-109

Outputs:
    random_manifest_seed100.csv
    ...
    random_manifest_seed109.csv
    random_manifest_inventory.csv
    random_split_protocol.json

Optional:
    seed100/train.csv
    seed100/val.csv
    seed100/test.csv
    ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    """Return SHA256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def calculate_split_sizes(
    n_rows: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    """
    Calculate deterministic row counts.

    val/test are rounded to nearest integer.
    train receives all remaining rows so that every row is assigned exactly once.
    """

    total = train_fraction + val_fraction + test_fraction

    if not np.isclose(total, 1.0):
        raise ValueError(
            "train_fraction + val_fraction + test_fraction "
            f"must equal 1.0, got {total}"
        )

    n_val = int(round(n_rows * val_fraction))
    n_test = int(round(n_rows * test_fraction))
    n_train = n_rows - n_val - n_test

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Dataset too small for requested split: "
            f"train={n_train}, val={n_val}, test={n_test}"
        )

    return n_train, n_val, n_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input dataset CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for manifests and inventory.",
    )

    parser.add_argument(
        "--first-seed",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of consecutive random seeds.",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--write-split-csvs",
        action="store_true",
        help="Also write train.csv, val.csv and test.csv for every seed.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Read input
    # ------------------------------------------------------------------

    source = args.input_csv.resolve()
    output = args.output_dir.resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}")

    output.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(
        source,
        dtype={"ID": str},
    )

    if "ID" not in data.columns:
        raise ValueError("Input CSV must contain column 'ID'.")

    if data["ID"].isna().any():
        raise ValueError("Input IDs contain missing values.")

    if data["ID"].duplicated().any():
        duplicated = data.loc[data["ID"].duplicated(), "ID"].tolist()

        raise ValueError(
            "Input IDs must be unique. "
            f"Duplicated IDs include: {duplicated[:10]}"
        )

    n_rows = len(data)

    n_train, n_val, n_test = calculate_split_sizes(
        n_rows=n_rows,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    print("=== Input dataset ===")
    print(f"Path:  {source}")
    print(f"Rows:  {n_rows}")
    print()
    print("=== Requested split ===")
    print(f"Train: {n_train}")
    print(f"Val:   {n_val}")
    print(f"Test:  {n_test}")

    # ------------------------------------------------------------------
    # Generate manifests
    # ------------------------------------------------------------------

    inventory = []

    for seed in range(
        args.first_seed,
        args.first_seed + args.seeds,
    ):
        # NumPy Generator gives deterministic permutation for each seed.
        rng = np.random.default_rng(seed)

        shuffled_indices = rng.permutation(n_rows)

        train_index = shuffled_indices[:n_train]

        val_start = n_train
        val_end = n_train + n_val

        val_index = shuffled_indices[val_start:val_end]
        test_index = shuffled_indices[val_end:]

        # --------------------------------------------------------------
        # Construct split assignment
        # --------------------------------------------------------------

        split_by_index = pd.Series(
            index=data.index,
            dtype="object",
        )

        split_by_index.iloc[train_index] = "train"
        split_by_index.iloc[val_index] = "val"
        split_by_index.iloc[test_index] = "test"

        if split_by_index.isna().any():
            raise RuntimeError(
                f"Seed {seed}: not every row was assigned."
            )

        # --------------------------------------------------------------
        # Basic integrity checks
        # --------------------------------------------------------------

        train_set = set(train_index.tolist())
        val_set = set(val_index.tolist())
        test_set = set(test_index.tolist())

        if train_set & val_set:
            raise RuntimeError(
                f"Seed {seed}: train/val row leakage."
            )

        if train_set & test_set:
            raise RuntimeError(
                f"Seed {seed}: train/test row leakage."
            )

        if val_set & test_set:
            raise RuntimeError(
                f"Seed {seed}: val/test row leakage."
            )

        all_rows = train_set | val_set | test_set

        if len(all_rows) != n_rows:
            raise RuntimeError(
                f"Seed {seed}: split does not cover all rows."
            )

        # --------------------------------------------------------------
        # Manifest
        # --------------------------------------------------------------

        manifest = pd.DataFrame(
            {
                "sample_id": data["ID"],
                "split": split_by_index,
                "original_row_index": data.index,
            }
        )

        # split_order follows the original CSV ordering within each split.
        # This is consistent with the current fifth-group manifest format.
        manifest["split_order"] = (
            manifest
            .groupby("split", sort=False)
            .cumcount()
        )

        manifest_path = (
            output
            / f"random_manifest_seed{seed}.csv"
        )

        manifest.to_csv(
            manifest_path,
            index=False,
        )

        # --------------------------------------------------------------
        # Verify saved manifest
        # --------------------------------------------------------------

        saved_manifest = pd.read_csv(
            manifest_path,
            dtype={"sample_id": str},
        )

        if len(saved_manifest) != n_rows:
            raise RuntimeError(
                f"Seed {seed}: saved manifest row count mismatch."
            )

        if saved_manifest["sample_id"].duplicated().any():
            raise RuntimeError(
                f"Seed {seed}: duplicate sample IDs in manifest."
            )

        counts = saved_manifest["split"].value_counts()

        if int(counts.get("train", 0)) != n_train:
            raise RuntimeError(
                f"Seed {seed}: incorrect train count."
            )

        if int(counts.get("val", 0)) != n_val:
            raise RuntimeError(
                f"Seed {seed}: incorrect val count."
            )

        if int(counts.get("test", 0)) != n_test:
            raise RuntimeError(
                f"Seed {seed}: incorrect test count."
            )

        # --------------------------------------------------------------
        # Optional physical CSVs
        # --------------------------------------------------------------

        if args.write_split_csvs:
            seed_dir = output / f"seed{seed}"
            seed_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Use manifest ordering rather than shuffled ordering,
            # matching original CSV row order inside each split.
            for split_name in ("train", "val", "test"):
                mask = split_by_index.eq(split_name)

                subset = data.loc[mask].copy()

                subset.to_csv(
                    seed_dir / f"{split_name}.csv",
                    index=False,
                )

        # --------------------------------------------------------------
        # Optional Fifth identity statistics
        # These are statistics only; they do NOT affect splitting.
        # --------------------------------------------------------------

        fifth_stats = {}

        if "Fifth_SMILE" in data.columns:
            for split_name in ("train", "val", "test"):
                mask = split_by_index.eq(split_name)

                fifth_stats[
                    f"{split_name}_raw_fifth_values"
                ] = int(
                    data.loc[
                        mask,
                        "Fifth_SMILE",
                    ].nunique(dropna=False)
                )

        inventory_row = {
            "seed": seed,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "rows": n_rows,
            "train_rows": n_train,
            "val_rows": n_val,
            "test_rows": n_test,
        }

        inventory_row.update(fifth_stats)

        inventory.append(inventory_row)

        print(
            f"seed={seed}: "
            f"train={n_train}, "
            f"val={n_val}, "
            f"test={n_test}"
        )

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    inventory_path = (
        output / "random_manifest_inventory.csv"
    )

    pd.DataFrame(inventory).to_csv(
        inventory_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    protocol = {
        "source": str(source),
        "source_sha256": sha256(source),
        "rows": n_rows,
        "split_type": "row-level random shuffle",
        "train_fraction": args.train_fraction,
        "validation_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "train_rows": n_train,
        "validation_rows": n_val,
        "test_rows": n_test,
        "first_seed": args.first_seed,
        "seeds": args.seeds,
        "random_generator": "numpy.random.default_rng(seed).permutation",
        "group_constraint": None,
        "fifth_component_disjoint": False,
        "target_columns_read_for_splitting": [],
        "external_feedback_read": False,
        "threshold_or_side_criterion_used": False,
        "invariant": (
            "Every input row occurs exactly once in train, validation, "
            "or test for each seed."
        ),
    }

    protocol_path = (
        output / "random_split_protocol.json"
    )

    protocol_path.write_text(
        json.dumps(
            protocol,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=== Finished ===")
    print(f"Inventory: {inventory_path}")
    print(f"Protocol:  {protocol_path}")


if __name__ == "__main__":
    main()