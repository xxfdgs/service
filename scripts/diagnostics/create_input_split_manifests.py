#!/usr/bin/env python3
"""Create reproducible 80/10/10 input-only train/validation/test manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build_manifest(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Make an exact 80/10/10 split, retaining the loader's source-row order."""
    if "ID" not in frame:
        raise ValueError("Input CSV must contain the unique ID column.")
    ids = frame["ID"].astype(str)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Input CSV requires a non-null, unique ID per row.")

    indices = np.arange(len(frame), dtype=int)
    train_validation, test_indices = train_test_split(
        indices, test_size=0.10, random_state=seed, shuffle=True,
    )
    train_indices, validation_indices = train_test_split(
        train_validation, test_size=len(test_indices) / len(train_validation),
        random_state=seed, shuffle=True,
    )

    parts = []
    for split, split_indices in (("train", train_indices), ("val", validation_indices),
                                 ("test", test_indices)):
        parts.append(pd.DataFrame({
            "sample_id": ids.iloc[split_indices].to_numpy(),
            "split": split,
            "original_row_index": split_indices,
            "split_order": np.arange(len(split_indices), dtype=int),
        }))
    manifest = pd.concat(parts, ignore_index=True)
    counts = manifest.groupby("split").size().to_dict()
    expected = {"train": int(.8 * len(frame)), "val": int(.1 * len(frame)),
                "test": int(.1 * len(frame))}
    if len(manifest) != len(frame) or manifest.sample_id.duplicated().any() or counts != expected:
        raise RuntimeError(f"Invalid split manifest counts={counts}, expected={expected}.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    arguments = parser.parse_args()

    input_csv = arguments.input_csv.resolve()
    frame = pd.read_csv(input_csv)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in arguments.seeds:
        output = arguments.output_dir / f"split_manifest_seed{seed}.csv"
        build_manifest(frame, seed).to_csv(output, index=False)
        print(output)


if __name__ == "__main__":
    main()
