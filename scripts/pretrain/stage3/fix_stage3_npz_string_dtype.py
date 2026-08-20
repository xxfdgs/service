#!/usr/bin/env python3
"""
Fix Stage-3 NPZ string metadata written as dtype=object.

Why this is needed
------------------
pandas Series.astype(str).to_numpy() may still produce a NumPy object array.
np.savez_compressed then stores that object array using pickle semantics, while
Stage 4 intentionally reads with allow_pickle=False.

This migration reads the trusted locally-generated Stage-3 files once with
allow_pickle=True, converts only string metadata to fixed-width Unicode arrays,
and atomically rewrites the NPZ files. Numeric targets are preserved unchanged.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def unicode_array(values) -> np.ndarray:
    return np.asarray([str(x) for x in values.tolist()], dtype=np.str_)


def rewrite_npz(path: Path, string_keys: set[str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as src:
        payload = {}
        for key in src.files:
            arr = src[key]
            if key in string_keys:
                arr = unicode_array(arr)
            payload[key] = arr

    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **payload)

    # Verify the rewritten file can be read under the Stage-4 policy.
    with np.load(tmp, allow_pickle=False) as check:
        for key in check.files:
            _ = check[key]

    os.replace(tmp, path)

    with np.load(path, allow_pickle=False) as check:
        dtypes = {key: str(check[key].dtype) for key in check.files}

    print(f"Fixed: {path}")
    print("  dtypes:", dtypes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.stage3_dir.resolve()

    rewrite_npz(
        root / "descriptor_targets_scaled.npz",
        {"stage2c_id", "descriptor_names", "split"},
    )
    rewrite_npz(
        root / "morgan_fp_1024.npz",
        {"stage2c_id", "split"},
    )

    print()
    print("Stage-3 NPZ metadata migration PASSED.")
    print("Both files are now readable with allow_pickle=False.")


if __name__ == "__main__":
    main()
