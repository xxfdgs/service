#!/usr/bin/env python3
"""Audit the fixed input-derived Fifth_class encoding used by O13-B.

The loader builds this vocabulary once from the original input CSV, before a
diagnostic manifest is applied.  This script makes that behaviour explicit and
rejects unrecognised non-missing labels rather than allowing an experiment to
silently map them to ``__unknown__``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ALLOWED = {"single", "double", "__unknown__"}


def canonical(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return "__unknown__"
    return str(value).strip().lower()


def fixed_vocabulary(classes: set[str]) -> dict[str, int]:
    """Mirror build_input_fifth_class_vocab exactly for this source CSV."""
    keys = sorted(classes)
    if "__unknown__" not in keys:
        keys.insert(0, "__unknown__")
    return {name: index for index, name in enumerate(keys)}


def manifest_rows(source: pd.DataFrame, manifest: Path, protocol: str,
                  seed: int) -> list[dict[str, object]]:
    frame = pd.read_csv(manifest, dtype={"sample_id": str})
    required = {"sample_id", "split"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{manifest} misses {sorted(missing)}")
    if frame.sample_id.duplicated().any():
        raise ValueError(f"{manifest} has duplicate sample_id values")
    source_fields = source[["ID", "Fifth", "fifth_class_canonical"]].rename(
        columns={"Fifth": "Fifth_identity"})
    joined = frame.merge(source_fields, left_on="sample_id", right_on="ID",
                         how="left", validate="one_to_one")
    if joined.ID.isna().any():
        raise ValueError(f"{manifest} refers to IDs absent from the locked input CSV")
    if len(joined) != len(source):
        raise ValueError(f"{manifest} does not cover the locked input CSV exactly once")
    rows = []
    for split, group in joined.groupby("split", sort=True):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"{manifest} has unexpected split {split!r}")
        for fifth_class, count in group.fifth_class_canonical.value_counts().sort_index().items():
            rows.append({
                "protocol": protocol,
                "split_seed": seed,
                "split": split,
                "fifth_class": fifth_class,
                "rows": int(count),
                "unique_fifth_identities": int(group.loc[
                    group.fifth_class_canonical.eq(fifth_class), "Fifth_identity"
                ].fillna("[missing]").nunique()),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--random-manifest-root", type=Path, required=True)
    parser.add_argument("--ood-manifest-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.input_csv, dtype={"ID": str})
    required = {"ID", "Fifth", "Fifth_SMILE", "Fifth_class", "mol%_Fifth"}
    if missing := required.difference(source.columns):
        raise ValueError(f"Locked input CSV misses {sorted(missing)}")
    if source.ID.duplicated().any():
        raise ValueError("Locked input CSV has duplicate IDs")
    source = source.copy()
    source["fifth_class_canonical"] = source.Fifth_class.map(canonical)
    classes = set(source.fifth_class_canonical)
    unexpected = classes.difference(ALLOWED)
    if unexpected:
        raise ValueError(
            "Refusing a silent Fifth_class unknown mapping; unexpected normalized "
            f"labels: {sorted(unexpected)}")
    vocabulary = fixed_vocabulary(classes)
    if vocabulary.get("__unknown__") != 0:
        raise RuntimeError("The placeholder class must occupy embedding ID 0")

    ratio = pd.to_numeric(source["mol%_Fifth"], errors="coerce")
    fifth_smiles = source.Fifth_SMILE.fillna("").astype(str).str.strip()
    placeholder_mask = source.fifth_class_canonical.eq("__unknown__")
    missing_smiles_mask = fifth_smiles.eq("")
    fr_mask = fifth_smiles.str.contains(r"\[Fr\]", regex=True, na=False)
    rows = []
    for protocol, root, prefix in (
        ("random", args.random_manifest_root, "split_manifest_seed"),
        ("fifth_identity_ood", args.ood_manifest_root, "fifth_identity_manifest_seed"),
    ):
        for seed in args.seeds:
            manifest = root / f"{prefix}{seed}.csv"
            if not manifest.is_file():
                raise FileNotFoundError(f"Missing frozen manifest: {manifest}")
            rows.extend(manifest_rows(source, manifest, protocol, seed))

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(
        ["protocol", "split_seed", "split", "fifth_class"]
    ).to_csv(output / "fifth_class_by_manifest.csv", index=False)
    source_counts = source.fifth_class_canonical.value_counts().sort_index()
    report = {
        "input_csv": str(args.input_csv.resolve()),
        "encoding": vocabulary,
        "encoding_rule": (
            "canonical label = stripped lowercase value; missing or blank = __unknown__; "
            "the vocabulary is built once from the locked input CSV before each split."
        ),
        "accepted_canonical_labels": sorted(ALLOWED),
        "source_class_row_counts": {key: int(value) for key, value in source_counts.items()},
        "placeholder_audit": {
            "placeholder_rows": int(placeholder_mask.sum()),
            "ratio_zero_or_missing_rows": int(ratio.fillna(0).eq(0).sum()),
            "missing_fifth_smiles_rows": int(missing_smiles_mask.sum()),
            "fr_smiles_rows": int(fr_mask.sum()),
            "placeholder_rows_all_ratio_zero_or_missing": bool(ratio[placeholder_mask].fillna(0).eq(0).all()),
            "placeholder_rows_all_missing_fifth_smiles": bool(missing_smiles_mask[placeholder_mask].all()),
        },
        "leakage_guard": (
            "All train/val/test rows use this same fixed mapping. Any non-missing label "
            "outside single/double causes this audit to fail; no split-specific vocabulary is built."
        ),
    }
    (output / "fifth_class_encoding_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
