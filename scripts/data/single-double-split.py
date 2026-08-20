#!/usr/bin/env python3
"""
Split training and new_validation datasets by Fifth_class.

Outputs:
    train_single.csv
    train_double.csv
    validation_single.csv
    validation_double.csv

Rows and columns are preserved exactly from the source CSVs.
Only rows whose Fifth_class is explicitly "single" or "double" are exported.
"""

from pathlib import Path
import argparse

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    """Read CSV while tolerating historical encoding differences."""
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"Unable to decode CSV: {path}")


def split_dataset(
    input_path: Path,
    output_single: Path,
    output_double: Path,
    dataset_name: str,
) -> None:
    df = read_csv(input_path)

    if "Fifth_class" not in df.columns:
        raise KeyError(
            f"{input_path} does not contain required column 'Fifth_class'.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Normalize only for deciding which rows belong to each class.
    # The original Fifth_class column itself is NOT modified.
    classes = (
        df["Fifth_class"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    single_mask = classes.eq("single")
    double_mask = classes.eq("double")
    known_mask = single_mask | double_mask

    single_df = df.loc[single_mask].copy()
    double_df = df.loc[double_mask].copy()
    other_df = df.loc[~known_mask].copy()

    output_single.parent.mkdir(parents=True, exist_ok=True)

    # index=False prevents pandas from adding an extra CSV index column.
    single_df.to_csv(output_single, index=False, encoding="utf-8-sig")
    double_df.to_csv(output_double, index=False, encoding="utf-8-sig")

    print(f"\n=== {dataset_name} ===")
    print(f"Input:   {input_path}")
    print(f"Total:   {len(df)}")
    print(f"single:  {len(single_df)} -> {output_single}")
    print(f"double:  {len(double_df)} -> {output_double}")
    print(f"other:   {len(other_df)}")

    if len(other_df):
        print("\nWARNING: rows with Fifth_class other than single/double:")
        print(other_df["Fifth_class"].value_counts(dropna=False).to_string())


def main():
    root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Split training and new_validation CSVs into single/double datasets."
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=root / "datasets_lrx/raw/input/20260812-sum-700.csv",
        help="Training CSV",
    )

    parser.add_argument(
        "--validation",
        type=Path,
        default=root / "datasets_lrx/raw/feedback/new_validation.csv",
        help="new_validation CSV",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "datasets_lrx/single_double_split",
        help="Output directory",
    )

    args = parser.parse_args()

    train_path = args.train.resolve()
    validation_path = args.validation.resolve()
    output_dir = args.output_dir.resolve()

    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")

    if not validation_path.is_file():
        raise FileNotFoundError(
            f"Validation CSV not found: {validation_path}"
        )

    split_dataset(
        input_path=train_path,
        output_single=output_dir / "train_single.csv",
        output_double=output_dir / "train_double.csv",
        dataset_name="TRAIN",
    )

    split_dataset(
        input_path=validation_path,
        output_single=output_dir / "validation_single.csv",
        output_double=output_dir / "validation_double.csv",
        dataset_name="NEW VALIDATION",
    )

    print("\n=== Finished ===")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()