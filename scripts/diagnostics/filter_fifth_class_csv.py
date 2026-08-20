#!/usr/bin/env python3
"""Create a reproducible Fifth_class-filtered copy of a labelled CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--output-csv', type=Path, required=True)
    parser.add_argument('--fifth-class', choices=('single', 'double'), required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.input_csv, dtype={'ID': str})
    if 'Fifth_class' not in source:
        raise ValueError(f'{args.input_csv} has no Fifth_class column.')
    values = source.Fifth_class.fillna('').astype(str).str.strip().str.lower()
    output = source.loc[values.eq(args.fifth_class)].copy()
    if output.empty:
        raise ValueError(f'No {args.fifth_class} rows were found.')
    if output.ID.duplicated().any():
        raise ValueError('Filtered source has duplicate ID values.')
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f'Wrote {len(output)} {args.fifth_class} rows: {args.output_csv}')


if __name__ == '__main__':
    main()
