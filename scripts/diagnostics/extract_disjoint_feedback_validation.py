#!/usr/bin/env python3
"""Export ID-disjoint portions of two labelled feedback validation tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={'ID': str})
    if 'ID' not in frame or frame.ID.isna().any() or frame.ID.duplicated().any():
        raise ValueError(f'{path} requires unique non-null ID values.')
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    first, second = load(args.first), load(args.second)
    first_ids, second_ids = set(first.ID), set(second.ID)
    first_only = first.loc[~first.ID.isin(second_ids)].copy()
    second_only = second.loc[~second.ID.isin(first_ids)].copy()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    first_only.to_csv(output / f'{args.first.stem}_only.csv', index=False)
    second_only.to_csv(output / f'{args.second.stem}_only.csv', index=False)
    audit = {
        'first': str(args.first.resolve()), 'second': str(args.second.resolve()),
        'first_rows': len(first), 'second_rows': len(second),
        'intersection_rows': len(first_ids & second_ids),
        'first_only_rows': len(first_only), 'second_only_rows': len(second_only),
        'second_is_subset_of_first': second_ids <= first_ids,
        'first_is_subset_of_second': first_ids <= second_ids,
    }
    (output / 'disjoint_audit.json').write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
