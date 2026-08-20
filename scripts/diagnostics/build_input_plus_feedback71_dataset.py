#!/usr/bin/env python3
"""Build a UTF-8 input dataset augmented with disjoint feedback labels.

Original fixed split membership is retained.  The feedback-only rows are
assigned to the training portion of every split seed, so they contribute to
training but never alter original validation/test membership.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_input(path: Path) -> pd.DataFrame:
    for encoding in ('utf-8', 'utf-8-sig', 'gb18030'):
        try:
            return pd.read_csv(path, dtype={'ID': str}, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError('input', b'', 0, 1, f'Cannot decode {path}')


def validate(frame: pd.DataFrame, name: str) -> None:
    if 'ID' not in frame or frame.ID.isna().any() or frame.ID.duplicated().any():
        raise ValueError(f'{name} requires unique, non-null ID values.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--feedback-csv', type=Path, required=True)
    parser.add_argument('--manifests-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    base = read_input(args.input_csv)
    feedback = pd.read_csv(args.feedback_csv, dtype={'ID': str})
    validate(base, 'input CSV')
    validate(feedback, 'feedback CSV')
    if set(base.ID).intersection(feedback.ID):
        raise ValueError('Input and feedback-only CSVs must not have overlapping IDs.')
    missing = set(base.columns).difference(feedback.columns)
    if missing:
        raise ValueError(f'Feedback CSV misses input columns: {sorted(missing)}')
    feedback = feedback.reindex(columns=base.columns)
    merged = pd.concat([base, feedback], ignore_index=True)
    validate(merged, 'merged CSV')

    output = args.output_dir.resolve()
    manifest_output = output / 'five_split_manifests_augmented'
    feedback_only_manifest_output = output / 'five_split_manifests_feedback_only'
    manifest_output.mkdir(parents=True, exist_ok=True)
    feedback_only_manifest_output.mkdir(parents=True, exist_ok=True)
    base.to_csv(output / 'input_20260703_sum_utf8.csv', index=False)
    merged.to_csv(output / 'input_20260703_sum_plus_feedback71.csv', index=False)
    feedback.to_csv(output / 'feedback_only_20260703_validation_nonoverlap.csv', index=False)

    for seed in range(100, 110):
        source = args.manifests_dir / f'split_manifest_seed{seed}.csv'
        manifest = pd.read_csv(source, dtype={'sample_id': str})
        required = {'sample_id', 'split', 'split_order'}
        if not required.issubset(manifest.columns):
            raise ValueError(f'{source} misses required columns: {sorted(required)}')
        if set(manifest.sample_id) != set(base.ID) or len(manifest) != len(base):
            raise ValueError(f'{source} does not exactly cover the original input CSV.')
        train_order = int(manifest.loc[manifest.split.eq('train'), 'split_order'].max()) + 1
        appended = pd.DataFrame({
            'sample_id': feedback.ID.to_numpy(),
            'split': 'train',
            'original_row_index': np.arange(len(base), len(merged), dtype=int),
            'split_order': np.arange(train_order, train_order + len(feedback), dtype=int),
        })
        columns = list(manifest.columns)
        augmented = pd.concat([manifest, appended.reindex(columns=columns)], ignore_index=True)
        if len(augmented) != len(merged) or augmented.sample_id.duplicated().any():
            raise RuntimeError(f'Invalid augmented manifest for seed {seed}.')
        augmented.to_csv(manifest_output / source.name, index=False)

        # A small-data ablation uses no original input rows: 57/7/7 for the
        # 71 feedback-only rows.  Each seed has an independent deterministic
        # partition, while every source row is represented exactly once.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(feedback))
        split_labels = np.full(len(feedback), 'train', dtype=object)
        split_labels[order[57:64]] = 'val'
        split_labels[order[64:]] = 'test'
        feedback_manifest = pd.DataFrame({
            'sample_id': feedback.ID.to_numpy(),
            'split': split_labels,
            'original_row_index': np.arange(len(feedback), dtype=int),
        })
        feedback_manifest['split_order'] = (
            feedback_manifest.groupby('split', sort=False).cumcount())
        if feedback_manifest['split'].value_counts().to_dict() != {'train': 57, 'val': 7, 'test': 7}:
            raise RuntimeError(f'Invalid feedback-only partition for seed {seed}.')
        feedback_manifest.to_csv(feedback_only_manifest_output / source.name, index=False)

    audit = {
        'base_input': str(args.input_csv.resolve()),
        'feedback_only': str(args.feedback_csv.resolve()),
        'base_rows': len(base), 'feedback_rows_added_to_train_per_seed': len(feedback),
        'merged_rows': len(merged), 'split_seeds': list(range(100, 110)),
        'original_val_test_membership_preserved': True,
        'feedback_only_split_sizes': {'train': 57, 'val': 7, 'test': 7},
    }
    (output / 'dataset_audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
