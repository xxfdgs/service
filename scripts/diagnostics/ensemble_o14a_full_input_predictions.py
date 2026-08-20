#!/usr/bin/env python3
"""Ensemble selected O14-A Full checkpoints over the original 700-row input.

Every selected checkpoint already writes one prediction for every source row.
This utility joins those exports by ``sample_id`` and never reruns inference.
The saved split counts make clear that this is an in-sample-inclusive diagnostic
ensemble, not an outer-test estimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGETS = ('Norm_before', 'Norm_after')


def run_path(root: Path, target: str, seed: int) -> Path:
    slug = target.lower()
    title = 'NormBefore' if target == 'Norm_before' else 'NormAfter'
    return root / 'A0' / 'full' / slug / f'O14A0Full_FifthOOD_{title}_seed{seed}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(100, 110)))
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / 'input_700_ensemble').resolve()
    output.mkdir(parents=True, exist_ok=True)

    wide = None
    long_parts = []
    for target in TARGETS:
        seed_frames = []
        for split_seed in args.seeds:
            path = run_path(root, target, split_seed) / 'threshold_predictions.csv'
            if not path.is_file():
                raise FileNotFoundError(f'Missing selected-checkpoint predictions: {path}')
            frame = pd.read_csv(path, dtype={'sample_id': str})
            needed = {'sample_id', 'Fifth', 'fifth_class', 'split', 'true_norm', 'pred_norm'}
            if missing := needed.difference(frame.columns):
                raise ValueError(f'{path} misses {sorted(missing)}')
            if len(frame) != 700 or frame.sample_id.duplicated().any():
                raise ValueError(f'{path} does not contain exactly one prediction per 700-row input sample.')
            part = frame[['sample_id', 'Fifth', 'fifth_class', 'split', 'true_norm', 'pred_norm']].copy()
            part['split_seed'] = int(split_seed)
            part['target'] = target
            long_parts.append(part)
            seed_frames.append(part)
        combined = pd.concat(seed_frames, ignore_index=True)
        counts = combined.groupby('sample_id', sort=False).size()
        if not counts.eq(len(args.seeds)).all():
            raise RuntimeError(f'{target} does not have exactly one value from every requested seed.')
        metadata = combined.groupby('sample_id', sort=False).agg(
            Fifth=('Fifth', 'first'), fifth_class=('fifth_class', 'first'),
            **{f'true_{target}': ('true_norm', 'first')},
        )
        if combined.groupby('sample_id')['true_norm'].nunique(dropna=False).gt(1).any():
            raise RuntimeError(f'{target} true labels differ across selected checkpoint exports.')
        prediction = combined.groupby('sample_id')['pred_norm'].agg(['mean', 'std', 'count']).rename(columns={
            'mean': f'pred_{target}_mean', 'std': f'pred_{target}_std_10models',
            'count': f'pred_{target}_n_models',
        })
        # Each model's membership is saved separately because a row is train,
        # val, or test depending on the frozen split seed.
        memberships = combined.pivot(index='sample_id', columns='split_seed', values='split')
        memberships.columns = [f'{target}_split_seed{seed}' for seed in memberships.columns]
        target_wide = metadata.join(prediction).join(memberships).reset_index()
        if wide is None:
            wide = target_wide
        else:
            wide = wide.merge(target_wide, on=['sample_id', 'Fifth', 'fifth_class'], how='inner', validate='one_to_one')
    if len(wide) != 700:
        raise RuntimeError(f'Expected 700 joined rows, found {len(wide)}.')
    wide.to_csv(output / 'o14a0_full_700row_ensemble_predictions.csv', index=False)
    pd.concat(long_parts, ignore_index=True).to_csv(
        output / 'o14a0_full_700row_predictions_by_checkpoint.csv', index=False)
    summary = pd.DataFrame({
        'target': list(TARGETS), 'rows': [len(wide)] * len(TARGETS),
        'models_per_row': [len(args.seeds)] * len(TARGETS),
        'mean_model_std': [wide[f'pred_{target}_std_10models'].mean() for target in TARGETS],
        'max_model_std': [wide[f'pred_{target}_std_10models'].max() for target in TARGETS],
    })
    summary.to_csv(output / 'o14a0_full_700row_ensemble_summary.csv', index=False)
    print(f'Wrote {len(wide)} rows × {len(args.seeds)} models: {output}')


if __name__ == '__main__':
    main()
