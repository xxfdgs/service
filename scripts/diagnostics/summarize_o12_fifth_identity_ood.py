#!/usr/bin/env python3
"""Compare locked O12 random-split and Fifth-identity OOD diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ('mae', 'r2', 'pearson', 'spearman')


def aggregate(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {'split_seed', 'split', 'target', *METRICS}
    if missing := required.difference(frame.columns):
        raise ValueError(f'{label} metrics miss columns: {sorted(missing)}')
    grouped = frame.groupby(['split', 'target'], as_index=False)[list(METRICS)].agg(['mean', 'std'])
    grouped.columns = ['split', 'target', *[
        f'{metric}_{stat}' for metric in METRICS for stat in ('mean', 'std')]]
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-metrics', type=Path, required=True)
    parser.add_argument('--ood-metrics', type=Path, required=True)
    parser.add_argument('--ood-predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    random = pd.read_csv(args.random_metrics)
    ood = pd.read_csv(args.ood_metrics)
    summary = aggregate(random, 'Random').merge(
        aggregate(ood, 'Fifth-OOD'), on=['split', 'target'], suffixes=('_random', '_fifth_ood'),
        validate='one_to_one')
    for metric in METRICS:
        summary[f'{metric}_mean_delta_ood_minus_random'] = (
            summary[f'{metric}_mean_fifth_ood'] - summary[f'{metric}_mean_random'])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / 'random_vs_fifth_identity_ood_metrics.csv', index=False)

    prediction = pd.read_csv(args.ood_predictions)
    required_prediction = {'split_seed', 'split', 'target', 'y_true', 'y_pred'}
    if missing := required_prediction.difference(prediction.columns):
        raise ValueError(f'OOD predictions miss columns: {sorted(missing)}')
    rows = []
    for keys, frame in prediction.groupby(['split_seed', 'split', 'target'], sort=True):
        truth_std = float(frame.y_true.std(ddof=0))
        prediction_std = float(frame.y_pred.std(ddof=0))
        rows.append({
            'split_seed': keys[0], 'split': keys[1], 'target': keys[2], 'n': len(frame),
            'true_mean': float(frame.y_true.mean()), 'prediction_mean': float(frame.y_pred.mean()),
            'true_std': truth_std, 'prediction_std': prediction_std,
            'prediction_to_true_std_ratio': prediction_std / truth_std if truth_std else np.nan,
            'mean_bias': float((frame.y_pred - frame.y_true).mean()),
            'collapse_flag_std_ratio_below_0_10': bool(truth_std and prediction_std / truth_std < .10),
        })
    collapse_per_seed = pd.DataFrame(rows)
    collapse_per_seed.to_csv(output / 'fifth_identity_ood_prediction_dispersion_by_seed.csv', index=False)
    collapse = collapse_per_seed.groupby(['split', 'target'], as_index=False).agg(
        seeds=('split_seed', 'nunique'),
        mean_prediction_to_true_std_ratio=('prediction_to_true_std_ratio', 'mean'),
        std_prediction_to_true_std_ratio=('prediction_to_true_std_ratio', 'std'),
        mean_bias=('mean_bias', 'mean'), std_bias=('mean_bias', 'std'),
        collapse_seed_count=('collapse_flag_std_ratio_below_0_10', 'sum'),
    )
    collapse.to_csv(output / 'fifth_identity_ood_prediction_dispersion_summary.csv', index=False)

    test = summary.loc[summary.split.eq('test')].copy()
    lines = [
        '# O12 Fifth-identity OOD benchmark summary', '',
        'Values are mean ± sample standard deviation across seeds 100–109.',
        'Metric deltas are Fifth-OOD minus random-split; positive MAE delta and negative R²/Spearman delta indicate OOD degradation.', '',
        '| target | random test MAE | OOD test MAE | Δ MAE | random test R² | OOD test R² | Δ R² | random Spearman | OOD Spearman | Δ Spearman |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in test.itertuples(index=False):
        lines.append(
            f'| {row.target} | {row.mae_mean_random:.3f} ± {row.mae_std_random:.3f} '
            f'| {row.mae_mean_fifth_ood:.3f} ± {row.mae_std_fifth_ood:.3f} '
            f'| {row.mae_mean_delta_ood_minus_random:+.3f} '
            f'| {row.r2_mean_random:.3f} | {row.r2_mean_fifth_ood:.3f} '
            f'| {row.r2_mean_delta_ood_minus_random:+.3f} '
            f'| {row.spearman_mean_random:.3f} | {row.spearman_mean_fifth_ood:.3f} '
            f'| {row.spearman_mean_delta_ood_minus_random:+.3f} |')
    (output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(test.to_string(index=False))
    print('\nPrediction dispersion (test):')
    print(collapse.loc[collapse.split.eq('test')].to_string(index=False))


if __name__ == '__main__':
    main()
