#!/usr/bin/env python3
"""Strict paired O12 vs O13-A comparison for random and Fifth-OOD splits.

O13-A is defined solely as O12 with final fifth-only fusion.  This script
does no training or model selection; it compares corresponding seed metrics
and derives prediction-dispersion ratios from saved per-sample predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ('mae', 'r2', 'pearson', 'spearman')
FOCUS_TARGETS = ('EE_before', 'EE_after', 'mRNA_Recovery_Efficiency')


def per_seed(metrics_path: Path, predictions_path: Path, label: str) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    required_metrics = {'target_group', 'split_seed', 'split', 'target', *METRICS}
    required_predictions = {
        'target_group', 'split_seed', 'split', 'target', 'y_true', 'y_pred',
    }
    if missing := required_metrics.difference(metrics.columns):
        raise ValueError(f'{label} metrics miss columns: {sorted(missing)}')
    if missing := required_predictions.difference(predictions.columns):
        raise ValueError(f'{label} predictions miss columns: {sorted(missing)}')
    expected_keys = ['target_group', 'split_seed', 'split', 'target']
    ratio_rows = []
    for keys, frame in predictions.groupby(expected_keys, sort=False):
        ratio_rows.append({
            **dict(zip(expected_keys, keys)),
            'prediction_std': float(frame.y_pred.std(ddof=0)),
            'true_std': float(frame.y_true.std(ddof=0)),
        })
    ratio = pd.DataFrame(ratio_rows)
    ratio['prediction_std_to_true_std'] = ratio.prediction_std / ratio.true_std
    if (ratio.true_std <= 0).any() or not np.isfinite(ratio.prediction_std_to_true_std).all():
        raise ValueError(f'{label} contains a zero true standard deviation or non-finite ratio.')
    keys = ['target_group', 'split_seed', 'split', 'target']
    metric = metrics[keys + list(METRICS)].copy()
    if metric.duplicated(keys).any() or ratio.duplicated(keys).any():
        raise ValueError(f'{label} has duplicate per-seed target rows.')
    output = metric.merge(ratio[keys + ['prediction_std_to_true_std']], on=keys,
                          how='inner', validate='one_to_one')
    if len(output) != len(metric):
        raise RuntimeError(f'{label} metric/prediction rows do not align one-to-one.')
    return output


def check_pairs(o12: pd.DataFrame, o13: pd.DataFrame, protocol: str) -> pd.DataFrame:
    keys = ['target_group', 'split_seed', 'split', 'target']
    left, right = set(map(tuple, o12[keys].to_numpy())), set(map(tuple, o13[keys].to_numpy()))
    if left != right:
        raise RuntimeError(
            f'{protocol}: O12/O13-A seed-target keys differ; '
            f'missing_in_o13={len(left - right)}, missing_in_o12={len(right - left)}')
    return o12.merge(o13, on=keys, suffixes=('_o12', '_o13a'), validate='one_to_one')


def summarize(paired: pd.DataFrame, protocol: str) -> pd.DataFrame:
    rows = []
    for (split, target), frame in paired.groupby(['split', 'target'], sort=True):
        row = {'protocol': protocol, 'split': split, 'target': target, 'seeds': len(frame)}
        for metric in (*METRICS, 'prediction_std_to_true_std'):
            left = frame[f'{metric}_o12']
            right = frame[f'{metric}_o13a']
            delta = right - left
            row.update({
                f'o12_{metric}_mean': float(left.mean()),
                f'o12_{metric}_std': float(left.std(ddof=1)),
                f'o13a_{metric}_mean': float(right.mean()),
                f'o13a_{metric}_std': float(right.std(ddof=1)),
                f'paired_delta_o13a_minus_o12_{metric}_mean': float(delta.mean()),
                f'paired_delta_o13a_minus_o12_{metric}_std': float(delta.std(ddof=1)),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def conclusion(summary: pd.DataFrame) -> list[str]:
    test_ood = summary.query("protocol == 'fifth_identity_ood' and split == 'test'").set_index('target')
    test_random = summary.query("protocol == 'random' and split == 'test'").set_index('target')
    named = test_ood.loc[list(FOCUS_TARGETS)]
    ood_r2_improved = named['paired_delta_o13a_minus_o12_r2_mean'].gt(0).sum()
    ood_rank_improved = named['paired_delta_o13a_minus_o12_spearman_mean'].gt(0).sum()
    ood_compression_reduced = named[
        'paired_delta_o13a_minus_o12_prediction_std_to_true_std_mean'].gt(0).sum()
    random_mae_worse = test_random.loc[list(FOCUS_TARGETS),
        'paired_delta_o13a_minus_o12_mae_mean'].gt(0).sum()
    if ood_r2_improved >= 2 and ood_rank_improved >= 2:
        first = 'fifth-only fusion improves Fifth-OOD generalization on the majority of pre-specified focus targets.'
        second = ('The result is consistent with a background-shortcut contribution only if this OOD gain is accompanied '
                  'by random-split degradation; see the paired random table.')
    else:
        first = 'fifth-only fusion does not improve Fifth-OOD generalization on the majority of pre-specified focus targets.'
        second = ('This outcome is more consistent with insufficient Fifth representation than with a removable '
                  'background-formulation shortcut.')
    third = (f'Focus-target counts: OOD R² improved {ood_r2_improved}/3; OOD Spearman improved '
             f'{ood_rank_improved}/3; prediction-dispersion ratio increased {ood_compression_reduced}/3; '
             f'random-split MAE worsened {random_mae_worse}/3.')
    return [first, second, third]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for model in ('o12', 'o13a'):
        for protocol in ('random', 'ood'):
            parser.add_argument(f'--{model}-{protocol}-metrics', type=Path, required=True)
            parser.add_argument(f'--{model}-{protocol}-predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    pairs = {}
    for protocol, argument_protocol in (('random', 'random'), ('fifth_identity_ood', 'ood')):
        o12 = per_seed(getattr(args, f'o12_{argument_protocol}_metrics'),
                       getattr(args, f'o12_{argument_protocol}_predictions'), f'O12 {protocol}')
        o13 = per_seed(getattr(args, f'o13a_{argument_protocol}_metrics'),
                       getattr(args, f'o13a_{argument_protocol}_predictions'), f'O13-A {protocol}')
        pairs[protocol] = check_pairs(o12, o13, protocol)
    paired = pd.concat([
        frame.assign(protocol=protocol) for protocol, frame in pairs.items()
    ], ignore_index=True)
    summary = pd.concat([summarize(frame, protocol) for protocol, frame in pairs.items()],
                        ignore_index=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / 'paired_per_seed_metrics_and_dispersion.csv', index=False)
    summary.to_csv(output / 'o12_vs_o13a_paired_summary.csv', index=False)

    lines = [
        '# O13-A fifth-only fusion paired diagnostic', '',
        'All values use the same seed and split membership within each protocol. '
        'Paired deltas are O13-A minus O12. For MAE, negative is better; for R², Pearson, '
        'Spearman, and prediction std/true std, positive is better.', '',
    ]
    for protocol in ('random', 'fifth_identity_ood'):
        lines.extend([
            f'## {protocol}: test split', '',
            '| target | O12 MAE | O13-A MAE | Δ MAE | O12 R² | O13-A R² | Δ R² | O12 Spearman | O13-A Spearman | Δ Spearman | O12 σpred/σtrue | O13-A σpred/σtrue | Δ ratio |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        ])
        table = summary.query('protocol == @protocol and split == "test"')
        for row in table.itertuples(index=False):
            lines.append(
                f'| {row.target} | {row.o12_mae_mean:.3f} ± {row.o12_mae_std:.3f} '
                f'| {row.o13a_mae_mean:.3f} ± {row.o13a_mae_std:.3f} '
                f'| {row.paired_delta_o13a_minus_o12_mae_mean:+.3f} '
                f'| {row.o12_r2_mean:.3f} | {row.o13a_r2_mean:.3f} '
                f'| {row.paired_delta_o13a_minus_o12_r2_mean:+.3f} '
                f'| {row.o12_spearman_mean:.3f} | {row.o13a_spearman_mean:.3f} '
                f'| {row.paired_delta_o13a_minus_o12_spearman_mean:+.3f} '
                f'| {row.o12_prediction_std_to_true_std_mean:.3f} '
                f'| {row.o13a_prediction_std_to_true_std_mean:.3f} '
                f'| {row.paired_delta_o13a_minus_o12_prediction_std_to_true_std_mean:+.3f} |')
        lines.append('')
    lines.extend(['## Pre-specified conclusion', '', *[f'- {item}' for item in conclusion(summary)], ''])
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary.query('split == "test"').to_string(index=False))


if __name__ == '__main__':
    main()
