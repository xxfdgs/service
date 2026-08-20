#!/usr/bin/env python3
"""Score a locked O12 ensemble on labelled external data without model access.

This is deliberately a post-inference diagnostic.  It consumes only the
already-generated arithmetic-mean ensemble predictions, verifies their
provenance, and joins labels after prediction.  It never trains, selects a
checkpoint, fits a scaler, or modifies a model/cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


TARGETS = [
    'EE_before', 'EE_after', 'Aerosolization_Efficiency',
    'mRNA_Recovery_Efficiency', 'Norm_before', 'Norm_after',
]
GROUPS = ('single', 'double')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def correlation(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.std(truth) == 0 or np.std(prediction) == 0:
        return float('nan')
    return float(function(truth, prediction).statistic)


def metrics(frame: pd.DataFrame, scope: str, group: str | None) -> dict:
    truth = frame.y_true.to_numpy(dtype=float)
    prediction = frame.y_pred.to_numpy(dtype=float)
    return {
        'scope': scope,
        'fifth_class': group if group is not None else 'all',
        'n': int(len(frame)),
        'mae': float(mean_absolute_error(truth, prediction)),
        'r2': float(r2_score(truth, prediction)) if len(frame) >= 2 else float('nan'),
        'pearson': correlation(pearsonr, truth, prediction),
        'spearman': correlation(spearmanr, truth, prediction),
        'true_mean': float(truth.mean()),
        'predicted_mean': float(prediction.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--labels-csv', type=Path, required=True)
    parser.add_argument('--prediction-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    baseline_root = args.baseline_root.resolve()
    labels_path = args.labels_csv.resolve()
    prediction_dir = args.prediction_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not labels_path.is_file():
        raise FileNotFoundError(f'Labels CSV does not exist: {labels_path}')
    if not baseline_root.is_dir():
        raise FileNotFoundError(f'Baseline root does not exist: {baseline_root}')

    labels = pd.read_csv(labels_path, dtype={'ID': str})
    required = {'ID', 'Fifth', 'Fifth_class', *TARGETS}
    if missing := required.difference(labels.columns):
        raise ValueError(f'Labels CSV misses columns: {sorted(missing)}')
    if labels.ID.isna().any() or labels.ID.duplicated().any():
        raise ValueError('External labels must have unique, non-null IDs.')
    if invalid := set(labels.Fifth_class.dropna().astype(str)).difference(GROUPS):
        raise ValueError(f'Fifth_class must be only {GROUPS}; found {sorted(invalid)}')

    prediction_tables = []
    for target_group, group_targets in {
        'core4': TARGETS[:4], 'norm2': TARGETS[4:],
    }.items():
        provenance_path = prediction_dir / f'provenance_{target_group}.json'
        prediction_path = prediction_dir / f'ensemble_mean_predictions_{target_group}.csv'
        if not provenance_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(
                f'Missing locked {target_group} prediction or provenance in {prediction_dir}')
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        if provenance.get('labels_used_for_model_input') is not False:
            raise RuntimeError(f'{target_group} predictions did not record label-free model input.')
        if provenance.get('source_sha256') != sha256(labels_path):
            raise RuntimeError(f'{target_group} predictions were not made from this exact labels CSV.')
        checkpoints = provenance.get('checkpoints', [])
        if len(checkpoints) != 10:
            raise RuntimeError(f'{target_group} provenance does not contain exactly ten checkpoints.')
        for item in checkpoints:
            checkpoint = Path(item['path'])
            try:
                checkpoint.relative_to(baseline_root)
            except ValueError as error:
                raise RuntimeError(
                    f'{target_group} checkpoint is outside locked baseline root: {checkpoint}') from error
            if not checkpoint.is_file() or sha256(checkpoint) != item.get('sha256'):
                raise RuntimeError(f'{target_group} checkpoint failed frozen-provenance verification: {checkpoint}')
        predictions = pd.read_csv(prediction_path, dtype={'ID': str})
        columns = ['ID', *[f'pred_{target}_mean' for target in group_targets]]
        if missing := set(columns).difference(predictions.columns):
            raise ValueError(f'{target_group} prediction columns missing: {sorted(missing)}')
        prediction_tables.append(predictions[columns])

    wide = labels[['ID', 'Fifth', 'Fifth_class', *TARGETS]].copy()
    for table in prediction_tables:
        if table.ID.duplicated().any() or set(table.ID) != set(wide.ID):
            raise ValueError('Prediction IDs do not match external-label IDs exactly.')
        wide = wide.merge(table, on='ID', how='left', validate='one_to_one')

    long_rows = []
    for sample_index, row in wide.reset_index(drop=True).iterrows():
        for target in TARGETS:
            true = float(row[target])
            prediction = float(row[f'pred_{target}_mean'])
            long_rows.append({
                'sample_index': sample_index,
                'ID': row.ID,
                'Fifth': row.Fifth,
                'Fifth_class': row.Fifth_class,
                'target': target,
                'y_true': true,
                'y_pred': prediction,
                'absolute_error': abs(prediction - true),
            })
    long = pd.DataFrame(long_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(output_dir / 'external_predictions_long.csv', index=False)

    result_rows = []
    for target, target_frame in long.groupby('target', sort=False):
        result_rows.append({'target': target, **metrics(target_frame, 'overall', None)})
        for group in GROUPS:
            result_rows.append({
                'target': target,
                **metrics(target_frame.loc[target_frame.Fifth_class.eq(group)], 'by_fifth_class', group),
            })
    result = pd.DataFrame(result_rows)
    result.to_csv(output_dir / 'metrics_overall_and_by_fifth_class.csv', index=False)
    result.loc[result.scope.eq('overall')].to_csv(output_dir / 'metrics_overall.csv', index=False)
    result.loc[result.scope.eq('by_fifth_class')].to_csv(
        output_dir / 'metrics_by_fifth_class.csv', index=False)

    means = (result.loc[result.scope.eq('by_fifth_class'),
                        ['target', 'fifth_class', 'n', 'true_mean', 'predicted_mean']]
             .sort_values(['target', 'fifth_class']))
    means.to_csv(output_dir / 'fifth_class_true_predicted_means.csv', index=False)

    overall = result.loc[result.scope.eq('overall')].set_index('target')
    by_group = result.loc[result.scope.eq('by_fifth_class')].pivot(
        index='target', columns='fifth_class', values=['true_mean', 'predicted_mean'])
    summary_rows = []
    for target in TARGETS:
        truth_delta = by_group.loc[target, ('true_mean', 'double')] - by_group.loc[target, ('true_mean', 'single')]
        prediction_delta = (by_group.loc[target, ('predicted_mean', 'double')]
                            - by_group.loc[target, ('predicted_mean', 'single')])
        direction_matches = bool(np.sign(truth_delta) == np.sign(prediction_delta)) if truth_delta and prediction_delta else None
        summary_rows.append({
            'target': target,
            'overall_spearman': overall.loc[target, 'spearman'],
            'single_spearman': result.query("target == @target and fifth_class == 'single'").iloc[0].spearman,
            'double_spearman': result.query("target == @target and fifth_class == 'double'").iloc[0].spearman,
            'double_minus_single_true_mean': truth_delta,
            'double_minus_single_predicted_mean': prediction_delta,
            'single_double_direction_matches': direction_matches,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / 'ranking_and_group_separation_summary.csv', index=False)

    matched = int(summary.single_double_direction_matches.eq(True).sum())
    lines = [
        '# Locked O12 external diagnostic summary', '',
        f'- Baseline root: `{baseline_root}`',
        f'- External data: `{labels_path}` ({len(labels)} samples; SHA-256 `{sha256(labels_path)}`)',
        '- Inference provenance verifies ten frozen selected-best checkpoints per target group; labels were excluded from model input.',
        f'- Single/double mean-shift direction matches the observed direction for {matched}/{len(TARGETS)} targets.',
        '- Ranking validity is reported as Spearman: overall uses all 26 samples; single and double use only their own group samples.', '',
        '| target | overall Spearman | single Spearman | double Spearman | group-shift direction match |',
        '|---|---:|---:|---:|:---:|',
    ]
    for row in summary.itertuples(index=False):
        match = 'yes' if row.single_double_direction_matches is True else 'no'
        lines.append(
            f'| {row.target} | {row.overall_spearman:.3f} | {row.single_spearman:.3f} '
            f'| {row.double_spearman:.3f} | {match} |')
    (output_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(result.to_string(index=False))
    print('\n' + summary.to_string(index=False))


if __name__ == '__main__':
    main()
