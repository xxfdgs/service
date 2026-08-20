#!/usr/bin/env python3
"""Aggregate matched-seed O14-A Full/Double Fifth-OOD diagnostics.

Only selected *regression-best* checkpoints are summarized here; for A1-A3,
the separately saved threshold-aware checkpoint remains a diagnostic artifact
until its validation-only selection rule is prospectively frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    'mae', 'rmse', 'r2', 'pearson', 'spearman', 'precision_gt1', 'recall_gt1',
    'specificity_gt1', 'f1_gt1', 'f2_gt1', 'false_negative_rate_gt1',
    'false_positive_rate_gt1', 'tp', 'tn', 'fp', 'fn',
]


def run_rows(root: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in root.glob('*/*/*/*/threshold_metrics_selected_checkpoint.csv'):
        # root / A0 / full / norm_before / run / threshold_metrics...
        run_dir = metrics_path.parent
        parts = metrics_path.relative_to(root).parts
        if len(parts) != 5:
            continue
        ablation, domain, target_slug, run_name, _ = parts
        settings_path = run_dir / 'run_settings.json'
        audit_path = run_dir / 'o14a_domain_audit.json'
        if not settings_path.is_file() or not audit_path.is_file():
            raise FileNotFoundError(f'Incomplete O14-A run: {run_dir}')
        settings = json.loads(settings_path.read_text())
        audit = json.loads(audit_path.read_text())
        if audit.get('status') != 'pass' or not audit.get('identity_leakage_pass'):
            raise RuntimeError(f'O14-A leakage audit did not pass: {run_dir}')
        metrics = pd.read_csv(metrics_path)
        selected = metrics.loc[(metrics.split.eq('test')) & (metrics.subset.eq('double'))
                               & (metrics.decision_source.eq('regression'))]
        if len(selected) != 1:
            raise RuntimeError(f'Expected exactly one double regression test row: {run_dir}')
        row = selected.iloc[0].to_dict()
        row.update({
            'ablation': ablation, 'training_domain': domain,
            'target': settings['single_target'], 'seed': int(run_name.rsplit('seed', 1)[1]),
            'run_dir': str(run_dir), 'manifest_path': settings['split_manifest'],
            'manifest_sha256': audit['manifest_sha256'],
            'input_sha256': audit['input_sha256'],
            'training_domain_audit': audit['training_domain'],
        })
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f'No completed O14-A threshold result found under {root}')
    return pd.DataFrame(rows).sort_values(['ablation', 'training_domain', 'target', 'seed'])


def aggregate(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    result = []
    for keys, frame in seed_metrics.groupby(['ablation', 'training_domain', 'target'], sort=True):
        row = dict(zip(['ablation', 'training_domain', 'target'], keys, strict=True))
        row['n_completed_seeds'] = int(len(frame))
        row['split_protocol'] = 'fifth_identity_ood'
        for metric in SUMMARY_COLUMNS:
            values = pd.to_numeric(frame[metric], errors='coerce')
            row[f'{metric}_mean'] = float(values.mean())
            row[f'{metric}_std'] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        result.append(row)
    return pd.DataFrame(result)


def paired_deltas(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = []
    # The Stage-1 scientific comparison is explicit; later within-domain
    # ablation comparisons are emitted only where both matched seeds exist.
    for target, target_frame in seed_metrics.groupby('target'):
        groups = {(ablation, domain): frame.set_index('seed')
                  for (ablation, domain), frame in target_frame.groupby(['ablation', 'training_domain'])}
        candidate_pairs = [(('A0', 'full'), ('A0', 'double'), 'A0_Double_minus_A0_Full')]
        for domain in ('full', 'double'):
            candidate_pairs.extend([
                (('A0', domain), ('A1', domain), f'A1_minus_A0_{domain}'),
                (('A1', domain), ('A2', domain), f'A2_minus_A1_{domain}'),
                (('A2', domain), ('A3', domain), f'A3_minus_A2_{domain}'),
            ])
        for left_key, right_key, label in candidate_pairs:
            if left_key not in groups or right_key not in groups:
                continue
            left, right = groups[left_key], groups[right_key]
            common = left.index.intersection(right.index).sort_values()
            if not len(common):
                continue
            for metric in ('mae', 'rmse', 'r2', 'pearson', 'spearman', 'precision_gt1',
                           'recall_gt1', 'f2_gt1', 'fn', 'fp'):
                delta = (pd.to_numeric(right.loc[common, metric], errors='coerce') -
                         pd.to_numeric(left.loc[common, metric], errors='coerce'))
                comparisons.append({
                    'target': target, 'comparison': label, 'metric': metric,
                    'definition': 'right minus left', 'n_matched_seeds': int(delta.notna().sum()),
                    'mean_delta': float(delta.mean()),
                    'std_delta': float(delta.std(ddof=1)) if delta.notna().sum() > 1 else np.nan,
                    'median_delta': float(delta.median()),
                    'improved_seed_count': int(
                        (delta < 0).sum() if metric in {'mae', 'rmse', 'fn', 'fp'} else (delta > 0).sum()),
                })
    return pd.DataFrame(comparisons)


def classifier_and_separation_rows(seed_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read selected-checkpoint classifier metrics and regression separation."""
    classifier_rows, separation_rows = [], []
    for _, item in seed_metrics.drop_duplicates('run_dir').iterrows():
        run_dir = Path(item.run_dir)
        base = {key: item[key] for key in ('ablation', 'training_domain', 'target', 'seed', 'run_dir')}
        metrics = pd.read_csv(run_dir / 'threshold_metrics_selected_checkpoint.csv')
        selected = metrics.loc[(metrics.split.eq('test')) & (metrics.subset.eq('double'))
                               & (metrics.decision_source.eq('classifier'))]
        if len(selected):
            if len(selected) != 1:
                raise RuntimeError(f'Ambiguous classifier metric row: {run_dir}')
            classifier_rows.append({**base, **selected.iloc[0].to_dict()})
        prediction = pd.read_csv(run_dir / 'threshold_predictions.csv')
        double = prediction.loc[(prediction.split.eq('test')) &
                                (prediction.fifth_class.eq('double'))]
        high = double.loc[double.true_gt1, 'pred_norm']
        low = double.loc[~double.true_gt1, 'pred_norm']
        high_mean = float(high.mean()) if len(high) else np.nan
        low_mean = float(low.mean()) if len(low) else np.nan
        separation_rows.append({
            **base, 'n': int(len(double)), 'true_gt1_count': int(len(high)),
            'true_le1_count': int(len(low)), 'mean_pred_true_gt1': high_mean,
            'mean_pred_true_le1': low_mean,
            'positive_margin': float((high - 1.0).mean()) if len(high) else np.nan,
            'separation': high_mean - low_mean if len(high) and len(low) else np.nan,
        })
    return pd.DataFrame(classifier_rows), pd.DataFrame(separation_rows)


def mean_std(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(['ablation', 'training_domain', 'target'], sort=True):
        row = dict(zip(['ablation', 'training_domain', 'target'], keys, strict=True))
        row['n_completed_seeds'] = int(len(part))
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors='coerce')
            row[f'{metric}_mean'] = float(values.mean())
            row[f'{metric}_std'] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / 'summary').resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_metrics = run_rows(root)
    summary = aggregate(seed_metrics)
    deltas = paired_deltas(seed_metrics)
    classifier, separation = classifier_and_separation_rows(seed_metrics)
    seed_metrics.to_csv(output / 'o14a_double_test_per_seed_metrics.csv', index=False)
    summary.to_csv(output / 'o14a_double_test_aggregate.csv', index=False)
    deltas.to_csv(output / 'o14a_matched_seed_deltas.csv', index=False)
    separation.to_csv(output / 'o14a_double_test_separation_per_seed.csv', index=False)
    mean_std(separation, ['mean_pred_true_gt1', 'mean_pred_true_le1', 'positive_margin', 'separation']).to_csv(
        output / 'o14a_double_test_separation_aggregate.csv', index=False)
    if not classifier.empty:
        classifier.to_csv(output / 'o14a_double_test_classifier_per_seed.csv', index=False)
        mean_std(classifier, ['precision_gt1', 'recall_gt1', 'f2_gt1', 'auroc_gt1', 'auprc_gt1']).to_csv(
            output / 'o14a_double_test_classifier_aggregate.csv', index=False)
    print(summary.to_string(index=False))
    if not deltas.empty:
        print('\nMatched-seed deltas (right minus left):')
        print(deltas.to_string(index=False))


if __name__ == '__main__':
    main()
