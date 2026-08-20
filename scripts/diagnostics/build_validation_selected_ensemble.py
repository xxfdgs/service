"""Build an input-only, validation-selected convex GraphGPS ensemble.

Weights are fitted independently for each target using validation predictions
only.  Test predictions are read only after the weights have been finalized;
the output records this protocol alongside train/validation/test metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGETS = [
    'EE_before', 'EE_after', 'Aerosolization_Efficiency',
    'mRNA_Recovery_Efficiency',
]
SPLITS = ['train', 'val', 'test']


def constrained_weights(truth: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Fit non-negative weights summing to one by validation MAE."""
    count = predictions.shape[1]
    starts = [np.full(count, 1.0 / count)]
    starts.extend(np.eye(count))
    candidates = []
    for start in starts:
        result = minimize(
            lambda weights: np.mean(np.abs(truth - predictions @ weights)),
            start, method='SLSQP', bounds=[(0.0, 1.0)] * count,
            constraints={'type': 'eq', 'fun': lambda weights: weights.sum() - 1.0},
            options={'ftol': 1e-10, 'maxiter': 5000},
        )
        if result.success and np.isfinite(result.fun):
            candidates.append((float(result.fun), result.x))
    if not candidates:
        raise RuntimeError('Validation ensemble optimization failed for every feasible start.')
    return min(candidates, key=lambda item: item[0])[1]


def correlation(function, truth, prediction):
    if len(truth) < 2 or np.std(truth) == 0 or np.std(prediction) == 0:
        return float('nan')
    return float(function(truth, prediction).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    args = parser.parse_args()

    if any('feedback' in value.lower() for value in args.runs):
        raise ValueError('Input-only ensemble rejects feedback-named inputs.')
    frames = {}
    for name in args.runs:
        path = args.experiments_root / name / 'predictions.csv'
        if not path.is_file():
            raise FileNotFoundError(path)
        frames[name] = pd.read_csv(path)
        if 'feedback' in path.read_text(encoding='utf-8', errors='ignore').lower():
            raise ValueError(f'Input-only ensemble rejected feedback text in {path}')

    output_rows, weight_rows, metric_rows = [], [], []
    for target in TARGETS:
        validation = []
        for name, frame in frames.items():
            subset = frame[(frame['split'] == 'val') & (frame['target'] == target)]
            validation.append(subset[['sample_id', 'y_true', 'y_pred']].rename(
                columns={'y_pred': name}))
        merged_validation = validation[0]
        for frame in validation[1:]:
            merged_validation = merged_validation.merge(frame.drop(columns='y_true'), on='sample_id', validate='one_to_one')
        weights = constrained_weights(
            merged_validation.y_true.to_numpy(float),
            merged_validation[list(frames)].to_numpy(float),
        )
        for name, weight in zip(frames, weights):
            weight_rows.append({'target': target, 'experiment': name, 'weight': float(weight)})

        for split in SPLITS:
            merged = None
            for name, frame in frames.items():
                subset = frame[(frame['split'] == split) & (frame['target'] == target)]
                subset = subset[['sample_id', 'y_true', 'y_pred']].rename(columns={'y_pred': name})
                merged = subset if merged is None else merged.merge(
                    subset.drop(columns='y_true'), on='sample_id', validate='one_to_one')
            merged['y_pred'] = merged[list(frames)].to_numpy(float) @ weights
            merged['split'] = split
            merged['target'] = target
            output_rows.append(merged[['sample_id', 'split', 'target', 'y_true', 'y_pred']])
            truth, prediction = merged.y_true.to_numpy(float), merged.y_pred.to_numpy(float)
            metric_rows.append({
                'split': split, 'target': target, 'n': len(merged),
                'mae': float(mean_absolute_error(truth, prediction)),
                'rmse': float(np.sqrt(mean_squared_error(truth, prediction))),
                'r2': float(r2_score(truth, prediction)),
                'pearson': correlation(pearsonr, truth, prediction),
                'spearman': correlation(spearmanr, truth, prediction),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(output_rows, ignore_index=True).to_csv(args.output_dir / 'predictions.csv', index=False)
    pd.DataFrame(weight_rows).to_csv(args.output_dir / 'validation_weights.csv', index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / 'metrics.csv', index=False)
    summary = metrics.groupby('split', as_index=False).agg(
        mean_mae=('mae', 'mean'), mean_r2=('r2', 'mean'),
        mean_pearson=('pearson', 'mean'), mean_spearman=('spearman', 'mean'),
    )
    summary.to_csv(args.output_dir / 'metrics_summary.csv', index=False)
    (args.output_dir / 'protocol.json').write_text(json.dumps({
        'input_only': True,
        'runs': args.runs,
        'selection_split': 'val',
        'objective': 'per-target convex ensemble MAE',
        'test_read_after_weight_selection_only': True,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
