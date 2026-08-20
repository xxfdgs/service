#!/usr/bin/env python3
"""Summarize inverse-transformed O12 later4 validation/test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGETS = [
    'Aerosolization_Efficiency', 'mRNA_Recovery_Efficiency',
    'Norm_before', 'Norm_after',
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.runs_root.resolve()
    rows, scalers = [], []
    for seed in range(100, 110):
        run_dir = root / f'O12_later4_split{seed}'
        settings_path = run_dir / 'run_settings.json'
        predictions_path = run_dir / 'predictions.csv'
        scaler_path = run_dir / 'target_scaler.json'
        checkpoint_path = run_dir / 'checkpoints' / 'selected_best.pt'
        if not all(path.is_file() for path in (settings_path, predictions_path, scaler_path, checkpoint_path)):
            raise FileNotFoundError(f'Incomplete later4 run: {run_dir}')
        settings = json.loads(settings_path.read_text(encoding='utf-8'))
        if settings.get('targets') != TARGETS or settings.get('target_normalization') != 'zscore':
            raise RuntimeError(f'Run is not a z-score normalized later4 checkpoint: {run_dir}')
        scaler = json.loads(scaler_path.read_text(encoding='utf-8'))
        if scaler.get('type') != 'zscore_train_only':
            raise RuntimeError(f'Unexpected scaler in {scaler_path}: {scaler.get("type")!r}')
        scalers.append({'split_seed': seed, **scaler})
        prediction = pd.read_csv(predictions_path)
        for split in ('val', 'test'):
            for target in TARGETS:
                part = prediction.loc[prediction['split'].eq(split) & prediction['target'].eq(target)]
                if len(part) != 70:
                    raise RuntimeError(f'Expected 70 {split}/{target} predictions in {run_dir}, got {len(part)}')
                truth, values = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
                rows.append({
                    'split_seed': seed, 'split': split, 'target': target, 'n': len(part),
                    'mae': mean_absolute_error(truth, values),
                    'rmse': mean_squared_error(truth, values) ** .5,
                    'r2': r2_score(truth, values),
                })
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(['split', 'target'], as_index=False).agg(
        completed_seeds=('split_seed', 'nunique'),
        mean_mae=('mae', 'mean'), std_mae=('mae', 'std'),
        mean_rmse=('rmse', 'mean'), std_rmse=('rmse', 'std'),
        mean_r2=('r2', 'mean'), std_r2=('r2', 'std'),
    )
    metrics.to_csv(root / 'validation_test_metrics_by_seed_target.csv', index=False)
    summary.to_csv(root / 'validation_test_metrics_target_average.csv', index=False)
    pd.DataFrame(scalers).to_json(root / 'target_scalers_by_seed.json', orient='records', indent=2)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
