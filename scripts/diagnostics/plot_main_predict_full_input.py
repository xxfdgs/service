#!/usr/bin/env python3
"""Summarize and plot four-property predictions produced by main_predict.py.

The input is the ``predicted_average_6props.csv`` generated at the end of a
``double_predict`` run.  It intentionally accepts both the ensemble column
names from main_predict.py and per-model prediction column names, which makes
it useful for one-repeat as well as multi-repeat prediction runs.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROPERTY_COLUMNS = {
    'EE_before': ('true_EE_before', 'pred_EE_before_average', 'pred_EE_before'),
    'EE_after': ('true_EE_after', 'pred_EE_after_average', 'pred_EE_after'),
    'Aero_Efficiency': (
        'true_Aero_Efficiency',
        'pred_Aero_Efficiency_average',
        'pred_Aero_Efficiency',
    ),
    'Recovery_Efficiency': (
        'true_Recovery_Efficiency',
        'pred_Recovery_Efficiency_average',
        'pred_Recovery_Efficiency',
    ),
}


def select_prediction_column(frame, average_column, single_model_column):
    if average_column in frame.columns:
        return average_column
    if single_model_column in frame.columns:
        return single_model_column
    raise ValueError(
        f'Missing prediction column. Expected {average_column!r} or '
        f'{single_model_column!r}.'
    )


def correlation_or_nan(function, true, pred):
    if len(true) < 2 or np.isclose(np.std(true), 0) or np.isclose(np.std(pred), 0):
        return np.nan
    return float(function(true, pred).statistic)


def main():
    parser = argparse.ArgumentParser(
        description='Create whole-input metrics and true-vs-prediction scatter plots.'
    )
    parser.add_argument('--predictions', required=True, type=Path,
                        help='main_predict.py output: predicted_average_6props.csv')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()

    prediction_path = args.predictions.resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(f'Prediction CSV not found: {prediction_path}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(prediction_path)
    if frame.empty:
        raise ValueError(f'Prediction CSV is empty: {prediction_path}')

    metrics = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for axis, (property_name, columns) in zip(axes.flat, PROPERTY_COLUMNS.items()):
        true_column, average_column, single_model_column = columns
        if true_column not in frame.columns:
            raise ValueError(f'Missing true-value column: {true_column!r}')
        prediction_column = select_prediction_column(
            frame, average_column, single_model_column
        )
        subset = frame[[true_column, prediction_column]].dropna()
        if subset.empty:
            raise ValueError(f'No finite rows available for {property_name}.')
        true = subset[true_column].to_numpy(dtype=float)
        pred = subset[prediction_column].to_numpy(dtype=float)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        # ``squared=False`` is unavailable in the sklearn version bundled in
        # the project environment, so calculate RMSE explicitly.
        rmse = float(np.sqrt(mean_squared_error(true, pred)))
        pearson = correlation_or_nan(pearsonr, true, pred)
        spearman = correlation_or_nan(spearmanr, true, pred)
        metrics.append({
            'property': property_name,
            'n_samples': len(subset),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'pearson_r': pearson,
            'spearman_rho': spearman,
            'true_column': true_column,
            'prediction_column': prediction_column,
        })

        lower = min(true.min(), pred.min())
        upper = max(true.max(), pred.max())
        padding = max((upper - lower) * 0.04, 1e-6)
        lower -= padding
        upper += padding
        axis.scatter(true, pred, s=18, alpha=0.72, edgecolors='none')
        axis.plot([lower, upper], [lower, upper], '--', color='black', lw=1.2,
                  label='y = x')
        axis.set(xlim=(lower, upper), ylim=(lower, upper),
                 xlabel='True value', ylabel='Predicted value')
        axis.set_title(f'{property_name}\nMAE={mae:.3f}, R²={r2:.3f}')
        axis.legend(loc='best')
        axis.grid(alpha=0.22)

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(args.output_dir / 'metrics_per_property.csv', index=False)
    summary = pd.DataFrame([{
        'n_samples_per_property': int(metrics_frame['n_samples'].iloc[0]),
        'mae_macro': metrics_frame['mae'].mean(),
        'rmse_macro': metrics_frame['rmse'].mean(),
        'r2_macro': metrics_frame['r2'].mean(),
        'pearson_r_macro': metrics_frame['pearson_r'].mean(),
        'spearman_rho_macro': metrics_frame['spearman_rho'].mean(),
    }])
    summary.to_csv(args.output_dir / 'metrics_summary.csv', index=False)
    fig.suptitle(f'Whole-input prediction diagnostics (n={len(frame)})', fontsize=14)
    fig.savefig(args.output_dir / 'true_vs_pred_scatter.png', dpi=220)
    fig.savefig(args.output_dir / 'true_vs_pred_scatter.pdf')
    plt.close(fig)

    print(metrics_frame.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('\nMacro summary')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print(f'Wrote diagnostics to: {args.output_dir}')


if __name__ == '__main__':
    main()
