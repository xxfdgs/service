#!/usr/bin/env python
"""Compare Mordred descriptor distributions for training and feedback inputs."""

import argparse
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for name, value in {'float': float, 'int': int, 'product': np.prod}.items():
    if not hasattr(np, name):
        setattr(np, name, value)

from mordred import Calculator, descriptors
from rdkit import Chem
from scipy.stats import levene


COMPONENT_COLUMNS = [
    'IL_SMILE', 'HL_SMILE', 'Chol_SMILE', 'PEG_SMILE', 'Fifth_SMILE',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare calculable 2D Mordred descriptor distributions.'
    )
    parser.add_argument('--train-csv',
                        default='datasets_lrx/raw/input/20260703_sum.csv')
    parser.add_argument('--feedback-csv',
                        default='datasets_lrx/raw/feedback/20260703_validation.csv')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--top-k', type=int, default=20)
    return parser.parse_args()


def collect_molecule_occurrences(csv_path, source):
    frame = pd.read_csv(csv_path)
    records = []
    for component in COMPONENT_COLUMNS:
        for row_index, smiles in frame[component].items():
            if pd.isna(smiles):
                continue
            molecule = Chem.MolFromSmiles(str(smiles))
            if molecule is not None:
                records.append({
                    'source': source,
                    'component': component,
                    'row_index': row_index,
                    'smiles': Chem.MolToSmiles(molecule, canonical=True),
                })
    return pd.DataFrame(records)


def calculate_one_descriptor_row(smiles, output_queue):
    calculator = Calculator(descriptors, ignore_3D=True)
    molecule = Chem.MolFromSmiles(smiles)
    result = calculator(molecule)
    row = {'smiles': smiles}
    for name, value in result.items():
        try:
            row[str(name)] = float(value)
        except (TypeError, ValueError):
            row[str(name)] = np.nan
    output_queue.put(row)


def calculate_descriptors(unique_smiles, timeout_seconds=30):
    rows = []
    failed_smiles = []
    context = mp.get_context('fork')
    for smiles in unique_smiles:
        output_queue = context.Queue()
        process = context.Process(
            target=calculate_one_descriptor_row,
            args=(smiles, output_queue),
        )
        process.start()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            failed_smiles.append(smiles)
            continue
        if process.exitcode != 0:
            failed_smiles.append(smiles)
            continue
        try:
            rows.append(output_queue.get(timeout=1))
        except Exception:
            failed_smiles.append(smiles)
    descriptor_frame = pd.DataFrame(rows).set_index('smiles')
    return descriptor_frame.apply(pd.to_numeric, errors='coerce'), failed_smiles


def compare_distributions(occurrences, descriptor_frame):
    descriptor_names = descriptor_frame.columns.tolist()
    expanded = occurrences.join(descriptor_frame, on='smiles')
    comparison_rows = []
    for name in descriptor_names:
        train_values = expanded.loc[expanded.source == 'train', name]
        feedback_values = expanded.loc[expanded.source == 'feedback', name]
        train_values = train_values[np.isfinite(train_values)]
        feedback_values = feedback_values[np.isfinite(feedback_values)]
        if len(train_values) < 2 or len(feedback_values) < 2:
            continue

        train_mean, feedback_mean = train_values.mean(), feedback_values.mean()
        train_variance = train_values.var(ddof=1)
        feedback_variance = feedback_values.var(ddof=1)
        pooled_std = np.sqrt((train_variance + feedback_variance) / 2.0)
        effect_size = (abs(train_mean - feedback_mean) / pooled_std
                       if pooled_std > 1e-12 else 0.0)
        variance_ratio = ((feedback_variance + 1e-12) /
                          (train_variance + 1e-12))
        log2_variance_ratio = np.log2(variance_ratio)
        levene_statistic, levene_p_value = levene(
            train_values, feedback_values, center='median'
        )
        comparison_rows.append({
            'descriptor': name,
            'train_count': len(train_values),
            'feedback_count': len(feedback_values),
            'train_mean': train_mean,
            'feedback_mean': feedback_mean,
            'train_variance': train_variance,
            'feedback_variance': feedback_variance,
            'effect_size': effect_size,
            'variance_ratio_feedback_to_train': variance_ratio,
            'abs_log2_variance_ratio': abs(log2_variance_ratio),
            'levene_statistic': levene_statistic,
            'levene_p_value': levene_p_value,
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison['distribution_difference_score'] = (
        comparison.effect_size + comparison.abs_log2_variance_ratio
    )
    comparison['distribution_similarity_score'] = (
        comparison.effect_size + comparison.abs_log2_variance_ratio
    )
    return comparison.sort_values('distribution_difference_score', ascending=False)


def plot_top_descriptors(comparison, output_dir, top_k):
    top = comparison.head(top_k).iloc[::-1]
    figure, axis = plt.subplots(figsize=(10, max(6, top_k * 0.35)))
    axis.barh(top.descriptor, top.distribution_difference_score, color='#b44742')
    axis.set(xlabel='|standardized mean difference| + |log2 variance ratio|',
             title='Mordred descriptors with largest train-feedback differences')
    figure.tight_layout()
    figure.savefig(output_dir / 'top_descriptor_differences.png', dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    scatter = axis.scatter(comparison.effect_size,
                           comparison.abs_log2_variance_ratio,
                           c=-np.log10(comparison.levene_p_value.clip(lower=1e-300)),
                           s=12, alpha=0.7, cmap='viridis')
    figure.colorbar(scatter, ax=axis, label='-log10(Levene p-value)')
    axis.set(xlabel='Absolute standardized mean difference',
             ylabel='Absolute log2 variance ratio',
             title='Train-feedback Mordred distribution differences')
    figure.tight_layout()
    figure.savefig(output_dir / 'mean_vs_variance_difference.png', dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir or (
        'results/mordred_distribution_' +
        datetime.now().strftime('%Y%m%d_%H%M%S')
    ))
    output_dir.mkdir(parents=True, exist_ok=False)

    train_occurrences = collect_molecule_occurrences(args.train_csv, 'train')
    feedback_occurrences = collect_molecule_occurrences(args.feedback_csv, 'feedback')
    occurrences = pd.concat([train_occurrences, feedback_occurrences],
                            ignore_index=True)
    descriptor_frame, failed_smiles = calculate_descriptors(
        sorted(occurrences.smiles.unique())
    )
    comparison = compare_distributions(occurrences, descriptor_frame)
    nearly_identical = comparison.sort_values(
        'distribution_similarity_score', ascending=True
    ).head(args.top_k)

    occurrences.to_csv(output_dir / 'molecule_occurrences.csv', index=False)
    pd.DataFrame({'smiles': failed_smiles}).to_csv(
        output_dir / 'failed_mordred_smiles.csv', index=False
    )
    descriptor_frame.to_csv(output_dir / 'mordred_descriptors_unique_smiles.csv')
    comparison.to_csv(output_dir / 'descriptor_distribution_comparison.csv',
                      index=False)
    comparison.head(args.top_k).to_csv(
        output_dir / 'most_different_descriptors.csv', index=False
    )
    nearly_identical.to_csv(
        output_dir / 'least_different_descriptors.csv', index=False
    )
    plot_top_descriptors(comparison, output_dir, args.top_k)

    report = {
        'train_csv': str(Path(args.train_csv).resolve()),
        'feedback_csv': str(Path(args.feedback_csv).resolve()),
        'train_molecule_occurrences': len(train_occurrences),
        'feedback_molecule_occurrences': len(feedback_occurrences),
        'unique_molecules': len(descriptor_frame),
        'molecules_without_complete_descriptor_row': len(failed_smiles),
        'numeric_2d_descriptors_compared': len(comparison),
        'top_different_descriptors': comparison.head(args.top_k).to_dict('records'),
        'least_different_descriptors': nearly_identical.to_dict('records'),
    }
    with open(output_dir / 'report.json', 'w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print('Most different descriptors:')
    print(comparison.head(args.top_k)[[
        'descriptor', 'effect_size', 'abs_log2_variance_ratio',
        'variance_ratio_feedback_to_train', 'levene_p_value',
    ]].to_string(index=False, float_format='{:.4g}'.format))
    print('\nLeast different descriptors:')
    print(nearly_identical[[
        'descriptor', 'effect_size', 'abs_log2_variance_ratio',
        'variance_ratio_feedback_to_train', 'levene_p_value',
    ]].to_string(index=False, float_format='{:.4g}'.format))
    print(f'Outputs saved to: {output_dir}')


if __name__ == '__main__':
    main()
