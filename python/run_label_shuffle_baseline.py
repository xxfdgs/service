#!/usr/bin/env python
"""Run a no-signal label-shuffle sanity check on the base training dataset."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


COMPONENT_COLUMNS = [
    'IL_SMILE', 'HL_SMILE', 'Chol_SMILE', 'PEG_SMILE', 'Fifth_SMILE',
]
RATIO_COLUMNS = [
    'mol%_IL', 'mol%_HL', 'mol%_Chol', 'mol%_PEG', 'mol%_Fifth',
]
TARGET_COLUMNS = [
    'EE_before', 'EE_after', 'Aerosolization_Efficiency',
    'mRNA_Recovery_Efficiency',
]
FINGERPRINT_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train ExtraTrees on row-shuffled labels as a sanity check.'
    )
    parser.add_argument('--base-csv',
                        default='datasets_lrx/raw/input/20260703_sum.csv')
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--output-dir', default=None)
    return parser.parse_args()


def featurize(frame, fingerprint_generator):
    feature_rows = []
    for _, row in frame.iterrows():
        component_features = []
        for component_column, ratio_column in zip(COMPONENT_COLUMNS,
                                                  RATIO_COLUMNS):
            fingerprint = np.zeros(FINGERPRINT_SIZE, dtype=np.uint8)
            smiles = row[component_column]
            if pd.notna(smiles):
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is not None:
                    DataStructs.ConvertToNumpyArray(
                        fingerprint_generator.GetFingerprint(molecule),
                        fingerprint,
                    )
            ratio = float(row[ratio_column]) / 100.0
            component_features.append(fingerprint * ratio)
        ratios = row[RATIO_COLUMNS].to_numpy(dtype=np.float32) / 100.0
        feature_rows.append(np.concatenate(component_features + [ratios]))
    return np.asarray(feature_rows, dtype=np.float32)


def create_model(seed, n_estimators):
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=8,
        max_features=0.7,
        random_state=seed,
        n_jobs=-1,
    )


def evaluate_run(features, targets, seed, test_size, n_estimators):
    indices = np.arange(len(features))
    train_index, test_index = train_test_split(
        indices, test_size=test_size, random_state=seed, shuffle=True
    )
    random_state = np.random.default_rng(seed)
    shuffled_targets = targets[train_index][random_state.permutation(len(train_index))]

    model = create_model(seed, n_estimators)
    model.fit(features[train_index], shuffled_targets)
    prediction = model.predict(features[test_index])

    metrics = []
    for property_index, property_name in enumerate(TARGET_COLUMNS):
        metrics.append({
            'seed': seed,
            'property': property_name,
            'mae': mean_absolute_error(targets[test_index, property_index],
                                       prediction[:, property_index]),
            'r2': r2_score(targets[test_index, property_index],
                           prediction[:, property_index]),
        })
    return metrics


def main():
    args = parse_args()
    if args.repeat < 1:
        raise ValueError('--repeat must be at least 1.')
    if not 0 < args.test_size < 1:
        raise ValueError('--test-size must be between 0 and 1.')

    output_dir = Path(args.output_dir or (
        'results/label_shuffle_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    ))
    output_dir.mkdir(parents=True, exist_ok=False)

    frame = pd.read_csv(args.base_csv)
    targets = frame[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=FINGERPRINT_SIZE
    )
    features = featurize(frame, generator)

    metric_rows = []
    for offset in range(args.repeat):
        metric_rows.extend(evaluate_run(
            features, targets, args.seed + offset, args.test_size,
            args.n_estimators,
        ))

    metrics = pd.DataFrame(metric_rows)
    summary = metrics.groupby('property')[['mae', 'r2']].agg(['mean', 'std'])
    summary_for_report = {
        property_name: {
            f'{metric}_{statistic}': float(value)
            for (metric, statistic), value in row.items()
        }
        for property_name, row in summary.iterrows()
    }
    metrics.to_csv(output_dir / 'per_run_metrics.csv', index=False)
    summary.to_csv(output_dir / 'summary_metrics.csv')
    report = {
        'base_csv': str(Path(args.base_csv).resolve()),
        'sample_count': len(frame),
        'repeat': args.repeat,
        'test_size': args.test_size,
        'label_shuffle': 'Row-wise permutation within each training split.',
        'metrics': summary_for_report,
    }
    with open(output_dir / 'report.json', 'w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print('Label-shuffle test metrics (mean ± std over repeats):')
    print(summary.to_string(float_format='{:.4f}'.format))
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    main()
