#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_DESCRIPTORS = ['SsNH3', 'SMR_VSA9', 'SlogP_VSA11', 'SlogP_VSA10',
                       'SMR_VSA10', 'TopoPSA', 'MW', 'nRot', 'nRing',
                       'nAromAtom', 'nHBDon', 'nHBAcc']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptor-table', default='results/mordred_train_feedback/mordred_descriptors_unique_smiles.csv')
    parser.add_argument('--occurrences', default='results/mordred_train_feedback/molecule_occurrences.csv')
    parser.add_argument('--output', default='results/mordred_train_feedback/mordred_selected_features.csv')
    parser.add_argument('--descriptors', nargs='*', default=DEFAULT_DESCRIPTORS)
    parser.add_argument('--comparison', default='results/mordred_train_feedback/descriptor_distribution_comparison.csv')
    parser.add_argument('--auto-top-k', type=int)
    args = parser.parse_args()
    table = pd.read_csv(args.descriptor_table)
    occurrences = pd.read_csv(args.occurrences)
    if args.auto_top_k:
        comparison = pd.read_csv(args.comparison)
        eligible = comparison[
            (comparison.train_variance > 1e-10)
            & (comparison.feedback_variance > 1e-10)
        ].sort_values('effect_size', ascending=False)
        selected = [name for name in eligible.descriptor
                    if name in table.columns][:args.auto_top_k]
    else:
        selected = [name for name in args.descriptors if name in table.columns]
    if not selected:
        raise ValueError('No requested Mordred descriptors were found.')
    train_smiles = occurrences.loc[occurrences.source == 'train', 'smiles']
    train_values = table.set_index('smiles').reindex(train_smiles)[selected]
    medians = train_values.median().fillna(0.0)
    means = train_values.fillna(medians).mean()
    stds = train_values.fillna(medians).std().replace(0, 1.0)
    values = table[['smiles'] + selected].copy().set_index('smiles').fillna(medians)
    values = (values - means) / stds
    values.columns = [f'feature_{index}' for index in range(len(selected))]
    values.reset_index().to_csv(args.output, index=False)
    Path(args.output).with_suffix('.json').write_text(pd.Series({'descriptors': selected, 'mean': means.to_dict(), 'std': stds.to_dict()}).to_json(), encoding='utf-8')
    print(f'Wrote {len(selected)} standardized Mordred features to {args.output}')


if __name__ == '__main__':
    main()
