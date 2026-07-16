#!/usr/bin/env python3
"""
Find data points in the input dataset that have identical (or near-identical)
inputs but potentially different output values.

Input identity is defined by: all 5 SMILES + all 5 mol% ratios.
"""

import os, sys
import numpy as np
import pandas as pd

INPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'datasets_lrx', 'raw', 'input', '20260703_sum.csv')

# Columns that define "input identity"
SMILE_COLS = ['IL_SMILE', 'HL_SMILE', 'Chol_SMILE', 'PEG_SMILE', 'Fifth_SMILE']
RATIO_COLS = ['mol%_IL', 'mol%_HL', 'mol%_Chol', 'mol%_PEG', 'mol%_Fifth']

# Output columns to check for discrepancies
OUTPUT_COLS = ['EE_before', 'EE_after', 'Aerosolization_Efficiency',
               'mRNA_Recovery_Efficiency', 'Norm_before', 'Norm_after']

# Tolerance for considering two ratio values "the same" (percentage points)
RATIO_TOLERANCE = 0.5


def build_input_key(row):
    """Build a canonical input key from a row: SMILES tuple + rounded ratios."""
    smiles_key = tuple(str(row[c]) if pd.notna(row[c]) else 'NAN' for c in SMILE_COLS)
    # Round ratios to tolerance to group near-identical inputs
    ratio_key = tuple(round(row[c] / RATIO_TOLERANCE) * RATIO_TOLERANCE
                      if pd.notna(row[c]) else -1.0
                      for c in RATIO_COLS)
    return (smiles_key, ratio_key)


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f'Dataset: {len(df)} rows, {INPUT_CSV}')
    print(f'Input columns:  {SMILE_COLS + RATIO_COLS}')
    print(f'Output columns: {OUTPUT_COLS}')
    print(f'Ratio tolerance: ±{RATIO_TOLERANCE} percentage points')
    print()

    # Build input keys
    df['_input_key'] = df.apply(build_input_key, axis=1)

    # Group by input key
    groups = df.groupby('_input_key')
    n_unique = len(groups)
    n_total = len(df)
    print(f'Unique input combinations: {n_unique} / {n_total} total rows')
    print()

    # Find groups with multiple rows
    duplicate_groups = {k: g for k, g in groups if len(g) > 1}
    n_duplicate = len(duplicate_groups)
    n_duplicate_rows = sum(len(g) for g in duplicate_groups.values())

    if n_duplicate == 0:
        print('No duplicate inputs found. All rows have unique input combinations.')
        return

    print(f'Duplicate input groups: {n_duplicate} groups, {n_duplicate_rows} rows total')
    print()

    # Within each group, check if outputs differ
    conflict_groups = []
    for key, group in duplicate_groups.items():
        outputs = group[OUTPUT_COLS].values
        # Check max deviation across outputs
        output_range = np.max(outputs, axis=0) - np.min(outputs, axis=0)
        # Normalize by per-column std if possible
        col_std = np.std(outputs, axis=0)
        col_std[col_std < 0.01] = 1.0
        normalized_range = output_range / col_std

        if np.any(normalized_range > 0.01):  # meaningful difference
            conflict_groups.append((key, group, output_range))

    if not conflict_groups:
        print('All duplicate inputs have identical outputs (within tolerance).')
        print('This is expected if the same experiment was recorded multiple times.')
        return

    print(f'{"="*100}')
    print(f'FOUND {len(conflict_groups)} INPUT GROUPS WITH CONFLICTING OUTPUTS')
    print(f'{"="*100}')
    print()

    for i, (key, group, output_range) in enumerate(conflict_groups):
        smiles_key, ratio_key = key
        print(f'--- Conflict Group {i+1} ({len(group)} rows) ---')
        print(f'  SMILES:')
        for j, col in enumerate(SMILE_COLS):
            smi = smiles_key[j]
            print(f'    {col}: {smi[:80]}{"..." if len(smi)>80 else ""}')
        print(f'  Ratios: {dict(zip(RATIO_COLS, ratio_key))}')
        print()

        # Show the differing outputs
        display_cols = ['ID'] + OUTPUT_COLS
        for _, row in group[display_cols].iterrows():
            print(f'  {row["ID"]:12s}  EE_b={row["EE_before"]:6.1f}  EE_a={row["EE_after"]:6.1f}  '
                  f'Aero={row["Aerosolization_Efficiency"]:6.1f}  Recov={row["mRNA_Recovery_Efficiency"]:6.1f}  '
                  f'Norm_b={row["Norm_before"]:5.2f}  Norm_a={row["Norm_after"]:5.2f}')

        # Compute output spread
        outputs = group[OUTPUT_COLS]
        print()
        print(f'  Output spread (max - min):')
        for col in OUTPUT_COLS:
            vals = outputs[col]
            spread = vals.max() - vals.min()
            if spread > 0.01:
                print(f'    {col}: {vals.min():.3f} ~ {vals.max():.3f}  (Δ={spread:.3f})')
        print()

    # Summary stats
    print(f'{"="*100}')
    print('SUMMARY')
    print(f'{"="*100}')
    total_conflict_rows = sum(len(g) for _, g, _ in conflict_groups)
    print(f'  Total conflict groups: {len(conflict_groups)}')
    print(f'  Total rows in conflicts: {total_conflict_rows}')
    print()

    # Per-output max discrepancy
    print('  Max output discrepancy per property:')
    for col in OUTPUT_COLS:
        max_delta = 0.0
        for _, group, _ in conflict_groups:
            vals = group[col]
            delta = vals.max() - vals.min()
            if delta > max_delta:
                max_delta = delta
        print(f'    {col}: max Δ = {max_delta:.3f}')


if __name__ == '__main__':
    main()
