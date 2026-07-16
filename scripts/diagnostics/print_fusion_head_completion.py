#!/usr/bin/env python3
"""Print the required terminal hand-off for the fusion/head study."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'results/fusion_head_redesign_exp'


def value(metrics, candidate, fold, column):
    frame = metrics.loc[(metrics.candidate == candidate) & (metrics.fold == fold)]
    return float(frame[column].mean())


def main():
    equivalence = json.loads((EXP / 'implementation/baseline_equivalence_test.json').read_text())
    metrics = pd.read_csv(EXP / 'stage1/fold_metrics.csv')
    b0 = {column: value(metrics, 'A0', 'fold_4', column) for column in ('mae', 'spearman', 'std_ratio')}
    b4 = {column: value(metrics, 'B4', 'fold_4', column) for column in ('mae', 'spearman', 'std_ratio')}
    print(f"1. baseline equivalence: {'PASS' if equivalence['pass'] else 'FAIL'}")
    print('2. best head: none (A1 used only as diagnostic head)')
    print('3. best fusion: none safe (B4 has fold-4-only local signal)')
    print('4. fold-4 no longer near-constant: no validated cross-fold solution')
    print(f"5. fold-4 std ratio: A0={b0['std_ratio']:.4f}, B4={b4['std_ratio']:.4f}")
    print(f"6. fold-4 MAE/Spearman: A0={b0['mae']:.4f}/{b0['spearman']:.4f}, B4={b4['mae']:.4f}/{b4['spearman']:.4f}")
    print('7. fold-0/fold-1: fold-0 degrades for all fold-4-improving variants; fold-1 skipped by protocol')
    print('8. untouched folds: not run; no Stage-1 candidate passed')
    print('9. full retraining: not recommended for any tested fusion/head variant')
    print('10. single-task experiment: not recommended from current evidence')
    print('11. final status: NO_SAFE_FUSION_HEAD_FIX')
    print(f'12. report: {EXP / "report.md"}')
    print('13. unfinished stages: repeat/fold-1/fold-2/fold-3/pooled skipped because the mandatory Stage-1 gates failed')


if __name__ == '__main__':
    main()
