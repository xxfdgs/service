#!/usr/bin/env python3
"""Aggregate completed fusion/head Stage-1 runs without reading outer tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_run(run_dir):
    settings = json.loads((run_dir / 'run_settings.json').read_text())
    summary_path = run_dir / 'summary.json'
    complete = summary_path.exists()
    summary = json.loads(summary_path.read_text()) if complete else {}
    return {**settings, **summary, 'run_dir': str(run_dir), 'complete': complete}


def macro_at_selected_epoch(run):
    metrics_path = Path(run['run_dir']) / 'epoch_metrics.csv'
    if not run['complete'] or not metrics_path.exists():
        return pd.DataFrame()
    metric = pd.read_csv(metrics_path)
    metric = metric.loc[(metric.split == 'val') & (metric.epoch == run['best_epoch'])].copy()
    metric['candidate'] = run['candidate']
    metric['fold'] = run['fold']
    metric['group'] = run['group']
    metric['fusion_type'] = run['fusion_type']
    metric['head_type'] = run['head_type']
    return metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-root', type=Path, required=True)
    args = parser.parse_args()
    stage_root = args.stage_root.resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in sorted(stage_root.glob('group_*/*/fold_*')):
        if (path / 'run_settings.json').exists():
            runs.append(read_run(path))
    # A crashed invocation can be restarted in an ``*_attemptN`` directory.
    # Keep every actual attempt in inventory, but select exactly one canonical
    # completed run per planned (candidate, fold) comparison.  Prefer the
    # conventional ``.../fold_N`` directory whenever it completed.
    by_key = {}
    for run in runs:
        key = (run['candidate'], run['fold'])
        by_key.setdefault(key, []).append(run)
    for key, records in by_key.items():
        completed = [record for record in records if record['complete']]
        preferred = next(
            (record for record in completed
             if Path(record['run_dir']).name == record['fold']),
            completed[0] if completed else None,
        )
        for record in records:
            record['canonical_for_selection'] = record is preferred
            record['attempt_status'] = ('selected' if record is preferred else
                                        ('superseded_duplicate' if record['complete'] else 'incomplete'))
    inventory = pd.DataFrame(runs)
    inventory.to_csv(stage_root / 'run_inventory.csv', index=False)
    selected_runs = [run for run in runs if run['canonical_for_selection']]
    metric_frames = [macro_at_selected_epoch(run) for run in selected_runs]
    metrics = pd.concat([frame for frame in metric_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in metric_frames) else pd.DataFrame()
    metrics.to_csv(stage_root / 'fold_metrics.csv', index=False)

    epoch_frames, prediction_frames = [], []
    for run in selected_runs:
        run_dir = Path(run['run_dir'])
        if run['complete'] and (run_dir / 'epoch_metrics.csv').exists():
            frame = pd.read_csv(run_dir / 'epoch_metrics.csv')
            for key in ('candidate', 'fold', 'group', 'fusion_type', 'head_type'):
                frame[key] = run[key]
            epoch_frames.append(frame)
        if run['complete'] and (run_dir / 'predictions.csv').exists():
            frame = pd.read_csv(run_dir / 'predictions.csv')
            for key in ('candidate', 'fold', 'group', 'fusion_type', 'head_type'):
                frame[key] = run[key]
            prediction_frames.append(frame)
    (pd.concat(epoch_frames, ignore_index=True) if epoch_frames else pd.DataFrame()).to_csv(stage_root / 'epoch_metrics.csv', index=False)
    (pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()).to_csv(stage_root / 'predictions.csv', index=False)

    if metrics.empty:
        comparison = pd.DataFrame()
    else:
        comparison = metrics.groupby(
            ['group', 'candidate', 'fusion_type', 'head_type', 'fold'], as_index=False
        ).agg({'mae': 'mean', 'rmse': 'mean', 'r2': 'mean', 'pearson': 'mean',
               'spearman': 'mean', 'std_ratio': 'mean', 'prediction_std': 'mean'})
    comparison.to_csv(stage_root / 'architecture_comparison.csv', index=False)

    selection = {'formal_head': None, 'head_selection_status': 'PENDING',
                 'diagnostic_head_if_no_formal_candidate': 'A1',
                 'completed_runs': int(sum(run['complete'] for run in selected_runs)),
                 'all_completed_attempts': int(inventory.complete.sum()) if len(inventory) else 0}
    if not metrics.empty:
        aggregate = metrics.groupby(['candidate', 'fold'], as_index=False).agg(
            mae=('mae', 'mean'), r2=('r2', 'mean'), spearman=('spearman', 'mean'),
            std_ratio=('std_ratio', 'mean'), finite=('mae', lambda x: bool(np.isfinite(x).all())))
        baseline = aggregate.loc[aggregate.candidate == 'A0'].set_index('fold')
        candidates = []
        for candidate in ('A1', 'A2', 'A3', 'A4'):
            value = aggregate.loc[aggregate.candidate == candidate].set_index('fold')
            if not {'fold_0', 'fold_4'}.issubset(value.index) or not {'fold_0', 'fold_4'}.issubset(baseline.index):
                continue
            f0, f4, b0, b4 = value.loc['fold_0'], value.loc['fold_4'], baseline.loc['fold_0'], baseline.loc['fold_4']
            target_rows = metrics.loc[(metrics.candidate == candidate) & (metrics.fold == 'fold_4')]
            nonconstant_targets = int((target_rows.std_ratio >= .10).sum())
            accepted = bool(
                f4.std_ratio >= b4.std_ratio * 1.5 and f4.mae <= b4.mae * 1.02
                and f4.spearman >= b4.spearman and f0.mae <= b0.mae * 1.02
                and f0.spearman >= b0.spearman - .02 and f0.finite and f4.finite
                and nonconstant_targets >= 2
            )
            candidates.append({'candidate': candidate, 'accepted': accepted,
                               'fold4_mae': f4.mae, 'fold4_r2': f4.r2,
                               'fold4_spearman': f4.spearman, 'fold4_std_ratio': f4.std_ratio,
                               'fold0_mae': f0.mae, 'fold0_r2': f0.r2,
                               'fold0_spearman': f0.spearman,
                               'fold4_nonconstant_targets': nonconstant_targets})
        selection['head_candidates'] = candidates
        accepted = [row for row in candidates if row['accepted']]
        if len(baseline) == 2 and len(candidates) == 4:
            if accepted:
                accepted.sort(key=lambda row: (row['fold4_mae'], -row['fold4_r2'], -row['fold4_spearman'], -row['fold4_std_ratio']))
                selection['formal_head'] = accepted[0]['candidate']
                selection['head_selection_status'] = 'FORMAL_CANDIDATE_SELECTED'
            else:
                selection['head_selection_status'] = 'NO_FORMAL_HEAD_CANDIDATE_USE_A1_DIAGNOSTIC'
    (stage_root / 'head_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    report = [
        '# Stage 1 aggregation', '',
        f"Completed runs: {selection['completed_runs']}",
        f"Head selection status: {selection['head_selection_status']}",
        f"Formal head: {selection['formal_head']}",
        '', 'Only train/validation predictions are included. Outer test was not read for selection.',
    ]
    (stage_root / 'stage1_report.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()
