#!/usr/bin/env python3
"""Combine per-run diagnostics into the required experiment-level dynamics CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FILES = (
    'epoch_metrics.csv', 'branch_statistics.csv', 'fusion_statistics.csv',
    'gate_statistics.csv', 'head_statistics.csv', 'gradient_statistics.csv',
    'collapse_events.csv',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = {name: [] for name in FILES}
    for settings_path in sorted(args.stage_root.glob('group_*/*/fold_*/run_settings.json')):
        run_dir = settings_path.parent
        settings = json.loads(settings_path.read_text())
        run_id = str(run_dir.relative_to(args.stage_root))
        for name in FILES:
            path = run_dir / name
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame.insert(0, 'run_id', run_id)
            for key in ('fold', 'group', 'candidate', 'fusion_type', 'head_type', 'architecture_name'):
                frame[key] = settings[key]
            groups[name].append(frame)
    for name, frames in groups.items():
        output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        output.to_csv(output_dir / name, index=False)
        print(f'{name}: {len(output)} rows')


if __name__ == '__main__':
    main()
