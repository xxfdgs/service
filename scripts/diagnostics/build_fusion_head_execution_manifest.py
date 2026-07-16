#!/usr/bin/env python3
"""Build a reproducible execution manifest from fusion/head experiment runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'results/fusion_head_redesign_exp'


def sha256(path):
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    dataset = ROOT / 'results/deduplicated_rebaseline/data_audit/dataset_with_sample_id.csv'
    records = [{
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'command': 'scripts/diagnostics/test_fusion_head_baseline_equivalence.py',
        'stage': 'implementation', 'architecture': 'legacy_baseline',
        'fusion_type': 'softmax_sum', 'head_type': 'baseline', 'fold': 'fold_4',
        'seed': 0, 'dataset_hash': sha256(dataset), 'manifest_hash': None,
        'feature_hash': None, 'config_hash': sha256(ROOT / 'results/deduplicated_rebaseline/graphgps_cv/configs/formula_identity_group_cv_fold_4_seed_0.yaml'),
        'checkpoint': str(ROOT / 'results/deduplicated_rebaseline/graphgps_cv/training/formula_identity_group_cv_fold_4_seed_0/0/ckpt/49.ckpt'),
        'status': 'completed' if json.loads((EXPERIMENT / 'implementation/baseline_equivalence_test.json').read_text())['pass'] else 'failed',
        'error': None, 'output_path': str(EXPERIMENT / 'implementation/baseline_equivalence_test.json'),
    }]
    for command, stage, output_path in (
        ('scripts/diagnostics/test_fusion_head_interfaces.py', 'implementation_tests', ROOT / 'scripts/diagnostics/test_fusion_head_interfaces.py'),
        ('scripts/diagnostics/write_fusion_head_parameter_counts.py', 'implementation', EXPERIMENT / 'implementation/parameter_counts.csv'),
        ('scripts/diagnostics/aggregate_fusion_head_stage1.py', 'stage1_aggregation', EXPERIMENT / 'stage1/stage1_report.md'),
        ('scripts/diagnostics/aggregate_fusion_head_dynamics.py', 'dynamics_aggregation', EXPERIMENT / 'dynamics/epoch_metrics.csv'),
        ('scripts/diagnostics/write_fusion_head_final_report.py', 'final_report', EXPERIMENT / 'report.md'),
    ):
        records.append({
            'timestamp': datetime.now(timezone.utc).isoformat(), 'command': command,
            'stage': stage, 'architecture': None, 'fusion_type': None, 'head_type': None,
            'fold': None, 'seed': 0, 'dataset_hash': sha256(dataset), 'manifest_hash': None,
            'feature_hash': None, 'config_hash': None, 'checkpoint': None, 'status': 'completed',
            'error': None, 'output_path': str(output_path),
        })
    for settings_path in sorted((EXPERIMENT / 'stage1').glob('group_*/*/fold_*/run_settings.json')):
        run_dir = settings_path.parent
        settings = json.loads(settings_path.read_text())
        source_config = Path(settings['source_config'])
        config = yaml.safe_load(source_config.read_text())
        manifest = Path(config['train']['manifest_path'])
        feature = Path(config.get('mordred_feature_path', '')) if config.get('mordred_feature_path') else None
        summary_path = run_dir / 'summary.json'
        status = 'completed' if summary_path.exists() else 'in_progress'
        records.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'command': 'scripts/diagnostics/run_fusion_head_experiment.py ' + ' '.join(
                f'--{key.replace("_", "-")} {value}' for key, value in settings.items()
                if key in {'fold', 'group', 'candidate', 'fusion_type', 'head_type'}),
            'stage': 'stage1_group_' + settings['group'].lower(),
            'architecture': settings['architecture_name'], 'fusion_type': settings['fusion_type'],
            'head_type': settings['head_type'], 'fold': settings['fold'], 'seed': int(config['seed']),
            'dataset_hash': sha256(dataset), 'manifest_hash': sha256(manifest),
            'feature_hash': sha256(feature), 'config_hash': sha256(source_config),
            'checkpoint': str(run_dir / 'checkpoints/selected_best.pt') if summary_path.exists() else None,
            'status': status, 'error': None, 'output_path': str(run_dir),
        })
    (EXPERIMENT / 'execution_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({'records': len(records), 'output': str(EXPERIMENT / 'execution_manifest.json')}))


if __name__ == '__main__':
    main()
