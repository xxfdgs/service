#!/usr/bin/env python3
"""Prove that the default fusion/head interface preserves a legacy checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphgps  # noqa: F401,E402
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from graphgps.determinism import configure_determinism  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402


TARGETS = [
    'EE_before', 'EE_after', 'Aerosolization_Efficiency',
    'mRNA_Recovery_Efficiency',
]


def prepare_batches(items, device):
    for batch in items:
        batch.to(device)
    return items


def evaluate(model, loader_groups, device):
    rows = []
    model.eval()
    with torch.no_grad():
        for split, split_index in (('train', 0), ('val', 1), ('test', 2)):
            for batches in zip(*[group[split_index] for group in loader_groups]):
                batches = prepare_batches(list(batches), device)
                prediction, labels = model(*batches)
                prediction = prediction.detach().cpu().reshape(-1, 4).numpy() * 100.0
                labels = labels.detach().cpu().reshape(-1, 4).numpy() * 100.0
                source_indices = batches[0].sample_uid.detach().cpu().numpy().reshape(-1)
                for row_index, source_index in enumerate(source_indices):
                    for target_index, target in enumerate(TARGETS):
                        rows.append({
                            'source_index': int(source_index), 'split': split,
                            'target': target,
                            'y_true': float(labels[row_index, target_index]),
                            'y_pred': float(prediction[row_index, target_index]),
                        })
    return pd.DataFrame(rows)


def sha256_state(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fold', default='fold_4')
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {output_dir}')
    (output_dir / 'cache').mkdir(parents=True, exist_ok=True)

    set_cfg_gps(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(args.config.resolve()), opts=[]))
    # Explicitly select the strict historical path, irrespective of defaults.
    cfg.model.fusion_type = 'softmax_sum'
    cfg.model.head_type = 'baseline'
    cfg.model.architecture_name = 'legacy_baseline'
    cfg.model.validate_redesign_inputs = False
    cfg.dataset.dir = str(output_dir / 'cache')
    cfg.dataset.cache_tag = f'fusion-head-equivalence-{args.fold}'
    cfg.dataset.cache_refresh = True
    cfg.run_dir = str(output_dir)
    cfg.out_dir = str(output_dir)
    configure_determinism(int(cfg.seed), bool(cfg.train.deterministic))

    with (output_dir / 'cache_build.log').open('w') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            loader_groups = create_loader_5()
    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model = create_model_gps().to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_state = payload['model_state']
    incompatible = model.load_state_dict(checkpoint_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError('strict checkpoint loading unexpectedly returned incompatible keys')
    predicted = evaluate(model, loader_groups, device)
    reference = pd.read_csv(args.reference_predictions)
    reference = reference.loc[:, ['source_index', 'split', 'target', 'y_true', 'y_pred']]
    merged = reference.merge(
        predicted, on=['source_index', 'split', 'target'], suffixes=('_reference', '_new'),
        how='outer', indicator=True,
    )
    only_reference = int((merged['_merge'] == 'left_only').sum())
    only_new = int((merged['_merge'] == 'right_only').sum())
    common = merged.loc[merged['_merge'] == 'both'].copy()
    # Historical CSVs are presentation-scale percentages, whereas the model
    # and checkpoint operate on normalized values.  Report both explicitly;
    # the strict <1e-6 gate applies to the latter.
    prediction_difference_percent = np.abs(common.y_pred_reference - common.y_pred_new)
    label_difference_percent = np.abs(common.y_true_reference - common.y_true_new)
    prediction_difference = prediction_difference_percent / 100.0
    label_difference = label_difference_percent / 100.0
    state_numel = int(sum(value.numel() for value in model.state_dict().values()))
    checkpoint_numel = int(sum(value.numel() for value in checkpoint_state.values()))
    report = {
        'config': str(args.config.resolve()),
        'checkpoint': str(args.checkpoint.resolve()),
        'reference_predictions': str(args.reference_predictions.resolve()),
        'fold': args.fold,
        'strict_load': True,
        'legacy_path_selected': bool(model.model.legacy_baseline),
        'model_parameter_numel': state_numel,
        'checkpoint_parameter_numel': checkpoint_numel,
        'state_dict_sha256': sha256_state(model.state_dict()),
        'checkpoint_state_sha256': sha256_state(checkpoint_state),
        'reference_rows': int(len(reference)),
        'new_rows': int(len(predicted)),
        'matched_rows': int(len(common)),
        'only_reference_rows': only_reference,
        'only_new_rows': only_new,
        'value_units': {
            'comparison_gate': 'normalized model output',
            'csv_presentation': 'percentage points',
        },
        'max_abs_prediction_difference_normalized': float(prediction_difference.max()) if len(common) else float('inf'),
        'max_abs_label_difference_normalized': float(label_difference.max()) if len(common) else float('inf'),
        'max_abs_prediction_difference_percentage_points': float(prediction_difference_percent.max()) if len(common) else float('inf'),
        'max_abs_label_difference_percentage_points': float(label_difference_percent.max()) if len(common) else float('inf'),
    }
    report['pass'] = bool(
        report['legacy_path_selected']
        and state_numel == checkpoint_numel
        and only_reference == 0 and only_new == 0
        and report['max_abs_prediction_difference_normalized'] < 1e-6
        and report['max_abs_label_difference_normalized'] < 1e-6
    )
    predicted.to_csv(output_dir / 'baseline_equivalence_predictions.csv', index=False)
    (output_dir / 'baseline_equivalence_test.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n')
    if not report['pass']:
        raise SystemExit(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
