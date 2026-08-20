#!/usr/bin/env python3
"""Infer ten normalized O12 later4 checkpoints on labelled feedback CSVs.

Each checkpoint's train-only per-target z-score scaler is applied in reverse
before metrics and plots are written. Feedback labels are never fed to models.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphgps  # noqa: F401,E402
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402
from scripts.diagnostics.predict_o12_o22_feedback_ensemble import (  # noqa: E402
    MORDRED_11, SMILES_COLUMNS, build_feedback_mordred_lookup,
)


TARGETS = [
    'Aerosolization_Efficiency', 'mRNA_Recovery_Efficiency',
    'Norm_before', 'Norm_after',
]
ALL_LABELS = ['EE_before', 'EE_after', *TARGETS]
REQUIRED_COLUMNS = {'ID', *SMILES_COLUMNS, 'mol%_IL', 'mol%_HL', 'mol%_Chol',
                    'mol%_PEG', 'mol%_Fifth', *TARGETS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stage(frame: pd.DataFrame, source: Path, output: Path) -> tuple[pd.DataFrame, Path]:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f'{source} misses required columns: {sorted(missing)}')
    original = frame.copy()
    original['ID'] = original['ID'].astype(str)
    if len(original) < 3 or original.ID.isna().any() or original.ID.duplicated().any():
        raise ValueError(f'{source} requires at least three unique non-null IDs.')
    if original[TARGETS].isna().any().any():
        raise ValueError(f'{source} has missing later4 labels.')
    model_input = original.copy()
    for column in ALL_LABELS:
        model_input[column] = 0.0
    path = output / 'feedback_model_input_labels_zeroed.csv'
    model_input.to_csv(path, index=False)
    return original, path


def write_manifest(frame: pd.DataFrame, output: Path) -> Path:
    split = np.full(len(frame), 'test', dtype=object)
    split[0], split[1] = 'train', 'val'
    path = output / 'feedback_loader_manifest.csv'
    pd.DataFrame({'ID': frame.ID, 'split': split,
                  'split_order': np.arange(len(frame), dtype=int)}).to_csv(path, index=False)
    return path


def build_context(config_path: Path, model_input: Path, manifest: Path,
                  cache_dir: Path, mordred_lookup: Path):
    set_cfg_gps(cfg)
    cfg.set_new_allowed(True)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(config_path.resolve()), opts=[]))
    if cfg.model.type != 'OneHotEmbedGPS' or int(cfg.property_num) != len(TARGETS):
        raise RuntimeError(f'Expected a four-target OneHotEmbedGPS config: {config_path}')
    if list(getattr(cfg, 'multi_task_target_indices', [])) != [2, 3, 4, 5]:
        raise RuntimeError(f'Checkpoint config is not the later4 target ordering: {config_path}')
    cfg.read_csv = str(model_input.resolve())
    cfg.dataset.dir = str(cache_dir.resolve())
    cfg.dataset.cache_tag = f'o12_later4_feedback_{config_path.parent.name}'
    cfg.dataset.cache_refresh = True
    cfg.dataset.diagnostic_split_path = str(manifest.resolve())
    cfg.dataset.diagnostic_id_column = 'ID'
    cfg.dataset.diagnostic_manifest_id_column = 'ID'
    cfg.mordred_feature_path = str(mordred_lookup.resolve())
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / 'cache_build.log').open('w', encoding='utf-8') as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            loaders = create_loader_5()
    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    return create_model_gps().to(device), loaders, device


def predict(model: torch.nn.Module, loaders, device: torch.device, rows: int) -> np.ndarray:
    collected: list[tuple[int, np.ndarray]] = []
    model.eval()
    with torch.no_grad():
        for loader_index in range(3):
            for batches in zip(*[group[loader_index] for group in loaders]):
                for suffix, batch in zip(('', '_2', '_3', '_4', '_5'), batches):
                    batch.split = 'feedback' + suffix
                    batch.to(device)
                output, _ = model(*batches)
                values = output.detach().cpu().reshape(-1, len(TARGETS)).numpy()
                source = batches[0].sample_uid.detach().cpu().numpy().reshape(-1)
                collected.extend((int(index), value) for index, value in zip(source, values))
    collected.sort(key=lambda item: item[0])
    if [index for index, _ in collected] != list(range(rows)):
        raise RuntimeError('Predicted rows do not align with feedback source rows.')
    return np.vstack([value for _, value in collected])


def inverse_normalization(values: np.ndarray, scaler: dict[str, object],
                          target_transform: str, target_scales: list[float]) -> np.ndarray:
    restored = values.astype(float, copy=True)
    if scaler.get('type') != 'zscore_train_only':
        raise RuntimeError(f'Expected z-score target scaler, got {scaler.get("type")!r}')
    restored = restored * np.asarray(scaler['std'], dtype=float) + np.asarray(scaler['mean'], dtype=float)
    if target_transform == 'log1p':
        restored = np.expm1(restored).clip(min=0)
    elif target_transform != 'identity':
        raise RuntimeError(f'Unsupported checkpoint target transform: {target_transform}')
    return restored * np.asarray(target_scales, dtype=float)


def calculate_metrics(frame: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    rows = []
    for index, target in enumerate(TARGETS):
        truth = frame[target].to_numpy(float)
        prediction = values[:, index]
        rows.append({'target': target, 'n': len(frame),
                     'mae': mean_absolute_error(truth, prediction),
                     'rmse': mean_squared_error(truth, prediction) ** .5,
                     'r2': r2_score(truth, prediction) if np.std(truth) else np.nan})
    return pd.DataFrame(rows)


def scatter(frame: pd.DataFrame, values: np.ndarray, metric: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    colors = ['#4c78a8', '#f58518', '#54a24b', '#e45756']
    for axis, target, color in zip(axes.flat, TARGETS, colors):
        truth, prediction = frame[target].to_numpy(float), values[:, TARGETS.index(target)]
        lower, upper = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
        padding = max((upper - lower) * .06, .1)
        limits = (lower - padding, upper + padding)
        row = metric.loc[metric.target.eq(target)].iloc[0]
        axis.scatter(truth, prediction, s=34, alpha=.82, color=color, edgecolor='#222', linewidth=.35)
        axis.plot(limits, limits, '--', color='#d62728', linewidth=1.35, label='y = x')
        axis.set(xlabel='True value', ylabel='Predicted value', xlim=limits, ylim=limits)
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=.25)
        axis.legend(loc='upper left', fontsize=8)
        axis.set_title(f'{target}\nMAE = {row.mae:.3f}, R² = {row.r2:.3f}')
    figure.suptitle('O12 later4, ten-checkpoint mean prediction', fontsize=13)
    figure.savefig(output / 'later4_true_vs_pred_ensemble.png', dpi=180, bbox_inches='tight')
    figure.savefig(output / 'later4_true_vs_pred_ensemble.pdf', bbox_inches='tight')
    plt.close(figure)


def run_dataset(source: Path, model_root: Path, output_root: Path, metadata: Path) -> dict[str, object]:
    output = output_root / source.stem
    output.mkdir(parents=True, exist_ok=True)
    original, staged_path = stage(pd.read_csv(source, dtype={'ID': str}), source, output)
    manifest = write_manifest(original, output)
    lookup = output / 'mordred11_feedback_standardized.csv'
    mordred_summary = build_feedback_mordred_lookup(original, metadata, lookup)
    specs, signature = [], None
    for seed in range(100, 110):
        run_dir = model_root / f'O12_later4_split{seed}'
        checkpoint = run_dir / 'checkpoints' / 'selected_best.pt'
        config, settings_path = run_dir / 'effective_config.yaml', run_dir / 'run_settings.json'
        scaler_path = run_dir / 'target_scaler.json'
        if not all(path.is_file() for path in (checkpoint, config, settings_path, scaler_path)):
            raise FileNotFoundError(f'Incomplete later4 checkpoint: {run_dir}')
        settings = json.loads(settings_path.read_text(encoding='utf-8'))
        scaler = json.loads(scaler_path.read_text(encoding='utf-8'))
        if settings.get('targets') != TARGETS or settings.get('target_normalization') != 'zscore':
            raise RuntimeError(f'Not a z-score later4 checkpoint: {run_dir}')
        current = {'target_scales': settings.get('target_scales'),
                   'target_transform': settings.get('target_transform'),
                   'model_type': settings.get('model_type'),
                   'gps_layers': settings.get('gps_layers'),
                   'graph_hidden_dim': settings.get('graph_hidden_dim'),
                   'component_vocab_sizes': settings.get('component_vocab_sizes')}
        if signature is None:
            signature = current
        elif current != signature:
            raise RuntimeError(f'Incompatible later4 ensemble member: {run_dir}')
        specs.append((seed, checkpoint, config, scaler))
    model, loaders, device = build_context(specs[0][2], staged_path, manifest, output / 'cache', lookup)
    all_predictions, checkpoint_metrics, long_rows = [], [], []
    for seed, checkpoint_path, _, scaler in specs:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        normalized = predict(model, loaders, device, len(original))
        values = inverse_normalization(normalized, scaler, signature['target_transform'], signature['target_scales'])
        all_predictions.append(values)
        metric = calculate_metrics(original, values)
        metric.insert(0, 'split_seed', seed)
        checkpoint_metrics.append(metric)
        for index, target in enumerate(TARGETS):
            long_rows.extend({'sample_id': sample_id, 'split_seed': seed, 'target': target,
                              'y_true': float(truth), 'y_pred': float(value)}
                             for sample_id, truth, value in zip(
                                 original.ID, original[target], values[:, index]))
    stacked = np.stack(all_predictions)
    mean, std = stacked.mean(axis=0), stacked.std(axis=0, ddof=0)
    ensemble = original.copy()
    for index, target in enumerate(TARGETS):
        ensemble[f'pred_{target}_mean'] = mean[:, index]
        ensemble[f'pred_{target}_std_10models'] = std[:, index]
    metric = calculate_metrics(original, mean)
    ensemble.to_csv(output / 'ensemble_mean_predictions.csv', index=False)
    metric.to_csv(output / 'metrics_ensemble.csv', index=False)
    pd.concat(checkpoint_metrics, ignore_index=True).to_csv(output / 'metrics_by_checkpoint.csv', index=False)
    pd.DataFrame(long_rows).to_csv(output / 'predictions_by_checkpoint.csv', index=False)
    scatter(original, mean, metric, output)
    (output / 'provenance.json').write_text(json.dumps({
        'feedback_csv': str(source.resolve()), 'feedback_sha256': sha256(source),
        'rows': len(original), 'targets': TARGETS,
        'ensemble': 'unweighted arithmetic mean over O12_later4_split100...109',
        'label_use': 'labels replaced by zero in model input and used only for metrics/plots',
        'target_scaling': 'each checkpoint train-only z-score scaler inverse-transformed before averaging',
        'model_signature': signature, 'mordred_features': MORDRED_11,
        'mordred_lookup': str(lookup.resolve()), 'mordred_summary': mordred_summary,
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return {'dataset': source.name, 'rows': len(original),
            **{f'{row.target}_mae': row.mae for row in metric.itertuples(index=False)},
            **{f'{row.target}_r2': row.r2 for row in metric.itertuples(index=False)},
            'output': str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--feedback-files', type=Path, nargs='+', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--mordred-metadata', type=Path,
                        default=ROOT / 'results/input_graphgps_optimization/features/mordred11_train_standardized.json')
    args = parser.parse_args()
    if not args.mordred_metadata.is_file():
        raise FileNotFoundError(f'Missing Mordred scaler metadata: {args.mordred_metadata}')
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = [run_dataset(path.resolve(), args.model_root.resolve(), args.output_root.resolve(),
                           args.mordred_metadata.resolve()) for path in args.feedback_files]
    pd.DataFrame(summary).to_csv(args.output_root / 'run_summary.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
