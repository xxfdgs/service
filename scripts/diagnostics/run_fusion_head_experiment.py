#!/usr/bin/env python3
"""Resumable, validation-only training for GraphGPS fusion/head variants.

The runner never reads the outer test split unless ``--include-test`` is
explicitly supplied.  It keeps the encoder, descriptors, and hidden width from
the supplied fold configuration and changes only the registered fusion/head
interface in ``GPSDoubleModel_multi4_cat_v0``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphgps  # noqa: F401,E402
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from graphgps.determinism import configure_determinism  # noqa: E402
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402
from torch_geometric.graphgym.optim import (OptimizerConfig, create_optimizer,
                                             create_scheduler)  # noqa: E402


TARGETS = [
    'EE_before', 'EE_after', 'Aerosolization_Efficiency',
    'mRNA_Recovery_Efficiency',
]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
BRANCHES = ('graph', 'descriptor', 'formula')


def property_losses(prediction, labels):
    return torch.stack([
        torch.mean(torch.abs(prediction[index::4] - labels[index::4]))
        for index in range(4)
    ])


def prepare_batches(items, split, device):
    for suffix, batch in zip(('', '_2', '_3', '_4', '_5'), items):
        batch.split = split + suffix
        batch.to(device)
    return items


def tensor_stats(value):
    value = value.detach().float()
    flat = value.reshape(value.shape[0], -1) if value.ndim else value.view(1, 1)
    finite = flat[torch.isfinite(flat)]
    if not finite.numel():
        return dict(mean=math.nan, std=math.nan, min=math.nan, max=math.nan,
                    norm=math.nan, zero_fraction=math.nan,
                    saturated_fraction=math.nan, nan_count=int(torch.isnan(flat).sum()),
                    inf_count=int(torch.isinf(flat).sum()), sample_distance=math.nan,
                    effective_dimension_fraction=math.nan,
                    near_zero_variance_dimension_fraction=math.nan,
                    below_005_fraction=math.nan, above_095_fraction=math.nan)
    dimension_std = flat.std(dim=0, unbiased=False)
    return dict(
        mean=float(finite.mean()), std=float(finite.std(unbiased=False)),
        min=float(finite.min()), max=float(finite.max()),
        norm=float(torch.linalg.vector_norm(finite)),
        zero_fraction=float((finite == 0).float().mean()),
        saturated_fraction=float((finite.abs() >= 10).float().mean()),
        nan_count=int(torch.isnan(flat).sum()), inf_count=int(torch.isinf(flat).sum()),
        sample_distance=(float(torch.pdist(flat).mean()) if flat.shape[0] > 1 else math.nan),
        effective_dimension_fraction=float((dimension_std > 1e-6).float().mean()),
        near_zero_variance_dimension_fraction=float((dimension_std < 1e-6).float().mean()),
        below_005_fraction=float((finite < .05).float().mean()),
        above_095_fraction=float((finite > .95).float().mean()),
    )


def safe_correlation(function, y, p):
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        return math.nan
    return float(function(y, p).statistic)


def safe_slope(y, p):
    if len(y) < 2 or np.std(y) == 0:
        return math.nan
    return float(np.polyfit(y, p, 1)[0])


def metric_rows(prediction, labels, split, epoch, loss, lr, best_epoch, best_loss,
                early_counter, architecture):
    rows = []
    for index, target in enumerate(TARGETS):
        y = labels[:, index] * 100.0
        p = prediction[:, index] * 100.0
        valid = np.isfinite(y) & np.isfinite(p)
        y, p = y[valid], p[valid]
        target_std = float(np.std(y, ddof=1)) if len(y) > 1 else math.nan
        pred_std = float(np.std(p, ddof=1)) if len(p) > 1 else math.nan
        rows.append({
            'epoch': epoch, 'split': split, 'target': target, 'architecture': architecture,
            'n_valid': int(len(y)), 'mae': float(mean_absolute_error(y, p)),
            'rmse': float(math.sqrt(mean_squared_error(y, p))),
            'r2': float(r2_score(y, p)) if len(y) > 1 and np.std(y) else math.nan,
            'pearson': safe_correlation(pearsonr, y, p),
            'spearman': safe_correlation(spearmanr, y, p),
            'residual_mean': float(np.mean(p - y)),
            'calibration_slope': safe_slope(y, p),
            'target_mean': float(np.mean(y)), 'target_std': target_std,
            'prediction_mean': float(np.mean(p)), 'prediction_std': pred_std,
            'std_ratio': pred_std / target_std if target_std else math.nan,
            'total_loss_normalized': float(loss), 'lr': float(lr),
            'best_epoch_candidate': best_epoch,
            'best_validation_loss_candidate': float(best_loss),
            'early_stopping_counter': early_counter,
        })
    return rows


def prediction_frame(prediction, labels, source_indices, manifest, split, epoch,
                     checkpoint, architecture):
    sample_ids = dict(zip(
        manifest.original_row_index.astype(int), manifest.sample_id.astype(str)))
    rows = []
    for row_index, source_index in enumerate(source_indices):
        for target_index, target in enumerate(TARGETS):
            rows.append({
                'sample_id': sample_ids[int(source_index)],
                'source_index': int(source_index), 'split': split, 'target': target,
                'epoch': int(epoch), 'checkpoint': checkpoint,
                'architecture': architecture,
                'y_true': float(labels[row_index, target_index] * 100.0),
                'y_pred': float(prediction[row_index, target_index] * 100.0),
            })
    return pd.DataFrame(rows)


def append_rows(rows, path, columns):
    if not rows:
        return
    pd.DataFrame(rows).reindex(columns=columns).to_csv(
        path, mode='a' if path.exists() else 'w', header=not path.exists(), index=False)


def initialise_csvs(run_dir):
    schemas = {
        'epoch_metrics.csv': ['epoch', 'split', 'target', 'architecture', 'n_valid', 'mae', 'rmse', 'r2', 'pearson', 'spearman', 'residual_mean', 'calibration_slope', 'target_mean', 'target_std', 'prediction_mean', 'prediction_std', 'std_ratio', 'total_loss_normalized', 'lr', 'best_epoch_candidate', 'best_validation_loss_candidate', 'early_stopping_counter'],
        'branch_statistics.csv': ['epoch', 'split', 'branch', 'kind', 'mean', 'std', 'min', 'max', 'norm', 'zero_fraction', 'saturated_fraction', 'nan_count', 'inf_count', 'sample_distance', 'effective_dimension_fraction', 'near_zero_variance_dimension_fraction', 'below_005_fraction', 'above_095_fraction'],
        'fusion_statistics.csv': ['epoch', 'split', 'fusion_type', 'branch', 'target', 'weight_mean', 'weight_std', 'weight_min', 'weight_max', 'entropy_mean', 'entropy_min', 'weight_above_095_fraction', 'weight_above_098_fraction'],
        'gate_statistics.csv': ['epoch', 'split', 'gate_type', 'mean', 'std', 'min', 'max', 'norm', 'zero_fraction', 'saturated_fraction', 'nan_count', 'inf_count', 'sample_distance', 'effective_dimension_fraction', 'near_zero_variance_dimension_fraction', 'below_005_fraction', 'above_095_fraction'],
        'head_statistics.csv': ['epoch', 'split', 'location', 'mean', 'std', 'min', 'max', 'norm', 'zero_fraction', 'saturated_fraction', 'nan_count', 'inf_count', 'sample_distance', 'effective_dimension_fraction', 'near_zero_variance_dimension_fraction', 'below_005_fraction', 'above_095_fraction'],
        'gradient_statistics.csv': ['epoch', 'split', 'module', 'parameter_count', 'grad_norm', 'grad_nan_count', 'grad_inf_count'],
        'collapse_events.csv': ['event', 'first_epoch', 'split', 'target_or_branch', 'observed', 'evidence'],
    }
    for name, columns in schemas.items():
        path = run_dir / name
        if not path.exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)
        else:
            # A runner update must not strand an already-resumable experiment
            # behind an older diagnostic schema.  Existing observations retain
            # their values; newly introduced diagnostics are explicitly NaN.
            existing = pd.read_csv(path)
            missing = [column for column in columns if column not in existing]
            if missing:
                for column in missing:
                    existing[column] = math.nan
                existing.reindex(columns=columns).to_csv(path, index=False)
    return schemas


def module_gradient_rows(core, epoch):
    # The input-only comparison also includes OneHotEmbedGPS.  It has no
    # legacy fusion/head diagnostic attributes, but its gradients still need
    # to be recorded rather than making the shared runner fail.
    if not hasattr(core, 'legacy_baseline'):
        modules = {'model': core}
        rows = []
        for name, module in modules.items():
            grads = [parameter.grad.detach().float().reshape(-1) for parameter in module.parameters()
                     if parameter.grad is not None]
            grad = torch.cat(grads) if grads else torch.empty(0)
            rows.append({
                'epoch': epoch, 'split': 'train', 'module': name,
                'parameter_count': int(sum(parameter.numel() for parameter in module.parameters())),
                'grad_norm': float(torch.linalg.vector_norm(grad)) if grad.numel() else math.nan,
                'grad_nan_count': int(torch.isnan(grad).sum()) if grad.numel() else 0,
                'grad_inf_count': int(torch.isinf(grad).sum()) if grad.numel() else 0,
            })
        return rows
    modules = {'encoder': core.gnn, 'ratio_encoder': core.ratio_encoder,
               'additive_delta': core.additive_delta_head}
    if core.legacy_baseline or core.head_ablation_with_legacy_fusion:
        modules.update({'legacy_main': core.FC_layers, 'legacy_direct': core.FC_layers_2mlp,
                        'legacy_middle': core.FC_layers_midle_mlp,
                        'legacy_gate': core.branch_weight_mlp})
    if core.head_ablation_with_legacy_fusion:
        modules['head'] = core.redesign_head
    else:
        if not core.legacy_baseline:
            modules.update({'fusion': core.redesign_fusion, 'head': core.redesign_head})
    rows = []
    for name, module in modules.items():
        grads = [parameter.grad.detach().float().reshape(-1) for parameter in module.parameters()
                 if parameter.grad is not None]
        grad = torch.cat(grads) if grads else torch.empty(0)
        rows.append({
            'epoch': epoch, 'split': 'train', 'module': name,
            'parameter_count': int(sum(parameter.numel() for parameter in module.parameters())),
            'grad_norm': float(torch.linalg.vector_norm(grad)) if grad.numel() else math.nan,
            'grad_nan_count': int(torch.isnan(grad).sum()) if grad.numel() else 0,
            'grad_inf_count': int(torch.isinf(grad).sum()) if grad.numel() else 0,
        })
    return rows


def diagnostic_rows(core, epoch, split):
    diag = getattr(core, 'last_diagnostics', {})
    branch_rows, fusion_rows, gate_rows, head_rows = [], [], [], []
    field_map = {
        'graph': 'graph_branch' if 'graph_branch' in diag else 'graph_input',
        'descriptor': 'descriptor_branch' if 'descriptor_branch' in diag else 'descriptor_input',
        'formula': 'formula_branch' if 'formula_branch' in diag else 'formula_input',
    }
    for branch, field in field_map.items():
        if field in diag:
            branch_rows.append({'epoch': epoch, 'split': split, 'branch': branch,
                                'kind': field, **tensor_stats(diag[field])})
    weights = diag.get('fusion_weights', diag.get('legacy_branch_weights'))
    if weights is not None:
        if weights.ndim == 2:
            weights = weights.unsqueeze(1)
        entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=-1)
        labels = TARGETS if weights.shape[1] == len(TARGETS) else ['all']
        for target_index, target in enumerate(labels):
            idx = target_index if weights.shape[1] == len(TARGETS) else 0
            for branch_index, branch in enumerate(('main', 'direct', 'middle')):
                value = weights[:, idx, branch_index]
                fusion_rows.append({
                    'epoch': epoch, 'split': split, 'fusion_type': core.fusion_type,
                    'branch': branch, 'target': target,
                    'weight_mean': float(value.mean()), 'weight_std': float(value.std(unbiased=False)),
                    'weight_min': float(value.min()), 'weight_max': float(value.max()),
                    'entropy_mean': float(entropy[:, idx].mean()),
                    'entropy_min': float(entropy[:, idx].min()),
                    'weight_above_095_fraction': float((value > .95).float().mean()),
                    'weight_above_098_fraction': float((value > .98).float().mean()),
                })
    if 'feature_gate' in diag:
        gate_rows.append({'epoch': epoch, 'split': split, 'gate_type': 'feature_gate',
                          **tensor_stats(diag['feature_gate'])})
    if weights is not None:
        gate_rows.append({'epoch': epoch, 'split': split, 'gate_type': 'softmax_weights',
                          **tensor_stats(weights)})
    for location, field in (('fused', 'fused'), ('head_input', 'head_input'),
                            ('head_output', 'head_output'), ('legacy_prediction', 'legacy_branch_predictions')):
        if field in diag:
            head_rows.append({'epoch': epoch, 'split': split, 'location': location,
                              **tensor_stats(diag[field])})
    return branch_rows, fusion_rows, gate_rows, head_rows


def evaluate(model, loaders, split, device, epoch, collect_diagnostics=False):
    model.eval()
    predictions, labels, sources = [], [], []
    diagnostic = ([], [], [], [])
    with torch.no_grad():
        for batch_index, batches in enumerate(zip(*[group[{'train': 0, 'val': 1, 'test': 2}[split]] for group in loaders])):
            batches = prepare_batches(list(batches), split, device)
            pred, label = model(*batches)
            if collect_diagnostics and batch_index == 0:
                diagnostic = diagnostic_rows(model.model, epoch, split)
            predictions.append(pred.detach().cpu().reshape(-1, 4).numpy())
            labels.append(label.detach().cpu().reshape(-1, 4).numpy())
            sources.append(batches[0].sample_uid.detach().cpu().numpy().reshape(-1))
    prediction, label = np.vstack(predictions), np.vstack(labels)
    loss = float(np.mean(np.abs(prediction - label), axis=0).sum())
    return prediction, label, np.concatenate(sources), loss, diagnostic


def copy_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def collapse_events(run_dir):
    metrics = pd.read_csv(run_dir / 'epoch_metrics.csv')
    fusion = pd.read_csv(run_dir / 'fusion_statistics.csv')
    gates = pd.read_csv(run_dir / 'gate_statistics.csv')
    gradients = pd.read_csv(run_dir / 'gradient_statistics.csv')
    events = []
    def first(frame, condition, event, split='any', target='any', evidence=''):
        selected = frame.loc[condition]
        events.append({
            'event': event,
            'first_epoch': int(selected.epoch.min()) if len(selected) else math.nan,
            'split': split, 'target_or_branch': target, 'observed': bool(len(selected)),
            'evidence': evidence,
        })
    first(metrics, metrics.std_ratio < .10, 'prediction_std_below_target_10pct', evidence='prediction std / target std < 0.10')
    if not fusion.empty:
        first(fusion, fusion.weight_max > .98, 'softmax_weight_above_0.98', evidence='a softmax branch weight > 0.98')
        first(fusion, fusion.entropy_mean < .05, 'softmax_entropy_below_0.05', evidence='mean branch entropy < 0.05')
    if not gates.empty:
        first(gates, gates.saturated_fraction > .95, 'gate_saturated', evidence='>95% gate values at absolute magnitude >=10')
    if not gradients.empty:
        first(gradients, gradients.grad_norm < 1e-8, 'gradient_near_zero', evidence='module gradient norm < 1e-8')
        first(gradients, gradients.grad_nan_count.gt(0) | gradients.grad_inf_count.gt(0), 'gradient_nonfinite', evidence='NaN or inf gradient')
    pd.DataFrame(events).to_csv(run_dir / 'collapse_events.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--fold', required=True)
    parser.add_argument('--group', choices=('A', 'B'), required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--fusion-type', choices=('softmax_sum', 'concat', 'concat_mlp', 'residual', 'gated_concat'), required=True)
    parser.add_argument('--head-type', choices=('baseline', 'linear', 'two_layer', 'residual_head', 'target_specific'), required=True)
    parser.add_argument('--execution-max-epochs', type=int, default=120)
    parser.add_argument('--base-lr', type=float, default=None,
                        help='Optional input-only experiment override for optim.base_lr.')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Optional input-only experiment override for optim.weight_decay.')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Optional input-only experiment override for train.batch_size.')
    parser.add_argument('--warmup-epochs', type=int, default=None,
                        help='Optional input-only experiment override for optim.num_warmup_epochs.')
    parser.add_argument('--head-hidden-dim', type=int, default=None,
                        help='Optional width for the redesigned prediction head.')
    parser.add_argument('--head-dropout', type=float, default=None,
                        help='Optional dropout for the redesigned prediction head.')
    parser.add_argument('--fusion-hidden-dim', type=int, default=None,
                        help='Optional width for the redesigned fusion module.')
    parser.add_argument('--fusion-dropout', type=float, default=None,
                        help='Optional dropout for the redesigned fusion module.')
    parser.add_argument('--model-type', type=str, default=None,
                        help='Optional registered GraphGPS network type override.')
    parser.add_argument('--use-mordred-features', action='store_true',
                        help='Enable a supplied input-only Mordred lookup.')
    parser.add_argument('--mordred-feature-path', type=Path, default=None)
    parser.add_argument('--mordred-feature-dim', type=int, default=None)
    parser.add_argument('--mordred-fifth-only', action='store_true')
    parser.add_argument('--coarse-grain-enable', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--restart-incomplete', action='store_true',
                        help='Reuse an incomplete run directory that has no resume state.')
    parser.add_argument('--chunk-epochs', type=int, default=None)
    parser.add_argument('--include-test', action='store_true')
    args = parser.parse_args()
    # Group A deliberately preserves the historical three-branch fusion and
    # replaces only its main prediction head.  That forward path has no
    # descriptor input, so accepting Mordred flags here would create a
    # misleading "descriptor" experiment that never consumes descriptors.
    if args.use_mordred_features and args.group == 'A':
        raise ValueError(
            'Mordred experiments must use group B (a descriptor-aware fusion); '
            'group A changes only the legacy prediction head.')
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and not args.resume and not args.restart_incomplete:
        raise FileExistsError(f'Refusing to overwrite existing run: {run_dir}')
    if args.restart_incomplete and (run_dir / 'resume_state.pt').exists():
        raise RuntimeError('restart-incomplete is unsafe once a resume state exists; use --resume.')
    for directory in (run_dir, run_dir / 'cache', run_dir / 'checkpoints'):
        directory.mkdir(parents=True, exist_ok=True)
    schemas = initialise_csvs(run_dir)

    set_cfg_gps(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(args.config.resolve()), opts=[]))
    cfg.model.fusion_type = args.fusion_type
    cfg.model.head_type = args.head_type
    cfg.model.architecture_name = ('legacy_baseline' if args.candidate == 'A0'
                                   else f'{args.group}_{args.candidate}_{args.fusion_type}_{args.head_type}')
    cfg.model.target_specific_heads = args.head_type == 'target_specific'
    cfg.model.validate_redesign_inputs = args.candidate != 'A0'
    for key, value in {
        'base_lr': args.base_lr,
        'weight_decay': args.weight_decay,
        'num_warmup_epochs': args.warmup_epochs,
    }.items():
        if value is not None:
            setattr(cfg.optim, key, value)
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.model_type is not None:
        cfg.model.type = args.model_type
    if args.use_mordred_features:
        if args.mordred_feature_path is None or args.mordred_feature_dim is None:
            raise ValueError('--use-mordred-features requires lookup path and dimension.')
        cfg.use_mordred_features = True
        cfg.mordred_feature_path = str(args.mordred_feature_path.resolve())
        cfg.mordred_feature_dim = int(args.mordred_feature_dim)
        cfg.mordred_fifth_only = bool(args.mordred_fifth_only)
    if args.coarse_grain_enable:
        cfg.coarse_grain_enable = True
    for key, value in {
        'head_hidden_dim': args.head_hidden_dim,
        'head_dropout': args.head_dropout,
        'fusion_hidden_dim': args.fusion_hidden_dim,
        'fusion_dropout': args.fusion_dropout,
    }.items():
        if value is not None:
            setattr(cfg.model, key, value)
    cfg.dataset.dir = str(run_dir / 'cache')
    cfg.dataset.cache_tag = f'fusion-head-{args.fold}-{args.group}-{args.candidate}'
    cfg.dataset.cache_refresh = not args.resume
    cfg.run_dir = str(run_dir)
    cfg.out_dir = str(run_dir)
    execution_max_epochs = int(args.execution_max_epochs)
    if execution_max_epochs <= 0 or execution_max_epochs > int(cfg.optim.max_epoch):
        raise ValueError('execution-max-epochs must be in [1, configured max_epoch]')
    configure_determinism(int(cfg.seed), bool(cfg.train.deterministic))
    if not args.resume:
        shutil.copy2(args.config, run_dir / 'source_config.yaml')
        with (run_dir / 'effective_config.yaml').open('w') as stream:
            cfg.dump(stream=stream)
        (run_dir / 'run_settings.json').write_text(json.dumps({
            'fold': args.fold, 'group': args.group, 'candidate': args.candidate,
            'fusion_type': args.fusion_type, 'head_type': args.head_type,
            'architecture_name': cfg.model.architecture_name,
            'execution_max_epochs': execution_max_epochs,
            'base_lr': cfg.optim.base_lr,
            'weight_decay': cfg.optim.weight_decay,
            'batch_size': cfg.train.batch_size,
            'num_warmup_epochs': cfg.optim.num_warmup_epochs,
            'head_hidden_dim': cfg.model.head_hidden_dim,
            'head_dropout': cfg.model.head_dropout,
            'fusion_hidden_dim': cfg.model.fusion_hidden_dim,
            'fusion_dropout': cfg.model.fusion_dropout,
            'model_type': cfg.model.type,
            'use_mordred_features': cfg.use_mordred_features,
            'mordred_feature_path': cfg.mordred_feature_path,
            'mordred_feature_dim': cfg.mordred_feature_dim,
            'mordred_fifth_only': cfg.mordred_fifth_only,
            'coarse_grain_enable': cfg.coarse_grain_enable,
            'outer_test_read_during_selection': False,
            'source_config': str(args.config.resolve()),
        }, indent=2) + '\n')
    elif not (run_dir / 'resume_state.pt').exists():
        raise FileNotFoundError('resume requested but resume_state.pt is absent')

    with (run_dir / 'cache_build.log').open('w') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            loaders = create_loader_5()
    # Some data-dependent settings (notably input-derived OneHot component
    # vocabulary sizes) are established while building the cache. Persist the
    # final runtime configuration instead of leaving only pre-loader defaults.
    with (run_dir / 'effective_config.yaml').open('w') as stream:
        cfg.dump(stream=stream)
    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model = create_model_gps().to(device)
    optimizer = create_optimizer(model.parameters(), OptimizerConfig(
        optimizer=cfg.optim.optimizer, base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay, momentum=cfg.optim.momentum))
    scheduler = create_scheduler(optimizer, ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler, steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs, train_mode=cfg.train.mode,
        eval_period=cfg.train.eval_period))
    manifest = pd.read_csv(cfg.train.manifest_path, dtype={'sample_id': str})
    checkpoint_metadata = {
        'fusion_type': args.fusion_type,
        'head_type': args.head_type,
        'architecture_name': cfg.model.architecture_name,
        'architecture_hash': hashlib.sha256(
            f'{cfg.model.architecture_name}|{args.fusion_type}|{args.head_type}'.encode()).hexdigest(),
        'manifest_path': str(Path(cfg.train.manifest_path).resolve()),
        'manifest_hash': file_sha256(cfg.train.manifest_path),
        'feature_hash': file_sha256(cfg.mordred_feature_path) if cfg.use_mordred_features else None,
        'config_hash': file_sha256(args.config),
    }
    start_epoch, best_loss, best_epoch, best_state = 0, math.inf, None, None
    early_reference, early_counter = math.inf, 0
    if args.resume:
        state = torch.load(run_dir / 'resume_state.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        start_epoch, best_loss, best_epoch, best_state = (state['next_epoch'], state['best_loss'],
                                                           state['best_epoch'], state['best_state'])
        early_reference, early_counter = state['early_reference'], state['early_counter']
        restore_rng_state(state['rng_state'])

    completed, last_epoch = False, start_epoch - 1
    for epoch in range(start_epoch, execution_max_epochs):
        model.train()
        train_pred, train_label, train_source = [], [], []
        train_diag = ([], [], [], [])
        train_grad = []
        for batch_index, batches in enumerate(zip(*[group[0] for group in loaders])):
            batches = prepare_batches(list(batches), 'train', device)
            optimizer.zero_grad()
            pred, label = model(*batches)
            loss = property_losses(pred, label).sum()
            loss.backward()
            if batch_index == 0:
                train_diag = diagnostic_rows(model.model, epoch, 'train')
                train_grad = module_gradient_rows(model.model, epoch)
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_pred.append(pred.detach().cpu().reshape(-1, 4).numpy())
            train_label.append(label.detach().cpu().reshape(-1, 4).numpy())
            train_source.append(batches[0].sample_uid.detach().cpu().numpy().reshape(-1))
        train_pred, train_label = np.vstack(train_pred), np.vstack(train_label)
        train_source = np.concatenate(train_source)
        train_loss = float(np.mean(np.abs(train_pred - train_label), axis=0).sum())
        val_pred, val_label, val_source, val_loss, val_diag = evaluate(
            model, loaders, 'val', device, epoch, collect_diagnostics=True)
        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, copy_state(model)
            torch.save({'model_state': best_state, 'epoch': epoch,
                        'validation_loss': val_loss, **checkpoint_metadata},
                       run_dir / 'checkpoints' / f'best_candidate_epoch_{epoch}.pt')
        if early_reference == math.inf:
            early_reference, early_counter = val_loss, 0
        elif val_loss < early_reference - float(cfg.train.early_stop_min_delta):
            early_reference, early_counter = val_loss, 0
        else:
            early_counter += 1
        lr = float(scheduler.get_last_lr()[0])
        metric = metric_rows(train_pred, train_label, 'train', epoch, train_loss, lr,
                             best_epoch, best_loss, early_counter, cfg.model.architecture_name)
        metric += metric_rows(val_pred, val_label, 'val', epoch, val_loss, lr,
                              best_epoch, best_loss, early_counter, cfg.model.architecture_name)
        append_rows(metric, run_dir / 'epoch_metrics.csv', schemas['epoch_metrics.csv'])
        for rows, name, schema in zip(
                train_diag + val_diag + (train_grad,),
                ('branch_statistics.csv', 'fusion_statistics.csv', 'gate_statistics.csv',
                 'head_statistics.csv', 'branch_statistics.csv', 'fusion_statistics.csv',
                 'gate_statistics.csv', 'head_statistics.csv', 'gradient_statistics.csv'),
                (schemas['branch_statistics.csv'], schemas['fusion_statistics.csv'], schemas['gate_statistics.csv'],
                 schemas['head_statistics.csv'], schemas['branch_statistics.csv'], schemas['fusion_statistics.csv'],
                 schemas['gate_statistics.csv'], schemas['head_statistics.csv'], schemas['gradient_statistics.csv'])):
            append_rows(rows, run_dir / name, schema)
        scheduler.step()
        torch.save({
            'next_epoch': epoch + 1, 'model_state': copy_state(model),
            'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
            'best_loss': best_loss, 'best_epoch': best_epoch, 'best_state': best_state,
            'early_reference': early_reference, 'early_counter': early_counter,
            'rng_state': save_rng_state(),
        }, run_dir / 'resume_state.pt')
        print(json.dumps({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                          'best_epoch': best_epoch, 'best_val_loss': best_loss,
                          'lr': lr, 'early_counter': early_counter}), flush=True)
        last_epoch = epoch
        if early_counter >= int(cfg.train.early_stop_patience) or epoch + 1 == execution_max_epochs:
            completed = True
            break
        if args.chunk_epochs is not None and epoch - start_epoch + 1 >= args.chunk_epochs:
            break

    if best_state is None:
        raise RuntimeError('No training epoch completed.')
    if not completed:
        return
    selected_checkpoint = run_dir / 'checkpoints' / 'selected_best.pt'
    torch.save({'model_state': best_state, 'epoch': best_epoch,
                'validation_loss': best_loss, **checkpoint_metadata}, selected_checkpoint)
    model.load_state_dict(best_state, strict=True)
    frames = []
    for split in ('train', 'val') + (('test',) if args.include_test else ()):
        pred, label, source, _, _ = evaluate(model, loaders, split, device, last_epoch)
        frames.append(prediction_frame(pred, label, source, manifest, split, best_epoch,
                                       str(selected_checkpoint), cfg.model.architecture_name))
    pd.concat(frames, ignore_index=True).to_csv(run_dir / 'predictions.csv', index=False)
    collapse_events(run_dir)
    summary = {
        'fold': args.fold, 'group': args.group, 'candidate': args.candidate,
        'fusion_type': args.fusion_type, 'head_type': args.head_type,
        'architecture_name': cfg.model.architecture_name, 'last_epoch': last_epoch,
        'best_epoch': best_epoch, 'best_validation_loss_normalized': best_loss,
        'outer_test_read_during_selection': False,
        'parameter_count': int(sum(parameter.numel() for parameter in model.parameters())),
        'core_model_parameter_count': int(sum(parameter.numel() for parameter in model.model.parameters())),
    }
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
