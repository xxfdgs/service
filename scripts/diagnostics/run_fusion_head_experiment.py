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
import torch.nn.functional as functional
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphgps  # noqa: F401,E402
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from scripts.pretrain.stage4.stage4_transfer import (  # noqa: E402
    load_stage4_encoder_into,
    load_stage4_comp5_encoder,
)
from graphgps.determinism import configure_determinism  # noqa: E402
from graphgps.component_aux import normalize_component_aux_components  # noqa: E402
from graphgps.lrx_add.csv_pyg_five_multi import (  # noqa: E402
    build_input_component_vocab,
    build_input_fifth_class_vocab,
)
from graphgps.norm_threshold import (  # noqa: E402
    classifier_metrics,
    crossing_false_negative_loss,
    double_high_underprediction_loss,
    high_target,
    threshold_decision_metrics,
    weighted_regression_loss,
)
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402
from torch_geometric.graphgym.optim import (OptimizerConfig, create_optimizer,
                                             create_scheduler)  # noqa: E402
from torch_geometric.loader import DataLoader as PyGDataLoader  # noqa: E402


CORE4_TARGETS = [
    'EE_before', 'EE_after', 'Aerosolization_Efficiency',
    'mRNA_Recovery_Efficiency',
]
LATER4_TARGETS = [
    'Aerosolization_Efficiency', 'mRNA_Recovery_Efficiency',
    'Norm_before', 'Norm_after',
]
TARGET_SETS = {
    # The first four labels are stored normalized by /100 in the PyG loader.
    'core4': (CORE4_TARGETS, [100.0, 100.0, 100.0, 100.0], [0, 1, 2, 3]),
    # The fifth and sixth labels are kept in their original units by the
    # loader, so their metrics must not inherit the historical *100 factor.
    'norm2': (['Norm_before', 'Norm_after'], [1.0, 1.0], [4, 5]),
    # The loader stores the first two selected labels as /100 and the two
    # Norm targets in their original units.  Reporting restores each target
    # with its own scale.
    'later4': (LATER4_TARGETS, [100.0, 100.0, 1.0, 1.0], [2, 3, 4, 5]),
}
ALL_TARGETS = CORE4_TARGETS + TARGET_SETS['norm2'][0]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


BRANCHES = ('graph', 'descriptor', 'formula')


def property_losses(prediction, labels, target_count, target_indices=None,
                    loss_type="mae", huber_beta=0.1):
    if loss_type not in {"mae", "huber", "mse"}:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    def loss(index):
        prediction_i, label_i = prediction[index::target_count], labels[index::target_count]
        if loss_type == "mae":
            return torch.mean(torch.abs(prediction_i - label_i))
        if loss_type == "mse":
            return torch.mean((prediction_i - label_i).square())
        return functional.smooth_l1_loss(prediction_i, label_i, beta=huber_beta)
    losses = torch.stack([
        loss(index)
        for index in range(target_count)
    ])
    return losses if target_indices is None else losses[target_indices]


def transform_targets(labels, target_transform, target_scaler=None):
    """Map loader labels to the continuous space optimized by the model."""
    if target_transform == "identity":
        transformed = labels
    elif target_transform == "log1p":
        if torch.any(labels < 0):
            raise ValueError("log1p target transform requires non-negative labels.")
        transformed = torch.log1p(labels)
    else:
        raise ValueError(f"Unsupported target transform: {target_transform}")
    if target_scaler is None or target_scaler['type'] == 'identity':
        return transformed
    mean = torch.as_tensor(target_scaler['mean'], dtype=transformed.dtype,
                           device=transformed.device)
    std = torch.as_tensor(target_scaler['std'], dtype=transformed.dtype,
                          device=transformed.device)
    if transformed.ndim == 1:
        if transformed.numel() % mean.numel():
            raise ValueError('Flattened target length is incompatible with the target scaler.')
        mean, std = mean.repeat(transformed.numel() // mean.numel()), std.repeat(
            transformed.numel() // std.numel())
    else:
        mean, std = mean.view(1, -1), std.view(1, -1)
    return (transformed - mean) / std


def inverse_predictions(prediction, target_transform, target_scaler=None):
    """Return predictions in the loader's original continuous target units."""
    transformed = prediction
    if target_scaler is not None and target_scaler['type'] != 'identity':
        mean = torch.as_tensor(target_scaler['mean'], dtype=prediction.dtype,
                               device=prediction.device)
        std = torch.as_tensor(target_scaler['std'], dtype=prediction.dtype,
                              device=prediction.device)
        if prediction.ndim == 1:
            if prediction.numel() % mean.numel():
                raise ValueError('Flattened prediction length is incompatible with the target scaler.')
            mean, std = mean.repeat(prediction.numel() // mean.numel()), std.repeat(
                prediction.numel() // std.numel())
        else:
            mean, std = mean.view(1, -1), std.view(1, -1)
        transformed = transformed * std + mean
    if target_transform == "identity":
        return transformed
    if target_transform == "log1p":
        return torch.expm1(transformed).clamp_min(0)
    raise ValueError(f"Unsupported target transform: {target_transform}")


def fit_target_scaler(data_path, manifest_path, data_id_column, manifest_id_column,
                      targets, target_scales, target_transform, normalization):
    """Fit a per-target scaler only on the manifest's outer-train rows."""
    if normalization == 'identity':
        return {'type': 'identity', 'mean': [0.0] * len(targets),
                'std': [1.0] * len(targets), 'fit_split': 'train'}
    frame = pd.read_csv(data_path, dtype={data_id_column: str})
    manifest = pd.read_csv(manifest_path, dtype={manifest_id_column: str})
    if data_id_column not in frame or manifest_id_column not in manifest:
        raise ValueError('Target-scaler ID column is missing from source data or split manifest.')
    if frame[data_id_column].duplicated().any():
        raise ValueError(f'Target-scaler source has duplicate {data_id_column} values.')
    train_ids = manifest.loc[manifest['split'].eq('train'), manifest_id_column]
    indexed = frame.set_index(data_id_column)
    missing = train_ids[~train_ids.isin(indexed.index)]
    if len(missing):
        raise ValueError(f'Target-scaler manifest IDs are absent from source data: {missing.iloc[:5].tolist()}')
    values = indexed.loc[train_ids, targets].to_numpy(dtype=float) / np.asarray(target_scales)
    if not np.isfinite(values).all():
        raise ValueError('Target-scaler train labels contain non-finite values.')
    if target_transform == 'log1p':
        if np.any(values < 0):
            raise ValueError('log1p target transform requires non-negative training labels.')
        values = np.log1p(values)
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=0)
    if np.any(std <= 0):
        bad = [targets[index] for index, value in enumerate(std) if value <= 0]
        raise ValueError(f'Cannot z-score constant training targets: {bad}')
    return {'type': 'zscore_train_only', 'mean': mean.tolist(), 'std': std.tolist(),
            'fit_split': 'train', 'fit_rows': int(len(values))}


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


def unpack_model_output(value):
    """Preserve the legacy two-value forward contract outside O14-A."""
    if not isinstance(value, tuple):
        raise TypeError('Model forward must return a tuple.')
    if len(value) == 2:
        prediction, label = value
        return prediction, label, None
    if len(value) == 3:
        prediction, label, high_logit = value
        return prediction, label, high_logit
    raise ValueError(f'Unexpected model forward arity: {len(value)}')


def source_metadata(input_csv, manifest):
    """Map loader source indices to IDs/names/classes without reading labels."""
    source = pd.read_csv(input_csv, dtype={'ID': str})
    required = {'ID', 'Fifth', 'Fifth_class'}
    if missing := required.difference(source.columns):
        raise ValueError(f'O14-A source metadata misses columns: {sorted(missing)}')
    if source.ID.duplicated().any() or manifest.sample_id.duplicated().any():
        raise ValueError('O14-A source/manifest IDs must be unique.')
    manifest_rows = manifest[['original_row_index', 'sample_id']].copy()
    manifest_rows['sample_id'] = manifest_rows.sample_id.astype(str)
    joined = manifest_rows.merge(
        source[['ID', 'Fifth', 'Fifth_class']], left_on='sample_id', right_on='ID', how='left',
        validate='one_to_one')
    if joined.ID.isna().any():
        raise ValueError('O14-A manifest has IDs absent from the input source.')
    joined['Fifth'] = joined.Fifth.fillna('__absent__')
    joined['fifth_class'] = joined.Fifth_class.fillna('__unknown__').astype(str).str.strip().str.lower()
    return joined.set_index('original_row_index')[['sample_id', 'Fifth', 'fifth_class']]


def _relative_to_repo(path):
    """Return a stable repository-relative spelling when available."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return None


def _ids_from_loader(loader):
    """Read ordered source indices from a PyG loader without model execution."""
    values = []
    for batch in loader:
        if not hasattr(batch, 'sample_uid'):
            raise RuntimeError('Loader batch is missing sample_uid provenance.')
        values.extend(batch.sample_uid.detach().cpu().view(-1).tolist())
    return [int(value) for value in values]


def apply_double_high_batch_oversampling(loaders, input_csv, manifest, manifest_path,
                                         batch_size, seed, run_dir):
    """Append train-only high-double examples to batches that otherwise lack one.

    Each original train row remains in exactly one batch.  Only a deterministic
    repeat of an eligible *training* row is appended to batches with no such
    row, so paired component loaders retain identical source-index order and
    validation/test loaders are untouched.
    """
    if batch_size <= 0:
        raise ValueError('Batch size must be positive for double-high oversampling.')
    source = pd.read_csv(input_csv, dtype={'ID': str})
    required = {'ID', 'Norm_before'}
    if missing := required.difference(source.columns):
        raise ValueError(f'Input CSV misses sampler fields: {sorted(missing)}')
    source_target = source.set_index('ID')['Norm_before']
    train_manifest = manifest.loc[manifest.split.eq('train'),
                                  ['original_row_index', 'sample_id', 'Fifth_class']].copy()
    train_manifest['sample_id'] = train_manifest.sample_id.astype(str)
    train_manifest['fifth_class'] = train_manifest.Fifth_class.fillna('').astype(str).str.lower()
    train_manifest['target'] = train_manifest.sample_id.map(source_target)
    if train_manifest.target.isna().any():
        raise RuntimeError('Sampler manifest IDs are absent from the input CSV.')
    high_source_indices = set(train_manifest.loc[
        train_manifest.fifth_class.eq('double') & train_manifest.target.gt(1.0),
        'original_row_index'].astype(int).tolist())
    if not high_source_indices:
        raise RuntimeError('Double-high batch oversampling found no eligible training rows.')

    ordered_ids = [_ids_from_loader(group[0]) for group in loaders]
    if any(ids != ordered_ids[0] for ids in ordered_ids[1:]):
        raise RuntimeError('Cannot apply paired sampler: component train loader order differs.')
    source_indices = ordered_ids[0]
    if len(source_indices) != len(set(source_indices)):
        raise RuntimeError('Cannot apply paired sampler: train source indices are not unique.')
    high_positions = [position for position, source_index in enumerate(source_indices)
                      if source_index in high_source_indices]
    if not high_positions:
        raise RuntimeError('No eligible double-high source indices reached the train loader.')

    rng = random.Random(int(seed))
    base_positions = list(range(len(source_indices)))
    rng.shuffle(base_positions)
    batches = [base_positions[start:start + batch_size]
               for start in range(0, len(base_positions), batch_size)]
    missing_batches = [index for index, batch in enumerate(batches)
                       if not any(position in high_positions for position in batch)]
    repeated_positions = []
    candidates = list(high_positions)
    rng.shuffle(candidates)
    for offset, batch_index in enumerate(missing_batches):
        repeated = candidates[offset % len(candidates)]
        batches[batch_index].append(repeated)
        repeated_positions.append(repeated)
    high_position_set = set(high_positions)
    post_high_counts = [sum(position in high_position_set for position in batch)
                        for batch in batches]
    if any(count < 1 for count in post_high_counts):
        raise RuntimeError('Double-high sampler failed to cover every training batch.')

    replacement_groups = []
    for group in loaders:
        train_loader = PyGDataLoader(
            group[0].dataset, batch_sampler=batches, num_workers=cfg.num_workers,
            pin_memory=True,
        )
        replacement_groups.append([train_loader, *group[1:]])
    audit = {
        'status': 'PASS',
        'strategy': 'append_one_deterministic_repeat_to_each_base_batch_without_double_high',
        'input_csv': str(Path(input_csv).resolve()),
        'manifest_path': str(Path(manifest_path).resolve()),
        'seed': int(seed),
        'batch_size': int(batch_size),
        'original_train_row_count': len(source_indices),
        'original_double_high_count': len(high_positions),
        'base_batch_count': len(batches),
        'base_batches_without_double_high': len(missing_batches),
        'base_batch_absence_fraction': len(missing_batches) / len(batches),
        'repeated_double_high_draw_count': len(repeated_positions),
        'total_train_draw_count': sum(len(batch) for batch in batches),
        'min_double_high_per_batch': min(post_high_counts),
        'max_double_high_per_batch': max(post_high_counts),
        'validation_or_test_resampling': False,
        'target_use': 'training-label eligibility only; no validation/test/external labels',
        'component_train_source_order_equal_before_sampling': True,
    }
    (Path(run_dir) / 'double_high_batch_sampler_audit.json').write_text(
        json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    return replacement_groups


def _id_difference(expected, actual):
    expected_set, actual_set = set(expected), set(actual)
    return {
        'expected_count': len(expected),
        'expected_unique_count': len(expected_set),
        'actual_count': len(actual),
        'actual_unique_count': len(actual_set),
        'missing_ids': sorted(expected_set - actual_set),
        'unexpected_ids': sorted(actual_set - expected_set),
        'duplicate_ids': sorted(
            pd.Series(actual, dtype='string').loc[
                pd.Series(actual, dtype='string').duplicated(keep=False)
            ].unique().tolist()
        ),
    }


def _reference_prediction_membership(reference_run_dir):
    """Read the selected prediction membership of a strict control run."""
    prediction_path = Path(reference_run_dir) / 'predictions.csv'
    if not prediction_path.is_file():
        raise FileNotFoundError(
            f'Membership reference run is missing predictions.csv: {prediction_path}'
        )
    frame = pd.read_csv(prediction_path, dtype={'sample_id': str})
    required = {'sample_id', 'split'}
    if missing := required.difference(frame.columns):
        raise ValueError(
            f'Membership reference predictions miss required columns: {sorted(missing)}'
        )
    result = {}
    for split in ('train', 'val', 'test'):
        ids = frame.loc[frame.split.eq(split), 'sample_id'].astype(str).tolist()
        if len(ids) != len(set(ids)):
            raise ValueError(
                f'Membership reference has duplicate {split} prediction IDs: '
                f'{prediction_path}'
            )
        result[split] = ids
    return prediction_path.resolve(), result


def membership_preflight_audit(run_dir, input_csv, manifest_path, loaders,
                               require_membership_count=None,
                               membership_reference_run_dir=None,
                               cache_before_build=None):
    """Fail closed unless CSV, frozen manifest, loaders, and optional control agree.

    This is intentionally performed after the processed PyG cache is rebuilt
    and before model construction/epoch 0.  It catches both loader filtering
    and stale-cache provenance errors rather than inferring membership later
    from exported predictions.
    """
    run_dir = Path(run_dir)
    input_csv, manifest_path = Path(input_csv).resolve(), Path(manifest_path).resolve()
    source = pd.read_csv(input_csv, dtype={'ID': str})
    manifest = pd.read_csv(manifest_path, dtype={'sample_id': str})
    required_source = {'ID'}
    required_manifest = {'sample_id', 'split', 'original_row_index'}
    if missing := required_source.difference(source.columns):
        raise ValueError(f'Membership source CSV misses columns: {sorted(missing)}')
    if missing := required_manifest.difference(manifest.columns):
        raise ValueError(f'Membership manifest misses columns: {sorted(missing)}')

    source_ids = source.ID.astype(str).tolist()
    manifest.sample_id = manifest.sample_id.astype(str)
    source_duplicates = sorted(source.loc[source.ID.duplicated(keep=False), 'ID'].unique().tolist())
    manifest_duplicates = sorted(
        manifest.loc[manifest.sample_id.duplicated(keep=False), 'sample_id'].unique().tolist())
    source_index_duplicates = sorted(
        manifest.loc[manifest.original_row_index.duplicated(keep=False), 'original_row_index']
        .astype(int).unique().tolist())
    valid_splits = {'train', 'val', 'test'}
    invalid_split_values = sorted(set(manifest.split.astype(str)) - valid_splits)
    expected = {
        split: manifest.loc[manifest.split.eq(split), 'sample_id'].astype(str).tolist()
        for split in ('train', 'val', 'test')
    }
    source_index_to_id = dict(zip(
        manifest.original_row_index.astype(int), manifest.sample_id.astype(str)
    ))

    component_results = {}
    loader_ids = {}
    for component_index, component_loaders in enumerate(loaders, start=1):
        component_results[str(component_index)] = {}
        for split_index, split in enumerate(('train', 'val', 'test')):
            source_indices = _ids_from_loader(component_loaders[split_index])
            unknown_indices = sorted(set(source_indices) - set(source_index_to_id))
            actual_ids = [source_index_to_id[index] for index in source_indices
                          if index in source_index_to_id]
            diff = _id_difference(expected[split], actual_ids)
            diff['unknown_source_indices'] = unknown_indices
            component_results[str(component_index)][split] = diff
            if component_index == 1:
                loader_ids[split] = actual_ids

    reference = None
    reference_checks = {}
    if membership_reference_run_dir is not None:
        reference_path, reference = _reference_prediction_membership(
            membership_reference_run_dir)
        for split in ('train', 'val', 'test'):
            reference_checks[split] = _id_difference(reference[split], loader_ids[split])
        reference_checks['prediction_path'] = str(reference_path)

    component_pass = all(
        not value['missing_ids'] and not value['unexpected_ids']
        and not value['duplicate_ids'] and not value['unknown_source_indices']
        for component in component_results.values() for value in component.values()
    )
    source_manifest_diff = _id_difference(source_ids, manifest.sample_id.astype(str).tolist())
    loader_union = [sample_id for split in ('train', 'val', 'test') for sample_id in loader_ids[split]]
    loader_union_diff = _id_difference(manifest.sample_id.astype(str).tolist(), loader_union)
    expected_count_pass = (
        require_membership_count is None
        or len(manifest) == int(require_membership_count)
    )
    reference_pass = all(
        not check['missing_ids'] and not check['unexpected_ids'] and not check['duplicate_ids']
        for split, check in reference_checks.items() if split in {'train', 'val', 'test'}
    )
    passed = bool(
        not source_duplicates and not manifest_duplicates and not source_index_duplicates
        and not invalid_split_values and not source_manifest_diff['missing_ids']
        and not source_manifest_diff['unexpected_ids'] and component_pass
        and not loader_union_diff['missing_ids'] and not loader_union_diff['unexpected_ids']
        and not loader_union_diff['duplicate_ids'] and expected_count_pass and reference_pass
    )

    cache_root = Path(str(cfg.dataset.dir)).resolve()
    processed_files = sorted(
        str(path.relative_to(cache_root))
        for path in cache_root.glob('**/processed/*.pt')
    ) if cache_root.exists() else []
    audit = {
        'status': 'PASS' if passed else 'FAIL',
        'stage': 'post_cache_pre_epoch0',
        'input_csv': {
            'absolute_path': str(input_csv), 'relative_path': _relative_to_repo(input_csv),
            'sha256': file_sha256(input_csv), 'row_count': int(len(source)),
            'unique_id_count': int(len(set(source_ids))), 'duplicate_ids': source_duplicates,
        },
        'manifest': {
            'absolute_path': str(manifest_path), 'relative_path': _relative_to_repo(manifest_path),
            'sha256': file_sha256(manifest_path), 'row_count': int(len(manifest)),
            'unique_sample_id_count': int(manifest.sample_id.nunique()),
            'duplicate_ids': manifest_duplicates,
            'duplicate_original_row_indices': source_index_duplicates,
            'invalid_split_values': invalid_split_values,
            'ids_by_split': expected,
        },
        'csv_vs_manifest': source_manifest_diff,
        'loader': {
            'ids_by_split_component1': loader_ids,
            'component_partition_checks': component_results,
            'union_check': loader_union_diff,
        },
        'membership_reference': reference_checks or None,
        'requirements': {
            'required_union_count': require_membership_count,
            'required_union_count_pass': expected_count_pass,
            'component_partitions_pass': component_pass,
            'reference_partitions_pass': reference_pass,
        },
        'effective_runtime_config': {
            'read_csv': str(Path(cfg.read_csv).resolve()),
            'component_vocab_source': str(Path(cfg.component_vocab_source).resolve()),
            'diagnostic_split_path': str(Path(cfg.dataset.diagnostic_split_path).resolve()),
            'diagnostic_id_column': str(cfg.dataset.diagnostic_id_column),
            'diagnostic_manifest_id_column': str(cfg.dataset.diagnostic_manifest_id_column),
            'component_vocab_sizes': list(cfg.component_vocab_sizes),
            'fifth_component_vocab_size': int(cfg.fifth_component_vocab_size),
            'fifth_class_vocab_size': int(cfg.fifth_class_vocab_size),
            'use_mordred_features': bool(cfg.use_mordred_features),
            'mordred_feature_dim': int(cfg.mordred_feature_dim),
            'mordred_feature_path': str(cfg.mordred_feature_path),
        },
        'cache_provenance': {
            'cache_directory': str(cache_root), 'cache_tag': str(cfg.dataset.cache_tag),
            'cache_refresh': bool(cfg.dataset.cache_refresh),
            'processed_files_before_build': cache_before_build or [],
            'processed_file_count_before_build': len(cache_before_build or []),
            'processed_file_count_after_build': len(processed_files),
            'processed_files_after_build': processed_files,
        },
    }
    audit_path = run_dir / 'membership_audit.json'
    audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    rows = []
    for split in ('train', 'val', 'test'):
        component_one = component_results['1'][split]
        reference_diff = reference_checks.get(split, {})
        rows.append({
            'stage': 'post_cache_pre_epoch0', 'split': split,
            'expected_count': component_one['expected_count'],
            'actual_count': component_one['actual_count'],
            'missing_ids': json.dumps(component_one['missing_ids']),
            'unexpected_ids': json.dumps(component_one['unexpected_ids']),
            'duplicate_ids': json.dumps(component_one['duplicate_ids']),
            'reference_missing_ids': json.dumps(reference_diff.get('missing_ids', [])),
            'reference_unexpected_ids': json.dumps(reference_diff.get('unexpected_ids', [])),
            'status': 'PASS' if (
                not component_one['missing_ids'] and not component_one['unexpected_ids']
                and not component_one['duplicate_ids']
                and not reference_diff.get('missing_ids', [])
                and not reference_diff.get('unexpected_ids', [])
            ) else 'FAIL',
        })
    pd.DataFrame(rows).to_csv(run_dir / 'membership_audit.csv', index=False)
    if not passed:
        raise RuntimeError(
            'Membership preflight failed before epoch 0; inspect '
            f'{audit_path} and {run_dir / "membership_audit.csv"}.'
        )
    print(f'[Membership preflight] PASS: {audit_path}', flush=True)
    return audit


def verify_prediction_membership(run_dir):
    """Append a selected-checkpoint export check to the pre-epoch audit."""
    run_dir = Path(run_dir)
    audit_path = run_dir / 'membership_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    prediction_path = run_dir / 'predictions.csv'
    predictions = pd.read_csv(prediction_path, dtype={'sample_id': str})
    checks = {}
    passed = True
    for split in ('train', 'val', 'test'):
        expected = audit['manifest']['ids_by_split'][split]
        actual = predictions.loc[predictions.split.eq(split), 'sample_id'].astype(str).tolist()
        diff = _id_difference(expected, actual)
        checks[split] = diff
        passed &= not diff['missing_ids'] and not diff['unexpected_ids'] and not diff['duplicate_ids']
    audit['prediction_export'] = {
        'path': str(prediction_path.resolve()), 'checks': checks,
        'status': 'PASS' if passed else 'FAIL',
    }
    audit['status'] = 'PASS' if audit['status'] == 'PASS' and passed else 'FAIL'
    audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    if not passed:
        raise RuntimeError(
            'Selected-checkpoint prediction export changed membership; inspect '
            f'{audit_path}.'
        )


def canonical_fifth_identity(value):
    """Canonical identity used only for the O14-A split leakage audit."""
    text = str(value).strip()
    if pd.isna(value) or text in {'', 'nan', '[Fr]'}:
        return '[Fr]'
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise ValueError(f'O14-A leakage audit found invalid Fifth_SMILE: {value!r}')
    return Chem.MolToSmiles(molecule, canonical=True)


def audit_o14a_domain(input_csv, manifest, training_domain, targets):
    """Fail loudly unless the active manifest is a complete identity-OOD fold.

    The check is target-free except for the saved training-domain availability
    summary.  It never changes membership or selects a model.
    """
    source = pd.read_csv(input_csv, dtype={'ID': str})
    required = {'ID', 'Fifth_SMILE', 'Fifth_class', *targets}
    if missing := required.difference(source.columns):
        raise ValueError(f'O14-A audit source misses columns: {sorted(missing)}')
    if source.ID.duplicated().any() or manifest.sample_id.duplicated().any():
        raise ValueError('O14-A audit requires unique source and manifest sample IDs.')
    if len(source) != len(manifest):
        raise ValueError('O14-A audit requires the manifest to cover the active input exactly once.')
    indexed = source.set_index(source.ID.astype(str), drop=False)
    ids = manifest.sample_id.astype(str)
    if set(ids) != set(indexed.index):
        raise ValueError('O14-A audit found mismatched source and manifest IDs.')
    joined = manifest[['sample_id', 'split']].copy()
    joined['sample_id'] = joined.sample_id.astype(str)
    joined = joined.join(indexed[['Fifth_SMILE', 'Fifth_class', *targets]], on='sample_id')
    joined['fifth_identity'] = joined.Fifth_SMILE.map(canonical_fifth_identity)
    joined['fifth_class'] = joined.Fifth_class.fillna('__unknown__').astype(str).str.strip().str.lower()
    if not set(joined.split).issubset({'train', 'val', 'test'}):
        raise ValueError('O14-A audit found a split other than train/val/test.')
    if not all((joined.split == split).any() for split in ('train', 'val', 'test')):
        raise ValueError('O14-A audit found an empty train/val/test split.')
    identities = {split: set(joined.loc[joined.split.eq(split), 'fifth_identity'])
                  for split in ('train', 'val', 'test')}
    overlap = {
        'train_val': sorted(identities['train'] & identities['val']),
        'train_test': sorted(identities['train'] & identities['test']),
        'val_test': sorted(identities['val'] & identities['test']),
    }
    if any(overlap.values()):
        raise RuntimeError(f'O14-A Fifth identity leakage: {overlap}')
    if training_domain == 'double' and not joined.fifth_class.eq('double').all():
        bad = joined.loc[~joined.fifth_class.eq('double'), 'fifth_class'].unique().tolist()
        raise RuntimeError(f'O14-A double domain contains non-double classes: {bad}')
    split_summary = {}
    for split in ('train', 'val', 'test'):
        subset = joined.loc[joined.split.eq(split)]
        target_summary = {}
        for target in targets:
            values = pd.to_numeric(subset[target], errors='coerce').dropna()
            target_summary[target] = {
                'valid_count': int(len(values)),
                'gt_threshold_count': int((values > 1.0).sum()),
                'le_threshold_count': int((values <= 1.0).sum()),
                'min': float(values.min()) if len(values) else None,
                'max': float(values.max()) if len(values) else None,
                'mean': float(values.mean()) if len(values) else None,
                'median': float(values.median()) if len(values) else None,
            }
        split_summary[split] = {
            'rows': int(len(subset)),
            'unique_fifth_identities': int(subset.fifth_identity.nunique()),
            'fifth_class_counts': subset.fifth_class.value_counts().to_dict(),
            'targets': target_summary,
        }
    return {
        'status': 'pass', 'training_domain': training_domain,
        'identity_leakage_pass': True,
        'identity_overlap_counts': {key: len(value) for key, value in overlap.items()},
        'identity_overlap_examples': overlap,
        'split_summary': split_summary,
    }


def threshold_metric_rows(prediction, labels, high_logits, source_indices, metadata,
                          split, epoch, target, target_scale, threshold,
                          regression_loss):
    """O14-A regression-threshold and auxiliary-classifier diagnostics."""
    source = np.asarray(source_indices, dtype=int)
    rows = metadata.reindex(source)
    if rows.isna().any().any():
        raise ValueError('O14-A prediction source indices do not match metadata.')
    y = np.asarray(labels[:, 0] * target_scale, dtype=float)
    p = np.asarray(prediction[:, 0] * target_scale, dtype=float)
    probabilities = None
    if high_logits is not None:
        logits = np.asarray(high_logits, dtype=float)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    records = []
    classes = rows.fifth_class.to_numpy(dtype=str)
    for subset, mask in (
            ('all', np.ones(len(rows), dtype=bool)),
            ('single', classes == 'single'),
            ('double', classes == 'double')):
        if not np.any(mask):
            continue
        y_sub, p_sub = y[mask], p[mask]
        base = {
            'epoch': int(epoch), 'split': split, 'target': target, 'subset': subset,
            'threshold': float(threshold), 'regression_loss_normalized': float(regression_loss),
            'mae': float(mean_absolute_error(y_sub, p_sub)),
            'rmse': float(mean_squared_error(y_sub, p_sub) ** 0.5),
            'r2': float(r2_score(y_sub, p_sub)) if len(y_sub) > 1 and np.std(y_sub) else math.nan,
            'pearson': safe_correlation(pearsonr, y_sub, p_sub),
            'spearman': safe_correlation(spearmanr, y_sub, p_sub),
        }
        records.append({**base, 'decision_source': 'regression',
                        **threshold_decision_metrics(y_sub, p_sub, threshold),
                        'auroc_gt1': math.nan, 'auprc_gt1': math.nan})
        if probabilities is not None:
            records.append({**base, 'decision_source': 'classifier',
                            **classifier_metrics(y_sub, probabilities[mask], threshold)})
    return records


def threshold_prediction_frame(prediction, labels, high_logits, source_indices, metadata,
                               split, epoch, checkpoint, target, target_scale, threshold):
    """Per-sample O14-A audit export; source columns are target-free metadata."""
    source = np.asarray(source_indices, dtype=int)
    meta = metadata.reindex(source)
    if meta.isna().any().any():
        raise ValueError('O14-A prediction source indices do not match metadata.')
    y = np.asarray(labels[:, 0] * target_scale, dtype=float)
    p = np.asarray(prediction[:, 0] * target_scale, dtype=float)
    if high_logits is None:
        probability = np.full(len(y), np.nan, dtype=float)
        classifier_prediction = pd.Series(pd.NA, index=np.arange(len(y)), dtype='boolean')
    else:
        logits = np.asarray(high_logits, dtype=float)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        classifier_prediction = probability >= 0.5
    result = meta.reset_index(names='source_index').copy()
    # Retain the source spelling required by downstream CSV consumers while
    # keeping the normalized lowercase field for subset filtering.
    result['Fifth_class'] = result['fifth_class']
    result['split'] = split
    result['seed_checkpoint_epoch'] = int(epoch)
    result['checkpoint'] = checkpoint
    result['target'] = target
    result['true_norm'] = y
    result['pred_norm'] = p
    result['true_gt1'] = y > float(threshold)
    result['pred_gt1_from_regression'] = p > float(threshold)
    result['prob_gt1_from_classifier'] = probability
    result['pred_gt1_from_classifier'] = classifier_prediction
    # Short aliases make the requested O14-A diagnostic schema directly
    # consumable while retaining the explicit historical column names.
    result['prob_gt1'] = result['prob_gt1_from_classifier']
    result['pred_gt1_regression'] = result['pred_gt1_from_regression']
    result['pred_gt1_classifier'] = result['pred_gt1_from_classifier']
    result['regression_error'] = p - y
    result['absolute_error'] = np.abs(p - y)
    result['threshold_crossing_error'] = result.true_gt1 & ~result.pred_gt1_from_regression
    return result


def double_high_magnitude_rows(frame):
    """Describe whether failures are near-threshold or extreme-tail errors."""
    double = frame.loc[(frame['split'].eq('test')) & (frame['fifth_class'].eq('double'))].copy()
    bins = [
        ('1.0_to_1.5', (double.true_norm > 1.0) & (double.true_norm <= 1.5)),
        ('1.5_to_2.0', (double.true_norm > 1.5) & (double.true_norm <= 2.0)),
        ('gt_2.0', double.true_norm > 2.0),
    ]
    rows = []
    for label, mask in bins:
        values = double.loc[mask]
        rows.append({
            'split': 'test', 'subset': 'double', 'magnitude_bin': label,
            'sample_count': int(len(values)),
            'mean_true_norm': float(values.true_norm.mean()) if len(values) else math.nan,
            'mean_pred_norm': float(values.pred_norm.mean()) if len(values) else math.nan,
            'mae': float(values.absolute_error.mean()) if len(values) else math.nan,
            'fraction_predicted_gt1': (
                float(values.pred_gt1_from_regression.mean()) if len(values) else math.nan),
        })
    return rows


def threshold_prediction_separation_rows(frame):
    """Quantify upward movement of true-high predictions without conflating FP."""
    records = []
    for split in ('train', 'val', 'test'):
        split_frame = frame.loc[frame.split.eq(split)]
        for subset, subset_frame in (
                ('all', split_frame),
                ('single', split_frame.loc[split_frame.fifth_class.eq('single')]),
                ('double', split_frame.loc[split_frame.fifth_class.eq('double')])):
            if subset_frame.empty:
                continue
            high = subset_frame.loc[subset_frame.true_gt1, 'pred_norm']
            low = subset_frame.loc[~subset_frame.true_gt1, 'pred_norm']
            high_mean = float(high.mean()) if len(high) else math.nan
            low_mean = float(low.mean()) if len(low) else math.nan
            records.append({
                'split': split, 'subset': subset, 'n': int(len(subset_frame)),
                'true_gt1_count': int(len(high)), 'true_le1_count': int(len(low)),
                'mean_pred_true_gt1': high_mean,
                'mean_pred_true_le1': low_mean,
                'positive_margin': float((high - 1.0).mean()) if len(high) else math.nan,
                'separation': high_mean - low_mean if len(high) and len(low) else math.nan,
            })
    return records


def threshold_selection_score(rows):
    """Prefer double F2 when its validation subset is informative, else all-sample F2."""
    frame = pd.DataFrame(rows)
    regression = frame.loc[frame['decision_source'].eq('regression')]
    double = regression.loc[regression['subset'].eq('double')]
    candidate = double.loc[double['n'].ge(3) & double['f2_gt1'].notna()]
    if candidate.empty:
        candidate = regression.loc[
            regression['subset'].eq('all') & regression['f2_gt1'].notna()]
    if candidate.empty:
        return -math.inf, 'none'
    selected = candidate.iloc[0]
    return float(selected['f2_gt1']), str(selected['subset'])


def metric_rows(prediction, labels, split, epoch, loss, lr, best_epoch, best_loss,
                early_counter, architecture, targets, target_scales):
    rows = []
    for index, target in enumerate(targets):
        y = labels[:, index] * target_scales[index]
        p = prediction[:, index] * target_scales[index]
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
                     checkpoint, architecture, targets, target_scales):
    sample_ids = dict(zip(
        manifest.original_row_index.astype(int), manifest.sample_id.astype(str)))
    rows = []
    for row_index, source_index in enumerate(source_indices):
        for target_index, target in enumerate(targets):
            rows.append({
                'sample_id': sample_ids[int(source_index)],
                'source_index': int(source_index), 'split': split, 'target': target,
                'epoch': int(epoch), 'checkpoint': checkpoint,
                'architecture': architecture,
                'y_true': float(labels[row_index, target_index] * target_scales[target_index]),
                'y_pred': float(prediction[row_index, target_index] * target_scales[target_index]),
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
        if getattr(core, 'use_norm_threshold_head', False):
            modules.update({
                'shared_fusion': core.fusion_backbone,
                'regression_head': core.regression_head,
                'norm_threshold_head': core.norm_threshold_head,
            })
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


def diagnostic_rows(core, epoch, split, targets):
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
        labels = targets if weights.shape[1] == len(targets) else ['all']
        for target_index, target in enumerate(labels):
            idx = target_index if weights.shape[1] == len(targets) else 0
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


def evaluate(model, loaders, split, device, epoch, target_count, targets,
             target_indices=None, collect_diagnostics=False,
             target_transform="identity", target_scaler=None):
    model.eval()
    predictions, labels, sources, high_logits = [], [], [], []
    diagnostic = ([], [], [], [])
    with torch.no_grad():
        for batch_index, batches in enumerate(zip(*[group[{'train': 0, 'val': 1, 'test': 2}[split]] for group in loaders])):
            batches = prepare_batches(list(batches), split, device)
            model_prediction, label, high_logit = unpack_model_output(model(*batches))
            pred = inverse_predictions(model_prediction, target_transform, target_scaler)
            if collect_diagnostics and batch_index == 0:
                diagnostic = diagnostic_rows(model.model, epoch, split, targets)
            predictions.append(pred.detach().cpu().reshape(-1, target_count).numpy())
            labels.append(label.detach().cpu().reshape(-1, target_count).numpy())
            sources.append(batches[0].sample_uid.detach().cpu().numpy().reshape(-1))
            if high_logit is not None:
                high_logits.append(high_logit.detach().cpu().reshape(-1).numpy())
    prediction, label = np.vstack(predictions), np.vstack(labels)
    if target_scaler is not None and target_scaler['type'] != 'identity':
        # `prediction` is already inverse-transformed, so reconstruct the
        # normalized validation space used for checkpoint selection.
        normalized_prediction = prediction
        if target_transform == 'log1p':
            normalized_prediction = np.log1p(np.clip(normalized_prediction, 0, None))
        normalized_prediction = ((normalized_prediction - np.asarray(target_scaler['mean']))
                                 / np.asarray(target_scaler['std']))
        normalized_label = transform_targets(
            torch.as_tensor(label), target_transform, target_scaler).numpy()
        loss_by_target = np.mean(np.abs(normalized_prediction - normalized_label), axis=0)
    else:
        loss_by_target = np.mean(np.abs(prediction - label), axis=0)
    loss = float(loss_by_target.sum() if target_indices is None
                 else loss_by_target[target_indices].sum())
    high_logit = np.concatenate(high_logits) if high_logits else None
    return prediction, label, np.concatenate(sources), loss, diagnostic, high_logit


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
    parser.add_argument('--input-csv', type=Path, default=None,
                        help='Optional UTF-8 training CSV override. The same CSV is also used '
                             'to construct component vocabularies for this new run.')
    parser.add_argument('--target-set', choices=tuple(TARGET_SETS), default='core4',
                        help='Label group: core4 (the historical first four percent labels) '
                             'or norm2 (Norm_before and Norm_after in original units), '
                             'or later4 (Aerosolization, Recovery, Norm_before, Norm_after).')
    parser.add_argument('--single-target', choices=ALL_TARGETS, default=None,
                        help='Train a true one-output model for one of the six labels. '
                             'Overrides --target-set and selects the matching CSV label.')
    parser.add_argument('--split-manifest', type=Path, default=None,
                        help='Optional explicit input-only train/val/test manifest. '
                             'This changes data membership independently of --seed.')
    parser.add_argument('--training-domain', choices=('full', 'double'), default='full',
                        help='O14-A audit label. double requires every active source row to have '
                             'Fifth_class=double and a Fifth-identity-OOD manifest.')
    parser.add_argument('--component-vocab-source', type=Path, default=None,
                        help='Optional frozen categorical-vocabulary source. This preserves the '
                             'full O13 categorical encoding when training a class-restricted domain.')
    parser.add_argument('--fold', required=True)
    parser.add_argument('--group', choices=('A', 'B'), required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--fusion-type', choices=('softmax_sum', 'concat', 'concat_mlp', 'residual', 'residual_concat', 'gated_concat', 'attention_concat'), required=True)
    parser.add_argument('--head-type', choices=('baseline', 'linear', 'two_layer', 'residual_head', 'target_specific'), required=True)
    parser.add_argument('--execution-max-epochs', type=int, default=120)
    parser.add_argument('--base-lr', type=float, default=None,
                        help='Optional input-only experiment override for optim.base_lr.')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Optional input-only experiment override for optim.weight_decay.')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Optional input-only experiment override for train.batch_size.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional deterministic seed override for an independent run.')
    parser.add_argument('--warmup-epochs', type=int, default=None,
                        help='Optional input-only experiment override for optim.num_warmup_epochs.')
    parser.add_argument('--early-stop-patience', type=int, default=None,
                        help='Optional number of consecutive non-improving validation epochs before stopping.')
    parser.add_argument('--head-hidden-dim', type=int, default=None,
                        help='Optional width for the redesigned prediction head.')
    parser.add_argument('--head-dropout', type=float, default=None,
                        help='Optional dropout for the redesigned prediction head.')
    parser.add_argument('--fusion-hidden-dim', type=int, default=None,
                        help='Optional width for the redesigned fusion module.')
    parser.add_argument('--fusion-dropout', type=float, default=None,
                        help='Optional dropout for the redesigned fusion module.')
    parser.add_argument('--ratio-polynomial-features', action='store_true',
                        help='Append train-label-free composition powers and interactions.')
    parser.add_argument('--fifth-only-fusion', action='store_true',
                        help='Use only the fifth GraphGPS branch, fifth ratio, and '
                             'fifth descriptors in the final regression fusion.')
    parser.add_argument('--loss-targets', nargs='+',
                        help='Train/select only these target(s); all selected-set predictions are recorded.')
    parser.add_argument('--training-loss', choices=('mae', 'huber', 'mse'), default='mae',
                        help='Backpropagation loss; validation checkpoint selection remains MAE.')
    parser.add_argument('--target-transform', choices=('identity', 'log1p'), default='identity',
                        help='Continuous target space used for backpropagation. Predictions '
                            'and checkpoint selection are always converted back to raw units.')
    parser.add_argument('--target-normalization', choices=('identity', 'zscore'), default='identity',
                        help='Optional per-target z-score fitted only on the outer training split. '
                             'Reports always use inverse-transformed values.')
    parser.add_argument('--huber-beta', type=float, default=0.1,
                        help='Normalized Smooth-L1 transition for --training-loss huber.')
    parser.add_argument('--enable-norm-threshold-aware', action='store_true',
                        help='Enable O14-A: a small shared-representation high-Norm classifier plus '
                             'threshold-aware training losses. Valid only for a single Norm target.')
    parser.add_argument('--norm-threshold-report-only', action='store_true',
                        help='For a regression-only single Norm control, export threshold diagnostics '
                             'without adding the O14-A classifier head or losses.')
    parser.add_argument('--norm-threshold', type=float, default=1.0,
                        help='Fixed physical Norm threshold for O14-A high-label and crossing loss.')
    parser.add_argument('--norm-cls-loss-weight', type=float, default=0.5,
                        help='O14-A BCEWithLogits auxiliary-classifier weight.')
    parser.add_argument('--norm-fn-loss-weight', type=float, default=1.0,
                        help='O14-A squared false-negative threshold-crossing penalty weight.')
    parser.add_argument('--norm-positive-reg-weight', type=float, default=1.0,
                        help='Optional O14-A regression weight for double samples with true Norm > threshold.')
    parser.add_argument('--norm-underprediction-weight', type=float, default=0.0,
                        help='Optional continuous squared underprediction penalty for double true-high Norm '
                             'samples.  This does not require or use the auxiliary classifier head.')
    parser.add_argument('--min-double-high-per-batch', type=int, default=0,
                        help='Train-only deterministic oversampling control. With value 1, append a '
                             'double true-high Norm training row to every otherwise-empty batch. '
                             'Valid only for a single Norm target with threshold reporting.')
    parser.add_argument('--norm-threshold-selection-mae-tolerance', type=float, default=0.10,
                        help='Maximum relative validation-MAE degradation allowed for the additional '
                             'O14-A threshold-aware checkpoint selection.')
    parser.add_argument('--gt-dropout', type=float, default=None,
                        help='Optional GraphGPS dropout override for architecture experiments.')
    parser.add_argument('--gt-attn-dropout', type=float, default=None,
                        help='Optional GraphGPS attention-dropout override for architecture experiments.')
    parser.add_argument('--gps-layers', type=int, default=None,
                        help='Optional GraphGPS layer-count override.')
    parser.add_argument('--graph-hidden-dim', type=int, default=None,
                        help='Optional shared GNN/GraphGPS hidden width. Must divide evenly by gt.n_heads.')
    parser.add_argument('--rwse-dim', type=int, default=None,
                        help='Optional RWSE positional-encoding width; it must be smaller than graph hidden width.')
    parser.add_argument('--model-type', type=str, default=None,
                        help='Optional registered GraphGPS network type override.')
    parser.add_argument('--graph-pooling', choices=('add', 'mean', 'max'), default=None,
                        help='Optional graph-level pooling override for registered GraphGPS models.')
    parser.add_argument('--use-mordred-features', action='store_true',
                        help='Enable a supplied input-only Mordred lookup.')
    parser.add_argument('--disable-mordred-features', action='store_true',
                        help='Explicitly disable Mordred even if the base YAML enables it.')
    parser.add_argument('--mordred-feature-path', type=Path, default=None)
    parser.add_argument('--mordred-feature-dim', type=int, default=None)
    parser.add_argument('--mordred-fifth-only', action='store_true')
    parser.add_argument('--use-fifth-mechanistic-descriptors', action='store_true',
                        help='Append a supplied seed-standardized, structure-only descriptor vector '
                             'to the component-5 fusion branch.')
    parser.add_argument('--fifth-mechanistic-descriptor-path', type=Path, default=None)
    parser.add_argument('--fifth-mechanistic-descriptor-dim', type=int, default=None)
    parser.add_argument('--use-fifth-semantic-features', action='store_true',
                        help='Add a supplied train-only-scaled chemistry-semantic vector to component 5 only.')
    parser.add_argument('--fifth-semantic-feature-path', type=Path, default=None)
    parser.add_argument('--fifth-semantic-feature-dim', type=int, default=None)
    parser.add_argument('--use-fifth-structured-features', action='store_true')
    parser.add_argument('--fifth-structured-feature-path', type=Path, default=None)
    parser.add_argument('--fifth-aa-embedding-dim', type=int, default=8)
    parser.add_argument('--fifth-terminal-embedding-dim', type=int, default=4)
    parser.add_argument('--fifth-aa-vocab-size', type=int, default=None,
                        help='Optional fixed structured-AA embedding table size. It may exceed, '
                             'but not be smaller than, the train-derived lookup IDs.')
    parser.add_argument('--fifth-terminal-vocab-size', type=int, default=None,
                        help='Optional fixed structured-terminal embedding table size. It may exceed, '
                             'but not be smaller than, the train-derived lookup IDs.')
    parser.add_argument('--use-component-aux-features', action='store_true',
                        help='Fuse input-only RDKit/Morgan component auxiliary features.')
    parser.add_argument('--component-aux-components', type=str, default=None,
                        help='One-based component positions that receive the auxiliary branch: '
                             '"all" (default), "fifth", or a comma-separated list such as '
                             '"1,3,5". Supplying this option enables component auxiliary features.')
    parser.add_argument('--strict-component-vocab', action='store_true',
                        help='Use only source-data component categories with no unknown row; '
                             'missing, unseen, or out-of-range IDs raise an error.')
    parser.add_argument('--use-fifth-identity-embedding', action='store_true',
                        help='Add an input-only categorical embedding to the fifth GraphGPS branch.')
    parser.add_argument('--use-fifth-class-embedding', action='store_true',
                        help='Add an input-derived Fifth_class embedding to the fifth GraphGPS branch.')
    parser.add_argument('--use-fifth-ratio-modulation', action='store_true',
                        help='Condition the fifth graph embedding on its formulation ratio.')
    parser.add_argument('--coarse-grain-enable', action='store_true')
    parser.add_argument('--output-activation', choices=('identity', 'sigmoid'),
                        default=None,
                        help='Readout activation. Defaults to sigmoid for bounded core4 '
                             'targets (including core4 single-task runs) and identity otherwise.')
    parser.add_argument(
        '--comp5-pretrained-checkpoint',
        type=Path,
        default=None,
        help=(
            'Optional Stage-4 Comp5GraphEncoder checkpoint. '
            'Loaded strictly into OneHotEmbedGPSModel.comp5_encoder '
            'after model construction and before optimizer construction.'
        ),
    )
    parser.add_argument(
        '--comp5-pretrain-label',
        type=str,
        default='P0_random',
        help='Audit label for Fifth-encoder initialization, e.g. P0_random, P1_PT_D, P2_PT_DF.',
    )
    parser.add_argument(
        '--comp5-lr',
        type=float,
        default=None,
        help=(
            'Optional differential learning rate for model.model.comp5_encoder. '
            'All remaining trainable parameters keep cfg.optim.base_lr. '
            'Requires --comp5-pretrained-checkpoint.'
        ),
    )
    parser.add_argument(
        '--frozen-comp5-aux-checkpoint',
        type=Path,
        default=None,
        help=(
            'Optional Stage-4 Comp5GraphEncoder checkpoint loaded into a '
            'second permanently frozen Fifth structural-prior branch. '
            'The normal comp5_encoder remains trainable.'
        ),
    )
    parser.add_argument(
        '--frozen-comp5-aux-label',
        type=str,
        default='none',
        help='Audit label for the optional frozen Fifth structural-prior branch.',
    )
    parser.add_argument(
        '--architecture-audit-only',
        action='store_true',
        help=(
            'Construct loaders/model/optimizer, run frozen-branch and optimizer '
            'partition audits, persist provenance, then exit before training. '
            'Use a disposable --run-dir for a Stage-8 preflight smoke audit.'
        ),
    )
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--restart-incomplete', action='store_true',
                        help='Reuse an incomplete run directory that has no resume state.')
    parser.add_argument('--reuse-existing-cache', action='store_true',
                        help='Reuse a pre-populated per-run processed cache instead of '
                             'deleting and rebuilding it on a fresh run.')
    parser.add_argument('--require-membership-count', type=int, default=None,
                        help='Fail before epoch 0 unless the frozen manifest and rebuilt '
                             'loader union contain exactly this many unique IDs.')
    parser.add_argument('--membership-reference-run-dir', type=Path, default=None,
                        help='Optional completed strict-control run directory. Before epoch 0, '
                             'require its train/val/test prediction ID sets to exactly match '
                             'the rebuilt loader partitions.')
    parser.add_argument('--require-fresh-cache', action='store_true',
                        help='Fail if a processed PyG cache already exists or cache refresh is '
                             'disabled. Intended for strict reproducibility reruns.')
    parser.add_argument('--chunk-epochs', type=int, default=None)
    parser.add_argument('--include-test', action='store_true')
    parser.add_argument('--tqdm-progress', action='store_true',
                        help='Show an epoch-level tqdm progress bar for this single seed run.')
    args = parser.parse_args()
    if args.single_target is None:
        targets, target_scales, target_indices_in_source = TARGET_SETS[args.target_set]
        target_count = len(targets)
        single_task_target_index = None
    else:
        targets = [args.single_target]
        target_scales = [100.0 if args.single_target in CORE4_TARGETS else 1.0]
        target_count = 1
        single_task_target_index = ALL_TARGETS.index(args.single_target)
        target_indices_in_source = [single_task_target_index]
    bounded_core_targets = (
        args.target_set == 'core4'
        if args.single_target is None
        else args.single_target in CORE4_TARGETS
    )
    if args.output_activation is None:
        args.output_activation = 'sigmoid' if bounded_core_targets else 'identity'
    elif bounded_core_targets and args.output_activation != 'sigmoid':
        raise ValueError(
            'Bounded core4 efficiency targets must use --output-activation sigmoid. '
            'The loader scales these labels to [0, 1]; sigmoid therefore restores '
            'predictions to the physical [0, 100] range after reporting-scale conversion.')
    if args.loss_targets and any(target not in targets for target in args.loss_targets):
        raise ValueError(
            f'--loss-targets must belong to --target-set {args.target_set}: {targets}')
    selected_target_indices = ([targets.index(target) for target in args.loss_targets]
                               if args.loss_targets else None)
    if args.huber_beta <= 0:
        raise ValueError('--huber-beta must be positive')
    args.norm_threshold_reporting = bool(
        args.enable_norm_threshold_aware or args.norm_threshold_report_only
        or args.norm_positive_reg_weight != 1.0 or args.norm_underprediction_weight != 0.0
        or args.min_double_high_per_batch != 0)
    if args.norm_threshold_reporting:
        if args.single_target not in {'Norm_before', 'Norm_after'} or target_count != 1:
            raise ValueError(
                'Norm threshold reporting requires --single-target Norm_before or Norm_after.')
        if args.norm_threshold <= 0:
            raise ValueError('--norm-threshold must be positive.')
    if args.norm_cls_loss_weight < 0 or args.norm_fn_loss_weight < 0 \
            or args.norm_underprediction_weight < 0:
        raise ValueError('Norm objective loss weights must be non-negative.')
    if args.norm_positive_reg_weight < 1.0:
        raise ValueError('--norm-positive-reg-weight must be at least 1.0.')
    if args.min_double_high_per_batch not in {0, 1}:
        raise ValueError('--min-double-high-per-batch currently supports only 0 or 1.')
    if args.enable_norm_threshold_aware:
        if args.norm_threshold_selection_mae_tolerance < 0:
            raise ValueError('--norm-threshold-selection-mae-tolerance must be non-negative.')
    # Group A deliberately preserves the historical three-branch fusion and
    # replaces only its main prediction head.  That forward path has no
    # descriptor input, so accepting Mordred flags here would create a
    # misleading "descriptor" experiment that never consumes descriptors.
    if args.use_mordred_features and args.group == 'A':
        raise ValueError(
            'Mordred experiments must use group B (a descriptor-aware fusion); '
            'group A changes only the legacy prediction head.')
    if args.use_fifth_mechanistic_descriptors:
        if args.group == 'A':
            raise ValueError('Fifth mechanistic descriptors require descriptor-aware group B fusion.')
        if args.fifth_mechanistic_descriptor_path is None or args.fifth_mechanistic_descriptor_dim is None:
            raise ValueError('--use-fifth-mechanistic-descriptors requires lookup path and dimension.')
        if args.fifth_mechanistic_descriptor_dim <= 0:
            raise ValueError('--fifth-mechanistic-descriptor-dim must be positive.')
    frozen_aux_checkpoint_path = None
    if args.frozen_comp5_aux_checkpoint is not None:
        frozen_aux_checkpoint_path = args.frozen_comp5_aux_checkpoint.resolve()
        if not frozen_aux_checkpoint_path.is_file():
            raise FileNotFoundError(
                f'Frozen Comp5 auxiliary checkpoint is missing: '
                f'{frozen_aux_checkpoint_path}'
            )
    if args.architecture_audit_only and args.resume:
        raise ValueError('--architecture-audit-only cannot be combined with --resume.')
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
    if args.input_csv is not None:
        input_csv = args.input_csv.resolve()
        if not input_csv.is_file():
            raise FileNotFoundError(f'Input CSV is missing: {input_csv}')
        cfg.read_csv = str(input_csv)
        # A fresh run must build categorical vocabularies from the same source
        # table that supplies its labels and fixed split membership.
        cfg.component_vocab_source = str(input_csv)
    if args.component_vocab_source is not None:
        vocabulary_source = args.component_vocab_source.resolve()
        if not vocabulary_source.is_file():
            raise FileNotFoundError(f'Component vocabulary source is missing: {vocabulary_source}')
        cfg.component_vocab_source = str(vocabulary_source)
    if args.split_manifest is not None:
        split_manifest = args.split_manifest.resolve()
        if not split_manifest.is_file():
            raise FileNotFoundError(f'Split manifest is missing: {split_manifest}')
        cfg.dataset.diagnostic_split_path = str(split_manifest)
        # Input benchmark manifests identify rows by their unique ID column.
        cfg.dataset.diagnostic_id_column = 'ID'
        cfg.dataset.diagnostic_manifest_id_column = 'sample_id'
    cfg.model.fusion_type = args.fusion_type
    cfg.model.head_type = args.head_type
    cfg.model.architecture_name = ('legacy_baseline' if args.candidate == 'A0'
                                   else f'{args.group}_{args.candidate}_{args.fusion_type}_{args.head_type}')
    cfg.model.target_specific_heads = args.head_type == 'target_specific'
    # Set before cache/model construction and effective-config persistence so
    # every checkpoint has a reconstructible Stage-8 topology.
    cfg.model.frozen_comp5_aux_enable = bool(
        args.frozen_comp5_aux_checkpoint is not None
    )
    cfg.model.frozen_comp5_aux_label = str(args.frozen_comp5_aux_label)
    cfg.model.frozen_comp5_aux_checkpoint_sha256 = (
        file_sha256(frozen_aux_checkpoint_path)
        if frozen_aux_checkpoint_path is not None else None
    )
    cfg.model.output_activation = args.output_activation
    cfg.property_num = target_count
    if target_count == 4:
        cfg.multi_task_target_indices = list(target_indices_in_source)
    if single_task_target_index is not None:
        cfg.single_task_target_index = single_task_target_index
        cfg.property_serial = single_task_target_index
    cfg.model.ratio_polynomial_features = bool(args.ratio_polynomial_features)
    cfg.model.fifth_only_fusion = bool(args.fifth_only_fusion)
    cfg.model.validate_redesign_inputs = args.candidate != 'A0'
    cfg.use_norm_threshold_head = bool(args.enable_norm_threshold_aware)
    cfg.norm_threshold = float(args.norm_threshold)
    cfg.norm_cls_loss_weight = float(args.norm_cls_loss_weight)
    cfg.norm_fn_loss_weight = float(args.norm_fn_loss_weight)
    cfg.norm_positive_reg_weight = float(args.norm_positive_reg_weight)
    for key, value in {
        'base_lr': args.base_lr,
        'weight_decay': args.weight_decay,
        'num_warmup_epochs': args.warmup_epochs,
    }.items():
        if value is not None:
            setattr(cfg.optim, key, value)
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.early_stop_patience is not None:
        if args.early_stop_patience <= 0:
            raise ValueError('--early-stop-patience must be positive.')
        cfg.train.early_stop_patience = int(args.early_stop_patience)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.model_type is not None:
        cfg.model.type = args.model_type
    if args.graph_pooling is not None:
        cfg.model.graph_pooling = args.graph_pooling
    if args.gps_layers is not None:
        if args.gps_layers <= 0:
            raise ValueError('--gps-layers must be positive.')
        cfg.gt.layers = int(args.gps_layers)
    if args.graph_hidden_dim is not None:
        hidden_dim = int(args.graph_hidden_dim)
        if hidden_dim <= 0 or hidden_dim % int(cfg.gt.n_heads) != 0:
            raise ValueError(
                '--graph-hidden-dim must be positive and divisible by cfg.gt.n_heads '
                f'({cfg.gt.n_heads}).')
        # OneHotEmbedGPS creates its component embeddings from gt.dim_hidden,
        # whereas its fifth-component GraphGPS encoder requires gt.dim_hidden
        # and gnn.dim_inner to be identical.
        cfg.gt.dim_hidden = hidden_dim
        cfg.gnn.dim_inner = hidden_dim
    if args.rwse_dim is not None:
        rwse_dim = int(args.rwse_dim)
        if rwse_dim <= 0 or rwse_dim >= int(cfg.gnn.dim_inner):
            raise ValueError('--rwse-dim must be positive and smaller than graph hidden width.')
        cfg.posenc_RWSE.dim_pe = rwse_dim
    if args.gt_dropout is not None:
        cfg.gt.dropout = float(args.gt_dropout)
    if args.gt_attn_dropout is not None:
        cfg.gt.attn_dropout = float(args.gt_attn_dropout)
    if args.use_mordred_features and args.disable_mordred_features:
        raise ValueError('Cannot enable and disable Mordred simultaneously.')
    if args.disable_mordred_features:
        cfg.use_mordred_features = False; cfg.mordred_feature_path = ''; cfg.mordred_feature_dim = 0
    if args.use_mordred_features:
        if args.mordred_feature_path is None or args.mordred_feature_dim is None:
            raise ValueError('--use-mordred-features requires lookup path and dimension.')
        cfg.use_mordred_features = True
        cfg.mordred_feature_path = str(args.mordred_feature_path.resolve())
        cfg.mordred_feature_dim = int(args.mordred_feature_dim)
        cfg.mordred_fifth_only = bool(args.mordred_fifth_only)
    if args.use_fifth_mechanistic_descriptors:
        descriptor_path = args.fifth_mechanistic_descriptor_path.resolve()
        if not descriptor_path.is_file():
            raise FileNotFoundError(f'Fifth descriptor lookup is missing: {descriptor_path}')
        cfg.use_fifth_mechanistic_descriptors = True
        cfg.fifth_mechanistic_descriptor_path = str(descriptor_path)
        cfg.fifth_mechanistic_descriptor_dim = int(args.fifth_mechanistic_descriptor_dim)
    if args.use_fifth_semantic_features:
        if args.fifth_semantic_feature_path is None or args.fifth_semantic_feature_dim is None:
            raise ValueError('--use-fifth-semantic-features requires lookup path and dimension.')
        semantic_path = args.fifth_semantic_feature_path.resolve()
        if not semantic_path.is_file() or int(args.fifth_semantic_feature_dim) <= 0:
            raise ValueError('Fifth semantic lookup must exist and have a positive dimension.')
        cfg.use_fifth_semantic_features = True
        cfg.fifth_semantic_feature_path = str(semantic_path)
        cfg.fifth_semantic_feature_dim = int(args.fifth_semantic_feature_dim)
    if args.use_fifth_structured_features:
        if args.fifth_structured_feature_path is None:
            raise ValueError('--use-fifth-structured-features requires --fifth-structured-feature-path.')
        structured_path = args.fifth_structured_feature_path.resolve()
        table = pd.read_csv(structured_path)
        required = {'smiles', 'aa_id', 'terminal_id', 'tail_length_normalized', 'tail_length_present_mask'}
        if not structured_path.is_file() or required.difference(table.columns):
            raise ValueError('O13G structured lookup is missing required columns.')
        cfg.use_fifth_structured_features = True
        cfg.architecture_family = 'O13G'
        cfg.fifth_structured_feature_path = str(structured_path)
        lookup_aa_size = int(table.aa_id.max()) + 1
        lookup_terminal_size = int(table.terminal_id.max()) + 1
        cfg.fifth_aa_vocab_size = int(args.fifth_aa_vocab_size or lookup_aa_size)
        cfg.fifth_terminal_vocab_size = int(args.fifth_terminal_vocab_size or lookup_terminal_size)
        if cfg.fifth_aa_vocab_size < lookup_aa_size:
            raise ValueError('--fifth-aa-vocab-size is smaller than an ID in the structured lookup.')
        if cfg.fifth_terminal_vocab_size < lookup_terminal_size:
            raise ValueError('--fifth-terminal-vocab-size is smaller than an ID in the structured lookup.')
        cfg.fifth_aa_embedding_dim = int(args.fifth_aa_embedding_dim)
        cfg.fifth_terminal_embedding_dim = int(args.fifth_terminal_embedding_dim)
    if args.coarse_grain_enable:
        cfg.coarse_grain_enable = True
    if args.use_component_aux_features or args.component_aux_components is not None:
        cfg.use_component_aux_features = True
    if args.component_aux_components is not None:
        cfg.component_aux_components = list(
            normalize_component_aux_components(args.component_aux_components))
    elif cfg.use_component_aux_features:
        # Older YAML files have no selector and historically applied aux to
        # every component.  Make that behavior explicit in each new run.
        cfg.component_aux_components = list(normalize_component_aux_components(
            getattr(cfg, 'component_aux_components', None)))
    if args.strict_component_vocab:
        cfg.component_vocab_strict = True
    if args.use_fifth_identity_embedding:
        cfg.use_fifth_identity_embedding = True
    if args.use_fifth_class_embedding:
        cfg.use_fifth_class_embedding = True
    if args.use_fifth_ratio_modulation:
        cfg.use_fifth_ratio_modulation = True
    for key, value in {
        'head_hidden_dim': args.head_hidden_dim,
        'head_dropout': args.head_dropout,
        'fusion_hidden_dim': args.fusion_hidden_dim,
        'fusion_dropout': args.fusion_dropout,
    }.items():
        if value is not None:
            setattr(cfg.model, key, value)
    cfg.dataset.dir = str(run_dir / 'cache')
    cfg.dataset.cache_tag = (
        f'fusion-head-{args.target_set}-{args.fold}-{args.group}-{args.candidate}')
    cfg.dataset.cache_refresh = not (args.resume or args.reuse_existing_cache)
    if args.reuse_existing_cache:
        print(
            'WARNING: --reuse-existing-cache uses pre-existing processed PyG files. '
            'Do not use it after changing a loader or graph feature transform.',
            flush=True,
        )
    cfg.run_dir = str(run_dir)
    cfg.out_dir = str(run_dir)
    # The loader normally discovers vocabulary sizes while rebuilding its
    # processed cache.  A resumed OneHotEmbedGPS run reuses that cache, so the
    # discovery callback is skipped and the default [1, 1, 1, 1] would create
    # an incompatible model.  Restore the same input-only vocabulary before
    # model construction in both fresh and resumed runs.
    if cfg.model.type == 'OneHotEmbedGPS':
        vocabulary_source = Path(str(cfg.component_vocab_source or cfg.read_csv))
        if not vocabulary_source.is_file():
            raise FileNotFoundError(
                f'OneHotEmbedGPS vocabulary source is missing: {vocabulary_source}')
        vocabulary_data = pd.read_csv(vocabulary_source)
        vocabularies = build_input_component_vocab(
            vocabulary_data,
            reserve_unknown=not bool(cfg.component_vocab_strict),
        )
        cfg.component_vocab_sizes = [len(vocabulary) for vocabulary in vocabularies[:4]]
        cfg.fifth_component_vocab_size = len(vocabularies[4])
        cfg.fifth_class_vocab_size = len(
            build_input_fifth_class_vocab(vocabulary_data))
        cfg.component_vocab_source = str(vocabulary_source.resolve())
    execution_max_epochs = int(args.execution_max_epochs)
    if execution_max_epochs <= 0 or execution_max_epochs > int(cfg.optim.max_epoch):
        raise ValueError('execution-max-epochs must be in [1, configured max_epoch]')
    target_scaler_path = run_dir / 'target_scaler.json'
    if args.resume:
        if not target_scaler_path.is_file():
            raise FileNotFoundError(f'Resume run misses target scaler: {target_scaler_path}')
        target_scaler = json.loads(target_scaler_path.read_text(encoding='utf-8'))
    else:
        target_scaler = fit_target_scaler(
            cfg.read_csv, cfg.dataset.diagnostic_split_path,
            cfg.dataset.diagnostic_id_column, cfg.dataset.diagnostic_manifest_id_column,
            targets, target_scales, args.target_transform, args.target_normalization)
        target_scaler_path.write_text(
            json.dumps(target_scaler, indent=2) + '\n', encoding='utf-8')
    configure_determinism(int(cfg.seed), bool(cfg.train.deterministic))
    if not args.resume:
        shutil.copy2(args.config, run_dir / 'source_config.yaml')
        with (run_dir / 'effective_config.yaml').open('w') as stream:
            cfg.dump(stream=stream)
        (run_dir / 'run_settings.json').write_text(json.dumps({
            'fold': args.fold, 'group': args.group, 'candidate': args.candidate,
            'target_set': args.target_set,
            'single_target': args.single_target,
            'targets': targets,
            'target_scales': target_scales,
            'target_scaler': target_scaler,
            'target_normalization': args.target_normalization,
            'property_num': target_count,
            'split_manifest': str(cfg.dataset.diagnostic_split_path),
            'input_csv': str(Path(cfg.read_csv).resolve()),
            'component_vocab_source': str(Path(cfg.component_vocab_source).resolve()),
            'training_domain': args.training_domain,
            'fusion_type': args.fusion_type, 'head_type': args.head_type,
            'architecture_name': cfg.model.architecture_name,
            'execution_max_epochs': execution_max_epochs,
            'base_lr': cfg.optim.base_lr,
            'weight_decay': cfg.optim.weight_decay,
            'batch_size': cfg.train.batch_size,
            'seed': cfg.seed,
            'num_warmup_epochs': cfg.optim.num_warmup_epochs,
            'early_stop_patience': cfg.train.early_stop_patience,
            'early_stop_min_delta': cfg.train.early_stop_min_delta,
            'head_hidden_dim': cfg.model.head_hidden_dim,
            'head_dropout': cfg.model.head_dropout,
            'fusion_hidden_dim': cfg.model.fusion_hidden_dim,
            'fusion_dropout': cfg.model.fusion_dropout,
            'output_activation': cfg.model.output_activation,
            'ratio_polynomial_features': cfg.model.ratio_polynomial_features,
            'fifth_only_fusion': cfg.model.fifth_only_fusion,
            'loss_targets': args.loss_targets or targets,
            'training_loss': args.training_loss,
            'target_transform': args.target_transform,
            'huber_beta': args.huber_beta,
            'enable_norm_threshold_aware': bool(args.enable_norm_threshold_aware),
            'norm_threshold_report_only': bool(args.norm_threshold_report_only),
            'norm_threshold': float(args.norm_threshold),
            'norm_cls_loss_weight': float(args.norm_cls_loss_weight),
            'norm_fn_loss_weight': float(args.norm_fn_loss_weight),
            'norm_positive_reg_weight': float(args.norm_positive_reg_weight),
            'norm_underprediction_weight': float(args.norm_underprediction_weight),
            'min_double_high_per_batch': int(args.min_double_high_per_batch),
            'norm_threshold_selection_mae_tolerance': float(args.norm_threshold_selection_mae_tolerance),
            'gps_layers': cfg.gt.layers,
            'graph_hidden_dim': cfg.gt.dim_hidden,
            'gnn_inner_dim': cfg.gnn.dim_inner,
            'rwse_dim': cfg.posenc_RWSE.dim_pe,
            'gt_dropout': cfg.gt.dropout,
            'gt_attn_dropout': cfg.gt.attn_dropout,
            'model_type': cfg.model.type,
            'use_mordred_features': cfg.use_mordred_features,
            'mordred_feature_path': cfg.mordred_feature_path,
            'mordred_feature_dim': cfg.mordred_feature_dim,
            'mordred_fifth_only': cfg.mordred_fifth_only,
            'frozen_comp5_aux_enable': bool(cfg.model.frozen_comp5_aux_enable),
            'frozen_comp5_aux_label': str(args.frozen_comp5_aux_label),
            'frozen_comp5_aux_checkpoint': (
                str(frozen_aux_checkpoint_path)
                if frozen_aux_checkpoint_path is not None else None
            ),
            'frozen_comp5_aux_checkpoint_sha256': (
                file_sha256(frozen_aux_checkpoint_path)
                if frozen_aux_checkpoint_path is not None else None
            ),
            'architecture_audit_only': bool(args.architecture_audit_only),
            'use_fifth_mechanistic_descriptors': cfg.use_fifth_mechanistic_descriptors,
            'fifth_mechanistic_descriptor_path': cfg.fifth_mechanistic_descriptor_path,
            'fifth_mechanistic_descriptor_dim': cfg.fifth_mechanistic_descriptor_dim,
            'use_fifth_semantic_features': cfg.use_fifth_semantic_features,
            'fifth_semantic_feature_path': cfg.fifth_semantic_feature_path,
            'fifth_semantic_feature_dim': cfg.fifth_semantic_feature_dim,
            'architecture_family': 'O13G' if cfg.use_fifth_structured_features else 'O12/O13',
            'use_fifth_structured_features': cfg.use_fifth_structured_features,
            'fifth_structured_feature_path': cfg.fifth_structured_feature_path,
            'use_fifth_aa_embedding': cfg.use_fifth_structured_features,
            'fifth_aa_vocab_size': cfg.fifth_aa_vocab_size,
            'fifth_terminal_vocab_size': cfg.fifth_terminal_vocab_size,
            'use_tail_length': cfg.use_fifth_structured_features,
            'use_aa_tail_interaction': cfg.use_fifth_structured_features,
            'use_component_aux_features': cfg.use_component_aux_features,
            'component_aux_components': list(cfg.component_aux_components),
            'component_vocab_strict': cfg.component_vocab_strict,
            'component_vocab_sizes': list(cfg.component_vocab_sizes),
            'use_fifth_identity_embedding': cfg.use_fifth_identity_embedding,
            'use_fifth_class_embedding': cfg.use_fifth_class_embedding,
            'fifth_class_vocab_size': cfg.fifth_class_vocab_size,
            'use_fifth_ratio_modulation': cfg.use_fifth_ratio_modulation,
            'coarse_grain_enable': cfg.coarse_grain_enable,
            'outer_test_read_during_selection': False,
            'reused_existing_cache': bool(args.reuse_existing_cache),
            'require_membership_count': args.require_membership_count,
            'membership_reference_run_dir': (
                str(args.membership_reference_run_dir.resolve())
                if args.membership_reference_run_dir is not None else None
            ),
            'require_fresh_cache': bool(args.require_fresh_cache),
            'source_config': str(args.config.resolve()),
        }, indent=2) + '\n')
    elif not (run_dir / 'resume_state.pt').exists():
        raise FileNotFoundError('resume requested but resume_state.pt is absent')

    cache_root = Path(str(cfg.dataset.dir)).resolve()
    cache_before_build = sorted(
        str(path.relative_to(cache_root))
        for path in cache_root.glob('**/processed/*.pt')
    ) if cache_root.exists() else []
    if args.require_fresh_cache:
        if args.reuse_existing_cache or not cfg.dataset.cache_refresh:
            raise RuntimeError(
                '--require-fresh-cache rejects --reuse-existing-cache and any '
                'configuration with cache_refresh=False.'
            )
        if cache_before_build:
            raise RuntimeError(
                '--require-fresh-cache found pre-existing processed PyG files: '
                f'{cache_before_build[:5]}'
            )
    with (run_dir / 'cache_build.log').open('w') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            loaders = create_loader_5()
    active_manifest_for_membership = Path(
        str(cfg.dataset.diagnostic_split_path or cfg.train.manifest_path)
    )
    membership_preflight_audit(
        run_dir,
        cfg.read_csv,
        active_manifest_for_membership,
        loaders,
        require_membership_count=args.require_membership_count,
        membership_reference_run_dir=args.membership_reference_run_dir,
        cache_before_build=cache_before_build,
    )
    evaluation_loaders = loaders
    if args.min_double_high_per_batch:
        sampler_manifest = pd.read_csv(active_manifest_for_membership, dtype={'sample_id': str})
        loaders = apply_double_high_batch_oversampling(
            loaders, cfg.read_csv, sampler_manifest, active_manifest_for_membership,
            cfg.train.batch_size, cfg.seed, run_dir,
        )
    # Some data-dependent settings (notably input-derived OneHot component
    # vocabulary sizes) are established while building the cache. Persist the
    # final runtime configuration instead of leaving only pre-loader defaults.
    with (run_dir / 'effective_config.yaml').open('w') as stream:
        cfg.dump(stream=stream)
    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model = create_model_gps().to(device)

    # ------------------------------------------------------------------
    # Stage-5 Fifth-encoder initialization.
    #
    # create_model_gps() returns GraphGymModule, whose `.model` is the
    # registered OneHotEmbedGPSModel.  Load BEFORE optimizer construction so
    # optimizer state starts from the transferred parameters.
    # ------------------------------------------------------------------
    comp5_init_metadata = {
        'label': str(args.comp5_pretrain_label),
        'mode': 'random',
        'checkpoint': None,
        'checkpoint_sha256': None,
        'strict_transfer_report': None,
    }

    if args.comp5_pretrained_checkpoint is not None:
        checkpoint_path = args.comp5_pretrained_checkpoint.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f'Comp5 pretrained checkpoint is missing: {checkpoint_path}'
            )
        if str(cfg.model.type) != 'OneHotEmbedGPS':
            raise ValueError(
                '--comp5-pretrained-checkpoint requires --model-type OneHotEmbedGPS '
                f'(effective cfg.model.type={cfg.model.type!r}).'
            )
        if not hasattr(model, 'model'):
            raise AttributeError(
                'create_model_gps() result has no `.model` GraphGym wrapper target.'
            )
        transfer_report = load_stage4_comp5_encoder(
            model.model,
            checkpoint_path,
            map_location='cpu',
        )
        comp5_init_metadata.update({
            'mode': 'stage4_pretrained_full_finetune',
            'checkpoint': str(checkpoint_path),
            'checkpoint_sha256': file_sha256(checkpoint_path),
            'strict_transfer_report': transfer_report,
        })

    if args.comp5_lr is not None:
        if args.comp5_lr <= 0:
            raise ValueError('--comp5-lr must be positive.')
        if args.comp5_pretrained_checkpoint is None:
            raise ValueError(
                '--comp5-lr is a Stage-6 pretrained-encoder option and requires '
                '--comp5-pretrained-checkpoint.'
            )

    comp5_init_metadata['differential_lr'] = bool(args.comp5_lr is not None)
    comp5_init_metadata['rest_lr'] = float(cfg.optim.base_lr)
    comp5_init_metadata['comp5_lr'] = (
        float(args.comp5_lr)
        if args.comp5_lr is not None
        else float(cfg.optim.base_lr)
    )

    (run_dir / 'comp5_initialization.json').write_text(
        json.dumps(comp5_init_metadata, indent=2) + '\n'
    )

    print(
        '[Stage5] Comp5 initialization: '
        f"{comp5_init_metadata['label']} "
        f"mode={comp5_init_metadata['mode']}"
    )
    if comp5_init_metadata['checkpoint'] is not None:
        print(
            '[Stage5] Strict Comp5 transfer PASS: '
            f"{comp5_init_metadata['checkpoint']}"
        )

    # ------------------------------------------------------------------
    # Stage-8 frozen PT-DF structural-prior branch.
    #
    # This is intentionally separate from ``comp5_encoder`` above: Stage-8
    # compares an independently trainable task encoder with a fixed Stage-4
    # structural representation, rather than fine-tuning the same weights.
    # ------------------------------------------------------------------
    frozen_comp5_aux_metadata = {
        'enabled': bool(args.frozen_comp5_aux_checkpoint is not None),
        'label': str(args.frozen_comp5_aux_label),
        'checkpoint': None,
        'checkpoint_sha256': None,
        'strict_transfer_report': None,
        'frozen_parameter_count': 0,
        'trainable_parameter_count': 0,
        'task_comp5_trainable_parameter_count': 0,
        'optimizer_parameter_count': None,
        'optimizer_includes_frozen_parameters': None,
        'optimizer_exact_trainable_partition': None,
        'frozen_training_after_model_train': None,
        'mordred_enabled': bool(cfg.use_mordred_features),
        'mordred_feature_dim': int(cfg.mordred_feature_dim),
        'topology': {
            'model_type': str(cfg.model.type),
            'frozen_comp5_aux_enable': bool(cfg.model.frozen_comp5_aux_enable),
            'frozen_aux_module_present': False,
            'task_and_frozen_encoder_distinct': None,
            'frozen_aux_embedding_dim': 0,
        },
    }
    if args.frozen_comp5_aux_checkpoint is not None:
        checkpoint_path = frozen_aux_checkpoint_path
        if checkpoint_path is None:
            raise RuntimeError('Frozen auxiliary checkpoint path was not resolved.')
        if str(cfg.model.type) != 'OneHotEmbedGPS' or not hasattr(model, 'model'):
            raise ValueError(
                '--frozen-comp5-aux-checkpoint requires --model-type OneHotEmbedGPS.'
            )
        core = model.model
        if not getattr(core, 'frozen_comp5_aux_enable', False):
            raise RuntimeError(
                'Frozen Comp5 auxiliary checkpoint requested but model topology '
                'did not enable frozen_comp5_aux_encoder.'
            )
        if not hasattr(core, 'frozen_comp5_aux_encoder'):
            raise AttributeError(
                'OneHotEmbedGPS is missing frozen_comp5_aux_encoder.'
            )
        if core.comp5_encoder is core.frozen_comp5_aux_encoder:
            raise RuntimeError(
                'Stage-8 task and frozen Comp5 encoders must be distinct modules.'
            )
        frozen_comp5_aux_metadata['topology'].update({
            'frozen_aux_module_present': True,
            'task_and_frozen_encoder_distinct': True,
            'frozen_aux_embedding_dim': int(core.frozen_comp5_aux_dim),
        })
        transfer_report = load_stage4_encoder_into(
            core.frozen_comp5_aux_encoder,
            checkpoint_path,
            map_location='cpu',
        )
        for parameter in core.frozen_comp5_aux_encoder.parameters():
            parameter.requires_grad_(False)
        core.frozen_comp5_aux_encoder.eval()
        frozen_parameter_count = int(sum(
            parameter.numel()
            for parameter in core.frozen_comp5_aux_encoder.parameters()
        ))
        frozen_trainable_count = int(sum(
            parameter.numel()
            for parameter in core.frozen_comp5_aux_encoder.parameters()
            if parameter.requires_grad
        ))
        task_comp5_trainable_count = int(sum(
            parameter.numel()
            for parameter in core.comp5_encoder.parameters()
            if parameter.requires_grad
        ))
        if frozen_trainable_count != 0:
            raise RuntimeError('Frozen Comp5 auxiliary encoder has trainable parameters.')
        if task_comp5_trainable_count <= 0:
            raise RuntimeError('Task-specific comp5_encoder is unexpectedly frozen.')
        frozen_comp5_aux_metadata.update({
            'checkpoint': str(checkpoint_path),
            'checkpoint_sha256': file_sha256(checkpoint_path),
            'strict_transfer_report': transfer_report,
            'frozen_parameter_count': frozen_parameter_count,
            'trainable_parameter_count': frozen_trainable_count,
            'task_comp5_trainable_parameter_count': task_comp5_trainable_count,
        })
        print(
            '[Stage8] Frozen Comp5 auxiliary transfer PASS: '
            f'{checkpoint_path}'
        )
        print(
            '[Stage8] Branch audit PASS: '
            f'frozen_params={frozen_parameter_count}, '
            f'frozen_trainable={frozen_trainable_count}, '
            f'task_comp5_trainable={task_comp5_trainable_count}'
        )
    optimizer_config = OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum,
    )

    optimizer_group_metadata = {
        'differential_lr': bool(args.comp5_lr is not None),
        'rest_lr': float(cfg.optim.base_lr),
        'comp5_lr': (
            float(args.comp5_lr)
            if args.comp5_lr is not None
            else float(cfg.optim.base_lr)
        ),
        'rest_trainable_parameter_count': None,
        'comp5_trainable_parameter_count': None,
        'total_trainable_parameter_count': int(
            sum(parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad)
        ),
        'optimizer_group_lrs_before_scheduler': None,
        'scheduler_base_lrs': None,
        'optimizer_group_lrs_after_scheduler_init': None,
    }

    if args.comp5_lr is None:
        # Passing all model.parameters() would let a frozen Stage-8 branch
        # enter the optimizer even though it receives no gradients.  Filter
        # explicitly so the optimizer set is exactly the trainable set.
        optimizer_parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not optimizer_parameters:
            raise RuntimeError('Model has no trainable parameters for optimizer.')
    else:
        if not hasattr(model, 'model') or not hasattr(model.model, 'comp5_encoder'):
            raise AttributeError(
                '--comp5-lr requires model.model.comp5_encoder, but the current '
                'GraphGym model does not expose that interface.'
            )

        comp5_parameters = [
            parameter
            for parameter in model.model.comp5_encoder.parameters()
            if parameter.requires_grad
        ]
        comp5_ids = {id(parameter) for parameter in comp5_parameters}

        rest_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in comp5_ids
        ]
        rest_ids = {id(parameter) for parameter in rest_parameters}
        all_trainable = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        all_trainable_ids = {id(parameter) for parameter in all_trainable}

        if not comp5_parameters:
            raise RuntimeError(
                'Differential LR requested but Comp5GraphEncoder has no '
                'trainable parameters.'
            )
        if not rest_parameters:
            raise RuntimeError(
                'Differential LR requested but no non-Comp5 trainable '
                'parameters were found.'
            )
        if comp5_ids & rest_ids:
            raise RuntimeError(
                'Comp5/rest optimizer parameter groups overlap.'
            )
        if (comp5_ids | rest_ids) != all_trainable_ids:
            raise RuntimeError(
                'Comp5/rest optimizer parameter groups do not exactly partition '
                'all trainable model parameters.'
            )
        if len(comp5_ids) != len(comp5_parameters):
            raise RuntimeError(
                'Duplicate Comp5 parameter objects detected.'
            )
        if len(rest_ids) != len(rest_parameters):
            raise RuntimeError(
                'Duplicate rest-model parameter objects detected.'
            )

        optimizer_group_metadata.update({
            'rest_trainable_parameter_count': int(
                sum(parameter.numel() for parameter in rest_parameters)
            ),
            'comp5_trainable_parameter_count': int(
                sum(parameter.numel() for parameter in comp5_parameters)
            ),
        })

        if (
            optimizer_group_metadata['rest_trainable_parameter_count']
            + optimizer_group_metadata['comp5_trainable_parameter_count']
            != optimizer_group_metadata['total_trainable_parameter_count']
        ):
            raise RuntimeError(
                'Optimizer parameter-count partition invariant failed.'
            )

        # Put the normal/base-LR group first so existing runner logging
        # scheduler.get_last_lr()[0] continues to report the rest-model LR.
        optimizer_parameters = [
            {
                'params': rest_parameters,
                'lr': float(cfg.optim.base_lr),
            },
            {
                'params': comp5_parameters,
                'lr': float(args.comp5_lr),
            },
        ]

    if args.comp5_lr is None:
        # Preserve the historical GraphGym optimizer path exactly for all
        # non-differential-LR experiments.
        optimizer = create_optimizer(
            optimizer_parameters,
            optimizer_config,
        )
    else:
        # GraphGym create_optimizer() filters its input as if every element is
        # a Parameter object and therefore cannot accept PyTorch parameter-
        # group dictionaries.  The registered optimizer used by this project
        # is AdamW, so use native torch.optim.AdamW for the differential-LR
        # branch only.
        optimizer_name = str(cfg.optim.optimizer).strip().lower().replace('_', '')
        if optimizer_name != 'adamw':
            raise ValueError(
                '--comp5-lr currently supports only the project AdamW optimizer; '
                f'effective cfg.optim.optimizer={cfg.optim.optimizer!r}.'
            )

        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=float(cfg.optim.base_lr),
            weight_decay=float(cfg.optim.weight_decay),
        )

    optimizer_group_metadata['optimizer_group_lrs_before_scheduler'] = [
        float(group['lr']) for group in optimizer.param_groups
    ]

    if args.comp5_lr is not None:
        actual = optimizer_group_metadata['optimizer_group_lrs_before_scheduler']
        expected = [float(cfg.optim.base_lr), float(args.comp5_lr)]
        if len(actual) != 2 or any(
            abs(got - want) > max(1e-12, abs(want) * 1e-9)
            for got, want in zip(actual, expected)
        ):
            raise RuntimeError(
                'create_optimizer did not preserve differential parameter-group '
                f'learning rates: expected={expected}, actual={actual}'
            )
    scheduler = create_scheduler(optimizer, ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler, steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs, train_mode=cfg.train.mode,
        eval_period=cfg.train.eval_period))
    optimizer_group_metadata['optimizer_group_lrs_after_scheduler_init'] = [
        float(group['lr']) for group in optimizer.param_groups
    ]
    scheduler_base_lrs = getattr(scheduler, 'base_lrs', None)
    if scheduler_base_lrs is not None:
        optimizer_group_metadata['scheduler_base_lrs'] = [
            float(value) for value in scheduler_base_lrs
        ]

    all_trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    all_trainable_ids = {id(parameter) for parameter in all_trainable_parameters}
    optimizer_parameters_flat = [
        parameter
        for group in optimizer.param_groups
        for parameter in group['params']
    ]
    optimizer_parameter_ids = {
        id(parameter) for parameter in optimizer_parameters_flat
    }
    if len(optimizer_parameter_ids) != len(optimizer_parameters_flat):
        raise RuntimeError('Optimizer contains duplicate parameter objects.')
    if optimizer_parameter_ids != all_trainable_ids:
        raise RuntimeError(
            'Optimizer parameter set does not exactly equal the model '
            'requires_grad=True parameter set.'
        )

    frozen_parameter_ids = set()
    if getattr(model, 'model', None) is not None and hasattr(
        model.model, 'frozen_comp5_aux_encoder'
    ):
        frozen_parameter_ids = {
            id(parameter)
            for parameter in model.model.frozen_comp5_aux_encoder.parameters()
        }
        model.train()
        if model.model.frozen_comp5_aux_encoder.training:
            raise RuntimeError(
                'model.train() re-enabled the frozen Comp5 auxiliary branch.'
            )
        frozen_comp5_aux_metadata.update({
            'optimizer_parameter_count': int(sum(
                parameter.numel() for parameter in optimizer_parameters_flat
            )),
            'optimizer_includes_frozen_parameters': bool(
                optimizer_parameter_ids & frozen_parameter_ids
            ),
            'optimizer_exact_trainable_partition': True,
            'frozen_training_after_model_train': bool(
                model.model.frozen_comp5_aux_encoder.training
            ),
        })
        if frozen_parameter_ids & optimizer_parameter_ids:
            raise RuntimeError(
                'Frozen Comp5 auxiliary parameters entered the optimizer.'
            )
    optimizer_group_metadata['optimizer_exact_trainable_partition'] = True
    optimizer_group_metadata['optimizer_parameter_count'] = int(sum(
        parameter.numel() for parameter in optimizer_parameters_flat
    ))

    optimizer_group_metadata_path = run_dir / 'optimizer_parameter_groups.json'

    frozen_aux_audit_path = run_dir / 'frozen_comp5_aux_initialization.json'
    if args.resume and (
        args.frozen_comp5_aux_checkpoint is not None
        or frozen_aux_audit_path.is_file()
    ):
        if not frozen_aux_audit_path.is_file():
            raise FileNotFoundError(
                'Resume requested but frozen auxiliary provenance is missing: '
                f'{frozen_aux_audit_path}'
            )
        saved_frozen_aux = json.loads(
            frozen_aux_audit_path.read_text(encoding='utf-8')
        )
        for field in (
            'enabled', 'label', 'checkpoint_sha256', 'mordred_enabled',
            'mordred_feature_dim', 'frozen_parameter_count',
            'trainable_parameter_count', 'task_comp5_trainable_parameter_count',
        ):
            if saved_frozen_aux.get(field) != frozen_comp5_aux_metadata.get(field):
                raise RuntimeError(
                    'Frozen auxiliary resume provenance mismatch for '
                    f'{field}: saved={saved_frozen_aux.get(field)!r}, '
                    f'requested={frozen_comp5_aux_metadata.get(field)!r}'
                )
        if saved_frozen_aux.get('topology') != frozen_comp5_aux_metadata.get('topology'):
            raise RuntimeError(
                'Frozen auxiliary resume topology mismatch; refusing to load '
                'a state produced by a different architecture.'
            )
        settings_path = run_dir / 'run_settings.json'
        if not settings_path.is_file():
            raise FileNotFoundError(
                f'Resume requested but run settings are missing: {settings_path}'
            )
        saved_settings = json.loads(settings_path.read_text(encoding='utf-8'))
        expected_sha = frozen_comp5_aux_metadata['checkpoint_sha256']
        if saved_settings.get('frozen_comp5_aux_checkpoint_sha256') != expected_sha:
            raise RuntimeError(
                'Frozen auxiliary resume checkpoint SHA256 mismatch in '
                'run_settings.json.'
            )
        if bool(saved_settings.get('frozen_comp5_aux_enable')) != bool(
            frozen_comp5_aux_metadata['enabled']
        ):
            raise RuntimeError(
                'Frozen auxiliary resume topology flag mismatch in '
                'run_settings.json.'
            )

    cfg.model.frozen_comp5_aux_parameter_count = int(
        frozen_comp5_aux_metadata['frozen_parameter_count']
    )
    cfg.model.frozen_comp5_aux_trainable_parameter_count = int(
        frozen_comp5_aux_metadata['trainable_parameter_count']
    )
    cfg.model.frozen_comp5_aux_optimizer_parameter_count = int(
        frozen_comp5_aux_metadata['optimizer_parameter_count'] or 0
    )
    cfg.model.frozen_comp5_aux_optimizer_includes_parameters = bool(
        frozen_comp5_aux_metadata['optimizer_includes_frozen_parameters']
    )
    # Persist post-construction parameter audits in both config and settings;
    # the first effective config dump occurs before a model exists.
    with (run_dir / 'effective_config.yaml').open('w') as stream:
        cfg.dump(stream=stream)
    run_settings_path = run_dir / 'run_settings.json'
    if not run_settings_path.is_file():
        raise FileNotFoundError(
            f'Run settings are missing before frozen provenance persistence: '
            f'{run_settings_path}'
        )
    run_settings = json.loads(run_settings_path.read_text(encoding='utf-8'))
    run_settings['frozen_comp5_aux_initialization'] = frozen_comp5_aux_metadata
    run_settings['strict_no_mordred'] = bool(not cfg.use_mordred_features)
    run_settings_path.write_text(
        json.dumps(run_settings, indent=2) + '\n', encoding='utf-8'
    )
    frozen_aux_audit_path.write_text(
        json.dumps(frozen_comp5_aux_metadata, indent=2) + '\n',
        encoding='utf-8',
    )

    if args.comp5_lr is not None:
        expected_ratio = float(args.comp5_lr) / float(cfg.optim.base_lr)

        # Prefer scheduler base_lrs for the hard audit.  Fall back to current
        # optimizer LRs when the scheduler implementation exposes no base_lrs.
        ratio_source = (
            optimizer_group_metadata['scheduler_base_lrs']
            if optimizer_group_metadata['scheduler_base_lrs'] is not None
            else optimizer_group_metadata['optimizer_group_lrs_after_scheduler_init']
        )
        if len(ratio_source) != 2 or ratio_source[0] == 0:
            raise RuntimeError(
                'Cannot audit differential LR ratio after scheduler creation: '
                f'{ratio_source}'
            )
        actual_ratio = float(ratio_source[1]) / float(ratio_source[0])
        if abs(actual_ratio - expected_ratio) > max(
            1e-12, abs(expected_ratio) * 1e-7
        ):
            raise RuntimeError(
                'Scheduler did not preserve the requested Comp5/rest LR ratio: '
                f'expected={expected_ratio}, actual={actual_ratio}, '
                f'LRs={ratio_source}'
            )

        if args.resume:
            if not optimizer_group_metadata_path.is_file():
                raise FileNotFoundError(
                    'Differential-LR resume requested but '
                    f'{optimizer_group_metadata_path} is missing.'
                )
            saved_group_metadata = json.loads(
                optimizer_group_metadata_path.read_text(encoding='utf-8')
            )
            for field in ('rest_lr', 'comp5_lr'):
                requested = float(optimizer_group_metadata[field])
                saved = float(saved_group_metadata[field])
                if abs(requested - saved) > max(1e-12, abs(saved) * 1e-9):
                    raise RuntimeError(
                        'Differential-LR resume provenance mismatch for '
                        f'{field}: saved={saved}, requested={requested}'
                    )
        else:
            optimizer_group_metadata_path.write_text(
                json.dumps(optimizer_group_metadata, indent=2) + '\n',
                encoding='utf-8',
            )

        print(
            '[Stage6] Differential LR audit PASS: '
            f"rest={optimizer_group_metadata['rest_lr']:.3e}, "
            f"comp5={optimizer_group_metadata['comp5_lr']:.3e}, "
            f"ratio={expected_ratio:.6f}, "
            f"rest_params={optimizer_group_metadata['rest_trainable_parameter_count']}, "
            f"comp5_params={optimizer_group_metadata['comp5_trainable_parameter_count']}"
        )
    elif not args.resume:
        optimizer_group_metadata_path.write_text(
            json.dumps(optimizer_group_metadata, indent=2) + '\n',
            encoding='utf-8',
        )

    # Prediction rows carry loader source indices.  They must be mapped through
    # the manifest actually supplied to this run, not a stale manifest path
    # inherited from the base YAML configuration.
    active_manifest_path = Path(str(cfg.dataset.diagnostic_split_path or cfg.train.manifest_path))
    if not active_manifest_path.is_file():
        raise FileNotFoundError(f'Active split manifest is missing: {active_manifest_path}')
    manifest = pd.read_csv(active_manifest_path, dtype={'sample_id': str})
    threshold_metadata = None
    double_class_id = -1
    if args.norm_threshold_reporting:
        threshold_metadata = source_metadata(cfg.read_csv, manifest)
        domain_audit = audit_o14a_domain(
            cfg.read_csv, manifest, args.training_domain, targets)
        domain_audit.update({
            'input_csv': str(Path(cfg.read_csv).resolve()),
            'input_sha256': file_sha256(cfg.read_csv),
            'manifest_path': str(active_manifest_path.resolve()),
            'manifest_sha256': file_sha256(active_manifest_path),
            'component_vocab_source': str(Path(cfg.component_vocab_source).resolve()),
        })
        (run_dir / 'o14a_domain_audit.json').write_text(
            json.dumps(domain_audit, indent=2) + '\n', encoding='utf-8')
        if args.enable_norm_threshold_aware or args.norm_positive_reg_weight != 1.0 \
                or args.norm_underprediction_weight != 0.0:
            class_source = pd.read_csv(cfg.component_vocab_source)
            double_class_id = build_input_fifth_class_vocab(class_source).get('double', -1)
        if not args.resume:
            (run_dir / 'norm_threshold_audit.json').write_text(json.dumps({
                'experiment': 'O14-A Threshold-Aware Norm Regression',
                'single_target': targets[0],
                'threshold': float(args.norm_threshold),
                'threshold_definition': 'high_target = (raw physical Norm > threshold)',
                'threshold_label_examples': {
                    '0.99': int(0.99 > args.norm_threshold),
                    '1.00': int(1.00 > args.norm_threshold),
                    '1.01': int(1.01 > args.norm_threshold),
                },
                'threshold_prediction_space': (
                    'inverse_predictions(model_prediction, target_transform, target_scaler); '
                    'the crossing loss never uses transformed/z-scored values'),
                'classification_head_enabled': bool(args.enable_norm_threshold_aware),
                'classification_head_attachment': (
                    'final shared fusion representation (fusion_backbone)'
                    if args.enable_norm_threshold_aware else None),
                'classification_loss': (
                    'BCEWithLogitsLoss(high_logit, raw_norm > threshold)'
                    if args.enable_norm_threshold_aware else None),
                'crossing_loss': 'mean(relu(threshold - raw_prediction[raw_norm > threshold]) ** 2)',
                'positive_regression_weight_policy': (
                    'only train minibatch rows with Fifth_class=double and raw_norm > threshold'),
                'double_class_id': int(double_class_id),
                'cls_loss_weight': float(args.norm_cls_loss_weight),
                'fn_loss_weight': float(args.norm_fn_loss_weight),
                'positive_regression_weight': float(args.norm_positive_reg_weight),
                'double_high_underprediction_weight': float(args.norm_underprediction_weight),
                'test_used_for_selection': False,
            }, indent=2) + '\n', encoding='utf-8')
    checkpoint_metadata = {
        'fusion_type': args.fusion_type,
        'head_type': args.head_type,
        'architecture_name': cfg.model.architecture_name,
        'target_set': args.target_set,
        'single_target': args.single_target,
        'targets': targets,
        'target_scales': target_scales,
        'target_scaler': target_scaler,
        'property_num': target_count,
        'loss_targets': args.loss_targets or targets,
        'target_transform': args.target_transform,
        'target_normalization': args.target_normalization,
        'enable_norm_threshold_aware': bool(args.enable_norm_threshold_aware),
        'norm_threshold_report_only': bool(args.norm_threshold_report_only),
        'norm_threshold': float(args.norm_threshold),
        'norm_cls_loss_weight': float(args.norm_cls_loss_weight),
        'norm_fn_loss_weight': float(args.norm_fn_loss_weight),
        'norm_positive_reg_weight': float(args.norm_positive_reg_weight),
        'use_mordred_features': bool(cfg.use_mordred_features),
        'mordred_feature_dim': int(cfg.mordred_feature_dim),
        'use_component_aux_features': bool(cfg.use_component_aux_features),
        'component_aux_components': list(cfg.component_aux_components),
        'architecture_hash': hashlib.sha256(
            f'{cfg.model.architecture_name}|{args.fusion_type}|{args.head_type}'.encode()).hexdigest(),
        'manifest_path': str(active_manifest_path.resolve()),
        'manifest_hash': file_sha256(active_manifest_path),
        'feature_hash': file_sha256(cfg.mordred_feature_path) if cfg.use_mordred_features else None,
        'fifth_mechanistic_feature_hash': (
            file_sha256(cfg.fifth_mechanistic_descriptor_path)
            if cfg.use_fifth_mechanistic_descriptors else None
        ),
        'fifth_semantic_feature_hash': (
            file_sha256(cfg.fifth_semantic_feature_path)
            if cfg.use_fifth_semantic_features else None
        ),
        'config_hash': file_sha256(args.config),
    }
    checkpoint_metadata['comp5_initialization'] = comp5_init_metadata
    checkpoint_metadata['frozen_comp5_aux_initialization'] = frozen_comp5_aux_metadata
    if args.architecture_audit_only:
        if frozen_comp5_aux_metadata['enabled']:
            required_audit_values = {
                'strict_transfer': (
                    frozen_comp5_aux_metadata['strict_transfer_report'] or {}
                ).get('strict') is True,
                'frozen_trainable_parameter_count': (
                    frozen_comp5_aux_metadata['trainable_parameter_count'] == 0
                ),
                'task_comp5_trainable_parameter_count': (
                    frozen_comp5_aux_metadata['task_comp5_trainable_parameter_count'] > 0
                ),
                'optimizer_excludes_frozen': (
                    frozen_comp5_aux_metadata['optimizer_includes_frozen_parameters'] is False
                ),
                'optimizer_exact_trainable_partition': (
                    frozen_comp5_aux_metadata['optimizer_exact_trainable_partition'] is True
                ),
                'frozen_eval_after_model_train': (
                    frozen_comp5_aux_metadata['frozen_training_after_model_train'] is False
                ),
                'distinct_encoders': (
                    frozen_comp5_aux_metadata['topology'].get(
                        'task_and_frozen_encoder_distinct'
                    ) is True
                ),
            }
            failed = [
                name for name, passed in required_audit_values.items()
                if not passed
            ]
            if failed:
                raise RuntimeError(
                    'Stage-8 architecture audit failed: ' + ', '.join(failed)
                )
        architecture_audit = {
            'status': 'PASS',
            'audit_only': True,
            'model_type': str(cfg.model.type),
            'use_mordred_features': bool(cfg.use_mordred_features),
            'mordred_feature_dim': int(cfg.mordred_feature_dim),
            'optimizer_parameter_groups': optimizer_group_metadata,
            'comp5_initialization': comp5_init_metadata,
            'frozen_comp5_aux_initialization': frozen_comp5_aux_metadata,
        }
        audit_path = run_dir / 'architecture_audit.json'
        audit_path.write_text(
            json.dumps(architecture_audit, indent=2) + '\n', encoding='utf-8'
        )
        print(
            '[Architecture audit] PASS: '
            f'frozen_aux={frozen_comp5_aux_metadata["enabled"]}, '
            f'mordred={cfg.use_mordred_features}, '
            f'output={audit_path}'
        )
        return
    start_epoch, best_loss, best_epoch, best_state = 0, math.inf, None, None
    early_reference, early_counter = math.inf, 0
    threshold_best_score, threshold_best_subset = -math.inf, 'none'
    threshold_best_loss, threshold_best_epoch, threshold_best_state = math.inf, None, None
    if args.resume:
        state = torch.load(run_dir / 'resume_state.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        start_epoch, best_loss, best_epoch, best_state = (state['next_epoch'], state['best_loss'],
                                                           state['best_epoch'], state['best_state'])
        early_reference, early_counter = state['early_reference'], state['early_counter']
        threshold_best_score = state.get('threshold_best_score', threshold_best_score)
        threshold_best_subset = state.get('threshold_best_subset', threshold_best_subset)
        threshold_best_loss = state.get('threshold_best_loss', threshold_best_loss)
        threshold_best_epoch = state.get('threshold_best_epoch', threshold_best_epoch)
        threshold_best_state = state.get('threshold_best_state', threshold_best_state)
        restore_rng_state(state['rng_state'])

    completed, last_epoch = False, start_epoch - 1
    epoch_iterator = range(start_epoch, execution_max_epochs)
    progress_bar = None
    if args.tqdm_progress:
        progress_bar = tqdm(
            epoch_iterator,
            total=execution_max_epochs - start_epoch,
            desc=f'{args.target_set} {args.fold}',
            unit='epoch', dynamic_ncols=True,
        )
        epoch_iterator = progress_bar
    for epoch in epoch_iterator:
        model.train()
        train_pred, train_label, train_source, train_high_logits = [], [], [], []
        train_objective_values = []
        train_diag = ([], [], [], [])
        train_grad = []
        for batch_index, batches in enumerate(zip(*[group[0] for group in loaders])):
            batches = prepare_batches(list(batches), 'train', device)
            optimizer.zero_grad()
            model_prediction, label, high_logit = unpack_model_output(model(*batches))
            transformed_label = transform_targets(
                label, args.target_transform, target_scaler)
            if args.enable_norm_threshold_aware or args.norm_positive_reg_weight != 1.0 \
                    or args.norm_underprediction_weight != 0.0:
                raw_prediction = inverse_predictions(
                    model_prediction, args.target_transform, target_scaler)
                class_ids = batches[4].fifth_class_id.view(-1).long()
                double_mask = class_ids.eq(int(double_class_id))
                if high_logit is None:
                    if args.enable_norm_threshold_aware:
                        raise RuntimeError('O14-A enabled but the model did not return a high-Norm logit.')
                regression_loss = weighted_regression_loss(
                    model_prediction, transformed_label, label, double_mask,
                    args.norm_positive_reg_weight, args.norm_threshold,
                    args.training_loss, args.huber_beta)
                underprediction_loss = double_high_underprediction_loss(
                    raw_prediction, label, double_mask, args.norm_threshold)
                if args.enable_norm_threshold_aware:
                    classification_loss = functional.binary_cross_entropy_with_logits(
                        high_logit.view(-1), high_target(label, args.norm_threshold))
                    crossing_loss = crossing_false_negative_loss(
                        raw_prediction, label, args.norm_threshold)
                    loss = (regression_loss
                            + args.norm_cls_loss_weight * classification_loss
                            + args.norm_fn_loss_weight * crossing_loss
                            + args.norm_underprediction_weight * underprediction_loss)
                else:
                    loss = regression_loss + args.norm_underprediction_weight * underprediction_loss
            else:
                loss = property_losses(model_prediction, transformed_label, target_count,
                                       selected_target_indices,
                                       args.training_loss, args.huber_beta).sum()
            loss.backward()
            if batch_index == 0:
                train_diag = diagnostic_rows(model.model, epoch, 'train', targets)
                train_grad = module_gradient_rows(model.model, epoch)
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_objective_values.append(float(loss.detach().cpu()))
            train_pred.append(
                inverse_predictions(model_prediction, args.target_transform, target_scaler)
                .detach().cpu().reshape(-1, target_count).numpy())
            train_label.append(label.detach().cpu().reshape(-1, target_count).numpy())
            train_source.append(batches[0].sample_uid.detach().cpu().numpy().reshape(-1))
            if high_logit is not None:
                train_high_logits.append(high_logit.detach().cpu().reshape(-1).numpy())
        train_pred, train_label = np.vstack(train_pred), np.vstack(train_label)
        train_source = np.concatenate(train_source)
        train_high_logits = np.concatenate(train_high_logits) if train_high_logits else None
        train_mae_by_target = np.mean(np.abs(train_pred - train_label), axis=0)
        train_loss = float(train_mae_by_target.sum() if selected_target_indices is None
                           else train_mae_by_target[selected_target_indices].sum())
        val_pred, val_label, val_source, val_loss, val_diag, val_high_logits = evaluate(
            model, evaluation_loaders, 'val', device, epoch, target_count, targets,
            selected_target_indices,
            collect_diagnostics=True,
            target_transform=args.target_transform, target_scaler=target_scaler)
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
                             best_epoch, best_loss, early_counter, cfg.model.architecture_name,
                             targets, target_scales)
        metric += metric_rows(val_pred, val_label, 'val', epoch, val_loss, lr,
                              best_epoch, best_loss, early_counter, cfg.model.architecture_name,
                              targets, target_scales)
        append_rows(metric, run_dir / 'epoch_metrics.csv', schemas['epoch_metrics.csv'])
        if args.norm_threshold_reporting:
            train_threshold_metrics = threshold_metric_rows(
                train_pred, train_label, train_high_logits, train_source, threshold_metadata,
                'train', epoch, targets[0], target_scales[0], args.norm_threshold,
                train_loss)
            val_threshold_metrics = threshold_metric_rows(
                val_pred, val_label, val_high_logits, val_source, threshold_metadata,
                'val', epoch, targets[0], target_scales[0], args.norm_threshold, val_loss)
            threshold_schema = [
                'epoch', 'split', 'target', 'subset', 'decision_source', 'threshold',
                'regression_loss_normalized', 'mae', 'rmse', 'r2', 'pearson', 'spearman',
                'n', 'tp', 'tn', 'fp', 'fn',
                'precision_gt1', 'recall_gt1', 'specificity_gt1', 'f1_gt1', 'f2_gt1',
                'accuracy_gt1', 'false_negative_rate_gt1', 'false_positive_rate_gt1',
                'auroc_gt1', 'auprc_gt1',
            ]
            append_rows(train_threshold_metrics + val_threshold_metrics,
                        run_dir / 'threshold_metrics.csv', threshold_schema)
            if args.enable_norm_threshold_aware:
                threshold_score, threshold_subset = threshold_selection_score(val_threshold_metrics)
                acceptable = val_loss <= best_loss * (1.0 + args.norm_threshold_selection_mae_tolerance)
                if (acceptable and (threshold_score > threshold_best_score
                                    or (threshold_score == threshold_best_score
                                        and val_loss < threshold_best_loss))):
                    threshold_best_score, threshold_best_subset = threshold_score, threshold_subset
                    threshold_best_loss, threshold_best_epoch = val_loss, epoch
                    threshold_best_state = copy_state(model)
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
            'threshold_best_score': threshold_best_score,
            'threshold_best_subset': threshold_best_subset,
            'threshold_best_loss': threshold_best_loss,
            'threshold_best_epoch': threshold_best_epoch,
            'threshold_best_state': threshold_best_state,
            'rng_state': save_rng_state(),
        }, run_dir / 'resume_state.pt')
        epoch_status = {'train_mae': f'{train_loss:.4f}',
                        'train_objective': f'{np.mean(train_objective_values):.4f}',
                        'val_mae': f'{val_loss:.4f}',
                        'best_epoch': best_epoch, 'best_val_mae': f'{best_loss:.4f}',
                        'lr': f'{lr:.2e}', 'patience': early_counter}
        if progress_bar is not None:
            progress_bar.set_postfix(epoch_status, refresh=False)
        else:
            print(json.dumps({'epoch': epoch, 'train_loss': train_loss,
                              'train_objective': float(np.mean(train_objective_values)), 'val_loss': val_loss,
                              'best_epoch': best_epoch, 'best_val_loss': best_loss,
                              'lr': lr, 'early_counter': early_counter}), flush=True)
        last_epoch = epoch
        if early_counter >= int(cfg.train.early_stop_patience) or epoch + 1 == execution_max_epochs:
            completed = True
            break
        if args.chunk_epochs is not None and epoch - start_epoch + 1 >= args.chunk_epochs:
            break

    if progress_bar is not None:
        progress_bar.close()

    if best_state is None:
        raise RuntimeError('No training epoch completed.')
    if not completed:
        return
    selected_checkpoint = run_dir / 'checkpoints' / 'selected_best.pt'
    torch.save({'model_state': best_state, 'epoch': best_epoch,
                'validation_loss': best_loss, **checkpoint_metadata}, selected_checkpoint)
    threshold_checkpoint = None
    if args.enable_norm_threshold_aware:
        if threshold_best_state is None:
            threshold_best_state, threshold_best_epoch = best_state, best_epoch
            threshold_best_loss, threshold_best_subset = best_loss, 'fallback_regression_best'
        threshold_checkpoint = run_dir / 'checkpoints' / 'best_threshold_aware.pt'
        torch.save({
            'model_state': threshold_best_state, 'epoch': threshold_best_epoch,
            'validation_loss': threshold_best_loss,
            'threshold_selection_f2_gt1': (
                threshold_best_score if math.isfinite(threshold_best_score) else None),
            'threshold_selection_subset': threshold_best_subset,
            **checkpoint_metadata,
        }, threshold_checkpoint)
    model.load_state_dict(best_state, strict=True)
    frames, threshold_frames, selected_threshold_metrics = [], [], []
    for split in ('train', 'val') + (('test',) if args.include_test else ()):
        pred, label, source, regression_loss, _, high_logits = evaluate(
            model, evaluation_loaders, split, device, last_epoch, target_count, targets,
            target_transform=args.target_transform, target_scaler=target_scaler)
        frames.append(prediction_frame(pred, label, source, manifest, split, best_epoch,
                                       str(selected_checkpoint), cfg.model.architecture_name,
                                       targets, target_scales))
        if args.norm_threshold_reporting:
            selected_threshold_metrics.extend(threshold_metric_rows(
                pred, label, high_logits, source, threshold_metadata, split, best_epoch,
                targets[0], target_scales[0], args.norm_threshold, regression_loss))
            threshold_frame = threshold_prediction_frame(
                pred, label, high_logits, source, threshold_metadata, split, best_epoch,
                str(selected_checkpoint), targets[0], target_scales[0], args.norm_threshold)
            threshold_frame['fold'] = args.fold
            threshold_frame['seed'] = int(cfg.seed)
            threshold_frames.append(threshold_frame)
    pd.concat(frames, ignore_index=True).to_csv(run_dir / 'predictions.csv', index=False)
    verify_prediction_membership(run_dir)
    if args.norm_threshold_reporting:
        selected_threshold = pd.DataFrame(selected_threshold_metrics)
        selected_threshold.to_csv(run_dir / 'threshold_metrics_selected_checkpoint.csv', index=False)
        threshold_predictions = pd.concat(threshold_frames, ignore_index=True)
        threshold_predictions.to_csv(run_dir / 'threshold_predictions.csv', index=False)
        threshold_predictions.loc[threshold_predictions.threshold_crossing_error].to_csv(
            run_dir / 'threshold_regression_false_negatives.csv', index=False)
        pd.DataFrame(double_high_magnitude_rows(threshold_predictions)).to_csv(
            run_dir / 'double_test_high_norm_magnitude_strata.csv', index=False)
        pd.DataFrame(threshold_prediction_separation_rows(threshold_predictions)).to_csv(
            run_dir / 'threshold_prediction_separation.csv', index=False)
    collapse_events(run_dir)
    summary = {
        'fold': args.fold, 'group': args.group, 'candidate': args.candidate,
        'fusion_type': args.fusion_type, 'head_type': args.head_type,
        'target_set': args.target_set, 'targets': targets,
        'single_target': args.single_target,
        'target_scales': target_scales, 'target_scaler': target_scaler,
        'target_normalization': args.target_normalization,
        'property_num': target_count,
        'architecture_name': cfg.model.architecture_name, 'last_epoch': last_epoch,
        'best_epoch': best_epoch, 'best_validation_loss_normalized': best_loss,
        'loss_targets': args.loss_targets or targets,
        'training_loss': args.training_loss, 'huber_beta': args.huber_beta,
        'target_transform': args.target_transform,
        'enable_norm_threshold_aware': bool(args.enable_norm_threshold_aware),
        'norm_threshold_report_only': bool(args.norm_threshold_report_only),
        'norm_threshold': float(args.norm_threshold),
        'norm_cls_loss_weight': float(args.norm_cls_loss_weight),
        'norm_fn_loss_weight': float(args.norm_fn_loss_weight),
        'norm_positive_reg_weight': float(args.norm_positive_reg_weight),
        'norm_underprediction_weight': float(args.norm_underprediction_weight),
        'norm_threshold_selection_mae_tolerance': float(args.norm_threshold_selection_mae_tolerance),
        'threshold_aware_checkpoint': str(threshold_checkpoint) if threshold_checkpoint else None,
        'threshold_aware_best_epoch': threshold_best_epoch if args.enable_norm_threshold_aware else None,
        'threshold_aware_selection_f2_gt1': (
            threshold_best_score if args.enable_norm_threshold_aware
            and math.isfinite(threshold_best_score) else None),
        'threshold_aware_selection_subset': threshold_best_subset if args.enable_norm_threshold_aware else None,
        'outer_test_read_during_selection': False,
        'parameter_count': int(sum(parameter.numel() for parameter in model.parameters())),
        'core_model_parameter_count': int(sum(parameter.numel() for parameter in model.model.parameters())),
    }
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
