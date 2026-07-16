#!/usr/bin/env python
"""Analyze formulation embeddings produced by a trained five-component GNN."""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import graphgps  # noqa: F401  Register custom GraphGPS components.
from graphgps.config.config_gps import set_cfg_gps
from graphgps.create_model_gps import create_model_gps
from graphgps.predicted_finetuning import set_new_cfg_allowed
from loader_5 import create_loader_5
from torch_geometric.graphgym.checkpoint import MODEL_STATE
from torch_geometric.graphgym.config import cfg


PROPERTY_NAMES = [
    'EE_before', 'EE_after', 'Aero_Efficiency', 'Recovery_Efficiency'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract and visualize GNN formulation embeddings.'
    )
    parser.add_argument('--config', required=True,
                        help='Training config or saved config.yaml for the checkpoint.')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to one .ckpt file.')
    parser.add_argument('--csv', required=True,
                        help='Absolute or repository-relative input CSV path.')
    parser.add_argument('--dataset-name', default='dataset',
                        help='Label used in output file names, e.g. train or feedback.')
    parser.add_argument('--component', default='all',
                        choices=['all', '1', '2', '3', '4', '5'],
                        help='Analyze all fused features or only one component embedding.')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for CSV, JSON, and PNG outputs.')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Extraction device. CPU is sufficient for analysis.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-pairs', type=int, default=50000,
                        help='Maximum random sample pairs per property correlation.')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def resolve_path(path):
    return os.path.abspath(path)


def configure(config_path, csv_path, cache_dir, batch_size, device):
    set_cfg_gps(cfg)
    set_new_cfg_allowed(cfg, True)
    cfg.merge_from_file(config_path)
    cfg.defrost()
    cfg.accelerator = device
    cfg.gpu_serial = 0
    cfg.devices = 1 if device == 'cuda' else 0
    cfg.num_workers = 0
    cfg.train.mode = 'double_predict'
    cfg.train.batch_size = batch_size
    cfg.read_csv = csv_path
    cfg.result_out = False
    cfg.dataset.dir = cache_dir
    cfg.wandb.use = False


def read_checkpoint_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu',
                            weights_only=False)
    return checkpoint[MODEL_STATE]


def extract_embeddings(model, loader_groups, device):
    captured = []

    def capture_fusion_input(_module, inputs):
        captured.append(inputs[0].detach().cpu())

    fusion_layer = (model.model.FC_layers[0]
                    if hasattr(model.model, 'FC_layers')
                    else model.model.fusion[0])
    handle = fusion_layer.register_forward_pre_hook(capture_fusion_input)
    embeddings, predictions, targets = [], [], []
    valid_loaders = [loader_group[1] for loader_group in loader_groups]
    model.eval()
    with torch.no_grad():
        for batches in zip(*valid_loaders):
            for batch in batches:
                batch.to(device)
            prediction, target = model(*batches)
            batch_size = batches[0].num_graphs
            embeddings.append(captured.pop().numpy())
            predictions.append(prediction.view(batch_size, -1).cpu().numpy())
            targets.append(target.view(batch_size, -1).cpu().numpy())
    handle.remove()
    return (np.concatenate(embeddings), np.concatenate(predictions),
            np.concatenate(targets))


def select_component_embeddings(embeddings, component, hidden_dim):
    if component == 'all':
        return embeddings

    component_index = int(component) - 1
    start = component_index * hidden_dim
    end = start + hidden_dim
    if embeddings.shape[1] < end:
        raise ValueError(
            f'Fusion embedding has {embeddings.shape[1]} dimensions; '
            f'cannot select component {component} with hidden_dim={hidden_dim}.'
        )
    return embeddings[:, start:end]


def pair_indices(sample_count, max_pairs, random_state):
    possible_pairs = sample_count * (sample_count - 1) // 2
    requested_pairs = min(max_pairs, possible_pairs)
    first = random_state.integers(0, sample_count, requested_pairs)
    second = random_state.integers(0, sample_count, requested_pairs)
    keep = first != second
    return first[keep], second[keep]


def plot_pca(points, values, property_name, output_dir):
    figure, axis = plt.subplots(figsize=(7, 6))
    scatter = axis.scatter(points[:, 0], points[:, 1], c=values,
                           cmap='viridis', s=24, alpha=0.8)
    figure.colorbar(scatter, ax=axis, label=property_name)
    axis.set(xlabel='PCA component 1', ylabel='PCA component 2',
             title=f'GNN formulation embeddings: {property_name}')
    figure.tight_layout()
    figure.savefig(output_dir / f'pca_{property_name}.png', dpi=180)
    plt.close(figure)


def plot_distance_relationship(distances, differences, property_name, output_dir,
                               correlation):
    figure, axis = plt.subplots(figsize=(7, 6))
    plot = axis.hexbin(distances, differences, gridsize=45, mincnt=1,
                       cmap='magma')
    figure.colorbar(plot, ax=axis, label='Pair count')
    axis.set(xlabel='Cosine distance between formulation embeddings',
             ylabel=f'Absolute {property_name} difference',
             title=f'{property_name}: Spearman r = {correlation:.3f}')
    figure.tight_layout()
    figure.savefig(output_dir / f'distance_vs_{property_name}.png', dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    checkpoint_path = Path(resolve_path(args.checkpoint))
    csv_path = resolve_path(args.csv)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    output_dir = Path(args.output_dir or
                      f'analysis/{checkpoint_path.parent.parent.parent.name}_'
                      f'{args.dataset_name}_seed{args.seed}')
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / 'dataset_cache'
    cache_dir.mkdir(exist_ok=True)

    device = torch.device(args.device if args.device == 'cpu' else 'cuda:0')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')

    configure(resolve_path(args.config), csv_path, str(cache_dir),
              args.batch_size, args.device)
    checkpoint_state = read_checkpoint_state(str(checkpoint_path))
    cfg.use_component_aux_features = any(
        'aux_feature_encoder' in key for key in checkpoint_state
    )
    loaders = create_loader_5()
    model = create_model_gps(to_device=False).to(device)
    model.load_state_dict(checkpoint_state, strict=True)

    embeddings, predictions, targets = extract_embeddings(model, loaders, device)
    sample_count = len(pd.read_csv(csv_path))
    embeddings = embeddings[:sample_count]
    predictions = predictions[:sample_count]
    targets = targets[:sample_count]
    embeddings = select_component_embeddings(
        embeddings, args.component, cfg.gt.dim_hidden
    )
    if targets.shape[1] != len(PROPERTY_NAMES):
        raise ValueError('This analyzer currently supports four-property models.')

    pca = PCA(n_components=2, random_state=args.seed)
    pca_points = pca.fit_transform(embeddings)
    embedding_frame = pd.DataFrame({
        'pca_1': pca_points[:, 0], 'pca_2': pca_points[:, 1]
    })
    for index, name in enumerate(PROPERTY_NAMES):
        embedding_frame[f'true_{name}'] = targets[:, index]
        embedding_frame[f'pred_{name}'] = predictions[:, index] * 100.0
        plot_pca(pca_points, targets[:, index], name, output_dir)
    embedding_frame.to_csv(output_dir / 'embeddings_pca.csv', index=False)
    np.save(output_dir / 'fusion_embeddings.npy', embeddings)

    random_state = np.random.default_rng(args.seed)
    first, second = pair_indices(len(embeddings), args.max_pairs, random_state)
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1,
                                          keepdims=True), 1e-12)
    distances = 1.0 - np.sum(normalized[first] * normalized[second], axis=1)
    report = {
        'checkpoint': str(checkpoint_path), 'csv': csv_path,
        'component': args.component,
        'sample_count': int(len(embeddings)),
        'embedding_dimension': int(embeddings.shape[1]),
        'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'properties': {}
    }
    for index, name in enumerate(PROPERTY_NAMES):
        differences = np.abs(targets[first, index] - targets[second, index])
        correlation, p_value = spearmanr(distances, differences)
        plot_distance_relationship(distances, differences, name, output_dir,
                                   correlation)
        report['properties'][name] = {
            'distance_target_spearman_r': float(correlation),
            'p_value': float(p_value),
            'target_mean': float(targets[:, index].mean()),
            'target_std': float(targets[:, index].std()),
        }
    with open(output_dir / 'embedding_report.json', 'w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Analysis outputs saved to: {output_dir}')


if __name__ == '__main__':
    main()
