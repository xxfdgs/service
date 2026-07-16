"""
Direct evaluation script: load OneHotEmbedGPS checkpoints (5 seeds),
run inference on the full feedback dataset, compute MAE and R^2.
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import graphgps
from graphgps.component_vocab import get_vocab_id, build_component_embeddings
from graphgps.config.config_gps import set_cfg_gps
from yacs.config import CfgNode as CN
from torch_geometric.graphgym.config import cfg as global_cfg
from torch_geometric.graphgym.register import network_dict
from graphgps.create_model_gps import create_model_gps

from torch_geometric.data import Data, Batch
from graph_feature import smiles2graph

CKPT_DIR = 'results/onehot_train'
FEEDBACK_CSV = 'datasets_lrx/raw/feedback/20260703_validation.csv'
CONFIG_YAML = 'configs/GPS/onehot_predict.yaml'
N_SEEDS = 5

COMPONENTS = ['IL_SMILE', 'HL_SMILE', 'Chol_SMILE', 'PEG_SMILE', 'Fifth_SMILE']
PROPERTIES = ['EE_before', 'EE_after', 'Aero_Efficiency', 'Recovery_Efficiency']


def load_model_for_seed(seed):
    """Load OneHotEmbedGPS model with checkpoint for a given seed."""
    # Re-init config
    cfg = CN()
    set_cfg_gps(cfg)
    cfg.merge_from_file(CONFIG_YAML)
    cfg.seed = seed

    # Update global cfg with relevant fields
    for key in ['gt', 'gnn', 'dataset', 'model', 'posenc_RWSE', 'property_num',
                'accelerator', 'gpu_serial']:
        if hasattr(cfg, key):
            setattr(global_cfg, key, getattr(cfg, key))

    global_cfg.share.dim_in = 1
    global_cfg.share.dim_out = 1

    # Create model
    model = create_model_gps(to_device=True, dim_in=1, dim_out=cfg.property_num)
    model.eval()

    # Load checkpoint
    ckpt_dir = os.path.join(CKPT_DIR, str(seed), 'ckpt')
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    if not ckpt_files:
        raise FileNotFoundError(f'No checkpoint for seed {seed}')
    ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f'  Seed {seed}: loaded {ckpt_files[0]}')
    return model


def smiles_to_batch(smiles_list, ratios):
    """Convert a list of SMILES strings to a PyG Batch object."""
    data_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Create dummy graph
            g = {'node_feat': np.zeros((1, 9), dtype=np.int64),
                 'edge_index': np.empty((2, 0), dtype=np.int64),
                 'edge_feat': np.empty((0, 1), dtype=np.int64),
                 'num_nodes': 1}
        else:
            g = smiles2graph(mol)

        x = torch.tensor(g['node_feat'], dtype=torch.long)
        edge_index = torch.tensor(g['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(g['edge_feat'], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=g['num_nodes'])
        data_list.append(data)

    batch = Batch.from_data_list(data_list)
    batch.ratio = torch.tensor(ratios, dtype=torch.float32)
    batch.to(torch.device('cuda', 0))
    return batch


def predict_feedback():
    """Run ensemble prediction on the full feedback dataset."""
    # Load feedback data
    fb = pd.read_csv(FEEDBACK_CSV)
    print(f'Feedback dataset: {len(fb)} samples')

    # Extract SMILES and true values
    true_values = fb[PROPERTIES].values  # [N, 4]

    # Load all 5 models
    models = []
    for seed in range(N_SEEDS):
        model = load_model_for_seed(seed)
        models.append(model)

    # Run prediction for each seed
    all_preds = []
    with torch.no_grad():
        for seed, model in enumerate(models):
            preds = []
            for idx in range(len(fb)):
                row = fb.iloc[idx]
                smiles_list = [row[c] for c in COMPONENTS]
                ratios = [float(row.get(f'ratio_{i+1}', 0.2)) for i in range(5)]
                # Normalize if all zeros
                if sum(ratios) > 0:
                    ratios = [r / sum(ratios) for r in ratios]
                else:
                    ratios = [0.2] * 5

                # Handle NaN SMILES (especially for component 5)
                clean_smiles = []
                clean_ratios = []
                for smi, r in zip(smiles_list, ratios):
                    if pd.isna(smi) or str(smi).strip() == '' or str(smi).lower() == 'nan':
                        smi = 'C'  # placeholder methane
                    clean_smiles.append(str(smi))
                    clean_ratios.append(r)

                # Create batch for each component
                batches = []
                for smi in clean_smiles:
                    b = smiles_to_batch([smi], [1.0])
                    batches.append(b)

                # Forward
                pred, _ = model(batches[0], batches[1], batches[2], batches[3], batches[4])
                # pred is flattened [out_dim * batch_size], reshape to [B, out_dim]
                pred = pred.view(-1, len(PROPERTIES))
                preds.append(pred.cpu().numpy())

            seed_preds = np.concatenate(preds, axis=0)  # [N, 4]
            all_preds.append(seed_preds)

    # Ensemble: average across seeds
    ensemble_preds = np.mean(all_preds, axis=0)  # [N, 4]

    # Compute metrics
    print('\n' + '=' * 80)
    print('OneHotEmbedGPS — Feedback Dataset Evaluation (5-seed ensemble)')
    print('=' * 80)
    print(f'{"Property":<25} {"MAE":>10} {"R²":>10}')
    print('-' * 50)

    for i, prop in enumerate(PROPERTIES):
        true = true_values[:, i]
        pred = ensemble_preds[:, i]

        # Remove any potential NaN
        mask = ~(np.isnan(true) | np.isnan(pred))
        true_clean = true[mask]
        pred_clean = pred[mask]

        mae = mean_absolute_error(true_clean, pred_clean)
        r2 = r2_score(true_clean, pred_clean)
        print(f'{prop:<25} {mae:>10.4f} {r2:>10.4f}')

    # Overall (sum of MAEs)
    sum_true = true_values.sum(axis=1)
    sum_pred = ensemble_preds.sum(axis=1)
    mask = ~(np.isnan(sum_true) | np.isnan(sum_pred))
    overall_mae = mean_absolute_error(sum_true[mask], sum_pred[mask])
    overall_r2 = r2_score(sum_true[mask], sum_pred[mask])
    print('-' * 50)
    print(f'{"MAE sum (overall)":<25} {overall_mae:>10.4f} {overall_r2:>10.4f}')

    # Per-seed variance
    print(f'\nPer-seed MAE sum: {[mean_absolute_error(true_values.sum(axis=1), p.sum(axis=1)) for p in all_preds]}')

    # Save results
    output = pd.DataFrame({
        'true_EE_before': true_values[:, 0],
        'pred_EE_before': ensemble_preds[:, 0],
        'true_EE_after': true_values[:, 1],
        'pred_EE_after': ensemble_preds[:, 1],
        'true_Aero_Efficiency': true_values[:, 2],
        'pred_Aero_Efficiency': ensemble_preds[:, 2],
        'true_Recovery_Efficiency': true_values[:, 3],
        'pred_Recovery_Efficiency': ensemble_preds[:, 3],
    })
    output.to_csv('results/onehot_feedback_predictions.csv', index=False)
    print(f'\nSaved predictions to results/onehot_feedback_predictions.csv')

    return output


if __name__ == '__main__':
    predict_feedback()
