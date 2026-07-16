# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Polymer property prediction service using GraphGPS (Graph Neural Networks + Transformer hybrids). Given 5 monomer components (as molecular graphs via SMILES) and their ratios, the model predicts multiple polymer properties (e.g., aerosolization efficiency, mRNA recovery efficiency, encapsulation efficiency before/after).

## How to Run

### Training

```bash
cd /home/puzexuan/study/code/blology_prediction/service
python main.py --cfg_file configs/GPS/direct_train.yaml --repeat 10
```

- `--cfg_file`: Path to YAML config (required). Available configs in `configs/GPS/`.
- `--repeat N`: Number of repeated runs with different random seeds (default multi-seed mode).
- `mark_done`: If set in config, renames the config file to `*_done` after completion to mark it as processed.
- The output directory is `results/<run_name>/<run_id>/`.

### Prediction / Inference

```bash
python main_predict.py --cfg_file configs/GPS/gps_predict.yaml --repeat 10
```

Uses a pretrained model checkpoint (set `pretrained.dir` in config) to run inference on new data (set `read_csv` to the input CSV path).

## Architecture

### Build Chain

The project extends **PyTorch Geometric's GraphGym** framework with custom **GraphGPS** modules. All custom modules are registered via `import graphgps` (the package's `__init__.py` imports all sub-modules, triggering `@register_*` decorators).

### Key Modules and Data Flow

1. **Entry Points** (`main.py` / `main_predict.py`):
   - Parse CLI args → load YAML config → run loop over seeds/splits.
   - In each run: create data loaders → create GPS model → optionally load pretrained weights → create optimizer/scheduler → dispatch to registered train mode.

2. **Data Loading** (`loader_5.py`, `loader_j.py`):
   - `loader_5.py`: Returns 5 separate DataLoaders — one for each monomer component in a 5-component polymer mixture. Each component gets its own graph dataset.
   - Data input format: CSV files in `datasets_lrx/raw/` containing SMILES columns for each monomer + ratio columns + target property columns.
   - Custom dataset loaders are registered in `graphgps.loader.dataset/` and discovered via `register.loader_dict`.

3. **Graph Feature Extraction** (`graph_feature.py`):
   - `smiles2graph(mol)`: Converts an RDKit Mol object into graph dict `{edge_index, edge_feat, node_feat, num_nodes}`.
   - Atom features: atomic number, chirality, degree, formal charge, num H, radical electrons, hybridization, aromaticity, ring membership (9-dim one-hot).
   - Bond features: bond type only (1-dim).
   - Feature dimensions are configured via `dataset.node_encoder_num_types` and `dataset.edge_encoder_num_types` in YAML.

4. **GPS Model** (`graphgps/network/gps_model.py`):
   - `GPSModel`: Standard single-graph GPS architecture — `FeatureEncoder` → optional `pre_mp` layers → stack of `GPSLayer` (GINE+Transformer blocks) → prediction head.
   - `FeatureEncoder`: Embeds discrete node/edge features via `nn.Embedding`, optional RWSE positional encoding, optional BatchNorm.

5. **Double/Multi GPS Models** (`graphgps/network/double_gps_*.py`):
   - Models for 5-component polymer inputs. Key variants:
     - `double_gps_cat_multi_2_expert5.py`: Shared GNN encoder for all 5 components → ratio-aware FiLM modulation → Transformer mixer for cross-component interaction → 3-branch prediction (main backbone, direct shortcut, component-5 expert) with learned gate fusion.
     - `double_gps_cat_multi_2_shared.py`: Simpler variant — shared ratio encoder, learned token importance pooling, 3 shared branches (attention-weighted, mean, max) with gate fusion.
     - Naming convention: `cat` = concatenation-based fusion, `sum` = summation-based, `multiN` = N-property output, `expertN` = expert branch for N-th component.
   - All double GPS models expect 5 batch inputs (`data1`–`data5`) from the 5 loaders.

6. **Model Registration** (`graphgps/create_model_gps.py`):
   - `create_model_gps()`: Factory function that instantiates the network class registered under `cfg.model.type` and wraps it in a `GraphGymModule` (PyTorch Lightning-compatible).
   - Set `model.type` in YAML config to select the network (e.g., `GPSModel`, `GPSDoubleModel_multi4_cat_v0`).

7. **Prediction Heads** (`graphgps/head/`):
   - Configurable via `gnn.head` in YAML. Common heads: `san_graph`, `feat_graph`, `fine_graph`, `multi_graph`, `double_graph`.
   - Heads apply global pooling (add/mean/max) to node embeddings, then an MLP to produce predictions.

8. **Training Loop** (`graphgps/train/train_five_multi.py`):
   - Custom training pipeline registered as `'double'` train mode.
   - Handles 5-loader batching: interleaves batches from all 5 loaders for each iteration.
   - Supports multi-property loss: `property_num` in config selects the loss function (L1 with NT-Xent contrastive for 1/6 properties, simple multi-L1 for 2/4 properties).
   - Implements SWA (Stochastic Weight Averaging) and early stopping at 200 patience epochs.

9. **Loss Functions** (`graphgps/loss/`, `graphgps/lrx_add/`):
   - `compute_loss_l1_CL_*.py`: L1 regression + NT-Xent contrastive loss (for 1/6 properties).
   - `compute_loss_multi4.py` / `compute_loss_multi2.py`: Plain L1 loss for 4/2 property regression.
   - Contrastive loss variants operate on graph embeddings from different model layers (GPS features, middle features, prediction features).

### Configuration System

- YAML-based via `yacs` (`.yaml` files + command-line overrides).
- Core config schema defined in `graphgps/config/config_gps.py` (`set_cfg_gps`).
- Key config sections:
  - `dataset`: Data format, encoders, positional encoding settings.
  - `train.mode`: `standard` (default GraphGym), `double` (5-component training), `double_predict` (inference), `double_multi` (multi-property).
  - `model.type`: Which registered network to use.
  - `gnn.head`: Which prediction head to use.
  - `gt`: Graph Transformer settings (layers, heads, hidden dim).
  - `property_num`: Number of target properties (1, 2, 4, or 6).
  - `pretrained.dir`: Path to pretrained model checkpoint directory.
  - `data_rate` / `data_rate_type`: Whether to use component ratio data.

### Results and Logging

- Per-run results stored in `results/<run_name>/<run_id>/` with `stats.json` files for train/val/test splits.
- Weights & Biases logging enabled via `wandb.use: True` in config.
- Post-training: `graphgps/lrx_add/read_json.py` generates result summary images (`result_picture`).

### Utility Scripts (`python/`)

- `distribution_map.py`: Generate property distribution heatmaps.
- `pred_csv.py`: Generate prediction CSV output.
- `png.py`: Create scatter plots of true vs. predicted values.
- `split_random.py`: Random train/val/test dataset splitting.
- `analyses_property.py`: Analyze prediction errors across properties.

## Dependencies

- **PyTorch + PyTorch Geometric** (GraphGym framework)
- **RDKit** (SMILES → molecular graph conversion)
- **yacs** (YAML configuration)
- **wandb** (experiment tracking)
- **ogb** (OGB dataset support, optional)

## Environment Notes

- Hardcoded paths in `main.py` and `main_predict.py` refer to `/home/lrx/dataset/` — these need to be updated for your environment.
- GPU configuration: `cfg.accelerator` selects device type, `cfg.gpu_serial` selects which GPU.
- Multi-worker data loading configured via `cfg.num_workers` (default 16).
