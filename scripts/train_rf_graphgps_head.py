#!/usr/bin/env python3
"""
Train a Random Forest / ExtraTrees prediction head on GraphGPS-extracted embeddings.

Two-stage workflow:
  Stage 1 — Extract embeddings from a frozen GraphGPS checkpoint.
  Stage 2 — Train tree ensembles on the extracted embeddings and evaluate.

Supported model types:
  - GPSDoubleModel_multi4_cat_v0  (standard 5-GraphGPS)
  - OneHotEmbedGPS                (one-hot comps 1-4 + GraphGPS comp 5)

Usage:
  # Full pipeline: train + feedback evaluation
  python scripts/train_rf_graphgps_head.py \
      --config configs/GPS/direct_train.yaml \
      --checkpoint results/fifth_component_weight2/direct_train \
      --output-dir results/rf_head_experiment

  # Feedback-only evaluation (using pre-extracted train embeddings)
  python scripts/train_rf_graphgps_head.py \
      --config configs/GPS/direct_train.yaml \
      --checkpoint results/fifth_component_weight2/direct_train \
      --train-embeddings results/rf_head_experiment/train_embeddings.npz \
      --feedback-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

import graphgps  # noqa — register custom modules
from graphgps.config.config_gps import set_cfg_gps
from yacs.config import CfgNode as CN
from torch_geometric.graphgym.config import cfg as global_cfg
from torch_geometric.graphgym.register import network_dict
from torch_geometric.graphgym import seed_everything

from torch_geometric.data import Data, Batch
from rdkit import Chem
from graph_feature import smiles2graph

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COLUMNS = ["EE_before", "EE_after", "Aerosolization_Efficiency",
                  "mRNA_Recovery_Efficiency"]
SMILE_COLS = ["IL_SMILE", "HL_SMILE", "Chol_SMILE", "PEG_SMILE", "Fifth_SMILE"]
RATIO_COLS = ["mol%_IL", "mol%_HL", "mol%_Chol", "mol%_PEG", "mol%_Fifth"]
PROPERTY_NAME_MAP = {
    "EE_before": "EE_before",
    "EE_after": "EE_after",
    "Aerosolization_Efficiency": "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency": "mRNA_Recovery_Efficiency",
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_model_config(config_path: str):
    """Load a YAML config and merge it into the global cfg."""
    cfg = CN()
    set_cfg_gps(cfg)
    cfg.merge_from_file(config_path)
    for section in ["gt", "gnn", "dataset", "model", "posenc_RWSE",
                     "property_num", "accelerator", "gpu_serial",
                     "use_mordred_features", "mordred_feature_dim",
                     "mordred_feature_path", "mordred_fifth_only",
                     "coarse_grain_enable", "use_component_aux_features"]:
        if hasattr(cfg, section):
            setattr(global_cfg, section, getattr(cfg, section))
    global_cfg.share.dim_in = 1
    global_cfg.share.dim_out = 1
    return cfg


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(checkpoint_dir: str, seed: int = 0,
                               model_type: str | None = None):
    """Load a GraphGPS model and its checkpoint for a given seed."""
    ckpt_dir = Path(checkpoint_dir) / str(seed) / "ckpt"
    ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpt_files:
        # Try direct seed dir
        ckpt_dir = Path(checkpoint_dir) / "ckpt"
        ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}/{seed}/ckpt/")

    ckpt_path = ckpt_files[-1]  # use latest epoch

    from graphgps.create_model_gps import create_model_gps
    model = create_model_gps(to_device=True, dim_in=1,
                             dim_out=global_cfg.property_num)
    ckpt = torch.load(str(ckpt_path), map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt_path


# ---------------------------------------------------------------------------
# Data: SMILES → PyG Batch
# ---------------------------------------------------------------------------
def smiles_to_batch(smiles_list, ratios):
    """Convert SMILES to a PyG Batch for one component."""
    data_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi)) if not pd.isna(smi) and str(smi).lower() != "nan" else None
        if mol is None:
            g = {"node_feat": np.zeros((1, 9), dtype=np.int64),
                 "edge_index": np.empty((2, 0), dtype=np.int64),
                 "edge_feat": np.empty((0, 1), dtype=np.int64),
                 "num_nodes": 1}
        else:
            g = smiles2graph(mol)
        d = Data(x=torch.tensor(g["node_feat"], dtype=torch.long),
                 edge_index=torch.tensor(g["edge_index"], dtype=torch.long),
                 edge_attr=torch.tensor(g["edge_feat"], dtype=torch.long),
                 num_nodes=g["num_nodes"])
        data_list.append(d)
    batch = Batch.from_data_list(data_list)
    batch.ratio = torch.tensor(ratios, dtype=torch.float32)
    batch.to(torch.device("cuda", 0))
    return batch


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def extract_embeddings(model, csv_path: str, model_type: str) -> dict:
    """Run the frozen model over all samples and extract pre-head embeddings.

    Returns:
        dict with keys: embeddings [N, D], labels [N, 4], sample_ids [N]
    """
    df = pd.read_csv(csv_path, encoding="gb18030") if "gb" in str(
        pd.read_csv(csv_path, nrows=0, encoding="utf-8", on_bad_lines="skip")
    ) else pd.read_csv(csv_path)

    all_embs = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for idx in tqdm(range(len(df)), desc="Extracting embeddings"):
            row = df.iloc[idx]
            smiles_list = [str(row[c]) if pd.notna(row[c]) else "C" for c in SMILE_COLS]
            ratios = [float(row.get(c, 20.0)) for c in RATIO_COLS]
            s = sum(ratios)
            ratios = [r / s if s > 0 else 0.2 for r in ratios]

            # Build batches for 5 components
            batches = [smiles_to_batch([smiles_list[i]], [ratios[i]]) for i in range(5)]

            # Run model and capture pre-head embedding via hook
            embedding_container = []

            def hook_fn(module, input_tensor, output_tensor):
                # input_tensor[0] is the input to fusion MLP
                embedding_container.append(input_tensor[0].detach().cpu().numpy().copy())

            # Attach hook to the fusion layer
            if "OneHot" in model_type or "onehot" in model_type.lower():
                hook = model.fusion[0].register_forward_hook(hook_fn)
            else:
                # For GPSDoubleModel, find the fusion/branch layer
                # The fusion_input is computed before being passed to branches
                # Hook into main_backbone or the first branch's first layer
                if hasattr(model, "main_backbone"):
                    hook = model.main_backbone[0].register_forward_hook(hook_fn)
                elif hasattr(model, "main_head"):
                    hook = model.main_head.register_forward_hook(hook_fn)
                else:
                    # Fallback: hook the fusion MLP
                    hook = model.fusion[0].register_forward_hook(hook_fn)

            _ = model(batches[0], batches[1], batches[2], batches[3], batches[4])

            hook.remove()

            if embedding_container:
                all_embs.append(embedding_container[0])

            # Get labels
            label = []
            for col in TARGET_COLUMNS:
                label.append(float(row[col]) if pd.notna(row[col]) else np.nan)
            all_labels.append(label)
            all_ids.append(str(row.get("ID", idx)))

    embeddings = np.concatenate(all_embs, axis=0)  # [N, D]
    labels = np.array(all_labels)  # [N, 4]

    return {"embeddings": embeddings, "labels": labels, "sample_ids": np.array(all_ids)}


# ---------------------------------------------------------------------------
# Tree model training & evaluation
# ---------------------------------------------------------------------------
def train_tree_head(train_emb: np.ndarray, train_labels: np.ndarray,
                    estimator, scaler=None):
    """Train a tree model on extracted embeddings for one target."""
    if scaler is not None:
        X = scaler.fit_transform(train_emb)
    else:
        X = train_emb
    y = train_labels
    mask = np.isfinite(y)
    estimator.fit(X[mask], y[mask])
    return estimator, scaler


def predict_tree_head(estimator, X: np.ndarray, scaler=None):
    """Predict with optional scaling."""
    if scaler is not None:
        X = scaler.transform(X)
    return estimator.predict(X)


def evaluate_all(embeddings: np.ndarray, labels: np.ndarray,
                 train_mask: np.ndarray, eval_mask: np.ndarray,
                 eval_name: str, estimators: dict):
    """Train on train_mask, evaluate on eval_mask, return metrics."""
    results = {}
    for i, target in enumerate(TARGET_COLUMNS):
        y_all = labels[:, i]
        train_y = y_all[train_mask]
        eval_y = y_all[eval_mask]

        # Train
        est = clone(estimators["ExtraTrees"])
        scaler = StandardScaler()
        fitted, scaler = train_tree_head(embeddings[train_mask], train_y, est, scaler)

        # Predict
        pred = predict_tree_head(fitted, embeddings[eval_mask], scaler)
        valid = np.isfinite(eval_y) & np.isfinite(pred)
        mae = mean_absolute_error(eval_y[valid], pred[valid])
        r2 = r2_score(eval_y[valid], pred[valid])

        results[target] = {"mae": mae, "r2": r2, "n": int(valid.sum()),
                           "estimator": fitted, "scaler": scaler}
        print(f"  {eval_name} {target:<30s} MAE={mae:.4f}  R²={r2:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="YAML config for the GraphGPS model")
    parser.add_argument("--checkpoint", required=True,
                        help="Directory containing per-seed checkpoint subdirs")
    parser.add_argument("--output-dir", default="results/rf_head_experiment")
    parser.add_argument("--train-csv", default="datasets_lrx/raw/input/20260703_sum.csv")
    parser.add_argument("--feedback-csv", default="datasets_lrx/raw/feedback/20260703_validation.csv")
    parser.add_argument("--test-split", type=float, default=0.15,
                        help="Fraction of training data to hold out as test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--train-embeddings", default=None,
                        help="Path to pre-extracted embeddings .npz (skip re-extraction)")
    parser.add_argument("--feedback-only", action="store_true",
                        help="Only evaluate on feedback using pre-extracted train embeddings")
    parser.add_argument("--ensemble-seeds", type=int, default=1,
                        help="Number of GraphGPS seeds to ensemble (1, 3, or 5)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    # Load config
    cfg = load_model_config(args.config)
    model_type = cfg.model.type
    property_num = cfg.property_num
    print(f"Model type: {model_type}")
    print(f"Property num: {property_num}")

    # -------------------------------------------------------------------
    # Stage 1: Extract embeddings from GraphGPS
    # -------------------------------------------------------------------
    if args.train_embeddings and args.feedback_only:
        print(f"\nLoading pre-extracted train embeddings from {args.train_embeddings}")
        train_data = dict(np.load(args.train_embeddings, allow_pickle=True))
        train_emb = train_data["embeddings"]
        train_labels = train_data["labels"]
        train_ids = train_data["sample_ids"]
    else:
        print("\n=== Stage 1: Extract GraphGPS embeddings ===")

        # Load model(s)
        ensemble_seeds = min(args.ensemble_seeds, 5)
        all_train_embs = []

        for seed_idx in range(ensemble_seeds):
            model, ckpt_path = load_model_from_checkpoint(args.checkpoint, seed_idx, model_type)
            print(f"  Loaded seed {seed_idx}: {ckpt_path}")

            train_csv = args.train_csv
            data = extract_embeddings(model, train_csv, model_type)
            all_train_embs.append(data["embeddings"])
            train_labels = data["labels"]
            train_ids = data["sample_ids"]

        # Ensemble: average embeddings across seeds
        train_emb = np.mean(all_train_embs, axis=0)
        print(f"  Train embeddings: {train_emb.shape}")

        # Save embeddings
        np.savez_compressed(output_dir / "train_embeddings.npz",
                            embeddings=train_emb, labels=train_labels,
                            sample_ids=train_ids)

    # -------------------------------------------------------------------
    # Feedback embeddings (if requested)
    # -------------------------------------------------------------------
    if args.feedback_csv and Path(args.feedback_csv).exists():
        print("\n=== Extract Feedback embeddings ===")
        fb_df = pd.read_csv(args.feedback_csv)
        fb_embs_all = []
        for seed_idx in range(ensemble_seeds if not args.feedback_only else 1):
            # Reuse the last loaded model for feedback-only mode
            if args.feedback_only:
                model, _ = load_model_from_checkpoint(args.checkpoint, 0, model_type)
            fb_data = extract_embeddings(model, args.feedback_csv, model_type)
            fb_embs_all.append(fb_data["embeddings"])
            fb_labels = fb_data["labels"]
            fb_ids = fb_data["sample_ids"]
        fb_emb = np.mean(fb_embs_all, axis=0)
        np.savez_compressed(output_dir / "feedback_embeddings.npz",
                            embeddings=fb_emb, labels=fb_labels, sample_ids=fb_ids)
        print(f"  Feedback embeddings: {fb_emb.shape}")

    # -------------------------------------------------------------------
    # Stage 2: Train RF head
    # -------------------------------------------------------------------
    print("\n=== Stage 2: Train Random Forest / ExtraTrees on embeddings ===")

    # Split train into train/test
    n_total = len(train_emb)
    n_test = int(n_total * args.test_split)
    indices = np.random.RandomState(args.seed).permutation(n_total)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    train_mask = np.zeros(n_total, dtype=bool)
    train_mask[train_idx] = True
    test_mask = np.zeros(n_total, dtype=bool)
    test_mask[test_idx] = True

    print(f"  Train: {train_mask.sum()}  Test: {test_mask.sum()}")

    estimators = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=args.n_estimators, min_samples_leaf=2,
            max_features=0.8, random_state=args.seed, n_jobs=args.n_jobs,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=args.n_estimators, min_samples_leaf=2,
            max_features=0.7, random_state=args.seed, n_jobs=args.n_jobs,
        ),
    }

    # Evaluate
    all_results = {}
    for est_name, est in estimators.items():
        print(f"\n--- {est_name} ---")
        print("  Training set evaluation:")
        train_results = evaluate_all(train_emb, train_labels, train_mask, train_mask,
                                     "train", {est_name: est})
        print("  Test set evaluation:")
        test_results = evaluate_all(train_emb, train_labels, train_mask, test_mask,
                                    "test", {est_name: est})
        all_results[est_name] = {"train": train_results, "test": test_results}

    # Feedback evaluation
    if "fb_emb" in dir():
        for est_name, est in estimators.items():
            print(f"\n--- {est_name} on Feedback ---")
            fb_mask = np.ones(len(fb_emb), dtype=bool)
            fb_results = evaluate_all(
                np.concatenate([train_emb, fb_emb]),
                np.concatenate([train_labels, fb_labels]),
                np.concatenate([train_mask, np.zeros(len(fb_emb), dtype=bool)]),
                np.concatenate([np.zeros(len(train_emb), dtype=bool), fb_mask]),
                "feedback", {est_name: est},
            )
            if est_name not in all_results:
                all_results[est_name] = {}
            all_results[est_name]["feedback"] = fb_results

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY — RF Head on GraphGPS Embeddings")
    print("=" * 90)

    summary_rows = []
    for est_name in estimators:
        for split in ["test"] + (["feedback"] if "fb_emb" in dir() else []):
            for target in TARGET_COLUMNS:
                r = all_results[est_name][split][target]
                summary_rows.append({
                    "model": f"GraphGPS+{est_name}", "split": split,
                    "target": target, "mae": r["mae"], "r2": r["r2"], "n": r["n"],
                })
                print(f"  {est_name:15s} {split:8s} {target:30s} "
                      f"MAE={r['mae']:.4f}  R²={r['r2']:.4f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "rf_head_summary.csv", index=False)

    # Overall MAE per split
    print("\n--- Overall (sum of 4 targets) ---")
    for est_name in estimators:
        for split in ["test"] + (["feedback"] if "fb_emb" in dir() else []):
            total_mae = sum(all_results[est_name][split][t]["mae"] for t in TARGET_COLUMNS)
            print(f"  {est_name:15s} {split:8s}  MAE_sum = {total_mae:.4f}")

    # Save metadata
    meta = {
        "config": args.config, "checkpoint": args.checkpoint,
        "model_type": model_type, "seed": args.seed,
        "embedding_dim": int(train_emb.shape[1]),
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "feedback_samples": int(len(fb_emb)) if "fb_emb" in dir() else 0,
        "n_estimators": args.n_estimators,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
