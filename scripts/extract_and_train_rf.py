#!/usr/bin/env python3
"""
Extract GraphGPS pre-head embeddings and train Random Forest / ExtraTrees head.

Workflow:
  1. Load trained GraphGPS checkpoint(s)
  2. Run through the native data pipeline, capturing pre-head embeddings via hook
  3. Train ExtraTreesRegressor / RandomForestRegressor on embeddings
  4. Evaluate on test split and feedback data

Usage:
  # Full pipeline: extract embeddings + train RF + evaluate
  python scripts/extract_and_train_rf.py \
      --config configs/GPS/direct_train.yaml \
      --checkpoint results/fifth_component_weight2/direct_train \
      --seeds 5

  # With feedback evaluation
  python scripts/extract_and_train_rf.py \
      --config configs/GPS/direct_train.yaml \
      --checkpoint results/fifth_component_weight2/direct_train \
      --feedback-csv datasets_lrx/raw/feedback/20260703_validation.csv \
      --seeds 5
"""

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

import graphgps  # noqa
from graphgps.config.config_gps import set_cfg_gps
from torch_geometric.graphgym.config import cfg as gcfg
from torch_geometric import seed_everything
from loader_5 import create_loader_5
from graphgps.create_model_gps import create_model_gps

TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]


def setup_config(config_path):
    set_cfg_gps(gcfg)
    gcfg.merge_from_file(config_path)
    gcfg.accelerator = "cuda"; gcfg.gpu_serial = 0
    gcfg.out_dir = "results"
    gcfg.train.mode = "double"
    # Use per-run cache to isolate Mordred / non-Mordred runs.
    # Each unique cache_tag produces a separate processed/ directory,
    # ensuring Mordred features are properly embedded in .pt files.
    gcfg.dataset.cache_per_run = True
    gcfg.dataset.cache_refresh = True
    if not hasattr(gcfg.dataset, "cache_tag") or not gcfg.dataset.cache_tag:
        gcfg.dataset.cache_tag = f"rf_extract_{Path(config_path).stem}"
    # Mordred / aux / coarse settings are controlled by the YAML config.
    # Do NOT force-disable them here — checkpoints trained with these
    # features must load with the matching config.


def load_model(checkpoint_dir, seed=0):
    ckpt_dir = Path(checkpoint_dir) / str(seed) / "ckpt"
    files = sorted(ckpt_dir.glob("*.ckpt"))
    if not files: raise FileNotFoundError(str(ckpt_dir))
    model = create_model_gps(dim_in=1, dim_out=gcfg.property_num)
    ckpt = torch.load(str(files[-1]), map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False); model.eval()
    return model


def find_hook_layer(model):
    """Return the module whose *input* we want to capture (pre-head fusion)."""
    net = model.model
    for name in ("FC_layers.0", "fusion.0", "main_backbone.0", "main_head"):
        for n, m in net.named_modules():
            if n == name:
                return m
    raise RuntimeError("Cannot find hook layer")


def extract_from_loaders(model, loaders_tuple, progress_desc="Extracting"):
    """Iterate the train loader, collecting embeddings + labels."""
    loaders, l2, l3, l4, l5 = loaders_tuple
    embs, lbls = [], []
    hook_layer = find_hook_layer(model)
    captured = []

    def _hook(m, inp):
        captured.append(inp[0].detach().cpu().numpy())
    handle = hook_layer.register_forward_pre_hook(_hook)

    device = torch.device("cuda", 0)
    n_batches = len(loaders[0])
    with torch.no_grad():
        for b1, b2, b3, b4, b5 in tqdm(zip(loaders[0], l2[0], l3[0], l4[0], l5[0]),
                                        total=n_batches, desc=progress_desc):
            for b, s in [(b1,"train"),(b2,"train_2"),(b3,"train_3"),(b4,"train_4"),(b5,"train_5")]:
                b.split = s; b.to(device)
            _, label = model(b1, b2, b3, b4, b5)
            lbls.append(label.detach().cpu().numpy())

    handle.remove()
    embs = np.concatenate(captured, axis=0)
    lbls = np.concatenate(lbls, axis=0)
    return embs, lbls


def run_extraction(checkpoint_dir, config_path, csv_path, output_dir, n_seeds):
    """Extract embeddings from trained checkpoints, average across seeds."""
    setup_config(config_path)
    gcfg.read_csv = csv_path
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_embs = []

    for s in range(n_seeds):
        gcfg.seed = s; seed_everything(s)
        print(f"\nSeed {s}: creating loaders...")
        loaders = create_loader_5()
        model = load_model(checkpoint_dir, s)
        embs, lbls = extract_from_loaders(model, loaders, f"Seed {s}")
        all_embs.append(embs)
        print(f"  Embeddings: {embs.shape}")

    avg_embs = np.mean(all_embs, axis=0)
    # Reshape labels from [N*4] to [N, 4] if flattened
    if lbls.ndim == 1:
        lbls = lbls.reshape(avg_embs.shape[0], -1)
    np.savez_compressed(out_dir / "train_embeddings.npz", embeddings=avg_embs, labels=lbls)
    print(f"\nSaved: {out_dir}/train_embeddings.npz  ({avg_embs.shape}, labels {lbls.shape})")
    return avg_embs, lbls


def run_feedback_extraction(checkpoint_dir, config_path, feedback_csv, output_dir):
    """Extract embeddings for feedback data (single seed)."""
    setup_config(config_path)
    gcfg.read_csv = "feedback/20260703_validation.csv"
    gcfg.seed = 0; seed_everything(0)
    out_dir = Path(output_dir)

    loaders = create_loader_5()
    model = load_model(checkpoint_dir, 0)
    embs, lbls = extract_from_loaders(model, loaders, "Feedback")
    if lbls.ndim == 1:
        lbls = lbls.reshape(embs.shape[0], -1)
    np.savez_compressed(out_dir / "feedback_embeddings.npz", embeddings=embs, labels=lbls)
    print(f"Feedback embeddings: {embs.shape}, labels {lbls.shape}")
    return embs, lbls


def train_eval_rf(train_emb, train_lbl, test_emb, test_lbl, estimators, seed, n_jobs, output_dir):
    """Train RF/ExtraTrees with 80/10/10 split, report train/val/test MAE."""
    n = len(train_emb)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(n * 0.10))
    n_val  = max(1, int(n * 0.10))
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    print(f"Split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    results = {}
    all_csv_rows = []

    for est_name, est_tmpl in estimators.items():
        print(f"\n{'='*60}\n  {est_name}\n{'='*60}")
        results[est_name] = {}
        total_train, total_val, total_test = 0, 0, 0

        for i, tgt in enumerate(TARGETS):
            scaler = StandardScaler(); est = clone(est_tmpl)
            Xtr = scaler.fit_transform(train_emb[train_idx]); ytr = train_lbl[train_idx, i]
            Xva = scaler.transform(train_emb[val_idx]);      yva = train_lbl[val_idx, i]
            Xte = scaler.transform(train_emb[test_idx]);     yte = train_lbl[test_idx, i]
            m_tr = np.isfinite(ytr); m_va = np.isfinite(yva); m_te = np.isfinite(yte)
            est.fit(Xtr[m_tr], ytr[m_tr])

            pred_tr = est.predict(Xtr[m_tr])
            mae_tr = mean_absolute_error(ytr[m_tr], pred_tr)

            pred_va = est.predict(Xva[m_va])
            mae_va = mean_absolute_error(yva[m_va], pred_va)

            pred_te = est.predict(Xte[m_te])
            mae_te = mean_absolute_error(yte[m_te], pred_te)
            r2_te  = r2_score(yte[m_te], pred_te)

            total_train += mae_tr; total_val += mae_va; total_test += mae_te
            print(f"  {tgt:<30s}  train={mae_tr:.4f}  val={mae_va:.4f}  test={mae_te:.4f}  R²={r2_te:.4f}")
            results[est_name][tgt] = {"train_mae": float(mae_tr), "val_mae": float(mae_va),
                                       "test_mae": float(mae_te), "test_r2": float(r2_te)}
            for j, (tv, pv) in enumerate(zip(yte[m_te], pred_te)):
                all_csv_rows.append({"model": est_name, "split": "test", "target": tgt,
                    "sample_idx": int(test_idx[m_te][j]), "y_true": float(tv), "y_pred": float(pv)})

        print(f"  {'─'*55}")
        print(f"  {'MAE_sum':<30s}  train={total_train:.4f}  val={total_val:.4f}  test={total_test:.4f}")

        if test_emb is not None:
            print(f"  --- Feedback ---")
            fb_mae = 0
            for i, tgt in enumerate(TARGETS):
                scaler = StandardScaler(); est = clone(est_tmpl)
                Xtr = scaler.fit_transform(train_emb[train_idx]); ytr = train_lbl[train_idx, i]
                Xfb = scaler.transform(test_emb); yfb = test_lbl[:, i]
                m_tr = np.isfinite(ytr); m_fb = np.isfinite(yfb)
                est.fit(Xtr[m_tr], ytr[m_tr]); pred = est.predict(Xfb[m_fb])
                mae = mean_absolute_error(yfb[m_fb], pred); r2 = r2_score(yfb[m_fb], pred)
                fb_mae += mae
                print(f"  FB    {tgt:<30s} MAE={mae:.4f}  R²={r2:.4f}")
                results[est_name][f"feedback_{tgt}"] = {"mae": float(mae), "r2": float(r2)}
                for j, (tv, pv) in enumerate(zip(yfb[m_fb], pred)):
                    all_csv_rows.append({"model": est_name, "split": "feedback", "target": tgt,
                        "sample_idx": int(j), "y_true": float(tv), "y_pred": float(pv)})
            print(f"  FB    {'MAE_sum':<30s} {fb_mae:.4f}")

    # -------------------------------------------------------------------
    # Save per-sample predictions CSV
    # -------------------------------------------------------------------
    pred_df = pd.DataFrame(all_csv_rows)
    csv_path = output_dir / "predictions_per_sample.csv"
    pred_df.to_csv(csv_path, index=False)
    print(f"\nSaved per-sample predictions: {csv_path}")

    # -------------------------------------------------------------------
    # Generate scatter plots (true vs predicted), one per property per split
    # -------------------------------------------------------------------
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    for model_name in pred_df["model"].unique():
        for split_name in pred_df["split"].unique():
            subset = pred_df[(pred_df["model"] == model_name) & (pred_df["split"] == split_name)]
            for tgt in TARGETS:
                prop = subset[subset["target"] == tgt]
                if len(prop) == 0:
                    continue
                true_vals = prop["y_true"].values
                pred_vals = prop["y_pred"].values
                mae = mean_absolute_error(true_vals, pred_vals)
                r2 = r2_score(true_vals, pred_vals)

                plt.figure(figsize=(7, 7))
                # Plot y=x reference line
                all_vals = np.concatenate([true_vals, pred_vals])
                vmin, vmax = all_vals.min(), all_vals.max()
                margin = (vmax - vmin) * 0.05
                plt.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                         "k--", alpha=0.3, linewidth=1)
                plt.plot(true_vals, pred_vals, "o", alpha=0.6, markersize=6)
                plt.title(f"{model_name} | {split_name} | {tgt}\nMAE={mae:.4f}  R²={r2:.4f}")
                plt.xlabel("True Value"); plt.ylabel("Predicted Value")
                plt.tight_layout()
                fname = f"{model_name}_{split_name}_{tgt}_scatter.png"
                plt.savefig(plot_dir / fname, dpi=150)
                plt.close()

    print(f"Saved scatter plots: {plot_dir}/")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/GPS/direct_train.yaml")
    p.add_argument("--checkpoint", default="results/fifth_component_weight2/direct_train")
    p.add_argument("--output-dir", default="results/rf_graphgps_head")
    p.add_argument("--feedback-csv", default=None)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--embeddings-only", action="store_true",
                   help="Only extract embeddings, skip RF training")
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    # -- Extract train embeddings --
    print("=" * 60 + "\nExtracting GraphGPS embeddings\n" + "=" * 60)
    train_emb, train_lbl = run_extraction(
        args.checkpoint, args.config, "input/20260703_sum_utf8.csv", out_dir, args.seeds)

    if args.embeddings_only:
        print("Embeddings saved. Done."); return

    # -- Extract feedback embeddings (optional) --
    fb_emb = fb_lbl = None
    if args.feedback_csv:
        print("\n" + "=" * 60 + "\nExtracting Feedback embeddings\n" + "=" * 60)
        fb_emb, fb_lbl = run_feedback_extraction(
            args.checkpoint, args.config, args.feedback_csv, out_dir)

    # -- Train RF / ExtraTrees --
    print("\n" + "=" * 60 + "\nTraining RF Head\n" + "=" * 60)
    estimators = {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=args.n_estimators, min_samples_leaf=2,
                                          max_features=0.8, random_state=args.seed, n_jobs=args.n_jobs),
        "RandomForest": RandomForestRegressor(n_estimators=args.n_estimators, min_samples_leaf=2,
                                              max_features=0.7, random_state=args.seed, n_jobs=args.n_jobs),
    }
    results = train_eval_rf(train_emb, train_lbl, fb_emb, fb_lbl, estimators, args.seed, args.n_jobs, out_dir)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # -------------------------------------------------------------------
    # Final summary: best model per target, overall
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL MODEL SELECTION")
    print("=" * 70)
    print(f"{'Model':<20} {'Property':<30} {'Train_MAE':>10} {'Val_MAE':>10} {'Test_MAE':>10} {'Test_R²':>10}")
    print("-" * 70)
    for est_name in estimators:
        for tgt in TARGETS:
            r = results[est_name][tgt]
            print(f"{est_name:<20} {tgt:<30} {r['train_mae']:>10.4f} {r['val_mae']:>10.4f} {r['test_mae']:>10.4f} {r['test_r2']:>10.4f}")
        # MAE sum row
        train_sum = sum(results[est_name][t]["train_mae"] for t in TARGETS)
        val_sum   = sum(results[est_name][t]["val_mae"]   for t in TARGETS)
        test_sum  = sum(results[est_name][t]["test_mae"]  for t in TARGETS)
        print(f"{' ':>20} {'MAE_sum':<30} {train_sum:>10.4f} {val_sum:>10.4f} {test_sum:>10.4f}")
        print("-" * 70)

    print(f"\nDone. Results: {out_dir}/results.json")


if __name__ == "__main__":
    main()
