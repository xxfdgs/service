#!/usr/bin/env python3
"""Unsupervised diagnostics for exported frozen GraphGPS representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, pairwise_distances, rbf_kernel

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS = [
    "graph_branch_raw", "descriptor_branch_raw", "formula_branch_raw",
    "graph_branch_projected", "descriptor_branch_projected", "formula_branch_projected",
    "fused_embedding", "head_hidden", "final_prediction",
]
EPOCH_ORDER = ["epoch_initial", "epoch_precollapse", "epoch_collapse", "epoch_best", "epoch_last"]


def effective_rank(values: np.ndarray) -> tuple[float, float, float, float]:
    centered = values - values.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2 or not np.any(centered):
        return 0.0, 0.0, 0.0, math.inf
    singular = np.linalg.svd(centered, compute_uv=False)
    eig = singular ** 2 / max(1, centered.shape[0] - 1)
    positive = eig[eig > max(eig.max(), 1.0) * 1e-12]
    if positive.size == 0:
        return 0.0, 0.0, 0.0, math.inf
    prob = positive / positive.sum()
    rank = float(np.exp(-(prob * np.log(prob)).sum()))
    return rank, float(positive.max()), float(positive.min()), float(positive.max() / positive.min())


def mean_pairwise_cosine(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    distances = cosine_distances(values)
    tri = distances[np.triu_indices_from(distances, 1)]
    return float(tri.mean()), float(np.partition(tri, 0)[0])


def group_distances(values: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    distance = pdist(values, metric="euclidean")
    left, right = np.triu_indices(len(values), 1)
    same = groups[left] == groups[right]
    return (float(distance[same].mean()) if same.any() else math.nan,
            float(distance[~same].mean()) if (~same).any() else math.nan)


def nearest_neighbor_distance(values: np.ndarray) -> float:
    if len(values) < 2:
        return math.nan
    distance = pairwise_distances(values, metric="euclidean")
    np.fill_diagonal(distance, np.inf)
    return float(distance.min(axis=1).mean())


def mmd_rbf(left: np.ndarray, right: np.ndarray, seed: int = 0) -> float:
    # Median-heuristic RBF MMD; cap only the kernel calculation, not the
    # representation statistics, so it remains a stable diagnostic.
    rng = np.random.default_rng(seed)
    if len(left) > 300:
        left = left[rng.choice(len(left), 300, replace=False)]
    if len(right) > 300:
        right = right[rng.choice(len(right), 300, replace=False)]
    joined = np.vstack([left, right])
    sample = joined if len(joined) <= 300 else joined[rng.choice(len(joined), 300, replace=False)]
    distances = pdist(sample)
    median = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
    gamma = 1.0 / max(median * median, 1e-12)
    xx = rbf_kernel(left, left, gamma=gamma)
    yy = rbf_kernel(right, right, gamma=gamma)
    xy = rbf_kernel(left, right, gamma=gamma)
    return float(xx.mean() + yy.mean() - 2.0 * xy.mean())


def load_archive(root: Path, fold: str, epoch_label: str, split: str, embedding: str):
    archive = np.load(root / "embeddings" / fold / epoch_label / f"{split}_{embedding}.npz", allow_pickle=False)
    return archive["embedding"].astype(np.float64), archive["group_id"].astype(str)


def make_plots(root: Path, stats: pd.DataFrame) -> None:
    figures = root / "representation_stats" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    epoch_position = {name: index for index, name in enumerate(EPOCH_ORDER)}
    plot_data = stats.loc[stats.split.isin(["train", "val"])].copy()
    plot_data["x"] = plot_data.epoch_label.map(epoch_position)
    for metric, filename, ylabel in [
        ("effective_rank", "effective_rank_vs_epoch.png", "effective rank"),
        ("embedding_std", "embedding_std_vs_epoch.png", "embedding element std"),
    ]:
        fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
        for ax, embedding in zip(axes.flat, EMBEDDINGS):
            subset = plot_data.loc[plot_data.embedding_name.eq(embedding)]
            for (fold, split), group in subset.groupby(["fold", "split"]):
                group = group.sort_values("x")
                ax.plot(group.x, group[metric], marker="o", label=f"{fold}/{split}")
            ax.set_title(embedding, fontsize=8)
            ax.set_xticks(range(len(EPOCH_ORDER)), [value.replace("epoch_", "") for value in EPOCH_ORDER], rotation=30, fontsize=7)
            ax.grid(alpha=.25)
        axes.flat[0].legend(fontsize=7)
        fig.supylabel(ylabel)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)

    # Raw/projected comparison uses validation values only; it is a displayed
    # diagnostic, never a candidate selection criterion by itself.
    pairs = [("graph_branch_raw", "graph_branch_projected"),
             ("descriptor_branch_raw", "descriptor_branch_projected"),
             ("formula_branch_raw", "formula_branch_projected")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    val = stats.loc[stats.split.eq("val")]
    for ax, (raw, projected) in zip(axes, pairs):
        merged = val.loc[val.embedding_name.isin([raw, projected])].pivot_table(
            index=["fold", "epoch_label"], columns="embedding_name", values="embedding_std")
        if raw in merged and projected in merged:
            ax.scatter(merged[raw], merged[projected])
            maximum = float(np.nanmax(merged[[raw, projected]].to_numpy()))
            ax.plot([0, maximum], [0, maximum], "k--", linewidth=.8)
        ax.set_xlabel(raw)
        ax.set_ylabel(projected)
        ax.set_title("validation std")
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(figures / "branch_raw_vs_projected_variance.png", dpi=160)
    plt.close(fig)


def pca_plot(root: Path) -> None:
    figures = root / "representation_stats" / "figures"
    selected = ["graph_branch_raw", "graph_branch_projected", "descriptor_branch_raw", "formula_branch_raw", "fused_embedding", "head_hidden"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, embedding in zip(axes.flat, selected):
        for fold, color in [("fold_0", "tab:blue"), ("fold_4", "tab:orange")]:
            train, _ = load_archive(root, fold, "epoch_best", "train", embedding)
            val, _ = load_archive(root, fold, "epoch_best", "val", embedding)
            pca = PCA(n_components=2, random_state=0).fit(train)
            ax.scatter(*pca.transform(train).T, s=4, alpha=.25, color=color, label=f"{fold} train")
            ax.scatter(*pca.transform(val).T, s=10, alpha=.65, marker="x", color=color, label=f"{fold} val")
        ax.set_title(embedding, fontsize=9)
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Train/validation PCA, epoch_best (PCA fitted on each fold's train only)")
    fig.tight_layout()
    fig.savefig(figures / "train_validation_pca.png", dpi=160)
    plt.close(fig)


def distance_plot(root: Path) -> None:
    figures = root / "representation_stats" / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, fold in zip(axes, ["fold_0", "fold_4"]):
        for embedding, color in [("graph_branch_projected", "tab:blue"), ("descriptor_branch_raw", "tab:green"),
                                 ("formula_branch_projected", "tab:purple"), ("fused_embedding", "tab:red")]:
            values, _ = load_archive(root, fold, "epoch_best", "val", embedding)
            # Normalize coordinates only for a comparable distribution plot.
            values = (values - values.mean(0)) / np.maximum(values.std(0), 1e-8)
            distance = pdist(values)
            ax.hist(distance, bins=35, density=True, histtype="step", linewidth=1.2, label=embedding, color=color)
        ax.set_title(fold + " validation")
        ax.set_xlabel("per-coordinate standardized pair distance")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "fused_vs_branch_distance_distribution.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/frozen_embedding_signal_exp")
    args = parser.parse_args()
    root = args.output_root.resolve()
    destination = root / "representation_stats"
    destination.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(root / "embeddings" / "embedding_index.csv")
    groups = index.loc[:, ["fold", "epoch_label", "split", "embedding_name", "group_id"]].drop_duplicates()
    rows, rank_rows, shift_rows = [], [], []
    for fold in ["fold_0", "fold_4"]:
        for epoch_label in EPOCH_ORDER:
            matrices: dict[tuple[str, str], np.ndarray] = {}
            for split in ("train", "val", "test"):
                for embedding in EMBEDDINGS:
                    values, group = load_archive(root, fold, epoch_label, split, embedding)
                    matrices[(split, embedding)] = values
                    mean = values.mean(axis=0)
                    std = values.std(axis=0, ddof=0)
                    pair_distance = pdist(values)
                    cosine_mean, nearest_cosine = mean_pairwise_cosine(values)
                    group_within, group_between = group_distances(values, group)
                    rank, max_eig, min_eig, condition = effective_rank(values)
                    pca_components = min(10, values.shape[0], values.shape[1])
                    pca = PCA(n_components=pca_components, random_state=0).fit(values)
                    summary = {
                        "fold": fold, "epoch_label": epoch_label, "split": split, "embedding_name": embedding,
                        "n_samples": len(values), "embedding_dim": values.shape[1],
                        "embedding_mean": float(mean.mean()), "embedding_std": float(values.std(ddof=0)),
                        "near_zero_variance_fraction": float((std < 1e-8).mean()),
                        "effective_rank": rank,
                        "covariance_max_eigenvalue": max_eig,
                        "covariance_min_nonzero_eigenvalue": min_eig,
                        "condition_number": condition,
                        "mean_norm": float(np.linalg.norm(values, axis=1).mean()),
                        "mean_euclidean_distance": float(pair_distance.mean()) if len(pair_distance) else math.nan,
                        "nearest_neighbor_distance": nearest_neighbor_distance(values),
                        "mean_cosine_distance": cosine_mean, "nearest_cosine_distance": nearest_cosine,
                        "group_within_distance": group_within, "group_between_distance": group_between,
                        **{f"pca_explained_variance_{i + 1}": float(value) if np.isfinite(value) else 0.0 for i, value in enumerate(pca.explained_variance_ratio_)},
                    }
                    rows.append(summary)
                    rank_rows.append({
                        "fold": fold, "epoch_label": epoch_label, "split": split, "embedding_name": embedding,
                        "effective_rank": rank, "covariance_max_eigenvalue": max_eig,
                        "covariance_min_nonzero_eigenvalue": min_eig, "condition_number": condition,
                        "near_zero_variance_fraction": summary["near_zero_variance_fraction"],
                    })
                    for dimension, (dimension_mean, dimension_std) in enumerate(zip(mean, std)):
                        rows.append({"fold": fold, "epoch_label": epoch_label, "split": split,
                                     "embedding_name": embedding, "dimension": dimension,
                                     "dimension_mean": float(dimension_mean), "dimension_std": float(dimension_std),
                                     "record_type": "per_dimension"})
            for embedding in EMBEDDINGS:
                train, val = matrices[("train", embedding)], matrices[("val", embedding)]
                covariance_train = np.cov(train, rowvar=False)
                covariance_val = np.cov(val, rowvar=False)
                denominator = max(float(np.linalg.norm(covariance_train, ord="fro")), 1e-12)
                shift_rows.append({
                    "fold": fold, "epoch_label": epoch_label, "embedding_name": embedding,
                    "train_n": len(train), "validation_n": len(val),
                    "center_distance": float(np.linalg.norm(train.mean(0) - val.mean(0))),
                    "covariance_relative_difference": float(np.linalg.norm(covariance_train - covariance_val, ord="fro") / denominator),
                    "mmd_rbf": mmd_rbf(train, val, seed=int.from_bytes(
                        hashlib.sha256(f"{fold}/{epoch_label}/{embedding}".encode()).digest()[:4], "little")),
                })
    stats = pd.DataFrame(rows)
    stats.loc[stats.get("record_type", "summary").fillna("summary").eq("summary")].to_csv(destination / "embedding_statistics.csv", index=False)
    stats.loc[stats.get("record_type", "summary").fillna("").eq("per_dimension")].to_csv(destination / "embedding_dimension_statistics.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(destination / "effective_rank.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(destination / "train_validation_shift.csv", index=False)
    rank = pd.DataFrame(rank_rows)
    collapse = rank.loc[rank.split.isin(["train", "val"])].copy()
    collapse["epoch_position"] = collapse.epoch_label.map({name: i for i, name in enumerate(EPOCH_ORDER)})
    collapse.to_csv(destination / "epoch_collapse_comparison.csv", index=False)
    make_plots(root, stats.loc[stats.get("record_type", "summary").fillna("summary").eq("summary")])
    pca_plot(root)
    distance_plot(root)
    manifest = root / "execution_manifest.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    records.append({"timestamp": pd.Timestamp.utcnow().isoformat(), "command": " ".join(sys.argv), "stage": "representation_statistics",
                    "fold": "fold_0,fold_4", "split": "train,val,test", "epoch": "all selected", "checkpoint": None,
                    "embedding_name": "all", "probe": None, "seed": 0, "dataset_hash": None, "manifest_hash": None,
                    "feature_hash": None, "config_hash": None, "checkpoint_hash": None, "embedding_hash": None,
                    "status": "completed", "error": None, "output_path": str(destination)})
    manifest.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
