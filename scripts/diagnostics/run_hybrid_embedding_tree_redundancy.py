#!/usr/bin/env python3
"""Outer-train-only redundancy and residual-signal diagnostics for hybrid CV."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prepare_hybrid_embedding_tree_experiment import (  # noqa: E402
    BASE, EMBEDDINGS, FOLDS, FROZEN, ROOT, TARGETS, archive, append_execution,
)


OUTPUT = ROOT / "results/hybrid_embedding_tree_exp"
FEATURE_ROOT = OUTPUT / "features"


def numeric_and_categorical(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column])]
    return numeric, [column for column in frame if column not in numeric]


def tree_preprocessor(frame: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric, categorical = numeric_and_categorical(frame)
    transforms: list[tuple[str, object, list[str]]] = []
    if numeric:
        steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if scale_numeric:
            steps.append(("scale", StandardScaler()))
        transforms.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transforms, sparse_threshold=0.0)


def group_splits(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    count = len(np.unique(groups))
    if count < 5:
        raise ValueError(f"GroupKFold needs at least five groups; got {count}")
    splitter = GroupKFold(n_splits=5)
    return list(splitter.split(np.zeros(len(groups)), groups=groups))


def load_global_features() -> dict[str, pd.DataFrame]:
    paths = {"F0": FEATURE_ROOT / "raw_11d_descriptor.csv", **{f"F{number}": FEATURE_ROOT / f"F{number}.csv" for number in range(1, 5)}}
    frames = {}
    for name, path in paths.items():
        frame = pd.read_csv(path, dtype={"sample_id": str}).set_index("sample_id", drop=True)
        if frame.index.has_duplicates or len(frame) != 700:
            raise ValueError(f"invalid feature snapshot {path}")
        frames[name] = frame
    return frames


def fold_manifest(fold: str) -> pd.DataFrame:
    path = BASE / "manifests/formula_identity_group_cv" / f"fold_{fold.split('_')[1]}.csv"
    manifest = pd.read_csv(path, dtype={"sample_id": str}).set_index("sample_id", drop=False)
    if manifest.index.has_duplicates or len(manifest) != 700:
        raise ValueError(f"invalid manifest {path}")
    return manifest


def fold_embedding(fold: str, embedding: str, sample_ids: pd.Index) -> pd.DataFrame:
    parts = []
    for split in ("train", "val"):
        data = archive(FROZEN / "embeddings" / fold / "epoch_best" / f"{split}_{embedding}.npz")
        part = pd.DataFrame(data["embedding"], index=pd.Index(data["sample_id"].astype(str), name="sample_id"))
        if part.index.has_duplicates:
            raise ValueError(f"duplicate embedding IDs: {fold}/{split}/{embedding}")
        parts.append(part)
    combined = pd.concat(parts, axis=0)
    if set(combined.index) != set(sample_ids):
        raise ValueError(f"embedding split mismatch: {fold}/{embedding}")
    combined = combined.loc[sample_ids]
    prefix = {"descriptor_branch_raw": "Edesc", "fused_embedding": "Efused", "graph_branch_raw": "Egraph"}[embedding]
    combined.columns = [f"{prefix}_{number:03d}" for number in range(combined.shape[1])]
    return combined


def standardized_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    left_scale = np.sqrt(np.square(left).sum(axis=0, keepdims=True))
    right_scale = np.sqrt(np.square(right).sum(axis=0, keepdims=True))
    return (left / np.maximum(left_scale, 1e-12)).T @ (right / np.maximum(right_scale, 1e-12))


def numeric_tree_columns(frame: pd.DataFrame) -> pd.DataFrame:
    numeric, categorical = numeric_and_categorical(frame)
    parts = [frame[numeric].astype(float)]
    if categorical:
        encoded = pd.get_dummies(frame[categorical].astype("string"), dummy_na=True, dtype=float)
        parts.append(encoded)
    merged = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
    return merged.fillna(merged.median(numeric_only=True)).fillna(0.0)


def correlations(fold: str, embedding: str, embed: pd.DataFrame, trees: dict[str, pd.DataFrame]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detail, summary = [], []
    for family, values in trees.items():
        design = numeric_tree_columns(values)
        pearson = standardized_correlation(embed.to_numpy(), design.to_numpy())
        embed_rank = np.column_stack([rankdata(embed.iloc[:, index]) for index in range(embed.shape[1])])
        design_rank = np.column_stack([rankdata(design.iloc[:, index]) for index in range(design.shape[1])])
        spearman = standardized_correlation(embed_rank, design_rank)
        abs_pearson = np.abs(pearson)
        max_by_embedding = abs_pearson.max(axis=1)
        for dim, feature_index in np.ndindex(pearson.shape):
            detail.append({"outer_fold": fold, "embedding_name": embedding, "tree_feature_family": family,
                           "embedding_dimension": int(dim), "tree_feature": design.columns[feature_index],
                           "pearson": float(pearson[dim, feature_index]), "spearman": float(spearman[dim, feature_index])})
        summary.append({"outer_fold": fold, "embedding_name": embedding, "tree_feature_family": family,
                        "embedding_dim": embed.shape[1], "tree_encoded_dim": design.shape[1],
                        "max_abs_pearson": float(abs_pearson.max()), "mean_max_abs_pearson": float(max_by_embedding.mean()),
                        "fraction_embedding_dims_max_abs_r_gt_080": float((max_by_embedding > .80).mean()),
                        "fraction_embedding_dims_max_abs_r_gt_090": float((max_by_embedding > .90).mean()),
                        "fraction_embedding_dims_max_abs_r_gt_095": float((max_by_embedding > .95).mean())})
    return detail, summary


def reconstruction_metrics(fold: str, source_name: str, source: pd.DataFrame, target_name: str, target: pd.DataFrame,
                           groups: np.ndarray, models: tuple[str, ...], direction: str) -> list[dict[str, object]]:
    rows = []
    splits = group_splits(groups)
    # Target PCA is fit only on each inner-training partition. For the reverse
    # direction, PCA caps high-dimensional F3/F4 targets without leaking it.
    for model_name in models:
        predicted = np.zeros_like(target.to_numpy(float))
        pca_dims: list[int] = []
        component_r2: list[float] = []
        for train, validation in splits:
            x_train, x_val = source.iloc[train], source.iloc[validation]
            y_train, y_val = target.iloc[train].to_numpy(float), target.iloc[validation].to_numpy(float)
            n_components = min(10, y_train.shape[1], len(train) - 1)
            pca = PCA(n_components=n_components, random_state=0).fit(y_train)
            transformed_train, transformed_val = pca.transform(y_train), pca.transform(y_val)
            preprocessor = tree_preprocessor(x_train, scale_numeric=model_name == "Ridge")
            estimator = Ridge(alpha=1.0) if model_name == "Ridge" else RandomForestRegressor(
                n_estimators=300, min_samples_leaf=3, max_features=.7, random_state=0, n_jobs=4)
            fitted = Pipeline([("preprocess", preprocessor), ("model", estimator)]).fit(x_train, transformed_train)
            estimated = fitted.predict(x_val)
            predicted[validation] = pca.inverse_transform(estimated)
            pca_dims.append(n_components)
            component_r2.extend(r2_score(transformed_val, estimated, multioutput="raw_values").tolist())
        raw_target = target.to_numpy(float)
        dim_r2 = r2_score(raw_target, predicted, multioutput="raw_values")
        rows.append({"outer_fold": fold, "direction": direction, "source_family": source_name, "target_family": target_name,
                     "model": model_name, "n_outer_train": len(source), "source_dim": source.shape[1], "target_dim": target.shape[1],
                     "inner_splits": len(splits), "pca_target_components_min": min(pca_dims), "pca_target_components_max": max(pca_dims),
                     "reconstruction_r2_mean": float(np.mean(dim_r2)), "reconstruction_r2_median": float(np.median(dim_r2)),
                     "reconstruction_r2_fraction_positive": float((dim_r2 > 0).mean()),
                     "inner_pc_r2_mean": float(np.mean(component_r2)), "inner_pc_r2_fraction_positive": float((np.asarray(component_r2) > 0).mean()),
                     "hardest_dimension_r2": float(np.min(dim_r2)), "easiest_dimension_r2": float(np.max(dim_r2))})
    return rows


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    return {"mae": float(mean_absolute_error(y, prediction)), "r2": float(r2_score(y, prediction)),
            "pearson": float(pearsonr(y, prediction).statistic) if np.std(prediction) > 0 else np.nan,
            "spearman": float(spearmanr(y, prediction).statistic) if np.std(prediction) > 0 else np.nan,
            "kendall_tau": float(kendalltau(y, prediction).statistic) if np.std(prediction) > 0 else np.nan,
            "prediction_std": float(np.std(prediction, ddof=1)), "target_std": float(np.std(y, ddof=1))}


def residual_screen(fold: str, dataset: pd.DataFrame, manifest: pd.DataFrame, features: dict[str, pd.DataFrame], embeddings: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids = manifest.loc[manifest.split.isin(["train", "val"]), "sample_id"].astype(str)
    groups = manifest.loc[train_ids, "group_id"].to_numpy(str)
    splits = group_splits(groups)
    # Fixed cross-fitted F2 ExtraTrees reference. This is screening-only; it is
    # never selected using any outer-test information and does not use in-sample residuals.
    tree_features = features["F2"].loc[train_ids]
    records, screening = [], []
    for target in TARGETS:
        y = dataset.loc[train_ids, target].to_numpy(float)
        tree_prediction = np.zeros(len(train_ids), float)
        for inner_fold, (train, validation) in enumerate(splits):
            fitted = Pipeline([
                ("preprocess", tree_preprocessor(tree_features.iloc[train], scale_numeric=False)),
                ("model", ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=.8, random_state=inner_fold, n_jobs=4)),
            ]).fit(tree_features.iloc[train], y[train])
            tree_prediction[validation] = fitted.predict(tree_features.iloc[validation])
        residual = y - tree_prediction
        for sample_id, group, truth, tree_pred, residual_value in zip(train_ids, groups, y, tree_prediction, residual):
            records.append({"outer_fold": fold, "sample_id": sample_id, "group_id": group, "target": target,
                            "tree_feature_family": "F2", "tree_model": "ExtraTrees_fixed_screening",
                            "y_true": truth, "tree_oof_pred": tree_pred, "residual": residual_value})
        for embedding_name, embedding in embeddings.items():
            values = embedding.to_numpy(float)
            dim_pearson = np.array([pearsonr(values[:, dim], residual).statistic if np.std(values[:, dim]) else np.nan for dim in range(values.shape[1])])
            dim_spearman = np.array([spearmanr(values[:, dim], residual).statistic if np.std(values[:, dim]) else np.nan for dim in range(values.shape[1])])
            for model_name, model in [("Ridge", Ridge(alpha=1.0)), ("ExtraTrees", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=.7, random_state=0, n_jobs=4))]:
                prediction = np.zeros(len(residual))
                for train, validation in splits:
                    frame = embedding.iloc[train]
                    pipeline = Pipeline([("preprocess", tree_preprocessor(frame, scale_numeric=model_name == "Ridge")), ("model", model)])
                    pipeline.fit(frame, residual[train])
                    prediction[validation] = pipeline.predict(embedding.iloc[validation])
                screening.append({"outer_fold": fold, "target": target, "embedding_name": embedding_name,
                                  "residual_probe_model": model_name, "n_outer_train": len(residual),
                                  "max_abs_embedding_residual_pearson": float(np.nanmax(np.abs(dim_pearson))),
                                  "max_abs_embedding_residual_spearman": float(np.nanmax(np.abs(dim_spearman))),
                                  **{f"residual_{key}": value for key, value in metrics(residual, prediction).items()}})
    return pd.DataFrame(records), pd.DataFrame(screening)


def main() -> None:
    prerequisite = OUTPUT / "prerequisite_summary.json"
    if not prerequisite.is_file():
        raise RuntimeError("Run prepare_hybrid_embedding_tree_experiment.py first")
    redundancy = OUTPUT / "redundancy"
    residual_output = OUTPUT / "residual"
    redundancy.mkdir(parents=True, exist_ok=True)
    residual_output.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(BASE / "data_audit/dataset_with_sample_id.csv", dtype={"sample_id": str}).set_index("sample_id", drop=False)
    features = load_global_features()
    details: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    embedding_reconstruction: list[dict[str, object]] = []
    reverse_reconstruction: list[dict[str, object]] = []
    residual_records, residual_screens = [], []
    for fold in FOLDS:
        manifest = fold_manifest(fold)
        train_ids = manifest.loc[manifest.split.isin(["train", "val"]), "sample_id"].astype(str)
        groups = manifest.loc[train_ids, "group_id"].to_numpy(str)
        tree_train = {name: values.loc[train_ids] for name, values in features.items() if name != "F0"}
        embedding_train = {embedding: fold_embedding(fold, embedding, train_ids) for embedding in EMBEDDINGS}
        for embedding_name, values in embedding_train.items():
            detail, summary = correlations(fold, embedding_name, values, tree_train)
            details.extend(detail)
            summaries.extend(summary)
            for family, tree_values in tree_train.items():
                embedding_reconstruction.extend(reconstruction_metrics(
                    fold, family, tree_values, embedding_name, values, groups, ("Ridge", "RandomForest"), "tree_to_embedding"))
                reverse_reconstruction.extend(reconstruction_metrics(
                    fold, embedding_name, values, family, numeric_tree_columns(tree_values), groups, ("Ridge", "RandomForest"), "embedding_to_tree_numeric"))
        records, screen = residual_screen(fold, dataset, manifest, features, embedding_train)
        residual_records.append(records)
        residual_screens.append(screen)
    pd.DataFrame(details).to_csv(redundancy / "feature_embedding_correlations.csv", index=False)
    pd.DataFrame(summaries).to_csv(redundancy / "feature_embedding_correlation_summary.csv", index=False)
    pd.DataFrame(embedding_reconstruction).to_csv(redundancy / "embedding_reconstruction_metrics.csv", index=False)
    pd.DataFrame(reverse_reconstruction).to_csv(redundancy / "tree_feature_reconstruction_metrics.csv", index=False)
    pd.concat(residual_records, ignore_index=True).to_csv(residual_output / "crossfitted_tree_predictions.csv", index=False)
    screen = pd.concat(residual_screens, ignore_index=True)
    screen.to_csv(redundancy / "residual_signal_screening.csv", index=False)
    # This report is evidence-only and intentionally does not look at outer test.
    corr = pd.DataFrame(summaries)
    report = ["# Hybrid feature redundancy audit", "", "All calculations use each fold's outer-train only; no outer-test labels were read.",
              "", "## Correlation summary", "", corr.groupby(["embedding_name", "tree_feature_family"], as_index=False).agg(
                  mean_max_abs_pearson=("mean_max_abs_pearson", "mean"),
                  mean_fraction_gt_090=("fraction_embedding_dims_max_abs_r_gt_090", "mean")).to_csv(index=False),
              "", "## Residual screening", "", screen.groupby(["target", "embedding_name", "residual_probe_model"], as_index=False).agg(
                  residual_mae=("residual_mae", "mean"), residual_r2=("residual_r2", "mean"),
                  residual_spearman=("residual_spearman", "mean")).to_csv(index=False),
              "", "`descriptor_branch_raw` is audited separately against the original 5 × 11 raw descriptor input in `audit/sample_alignment_audit.csv`."]
    (redundancy / "redundancy_report.md").write_text("\n".join(report), encoding="utf-8")
    append_execution(OUTPUT, stage="outer_train_redundancy_and_residual_screening", target="all", outer_fold="all",
                     feature_family="F1,F2,F3,F4", embedding_name=",".join(EMBEDDINGS), model="Ridge,RandomForest,ExtraTrees",
                     status="completed", output_path=str(redundancy))
    print("REDUNDANCY_COMPLETE", redundancy)


if __name__ == "__main__":
    main()
