#!/usr/bin/env python3
"""Leakage-safe inner GroupKFold evaluator for the hybrid tree experiment.

This is deliberately a *development-only* runner.  It reads train+val rows of
one fixed outer fold and never opens that fold's test embedding or test label.
Shards are independent so the full pre-registered grid can be scheduled in
small, resumable jobs without sharing mutable result files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prepare_hybrid_embedding_tree_experiment import (  # noqa: E402
    BASE, FROZEN, ROOT, TARGETS, archive, sha256_file,
)


OUTPUT = ROOT / "results/hybrid_embedding_tree_exp"
FEATURE_ROOT = OUTPUT / "features"
EMBEDDING_NAMES = {
    "E_desc": "descriptor_branch_raw",
    "E_fused": "fused_embedding",
    "E_graph": "graph_branch_raw",
}
LOCKED_EMBEDDING = {
    "EE_before": "E_desc",
    "EE_after": "E_desc",
    "Aerosolization_Efficiency": "E_desc",
    "mRNA_Recovery_Efficiency": "E_fused",
}
FAMILY_SPECS = {
    "A0": ("F0",), "A1": ("F1",), "A2": ("F2",), "A3": ("F3",), "A4": ("F4",),
    "A5": ("E_desc",), "A6": ("E_fused",), "A7": ("E_graph",),
    "B1": ("F0", "E_desc"), "B2": ("F1", "E_desc"), "B3": ("F2", "E_desc"),
    "B4": ("F3", "E_desc"), "B5": ("F4", "E_desc"), "B6": ("F1", "E_fused"),
    "B7": ("F2", "E_fused"), "B8": ("F3", "E_fused"), "B9": ("F4", "E_fused"),
    "B10": ("F2", "E_desc", "E_fused"),
}
TREE_BASES = ("F1", "F2", "F3", "F4")
MODEL_ORDER = {"M0": 0, "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 6}


def json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    residual = y_pred - y_true
    prediction_std = float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else 0.0
    target_std = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 0.0
    variable = prediction_std > 1e-12 and target_std > 1e-12
    slope = float(np.cov(y_true, y_pred, ddof=1)[0, 1] / np.var(y_true, ddof=1)) if target_std > 1e-12 else np.nan
    intercept = float(y_pred.mean() - slope * y_true.mean()) if np.isfinite(slope) else np.nan
    lower_cut, upper_cut = np.quantile(y_true, (.2, .8))
    lower_mask, upper_mask = y_true <= lower_cut, y_true >= upper_cut
    tail_count = max(1, int(np.ceil(.2 * len(y_true))))
    truth_order = np.argsort(y_true, kind="mergesort")
    prediction_order = np.argsort(y_pred, kind="mergesort")
    top_truth, top_prediction = set(truth_order[-tail_count:]), set(prediction_order[-tail_count:])
    bottom_truth, bottom_prediction = set(truth_order[:tail_count]), set(prediction_order[:tail_count])
    return {
        "n": int(len(y_true)), "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))), "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(pearsonr(y_true, y_pred).statistic) if variable else np.nan,
        "spearman": float(spearmanr(y_true, y_pred).statistic) if variable else np.nan,
        "kendall_tau": float(kendalltau(y_true, y_pred).statistic) if variable else np.nan,
        "prediction_mean": float(np.mean(y_pred)), "target_mean": float(np.mean(y_true)),
        "prediction_std": prediction_std, "target_std": target_std,
        "std_ratio": float(prediction_std / target_std) if target_std > 1e-12 else np.nan,
        "calibration_intercept": intercept, "calibration_slope": slope,
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
        "lower_tail_mae": float(mean_absolute_error(y_true[lower_mask], y_pred[lower_mask])),
        "upper_tail_mae": float(mean_absolute_error(y_true[upper_mask], y_pred[upper_mask])),
        "top20_recall": float(len(top_truth.intersection(top_prediction)) / tail_count),
        "bottom20_recall": float(len(bottom_truth.intersection(bottom_prediction)) / tail_count),
    }


def feature_frames() -> dict[str, pd.DataFrame]:
    paths = {"F0": FEATURE_ROOT / "raw_11d_descriptor.csv"}
    paths.update({f"F{number}": FEATURE_ROOT / f"F{number}.csv" for number in range(1, 5)})
    frames: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        frame = pd.read_csv(path, dtype={"sample_id": str}).set_index("sample_id", drop=True)
        if len(frame) != 700 or frame.index.has_duplicates:
            raise RuntimeError(f"Invalid audited feature frame: {path}")
        frames[name] = frame
    return frames


def outer_manifest(fold: int) -> pd.DataFrame:
    path = BASE / "manifests/formula_identity_group_cv" / f"fold_{fold}.csv"
    frame = pd.read_csv(path, dtype={"sample_id": str}).set_index("sample_id", drop=False)
    if len(frame) != 700 or frame.index.has_duplicates:
        raise RuntimeError(f"Invalid fixed manifest: {path}")
    return frame


def outer_train_embedding(fold: int, embedding_alias: str, sample_ids: pd.Index) -> pd.DataFrame:
    source = EMBEDDING_NAMES[embedding_alias]
    chunks = []
    for split in ("train", "val"):
        data = archive(FROZEN / "embeddings" / f"fold_{fold}" / "epoch_best" / f"{split}_{source}.npz")
        rows = pd.DataFrame(data["embedding"], index=pd.Index(data["sample_id"].astype(str), name="sample_id"))
        chunks.append(rows)
    frame = pd.concat(chunks)
    if frame.index.has_duplicates or set(frame.index) != set(sample_ids):
        raise RuntimeError(f"Embedding alignment failure for fold_{fold}/{embedding_alias}")
    frame = frame.loc[sample_ids]
    frame.columns = [f"{embedding_alias}_{index:03d}" for index in range(frame.shape[1])]
    return frame


def compose_features(family: str, target: str, bases: dict[str, pd.DataFrame], embeddings: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, object]]:
    if family == "B11":
        raise ValueError("B11 requires a concrete --b11-base selector in a shard")
    parts = []
    for item in FAMILY_SPECS[family]:
        part = bases[item] if item.startswith("F") else embeddings[item]
        copied = part.copy()
        if item.startswith("F"):
            copied.columns = [f"{item}__{column}" for column in copied.columns]
        parts.append(copied)
    frame = pd.concat(parts, axis=1)
    return frame, {"feature_blocks": list(FAMILY_SPECS[family]), "b11_base": None, "locked_embedding": None}


def compose_b11(target: str, base: str, bases: dict[str, pd.DataFrame], embeddings: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, object]]:
    locked = LOCKED_EMBEDDING[target]
    tree = bases[base].copy()
    tree.columns = [f"{base}__{column}" for column in tree.columns]
    frame = pd.concat([tree, embeddings[locked]], axis=1)
    return frame, {"feature_blocks": [base, locked], "b11_base": base, "locked_embedding": locked}


def type_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [column for column in frame if not pd.api.types.is_numeric_dtype(frame[column])]
    return [column for column in frame if column not in categorical], categorical


def preprocessor(frame: pd.DataFrame, *, scale: bool) -> ColumnTransformer:
    numeric, categorical = type_columns(frame)
    transforms: list[tuple[str, object, list[str]]] = []
    if numeric:
        steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if scale:
            steps.append(("scale", StandardScaler()))
        transforms.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transforms, sparse_threshold=0.0)


def inner_splits(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(np.unique(groups)) < 5:
        raise RuntimeError("Outer train has fewer than five formula groups")
    return list(GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups))


def transformed_partitions(frame: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]], *, scale: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    result = []
    for train, validation in splits:
        fitted = preprocessor(frame.iloc[train], scale=scale).fit(frame.iloc[train])
        result.append((fitted.transform(frame.iloc[train]), fitted.transform(frame.iloc[validation])))
    return result


def blank_predictions(n: int, params: list[dict[str, object]]) -> dict[str, np.ndarray]:
    return {json_text(param): np.full(n, np.nan, dtype=float) for param in params}


def forest_prefix_prediction(model: object, values: np.ndarray, count: int) -> np.ndarray:
    # With a fixed random_state sklearn's first 300 trees of a 600-tree forest
    # are identical to a direct 300-tree fit.  Reusing them halves the fixed
    # pre-registered n_estimators grid while preserving its predictions.
    trees = model.estimators_[:count]
    return np.mean(np.vstack([tree.predict(values) for tree in trees]), axis=0)


def evaluate_model(model_name: str, frame: pd.DataFrame, y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]], seed: int,
                   tree_n_jobs: int) -> tuple[list[dict[str, object]], dict[str, np.ndarray], list[dict[str, object]]]:
    """Return parameter definitions, OOF vectors, and explicit N/A records."""
    n = len(y)
    numeric, categorical = type_columns(frame)
    skipped: list[dict[str, object]] = []
    if model_name == "M0":
        params = [{"kind": "train_mean"}]
        predicted = blank_predictions(n, params)
        for train, validation in splits:
            predicted[json_text(params[0])][validation] = float(np.mean(y[train]))
        return params, predicted, skipped

    if model_name == "M1":
        params = [{"alpha": value} for value in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)]
        predicted = blank_predictions(n, params)
        partitions = transformed_partitions(frame, splits, scale=True)
        for split_number, ((train, validation), (x_train, x_val)) in enumerate(zip(splits, partitions)):
            for param in params:
                model = Ridge(alpha=float(param["alpha"]))
                model.fit(x_train, y[train])
                predicted[json_text(param)][validation] = model.predict(x_val)
        return params, predicted, skipped

    if model_name == "M2":
        params = [{"alpha": alpha, "l1_ratio": ratio} for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0) for ratio in (0.0, 0.1, 0.5, 0.9, 1.0)]
        predicted = blank_predictions(n, params)
        partitions = transformed_partitions(frame, splits, scale=True)
        for split_number, ((train, validation), (x_train, x_val)) in enumerate(zip(splits, partitions)):
            for param in params:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    # The grid fixes alpha and l1_ratio.  A moderate tolerance
                    # and bounded solver work keep high-dimensional fused
                    # feature shards reproducible and finishable; any solver
                    # nonconvergence remains a recorded candidate outcome,
                    # rather than silently changing the hyperparameter grid.
                    model = ElasticNet(alpha=float(param["alpha"]), l1_ratio=float(param["l1_ratio"]), max_iter=1000,
                                       tol=1e-3, random_state=seed + split_number)
                    model.fit(x_train, y[train])
                predicted[json_text(param)][validation] = model.predict(x_val)
        return params, predicted, skipped

    if model_name in {"M3", "M4"}:
        params = [{"n_estimators": count, "max_depth": depth, "min_samples_leaf": leaf, "max_features": features}
                  for count in (300, 600) for depth in (None, 5, 10, 20) for leaf in (1, 2, 3, 5, 10) for features in (1.0, "sqrt", 0.5)]
        predicted = blank_predictions(n, params)
        partitions = transformed_partitions(frame, splits, scale=False)
        estimator_type = RandomForestRegressor if model_name == "M3" else ExtraTreesRegressor
        for split_number, ((train, validation), (x_train, x_val)) in enumerate(zip(splits, partitions)):
            for depth in (None, 5, 10, 20):
                for leaf in (1, 2, 3, 5, 10):
                    for features in (1.0, "sqrt", 0.5):
                        model = estimator_type(n_estimators=600, max_depth=depth, min_samples_leaf=leaf, max_features=features,
                                               random_state=seed + split_number, n_jobs=tree_n_jobs)
                        model.fit(x_train, y[train])
                        for count, prediction in ((300, forest_prefix_prediction(model, x_val, 300)), (600, model.predict(x_val))):
                            param = {"n_estimators": count, "max_depth": depth, "min_samples_leaf": leaf, "max_features": features}
                            predicted[json_text(param)][validation] = prediction
        return params, predicted, skipped

    if model_name == "M5":
        params = [{"learning_rate": rate, "max_leaf_nodes": leaves, "min_samples_leaf": min_leaf, "l2_regularization": l2}
                  for rate in (0.03, 0.05, 0.1) for leaves in (7, 15, 31) for min_leaf in (5, 10, 20) for l2 in (0.0, 0.1, 1.0, 10.0)]
        predicted = blank_predictions(n, params)
        partitions = transformed_partitions(frame, splits, scale=False)
        for split_number, ((train, validation), (x_train, x_val)) in enumerate(zip(splits, partitions)):
            for param in params:
                model = HistGradientBoostingRegressor(**param, random_state=seed + split_number)
                model.fit(x_train, y[train])
                predicted[json_text(param)][validation] = model.predict(x_val)
        return params, predicted, skipped

    if model_name == "M6":
        params = [{"n_components": value} for value in (2, 4, 8, 16, 32)]
        predicted = blank_predictions(n, params)
        if categorical:
            for param in params:
                skipped.append({"params": param, "status": "NOT_APPLICABLE_CATEGORICAL_FEATURES", "detail": "PLS is restricted to continuous features"})
            return [], {}, skipped
        partitions = transformed_partitions(frame, splits, scale=True)
        allowed = []
        for param in params:
            component_count = int(param["n_components"])
            if component_count > len(numeric) or any(component_count > min(len(train) - 1, len(numeric)) for train, _ in splits):
                skipped.append({"params": param, "status": "NOT_APPLICABLE_COMPONENT_LIMIT", "detail": "n_components exceeds feature or inner-train dimensionality"})
            else:
                allowed.append(param)
        predicted = blank_predictions(n, allowed)
        for (train, validation), (x_train, x_val) in zip(splits, partitions):
            for param in allowed:
                model = PLSRegression(n_components=int(param["n_components"]), scale=False, max_iter=1000)
                model.fit(x_train, y[train])
                predicted[json_text(param)][validation] = np.ravel(model.predict(x_val))
        return allowed, predicted, skipped
    raise KeyError(model_name)


def selected_param(params: list[dict[str, object]], predictions: dict[str, np.ndarray], y: np.ndarray) -> dict[str, object]:
    scored = []
    for param in params:
        value = predictions[json_text(param)]
        metrics = metric_dict(y, value)
        scored.append((metrics["mae"], -metrics["r2"], param, metrics))
    return min(scored, key=lambda item: (item[0], item[1], json_text(item[2])))[2]


def run_fold(arguments: argparse.Namespace) -> None:
    prerequisite = OUTPUT / "prerequisite_summary.json"
    if not prerequisite.is_file():
        raise RuntimeError("Run prepare_hybrid_embedding_tree_experiment.py first")
    if arguments.stage not in {"stage1", "stage2"}:
        raise ValueError(arguments.stage)
    families = [value.strip() for value in arguments.families.split(",") if value.strip()]
    models = [value.strip() for value in arguments.models.split(",") if value.strip()]
    allowed_families = set(FAMILY_SPECS) | {"B11"}
    if set(families) - allowed_families or set(models) - set(MODEL_ORDER):
        raise ValueError("Unknown family or model")
    b11_bases = [value.strip() for value in arguments.b11_bases.split(",") if value.strip()]
    if "B11" in families and (not b11_bases or set(b11_bases) - set(TREE_BASES)):
        raise ValueError("B11 needs --b11-bases selected from F1,F2,F3,F4")
    pipeline_spec = None
    if arguments.pipeline_spec is not None:
        pipeline_spec = pd.read_csv(arguments.pipeline_spec)
        required_spec_columns = {"target", "feature_family", "model"}
        if missing := required_spec_columns - set(pipeline_spec.columns):
            raise ValueError(f"Pipeline spec missing columns: {sorted(missing)}")
        if "b11_base" not in pipeline_spec:
            pipeline_spec["b11_base"] = np.nan
        pipeline_spec = pipeline_spec.loc[pipeline_spec.target.isin(TARGETS)].copy()
        if pipeline_spec.empty or set(pipeline_spec.feature_family) - allowed_families or set(pipeline_spec.model) - set(MODEL_ORDER):
            raise ValueError("Pipeline spec contains no valid target/family/model rows")

    stage_root = OUTPUT / arguments.stage / "shards" / arguments.tag
    stage_root.mkdir(parents=True, exist_ok=True)
    done = stage_root / f"fold_{arguments.fold}_metrics.csv"
    if done.exists() and not arguments.force:
        print("ALREADY_COMPLETE", done)
        return

    dataset = pd.read_csv(BASE / "data_audit/dataset_with_sample_id.csv", dtype={"sample_id": str}).set_index("sample_id", drop=False)
    manifest = outer_manifest(arguments.fold)
    train_ids = pd.Index(manifest.loc[manifest.split.isin(["train", "val"]), "sample_id"].astype(str), name="sample_id")
    groups = manifest.loc[train_ids, "group_id"].to_numpy(str)
    if len(train_ids) != 560 or set(manifest.loc[manifest.split.eq("test"), "sample_id"]).intersection(train_ids):
        raise RuntimeError("Invalid sealed outer split")
    # The test dataframe/labels are intentionally never indexed below.
    bases = {name: frame.loc[train_ids] for name, frame in feature_frames().items()}
    embeddings = {alias: outer_train_embedding(arguments.fold, alias, train_ids) for alias in EMBEDDING_NAMES}
    splits = inner_splits(groups)
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, object]] = []

    for target in TARGETS:
        y = dataset.loc[train_ids, target].to_numpy(float)
        family_jobs: list[tuple[str, str | None, pd.DataFrame, dict[str, object], list[str]]] = []
        if pipeline_spec is None:
            requested = [(family, base, models) for family in families for base in (b11_bases if family == "B11" else [None])]
        else:
            requested = []
            for _, row in pipeline_spec.loc[pipeline_spec.target.eq(target)].iterrows():
                selected_base = None if pd.isna(row.b11_base) else str(row.b11_base)
                if row.feature_family == "B11" and selected_base not in TREE_BASES:
                    raise ValueError(f"B11 pipeline spec needs F1--F4 b11_base for {target}")
                requested.append((str(row.feature_family), selected_base, [str(row.model)]))
        for family, concrete_b11_base, selected_models in requested:
            if family == "B11":
                x, provenance = compose_b11(target, str(concrete_b11_base), bases, embeddings)
                family_jobs.append((family, concrete_b11_base, x, provenance, selected_models))
            else:
                x, provenance = compose_features(family, target, bases, embeddings)
                family_jobs.append((family, None, x, provenance, selected_models))
        for family, b11_base, x, provenance, selected_models in family_jobs:
            feature_hash = hash_text(json_text({"columns": list(x.columns), "dtypes": [str(x[column].dtype) for column in x.columns]}))
            for model_name in selected_models:
                params, predictions, skipped = evaluate_model(model_name, x, y, splits, arguments.seed, arguments.tree_n_jobs)
                for item in skipped:
                    skipped_rows.append({"stage": arguments.stage, "outer_fold": arguments.fold, "target": target,
                                         "feature_family": family, "b11_base": b11_base, "model": model_name,
                                         "params_json": json_text(item["params"]), "status": item["status"], "detail": item["detail"]})
                if not params:
                    continue
                for param in params:
                    prediction = predictions[json_text(param)]
                    if not np.isfinite(prediction).all():
                        raise RuntimeError(f"Non-finite inner prediction: {family}/{model_name}/{param}")
                    record = {"stage": arguments.stage, "outer_fold": arguments.fold, "target": target,
                              "feature_family": family, "b11_base": b11_base, "model": model_name,
                              "params_json": json_text(param), "feature_dim_raw": int(x.shape[1]),
                              "feature_blocks_json": json_text(provenance["feature_blocks"]), "feature_hash": feature_hash,
                              "embedding_locked": provenance["locked_embedding"], "inner_splits": len(splits),
                              "selection_metric": "inner_groupkfold_oof_mae", "status": "COMPLETED", **metric_dict(y, prediction)}
                    metrics_rows.append(record)
                chosen = selected_param(params, predictions, y)
                chosen_prediction = predictions[json_text(chosen)]
                prediction_rows.append(pd.DataFrame({
                    "stage": arguments.stage, "outer_fold": arguments.fold, "sample_id": train_ids,
                    "group_id": groups, "target": target, "feature_family": family, "b11_base": b11_base,
                    "model": model_name, "params_json": json_text(chosen), "y_true": y,
                    "inner_oof_pred": chosen_prediction,
                }))

    metrics = pd.DataFrame(metrics_rows).sort_values(["target", "feature_family", "b11_base", "model", "mae"], na_position="first")
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    metrics.to_csv(done, index=False)
    predictions.to_csv(stage_root / f"fold_{arguments.fold}_selected_inner_oof_predictions.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(stage_root / f"fold_{arguments.fold}_not_applicable.csv", index=False)
    local_manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "command": list(map(str, [Path(__file__).resolve(), *__import__("sys").argv[1:]])),
        "stage": arguments.stage, "outer_fold": arguments.fold, "tag": arguments.tag, "families": families,
        "models": models, "b11_bases": b11_bases, "pipeline_spec": None if arguments.pipeline_spec is None else str(arguments.pipeline_spec.resolve()), "dataset_hash": sha256_file(BASE / "data_audit/dataset_with_sample_id.csv"),
        "manifest_hash": sha256_file(BASE / "manifests/formula_identity_group_cv" / f"fold_{arguments.fold}.csv"),
        "test_labels_read": False, "status": "COMPLETED", "outputs": [str(done), str(stage_root / f"fold_{arguments.fold}_selected_inner_oof_predictions.csv")],
    }
    (stage_root / f"fold_{arguments.fold}_execution.json").write_text(json.dumps(local_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("STAGE_DEVELOPMENT_COMPLETE", arguments.stage, arguments.tag, f"fold_{arguments.fold}", len(metrics), len(predictions))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--families", required=True, help="Comma-separated A0..B11 family labels")
    parser.add_argument("--models", default="M0,M1,M2,M3,M4,M5,M6")
    parser.add_argument("--b11-bases", default="F1,F2,F3,F4")
    parser.add_argument("--pipeline-spec", type=Path, help="CSV target,feature_family,b11_base,model; evaluates only those locked target-specific pairs.")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--tree-n-jobs", type=int, default=1, help="Threads per RandomForest/ExtraTrees fit; use 1 for many process shards.")
    parser.add_argument("--force", action="store_true")
    run_fold(parser.parse_args())


if __name__ == "__main__":
    main()
