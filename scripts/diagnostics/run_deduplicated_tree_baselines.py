#!/usr/bin/env python3
"""Run leakage-safe nested Group-CV low-dimensional baselines on new artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from audit_deduplicated_dataset import TARGETS, append_execution, sha256_file, sha256_text  # noqa: E402


PROTOCOLS = ("fifth_component_group_cv", "formula_identity_group_cv")
FEATURES = ("F1_ratio_only", "F2_identity_ratio", "F3_physchem_weighted", "F4_physchem_interactions")
UNKNOWN = "__UNKNOWN_IN_OUTER_TRAIN__"


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    variable = np.nanstd(y_true) > 1e-12 and np.nanstd(y_pred) > 1e-12
    pearson = pearsonr(y_true, y_pred).statistic if len(y_true) > 1 and variable else np.nan
    spearman = spearmanr(y_true, y_pred).statistic if len(y_true) > 1 and variable else np.nan
    return {"mae": float(mean_absolute_error(y_true, y_pred)), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan, "pearson": float(pearson), "spearman": float(spearman)}


def bootstrap_mae_ci(errors: pd.Series, seed: int, repeats: int = 2000) -> tuple[float, float]:
    values = errors.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(repeats)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def require_inputs(output_dir: Path) -> tuple[pd.DataFrame, dict[str, object], dict[str, pd.DataFrame]]:
    source = json.loads((output_dir / "data_source.json").read_text(encoding="utf-8"))
    if source.get("audit_status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise RuntimeError("Baseline run blocked by audit status.")
    if sha256_file(Path(source["dataset_path"])) != source["dataset_sha256"]:
        raise RuntimeError("Raw dataset changed since audit.")
    dataset = pd.read_csv(output_dir / "data_audit" / "dataset_with_sample_id.csv", dtype={"sample_id": str})
    if dataset.sample_id.duplicated().any() or sha256_text("\n".join(sorted(dataset.sample_id))) != source["sample_id_hash"]:
        raise RuntimeError("Audited sample IDs are invalid or stale.")
    features: dict[str, pd.DataFrame] = {}
    expected = set(dataset.sample_id)
    for feature_name in FEATURES:
        table = pd.read_csv(output_dir / "artifacts" / f"{feature_name}.csv", dtype={"sample_id": str})
        if table.sample_id.duplicated().any() or set(table.sample_id) != expected:
            raise RuntimeError(f"Feature cache {feature_name} does not exactly match sample IDs.")
        features[feature_name] = table.set_index("sample_id", drop=True)
    return dataset.set_index("sample_id", drop=False), source, features


def estimator(model: str, seed: int, n_jobs: int):
    if model == "Ridge":
        return Ridge(alpha=1.0)
    if model == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.8, random_state=seed, n_jobs=n_jobs)
    if model == "RandomForest":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.7, random_state=seed, n_jobs=n_jobs)
    raise KeyError(model)


def split_feature_columns(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [column for column in features.columns if is_object_dtype(features[column]) or is_string_dtype(features[column])]
    numeric = [column for column in features.columns if column not in categorical]
    return numeric, categorical


def map_unknown_categories(train: pd.DataFrame, *others: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Create an explicit unknown column using only the current outer-training categories."""
    train = train.copy()
    _, categorical = split_feature_columns(train)
    category_values: dict[str, set[str]] = {}
    for column in categorical:
        train[column] = train[column].astype("string").fillna(UNKNOWN)
        category_values[column] = set(train[column].astype(str))
    mapped: list[pd.DataFrame] = [train]
    for other in others:
        current = other.copy()
        for column in categorical:
            value = current[column].astype("string").fillna(UNKNOWN).astype(str)
            current[column] = value.where(value.isin(category_values[column]), UNKNOWN)
        mapped.append(current)
    return tuple(mapped)


def pipeline_for(train_features: pd.DataFrame, model_name: str, seed: int, n_jobs: int) -> Pipeline:
    numeric, categorical = split_feature_columns(train_features)
    transforms: list[tuple[str, object, list[str]]] = []
    if numeric:
        transforms.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric))
    if categorical:
        categories = [sorted(set(train_features[column].astype(str)) | {UNKNOWN}) for column in categorical]
        transforms.append(("categorical", Pipeline([("onehot", OneHotEncoder(categories=categories, handle_unknown="error"))]), categorical))
    return Pipeline([("preprocess", ColumnTransformer(transforms, sparse_threshold=0.2)), ("model", estimator(model_name, seed, n_jobs))])


def row_predictions(protocol: str, fold: int, target: str, model: str, feature_set: str, sample_ids: pd.Index,
                    y_true: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    result = pd.DataFrame({"protocol": protocol, "outer_fold": fold, "target": target, "model": model,
                           "feature_set": feature_set, "sample_id": sample_ids.astype(str), "y_true": y_true.to_numpy(dtype=float),
                           "y_pred": np.asarray(predictions, dtype=float)})
    result["absolute_error"] = (result.y_true - result.y_pred).abs()
    return result


def importances(fitted: Pipeline, model_name: str, protocol: str, fold: int, target: str, feature_set: str) -> list[dict[str, object]]:
    model = fitted.named_steps["model"]
    names = fitted.named_steps["preprocess"].get_feature_names_out()
    if model_name in {"ExtraTrees", "RandomForest"}:
        values, kind = model.feature_importances_, "feature_importance"
    else:
        values, kind = np.abs(np.ravel(model.coef_)), "absolute_coefficient"
    return [{"protocol": protocol, "outer_fold": fold, "target": target, "model": model_name, "feature_set": feature_set,
             "feature": name, "importance_type": kind, "importance": float(value)} for name, value in zip(names, values)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "deduplicated_rebaseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Run one fold of one protocol to validate the complete pipeline.")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    dataset, source, features = require_inputs(output_dir)
    result_dir = output_dir / ("tree_baselines_smoke" if arguments.smoke else "tree_baselines")
    result_dir.mkdir(parents=True, exist_ok=True)
    protocols = (PROTOCOLS[0],) if arguments.smoke else PROTOCOLS
    metric_records: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    validation_records: list[dict[str, object]] = []
    importance_records: list[dict[str, object]] = []
    for protocol in protocols:
        paths = sorted((output_dir / "manifests" / protocol).glob("fold_*.csv"))
        if arguments.smoke:
            paths = paths[:1]
        for manifest_path in paths:
            manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
            if set(manifest.dataset_sha256) != {source["dataset_sha256"]}:
                raise RuntimeError(f"Wrong dataset hash in {manifest_path}")
            fold = int(manifest.outer_fold.iloc[0])
            split_ids = {name: pd.Index(manifest.loc[manifest.split == name, "sample_id"].astype(str)) for name in ("train", "val", "test")}
            if any(not ids.isin(dataset.index).all() for ids in split_ids.values()):
                raise RuntimeError(f"Manifest IDs do not map to dataset: {manifest_path}")
            outer_train_ids = split_ids["train"].append(split_ids["val"])
            for target in TARGETS:
                y_train = dataset.loc[split_ids["train"], target].astype(float)
                y_val = dataset.loc[split_ids["val"], target].astype(float)
                y_test = dataset.loc[split_ids["test"], target].astype(float)
                y_outer_train = dataset.loc[outer_train_ids, target].astype(float)
                for baseline, prediction in (("TrainMean", np.full(len(y_test), y_outer_train.mean())), ("TrainMedian", np.full(len(y_test), y_outer_train.median()))):
                    rows = row_predictions(protocol, fold, target, baseline, "none", split_ids["test"], y_test, prediction)
                    prediction_frames.append(rows)
                    metric_records.append({"protocol": protocol, "outer_fold": fold, "target": target, "model": baseline, "feature_set": "none", "n_test": len(rows), **metric_dict(rows.y_true, rows.y_pred)})
                model_candidates: list[tuple[str, str, float]] = []
                chosen_by_model: dict[str, tuple[str, float]] = {}
                for model_name in ("Ridge", "ExtraTrees", "RandomForest"):
                    scores: list[tuple[str, float]] = []
                    for feature_name, table in features.items():
                        train_x, val_x = map_unknown_categories(table.loc[split_ids["train"]], table.loc[split_ids["val"]])
                        fitted = pipeline_for(train_x, model_name, arguments.seed + fold, arguments.n_jobs)
                        fitted.fit(train_x, y_train)
                        validation_prediction = fitted.predict(val_x)
                        score = metric_dict(y_val.to_numpy(), validation_prediction)["mae"]
                        scores.append((feature_name, score))
                        validation_records.append({"protocol": protocol, "outer_fold": fold, "target": target, "model": model_name,
                                                   "feature_set": feature_name, "validation_mae": score})
                    selected_feature, selected_mae = min(scores, key=lambda value: value[1])
                    chosen_by_model[model_name] = (selected_feature, selected_mae)
                    model_candidates.append((model_name, selected_feature, selected_mae))
                    table = features[selected_feature]
                    train_x, test_x = map_unknown_categories(table.loc[outer_train_ids], table.loc[split_ids["test"]])
                    fitted = pipeline_for(train_x, model_name, arguments.seed + fold, arguments.n_jobs)
                    fitted.fit(train_x, y_outer_train)
                    test_prediction = fitted.predict(test_x)
                    rows = row_predictions(protocol, fold, target, model_name, selected_feature, split_ids["test"], y_test, test_prediction)
                    prediction_frames.append(rows)
                    metric_records.append({"protocol": protocol, "outer_fold": fold, "target": target, "model": model_name,
                                           "feature_set": selected_feature, "n_test": len(rows), "validation_mae": selected_mae,
                                           **metric_dict(rows.y_true, rows.y_pred)})
                    importance_records.extend(importances(fitted, model_name, protocol, fold, target, selected_feature))
                selected_model, selected_feature, selected_mae = min(model_candidates, key=lambda value: value[2])
                selected_rows = prediction_frames[-(1 if selected_model == "RandomForest" else 2 if selected_model == "ExtraTrees" else 3)].copy()
                selected_rows["model"] = "NestedSelectedBaseline"
                selected_rows["nested_selected_model"] = selected_model
                selected_rows["nested_validation_mae"] = selected_mae
                prediction_frames.append(selected_rows)
                metric_records.append({"protocol": protocol, "outer_fold": fold, "target": target, "model": "NestedSelectedBaseline",
                                       "feature_set": selected_feature, "nested_selected_model": selected_model, "validation_mae": selected_mae,
                                       "n_test": len(selected_rows), **metric_dict(selected_rows.y_true, selected_rows.y_pred)})
    metrics = pd.DataFrame(metric_records)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    validations = pd.DataFrame(validation_records)
    metrics.to_csv(result_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(result_dir / "oof_predictions.csv", index=False)
    validations.to_csv(result_dir / "validation_feature_selection.csv", index=False)
    pd.DataFrame(importance_records).to_csv(result_dir / "feature_importance_by_fold.csv", index=False)
    pooled_rows: list[dict[str, object]] = []
    for (protocol, target, model), group in predictions.groupby(["protocol", "target", "model"]):
        lower, upper = bootstrap_mae_ci(group.absolute_error, arguments.seed)
        mean_metric = predictions.loc[(predictions.protocol == protocol) & (predictions.target == target) & (predictions.model == "TrainMean")]
        train_mean_mae = metric_dict(mean_metric.y_true, mean_metric.y_pred)["mae"]
        pooled_rows.append({"protocol": protocol, "target": target, "model": model, "n": len(group),
                            "feature_set": "fold_selected" if group.feature_set.nunique() > 1 else group.feature_set.iloc[0],
                            **metric_dict(group.y_true, group.y_pred), "mae_ci95_low": lower, "mae_ci95_high": upper,
                            "relative_mae_improvement_vs_train_mean": float((train_mean_mae - metric_dict(group.y_true, group.y_pred)["mae"]) / train_mean_mae),
                            "completed_folds": int(group.outer_fold.nunique())})
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(result_dir / "pooled_oof_metrics.csv", index=False)
    selection_summary = validations.groupby(["protocol", "target", "model", "feature_set"], as_index=False).agg(mean_inner_validation_mae=("validation_mae", "mean"), selected_folds=("outer_fold", "nunique"))
    selected_rows: list[dict[str, object]] = []
    for (protocol, target), group in selection_summary.groupby(["protocol", "target"]):
        best = group.loc[group.mean_inner_validation_mae.idxmin()]
        selected_rows.append({"protocol": protocol, "target": target, "selected_model": best.model, "selected_feature_set": best.feature_set,
                              "mean_inner_validation_mae": best.mean_inner_validation_mae, "selection_rule": "minimum mean inner-validation MAE; feedback not read"})
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(result_dir / "selected_baseline_by_target.csv", index=False)
    report = ["# Deduplicated Tree Baselines", "", "- All identity encoders are fit in each outer-train partition; unseen validation/test identities map to an explicit unknown column.",
              "- Feature/model selection uses inner validation only; outer test labels are used only once for metrics.",
              "- No feedback CSV was read.", f"- Protocols completed: {', '.join(protocols)}."]
    (result_dir / "tree_baseline_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    append_execution(output_dir, {"timestamp": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
        "dataset_path": str(output_dir / "data_audit" / "dataset_with_sample_id.csv"), "dataset_sha256": source["dataset_sha256"],
        "protocol": "smoke_fifth" if arguments.smoke else "both", "fold": "0" if arguments.smoke else "all", "seed": arguments.seed,
        "manifest_sha256": sha256_file(output_dir / "manifests" / "manifest_integrity.csv"), "feature_hash": sha256_file(output_dir / "artifacts" / "feature_schema.json"),
        "config_hash": None, "checkpoint": None, "status": "PASS", "error": None, "output_path": str(result_dir)})
    print(f"Wrote {result_dir}")


if __name__ == "__main__":
    main()
