#!/usr/bin/env python3
"""Run leakage-safe tabular regressors across all diagnostic split definitions."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    add_common_arguments, build_feature_frames, discover_schema, load_frames,
    load_mordred_table, metric_dict,
)


def _preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    """Construct fold-local numeric/categorical preprocessing."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("variance", VarianceThreshold()),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ], sparse_threshold=0.2)


def _models(seed: int, n_jobs: int) -> dict[str, object]:
    """Return fixed, non-test-tuned model specifications."""
    models: dict[str, object] = {
        "KNN_k1": KNeighborsRegressor(n_neighbors=1, weights="distance"),
        "KNN_k3": KNeighborsRegressor(n_neighbors=3, weights="distance"),
        "KNN_k5": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "KNN_k10": KNeighborsRegressor(n_neighbors=10, weights="distance"),
        "Ridge_alpha1": Ridge(alpha=1.0),
        "ElasticNet_alpha0.1_l1_0.5": ElasticNet(
            alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=seed,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.7,
            random_state=seed, n_jobs=n_jobs,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=2, max_features=0.8,
            random_state=seed, n_jobs=n_jobs,
        ),
    }
    return models


def _pipeline(preprocessor: ColumnTransformer, estimator: object) -> Pipeline:
    """Keep all supervised selection and transformations inside training fit."""
    return Pipeline([
        ("preprocess", clone(preprocessor)),
        ("select", SelectKBest(score_func=f_regression, k=128)),
        ("regressor", estimator),
    ])


def _prediction_rows(
    split_name: str, evaluation_set: str, target: str, model_name: str,
    sample_ids: pd.Series, true_values: pd.Series, predictions: np.ndarray,
) -> pd.DataFrame:
    output = pd.DataFrame({
        "split_name": split_name,
        "evaluation_set": evaluation_set,
        "target": target,
        "model": model_name,
        "diagnostic_sample_id": sample_ids.astype(str).to_numpy(),
        "y_true": true_values.to_numpy(dtype=float),
        "y_pred": predictions.astype(float),
    })
    output["absolute_error"] = (output["y_true"] - output["y_pred"]).abs()
    return output


def _append_metrics(
    records: list[dict[str, object]], split_name: str, evaluation_set: str,
    target: str, model_name: str, true_values: pd.Series, predictions: np.ndarray,
    mean_mae: float,
) -> None:
    metrics = metric_dict(true_values, predictions)
    records.append({
        "split_name": split_name,
        "evaluation_set": evaluation_set,
        "target": target,
        "model": model_name,
        "n_samples": int(np.isfinite(true_values.to_numpy(dtype=float)).sum()),
        **metrics,
        "mae_improvement_vs_train_mean": float(mean_mae - metrics["mae"]),
        "status": "ok",
    })


def _evaluate_partition(
    split_name: str, train_indices: pd.Index, evaluation_indices: pd.Index,
    evaluation_set: str, frame: pd.DataFrame, features: pd.DataFrame,
    preprocessor: ColumnTransformer, models: dict[str, object], targets: list[str],
    metric_records: list[dict[str, object]], prediction_records: list[pd.DataFrame],
) -> None:
    """Fit one model per target only on train_indices and score a held-out set."""
    for target in targets:
        valid_train = frame.loc[train_indices, target].notna()
        target_train_indices = train_indices[valid_train]
        true_values = frame.loc[evaluation_indices, target]
        if len(target_train_indices) < 10:
            metric_records.append({
                "split_name": split_name, "evaluation_set": evaluation_set,
                "target": target, "model": "all", "status": "skipped_too_few_training_labels",
            })
            continue
        train_targets = frame.loc[target_train_indices, target].astype(float)
        train_features = features.loc[target_train_indices]
        evaluation_features = features.loc[evaluation_indices]
        mean_prediction = np.full(len(evaluation_indices), train_targets.mean(), dtype=float)
        median_prediction = np.full(len(evaluation_indices), train_targets.median(), dtype=float)
        mean_mae = metric_dict(true_values, mean_prediction)["mae"]
        for model_name, predictions in {
            "TrainMean": mean_prediction, "TrainMedian": median_prediction,
        }.items():
            _append_metrics(
                metric_records, split_name, evaluation_set, target, model_name,
                true_values, predictions, mean_mae,
            )
            prediction_records.append(_prediction_rows(
                split_name, evaluation_set, target, model_name,
                frame.loc[evaluation_indices, "diagnostic_sample_id"], true_values, predictions,
            ))

        for model_name, estimator in models.items():
            try:
                fitted_model = _pipeline(preprocessor, clone(estimator)).fit(train_features, train_targets)
                predictions = fitted_model.predict(evaluation_features)
                _append_metrics(
                    metric_records, split_name, evaluation_set, target, model_name,
                    true_values, predictions, mean_mae,
                )
                prediction_records.append(_prediction_rows(
                    split_name, evaluation_set, target, model_name,
                    frame.loc[evaluation_indices, "diagnostic_sample_id"], true_values, predictions,
                ))
            except Exception as error:  # Preserve other diagnostics if one model is unsupported.
                metric_records.append({
                    "split_name": split_name, "evaluation_set": evaluation_set,
                    "target": target, "model": model_name, "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-mordred-features", type=int, default=256)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    split_dir = output_dir / "splits"
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Split directory is missing: {split_dir}. Run build_generalization_splits.py first."
        )
    schema = discover_schema(arguments.train_csv, arguments.feedback_csv)
    train_frame, feedback_frame = load_frames(schema)
    mordred_frame = load_mordred_table(schema)
    train_numeric, train_categorical = build_feature_frames(
        train_frame, schema, mordred_frame, arguments.max_mordred_features
    )
    feedback_numeric, feedback_categorical = build_feature_frames(
        feedback_frame, schema, mordred_frame, arguments.max_mordred_features
    )
    train_features = pd.concat([train_numeric, train_categorical], axis=1)
    feedback_features = pd.concat([feedback_numeric, feedback_categorical], axis=1)
    preprocessor = _preprocessor(list(train_numeric.columns), list(train_categorical.columns))
    models = _models(arguments.seed, arguments.n_jobs)
    optional_availability = {
        "CatBoostRegressor": importlib.util.find_spec("catboost") is not None,
        "LightGBM": importlib.util.find_spec("lightgbm") is not None,
        "XGBoost": importlib.util.find_spec("xgboost") is not None,
    }
    metric_records: list[dict[str, object]] = []
    prediction_records: list[pd.DataFrame] = []

    split_paths = sorted(split_dir.glob("*_split.csv"))
    if not split_paths:
        raise FileNotFoundError(f"No split CSVs found in {split_dir}")
    frame_by_sample_id = train_frame.set_index("diagnostic_sample_id", drop=False)
    feature_by_sample_id = train_features.copy()
    feature_by_sample_id.index = train_frame["diagnostic_sample_id"]
    for split_path in split_paths:
        split_frame = pd.read_csv(split_path, dtype={"diagnostic_sample_id": str})
        split_name = split_path.stem
        if set(split_frame["diagnostic_sample_id"]) != set(train_frame["diagnostic_sample_id"]):
            raise ValueError(f"{split_path} does not cover exactly the training samples.")
        training_ids = split_frame.loc[split_frame["split"] == "train", "diagnostic_sample_id"]
        for evaluation_set in ("val", "test"):
            evaluation_ids = split_frame.loc[
                split_frame["split"] == evaluation_set, "diagnostic_sample_id"
            ]
            _evaluate_partition(
                split_name, pd.Index(training_ids), pd.Index(evaluation_ids), evaluation_set,
                frame_by_sample_id, feature_by_sample_id, preprocessor, models, schema.targets,
                metric_records, prediction_records,
            )

    # A full-source fit provides an external feedback baseline without using its labels.
    feedback_frame_by_id = feedback_frame.set_index("diagnostic_sample_id", drop=False)
    feedback_features_by_id = feedback_features.copy()
    feedback_features_by_id.index = feedback_frame["diagnostic_sample_id"]
    combined_frame = pd.concat([frame_by_sample_id, feedback_frame_by_id], axis=0)
    combined_features = pd.concat([feature_by_sample_id, feedback_features_by_id], axis=0)
    _evaluate_partition(
        "full_train", frame_by_sample_id.index, feedback_frame_by_id.index, "feedback",
        combined_frame, combined_features, preprocessor, models, schema.targets,
        metric_records, prediction_records,
    )

    metric_frame = pd.DataFrame(metric_records)
    prediction_frame = pd.concat(prediction_records, ignore_index=True)
    metric_frame.to_csv(output_dir / "baseline_metrics_long.csv", index=False)
    prediction_frame.to_csv(output_dir / "baseline_predictions.csv", index=False)
    summary_frame = metric_frame.loc[metric_frame["status"] == "ok"].groupby(
        ["evaluation_set", "target", "model"], as_index=False
    ).agg(
        mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"),
        mean_mae_improvement_vs_train_mean=("mae_improvement_vs_train_mean", "mean"),
        n_runs=("split_name", "nunique"),
    ).sort_values(["evaluation_set", "target", "mean_mae"])
    summary_frame.to_csv(output_dir / "baseline_metrics_summary.csv", index=False)
    pd.DataFrame([
        {"model": model_name, "available": is_available,
         "reason": "not installed; skipped without changing the environment" if not is_available else "available but not selected to avoid unvalidated optional dependency behavior"}
        for model_name, is_available in optional_availability.items()
    ]).to_csv(output_dir / "optional_model_availability.csv", index=False)
    print(f"Completed {len(metric_frame)} baseline metric rows in {output_dir}")
    print("Optional model availability:", optional_availability)


if __name__ == "__main__":
    main()
