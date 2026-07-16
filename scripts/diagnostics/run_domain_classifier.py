#!/usr/bin/env python3
"""Quantify train-feedback covariate shift with leakage-safe domain classifiers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    add_common_arguments, build_feature_frames, discover_schema, load_frames,
    load_mordred_table,
)


def _preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    """Create preprocessing fitted exclusively within each cross-validation fold."""
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


def _calibration_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Report Brier score and an equal-width expected calibration error."""
    bin_edges = np.linspace(0.0, 1.0, 11)
    expected_calibration_error = 0.0
    for lower_edge, upper_edge in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probabilities >= lower_edge) & (
            probabilities < upper_edge if upper_edge < 1.0 else probabilities <= upper_edge
        )
        if mask.any():
            expected_calibration_error += abs(
                labels[mask].mean() - probabilities[mask].mean()
            ) * mask.mean()
    return {
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_10_bins": float(expected_calibration_error),
    }


def _metric_row(model_name: str, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | str]:
    predictions = (probabilities >= 0.5).astype(int)
    calibration = _calibration_metrics(labels, probabilities)
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    specificity = true_negative / max(1, true_negative + false_positive)
    sensitivity = true_positive / max(1, true_positive + false_negative)
    return {
        "model": model_name,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float((specificity + sensitivity) / 2),
        **calibration,
    }


def _top_permutation_importance(
    fitted_pipeline: Pipeline, features: pd.DataFrame, labels: np.ndarray,
    n_features: int, seed: int,
) -> pd.DataFrame:
    """Compute AUC-drop permutation importance for the most promising raw fields."""
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    classifier = fitted_pipeline.named_steps["classifier"]
    transformed_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    order = np.argsort(importances)[::-1]
    raw_features: list[str] = []
    for transformed_name in transformed_names[order]:
        raw_name = transformed_name.split("__", maxsplit=1)[-1]
        # One-hot names include a category suffix. Restricting to the numerical
        # prefix is intentional; categorical indicators are still listed below.
        matching_columns = [
            column for column in features.columns
            if raw_name == column or raw_name.startswith(f"{column}_")
        ]
        if matching_columns and matching_columns[0] not in raw_features:
            raw_features.append(matching_columns[0])
        if len(raw_features) >= n_features:
            break
    baseline_auc = roc_auc_score(labels, fitted_pipeline.predict_proba(features)[:, 1])
    random_generator = np.random.default_rng(seed)
    records: list[dict[str, float | str]] = []
    for feature_name in raw_features:
        permuted = features.copy()
        permuted[feature_name] = random_generator.permutation(permuted[feature_name].to_numpy())
        permuted_auc = roc_auc_score(labels, fitted_pipeline.predict_proba(permuted)[:, 1])
        records.append({
            "model": "ExtraTreesClassifier", "feature": feature_name,
            "importance_type": "permutation_auc_drop",
            "importance": float(baseline_auc - permuted_auc),
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-mordred-features", type=int, default=256)
    parser.add_argument("--permutation-features", type=int, default=40)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = discover_schema(arguments.train_csv, arguments.feedback_csv)
    train_frame, feedback_frame = load_frames(schema)
    mordred_frame = load_mordred_table(schema)
    train_numeric, train_categorical = build_feature_frames(
        train_frame, schema, mordred_frame, arguments.max_mordred_features
    )
    feedback_numeric, feedback_categorical = build_feature_frames(
        feedback_frame, schema, mordred_frame, arguments.max_mordred_features
    )
    numeric_features = pd.concat([train_numeric, feedback_numeric], ignore_index=True)
    categorical_features = pd.concat([train_categorical, feedback_categorical], ignore_index=True)
    feature_frame = pd.concat([numeric_features, categorical_features], axis=1)
    labels = np.concatenate([np.zeros(len(train_frame), dtype=int), np.ones(len(feedback_frame), dtype=int)])
    sample_ids = pd.concat([
        train_frame[["diagnostic_sample_id"]], feedback_frame[["diagnostic_sample_id"]],
    ], ignore_index=True)
    domains = np.where(labels == 1, "feedback", "train")

    preprocessor = _preprocessor(list(numeric_features.columns), list(categorical_features.columns))
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="liblinear", random_state=arguments.seed,
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=arguments.seed, n_jobs=arguments.n_jobs,
        ),
    }
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=arguments.seed)
    metrics: list[dict[str, float | str]] = []
    prediction_records: list[pd.DataFrame] = []
    fitted_pipelines: dict[str, Pipeline] = {}
    for model_name, estimator in models.items():
        pipeline = Pipeline([("preprocess", clone(preprocessor)), ("classifier", estimator)])
        probabilities = cross_val_predict(
            pipeline, feature_frame, labels, cv=splitter, method="predict_proba",
            n_jobs=1,
        )[:, 1]
        metrics.append(_metric_row(model_name, labels, probabilities))
        prediction_records.append(pd.DataFrame({
            "diagnostic_sample_id": sample_ids["diagnostic_sample_id"],
            "domain": domains,
            "domain_label": labels,
            "model": model_name,
            "oof_domain_probability": probabilities,
        }))
        fitted_pipelines[model_name] = pipeline.fit(feature_frame, labels)

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output_dir / "domain_classifier_metrics.csv", index=False)
    prediction_frame = pd.concat(prediction_records, ignore_index=True)
    prediction_frame.to_csv(output_dir / "domain_classifier_predictions.csv", index=False)

    extra_tree_pipeline = fitted_pipelines["ExtraTreesClassifier"]
    extra_tree = extra_tree_pipeline.named_steps["classifier"]
    transformed_names = extra_tree_pipeline.named_steps["preprocess"].get_feature_names_out()
    feature_importance_frame = pd.DataFrame({
        "model": "ExtraTreesClassifier",
        "feature": transformed_names,
        "importance_type": "extra_trees_impurity",
        "importance": extra_tree.feature_importances_,
    }).sort_values("importance", ascending=False)
    permutation_frame = _top_permutation_importance(
        extra_tree_pipeline, feature_frame, labels, arguments.permutation_features, arguments.seed,
    )
    feature_importance_frame = pd.concat(
        [feature_importance_frame, permutation_frame], ignore_index=True
    )
    feature_importance_frame.to_csv(output_dir / "domain_feature_importance.csv", index=False)

    feedback_features = feature_frame.iloc[len(train_frame):]
    ood_scores = pd.DataFrame({
        "diagnostic_sample_id": feedback_frame["diagnostic_sample_id"],
        "ID": feedback_frame[schema.id_column].astype(str),
        "extra_trees_domain_probability": extra_tree_pipeline.predict_proba(feedback_features)[:, 1],
        "logistic_domain_probability": fitted_pipelines["LogisticRegression"].predict_proba(feedback_features)[:, 1],
    })
    ood_scores["ood_score"] = ood_scores["extra_trees_domain_probability"]
    ood_scores.to_csv(output_dir / "feedback_ood_scores.csv", index=False)

    print("Domain classifier cross-validated metrics:")
    print(metric_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
