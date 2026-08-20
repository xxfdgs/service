#!/usr/bin/env python3
"""Train an input-only, continuous Norm regression ensemble.

Hyperparameters are selected solely by MAE on leave-one-input-series-out
folds.  The targets are modeled as log1p continuous values.  No external
feedback table and no target threshold are read or used by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.build_input_only_o12_residual_tree_head import feature_frame


TARGETS = ("Norm_before", "Norm_after")
MAX_FEATURES = (0.35, 0.4, 0.5, 0.6)
MIN_SAMPLES_LEAF = (20, 30, 40)
FINAL_SEEDS = tuple(range(100, 110))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = [column for column in features if column.endswith("_key")]
    numeric = [column for column in features if column not in categorical]
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", "passthrough", numeric),
    ])


def model(seed: int, max_features: float, min_samples_leaf: int,
          estimators: int, jobs: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=seed,
        n_jobs=jobs,
    )


def predict_regression(estimator: RandomForestRegressor, features: np.ndarray) -> np.ndarray:
    return np.maximum(np.expm1(estimator.predict(features)), 0.0)


def select_hyperparameters(
    frame: pd.DataFrame,
    transformed: np.ndarray,
    series: pd.Series,
    estimators: int,
    jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    ordered_series = sorted(series.unique(), key=lambda value: int(value))
    for target in TARGETS:
        truth = frame[target].to_numpy(float)
        for max_features in MAX_FEATURES:
            for min_samples_leaf in MIN_SAMPLES_LEAF:
                for fold, held_out_series in enumerate(ordered_series):
                    train_indices = np.flatnonzero(series.ne(held_out_series).to_numpy())
                    validation_indices = np.flatnonzero(series.eq(held_out_series).to_numpy())
                    estimator = model(
                        1000 + fold,
                        max_features,
                        min_samples_leaf,
                        estimators,
                        jobs,
                    )
                    estimator.fit(transformed[train_indices], np.log1p(truth[train_indices]))
                    prediction = predict_regression(estimator, transformed[validation_indices])
                    absolute_error = np.abs(truth[validation_indices] - prediction)
                    rows.append({
                        "target": target,
                        "max_features": max_features,
                        "min_samples_leaf": min_samples_leaf,
                        "held_out_input_series": held_out_series,
                        "n": len(validation_indices),
                        "absolute_error_sum": float(absolute_error.sum()),
                        "mae": float(absolute_error.mean()),
                    })
    by_fold = pd.DataFrame(rows)
    summary = (
        by_fold.groupby(["target", "max_features", "min_samples_leaf"], as_index=False)
        .agg(
            folds=("held_out_input_series", "nunique"),
            n=("n", "sum"),
            absolute_error_sum=("absolute_error_sum", "sum"),
            mean_fold_mae=("mae", "mean"),
            std_fold_mae=("mae", "std"),
        )
    )
    summary["pooled_mae"] = summary.absolute_error_sum / summary.n
    summary["selected"] = False
    for target in TARGETS:
        candidates = summary.loc[summary.target.eq(target)].sort_values(
            ["pooled_mae", "mean_fold_mae", "max_features", "min_samples_leaf"]
        )
        summary.loc[candidates.index[0], "selected"] = True
    return by_fold, summary


def evaluate_fixed_input_splits(
    frame: pd.DataFrame,
    transformed: np.ndarray,
    selected: dict[str, dict[str, float | int]],
    manifests: Path,
    estimators: int,
    jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for split_seed in FINAL_SEEDS:
        manifest_path = manifests / f"split_manifest_seed{split_seed}.csv"
        manifest = pd.read_csv(manifest_path)
        train_indices = manifest.loc[
            manifest.split.eq("train"), "original_row_index"].to_numpy(int)
        for target in TARGETS:
            settings = selected[target]
            truth = frame[target].to_numpy(float)
            estimator = model(
                split_seed,
                float(settings["max_features"]),
                int(settings["min_samples_leaf"]),
                estimators,
                jobs,
            )
            estimator.fit(transformed[train_indices], np.log1p(truth[train_indices]))
            for split in ("val", "test"):
                indices = manifest.loc[
                    manifest.split.eq(split), "original_row_index"].to_numpy(int)
                prediction = predict_regression(estimator, transformed[indices])
                metric_rows.append({
                    "split_seed": split_seed,
                    "split": split,
                    "target": target,
                    "n": len(indices),
                    "mae": float(mean_absolute_error(truth[indices], prediction)),
                    "rmse": float(mean_squared_error(truth[indices], prediction) ** 0.5),
                    "r2": float(r2_score(truth[indices], prediction)),
                })
                prediction_rows.extend({
                    "split_seed": split_seed,
                    "source_index": int(index),
                    "sample_id": str(frame.iloc[index].ID),
                    "split": split,
                    "target": target,
                    "y_true": float(truth[index]),
                    "y_pred": float(value),
                } for index, value in zip(indices, prediction))
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-estimators", type=int, default=100)
    parser.add_argument("--evaluation-estimators", type=int, default=300)
    parser.add_argument("--final-estimators", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()

    input_csv = args.input_csv.resolve()
    manifests = args.manifests.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_csv, dtype={"ID": str})
    required = {"ID", "Norm_before", "Norm_after"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Input CSV lacks required columns: {sorted(missing)}")
    if frame.ID.duplicated().any() or frame[list(TARGETS)].isna().any().any():
        raise ValueError("Input IDs must be unique and Norm targets must be finite.")
    series = frame.ID.str.split("-", n=1).str[0]
    if series.nunique() < 3 or not series.str.fullmatch(r"\d+").all():
        raise ValueError("Input IDs must begin with a numeric series followed by '-'.")

    features = feature_frame(frame)
    preprocessor = build_preprocessor(features)
    transformed = preprocessor.fit_transform(features)
    cv_folds, cv_summary = select_hyperparameters(
        frame, transformed, series, args.selection_estimators, args.jobs)
    cv_folds.to_csv(output / "series_cv_metrics_by_fold.csv", index=False)
    cv_summary.to_csv(output / "series_cv_candidate_summary.csv", index=False)
    chosen = cv_summary.loc[cv_summary.selected].copy()
    if set(chosen.target) != set(TARGETS) or len(chosen) != len(TARGETS):
        raise RuntimeError("Expected exactly one selected configuration per target.")
    selected = {
        row.target: {
            "max_features": float(row.max_features),
            "min_samples_leaf": int(row.min_samples_leaf),
            "selection_pooled_mae": float(row.pooled_mae),
        }
        for row in chosen.itertuples(index=False)
    }

    split_metrics, split_predictions = evaluate_fixed_input_splits(
        frame, transformed, selected, manifests, args.evaluation_estimators, args.jobs)
    split_metrics.to_csv(output / "fixed_split_metrics.csv", index=False)
    split_predictions.to_csv(output / "fixed_split_predictions.csv", index=False)
    (
        split_metrics.groupby(["split", "target"], as_index=False)
        .agg(
            completed_seeds=("split_seed", "nunique"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            mean_r2=("r2", "mean"),
        )
        .to_csv(output / "fixed_split_metrics_summary.csv", index=False)
    )

    ensemble: dict[str, list[RandomForestRegressor]] = {}
    for target in TARGETS:
        truth = frame[target].to_numpy(float)
        settings = selected[target]
        ensemble[target] = []
        for seed in FINAL_SEEDS:
            estimator = model(
                seed,
                float(settings["max_features"]),
                int(settings["min_samples_leaf"]),
                args.final_estimators,
                args.jobs,
            )
            estimator.fit(transformed, np.log1p(truth))
            ensemble[target].append(estimator)
    artifact = {
        "preprocessor": preprocessor,
        "models": ensemble,
        "targets": TARGETS,
        "selected": selected,
        "input_columns": frame.columns.tolist(),
        "feature_columns": features.columns.tolist(),
        "target_transform": "log1p",
    }
    joblib.dump(artifact, output / "input_only_norm_log_rf_10seed.joblib", compress=3)
    protocol = {
        "input_only": True,
        "external_validation_read": False,
        "input_csv": str(input_csv),
        "input_sha256": sha256(input_csv),
        "input_rows": len(frame),
        "input_series": sorted(series.unique(), key=int),
        "selection": "pooled continuous MAE over leave-one-input-series-out folds",
        "selection_targets": list(TARGETS),
        "target_transform": "log1p continuous regression",
        "threshold_used_in_training_or_selection": False,
        "candidate_grid": {
            "max_features": list(MAX_FEATURES),
            "min_samples_leaf": list(MIN_SAMPLES_LEAF),
        },
        "selected": selected,
        "final_seeds": list(FINAL_SEEDS),
        "final_estimators_per_seed": args.final_estimators,
        "manifests": str(manifests),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(chosen[[
        "target", "max_features", "min_samples_leaf", "pooled_mae",
        "mean_fold_mae", "std_fold_mae"]].to_string(index=False))
    print()
    print(pd.read_csv(output / "fixed_split_metrics_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
