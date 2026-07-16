#!/usr/bin/env python3
"""Prepare a reproducible tabular-baseline versus standard-GraphGPS benchmark.

The script fixes one 80/10/10 random split of the supplied five-component
formulation data, evaluates leakage-safe tabular baselines, and materializes a
GraphGPS training YAML that consumes the exact same split manifest.  Run the
GraphGPS command printed by this script, export its held-out predictions with
``stage3_export_predictions.py``, then re-run this script with
``--graphgps-predictions`` to produce the final comparison table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import TARGET_COLUMNS, add_normalized_keys, discover_schema, metric_dict  # noqa: E402
from stable_formulation import build_stable_feature_sets  # noqa: E402


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a result input without loading it at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_dataset(path: Path) -> tuple[pd.DataFrame, str]:
    """Read an Excel-exported CSV without mutating the user-supplied source."""
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1,
                             f"Unable to decode {path} as UTF-8 or GB18030")


def make_pipeline(features: pd.DataFrame, estimator: object) -> Pipeline:
    """Keep all imputation and categorical encoding inside the training fold."""
    numeric_columns = features.select_dtypes(exclude="object").columns.tolist()
    categorical_columns = features.select_dtypes(include="object").columns.tolist()
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_columns))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers, sparse_threshold=0.2)),
        ("model", estimator),
    ])


def split_manifest(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Build one exact 80/10/10 split with immutable original row indices."""
    original_indices = np.arange(len(frame), dtype=int)
    train_val, test = train_test_split(
        original_indices, test_size=0.10, random_state=seed, shuffle=True,
    )
    train, val = train_test_split(
        train_val, test_size=len(test) / len(train_val), random_state=seed,
        shuffle=True,
    )
    parts: list[pd.DataFrame] = []
    for split_name, indices in (("train", train), ("val", val), ("test", test)):
        subset = pd.DataFrame({
            "sample_id": frame.iloc[indices]["ID"].astype(str).to_numpy(),
            "split": split_name,
            "original_row_index": indices,
            "split_order": np.arange(len(indices), dtype=int),
        })
        parts.append(subset)
    manifest = pd.concat(parts, ignore_index=True)
    if manifest["sample_id"].duplicated().any() or len(manifest) != len(frame):
        raise ValueError("The fixed split does not map one-to-one to dataset rows.")
    return manifest


def split_indices(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return original source row positions for every named data partition."""
    return {
        split_name: manifest.loc[manifest["split"] == split_name,
                                "original_row_index"].to_numpy(dtype=int)
        for split_name in ("train", "val", "test")
    }


def run_tabular_baselines(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    feature_sets: dict[str, pd.DataFrame],
    output_dir: Path,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    """Select tree feature sets on validation only and score untouched test rows."""
    indices = split_indices(manifest)
    candidates = {
        "EE_before": ("F2_identity_ratio", "F3_physchem_weighted"),
        "EE_after": ("F2_identity_ratio", "F4_physchem_interactions"),
        "Aerosolization_Efficiency": ("F2_identity_ratio",),
        "mRNA_Recovery_Efficiency": ("F2_identity_ratio",),
    }
    models = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.8,
            random_state=seed, n_jobs=n_jobs,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.7,
            random_state=seed, n_jobs=n_jobs,
        ),
    }
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        y_train = frame.iloc[indices["train"]][target].astype(float)
        y_val = frame.iloc[indices["val"]][target].astype(float)
        y_test = frame.iloc[indices["test"]][target].astype(float)
        test_ids = frame.iloc[indices["test"]]["ID"].astype(str).to_numpy()
        train_mean_prediction = np.full(len(y_test), y_train.mean())
        mean_rows = pd.DataFrame({
            "sample_id": test_ids, "split": "test", "target": target,
            "model": "TrainMean", "feature_set": "none", "y_true": y_test.to_numpy(),
            "y_pred": train_mean_prediction,
        })
        mean_rows["absolute_error"] = (mean_rows["y_true"] - mean_rows["y_pred"]).abs()
        prediction_rows.append(mean_rows)
        metric_rows.append({
            "split": "test", "target": target, "model": "TrainMean",
            "feature_set": "none", "validation_mae": np.nan, "n": len(mean_rows),
            **metric_dict(mean_rows["y_true"], mean_rows["y_pred"]),
        })
        for model_name, estimator in models.items():
            validation_scores: list[tuple[str, float]] = []
            for feature_name in candidates[target]:
                features = feature_sets[feature_name]
                fitted = make_pipeline(features, clone(estimator)).fit(
                    features.iloc[indices["train"]], y_train,
                )
                prediction = fitted.predict(features.iloc[indices["val"]])
                validation_scores.append((feature_name, metric_dict(y_val, prediction)["mae"]))
            feature_name, validation_mae = min(validation_scores, key=lambda value: value[1])
            features = feature_sets[feature_name]
            fitted = make_pipeline(features, clone(estimator)).fit(
                features.iloc[indices["train"]], y_train,
            )
            prediction = fitted.predict(features.iloc[indices["test"]])
            rows = pd.DataFrame({
                "sample_id": test_ids, "split": "test", "target": target,
                "model": model_name, "feature_set": feature_name,
                "y_true": y_test.to_numpy(), "y_pred": prediction,
            })
            rows["absolute_error"] = (rows["y_true"] - rows["y_pred"]).abs()
            prediction_rows.append(rows)
            metric_rows.append({
                "split": "test", "target": target, "model": model_name,
                "feature_set": feature_name, "validation_mae": validation_mae,
                "n": len(rows), **metric_dict(rows["y_true"], rows["y_pred"]),
            })
            selection_rows.append({
                "target": target, "model": model_name, "selected_feature_set": feature_name,
                "validation_mae": validation_mae,
                "candidates": "|".join(
                    f"{candidate}:{score:.8f}" for candidate, score in validation_scores
                ),
            })
    metrics = pd.DataFrame(metric_rows).sort_values(["target", "model"])
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics.to_csv(output_dir / "tabular_baseline_test_metrics.csv", index=False)
    predictions.to_csv(output_dir / "tabular_baseline_test_predictions.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output_dir / "tabular_feature_selection.csv", index=False)
    return metrics


def write_graphgps_config(
    output_dir: Path, dataset_path: Path, manifest_path: Path, seed: int, max_epochs: int,
    config_stem: str = "graphgps_standard", training_dir_name: str = "graphgps_training",
    cache_tag: str | None = None,
) -> Path:
    """Create the non-augmented GPU GraphGPS configuration for the shared split."""
    with (ROOT / "configs/GPS/direct_train.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["out_dir"] = str((output_dir / training_dir_name).resolve())
    config.update({
        "accelerator": "cuda", "devices": 1, "gpu_serial": 0, "num_workers": 0,
        "seed": seed, "read_csv": str(dataset_path.resolve()),
        "fifth_component_delta_weight": 1.0,
        "use_component_aux_features": False,
        "use_mordred_features": False,
        "coarse_grain_enable": False,
    })
    config["dataset"] = dict(config["dataset"])
    config["dataset"].update({
        "dir": str((ROOT / "datasets_lrx").resolve()),
        "diagnostic_split_path": str(manifest_path.resolve()),
        "diagnostic_id_column": "ID",
        "diagnostic_manifest_id_column": "sample_id",
        "cache_per_run": True,
        "cache_refresh": True,
        "cache_tag": cache_tag or f"new_dataset_benchmark_seed_{seed}",
    })
    config["optim"] = dict(config["optim"])
    config["optim"].update({
        "max_epoch": max_epochs,
        "num_warmup_epochs": min(50, max(1, max_epochs // 5)),
    })
    config["train"] = dict(config["train"])
    config["train"].update({
        "deterministic": True,
        "manifest_path": str(manifest_path.resolve()),
        "protocol": "new_dataset_random_80_10_10",
        "fold": "holdout",
        "early_stop_patience": 50,
    })
    path = output_dir / f"{config_stem}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def append_graphgps_comparison(
    output_dir: Path, baseline_metrics: pd.DataFrame, graphgps_path: Path,
) -> None:
    """Standardize GraphGPS exporter output and append it to the result table."""
    predictions = pd.read_csv(graphgps_path, dtype={"sample_id": str})
    expected = {"sample_id", "split", "target", "y_true", "y_pred"}
    if not expected.issubset(predictions.columns):
        raise ValueError(f"GraphGPS predictions lack required columns: {sorted(expected - set(predictions))}")
    test_predictions = predictions.loc[predictions["split"] == "test"].copy()
    if test_predictions.duplicated(["sample_id", "target"]).any():
        raise ValueError("GraphGPS test predictions contain duplicate sample/target pairs.")
    graphgps_rows: list[dict[str, object]] = []
    for target, group in test_predictions.groupby("target", sort=True):
        graphgps_rows.append({
            "split": "test", "target": target, "model": "GraphGPS_standard",
            "feature_set": "molecular_graphs_and_component_ratios",
            "validation_mae": np.nan, "n": len(group),
            **metric_dict(group["y_true"], group["y_pred"]),
        })
    graphgps_metrics = pd.DataFrame(graphgps_rows)
    graphgps_metrics.to_csv(output_dir / "graphgps_standard_test_metrics.csv", index=False)
    comparison = pd.concat([baseline_metrics, graphgps_metrics], ignore_index=True)
    comparison = comparison.sort_values(["target", "mae", "model"])
    comparison.to_csv(
        output_dir / "model_comparison_test_metrics.csv", index=False,
    )
    summary = comparison.groupby("model", as_index=False).agg(
        targets=("target", "nunique"), n_per_target=("n", "first"),
        mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"),
        mean_median_absolute_error=("median_absolute_error", "mean"),
        mean_pearson=("pearson", "mean"), mean_spearman=("spearman", "mean"),
    ).sort_values("mean_mae")
    summary.to_csv(output_dir / "model_comparison_test_summary.csv", index=False)


def _validate_regression_frame(frame: pd.DataFrame, label: str) -> None:
    """Ensure a frame contains the inputs and labels needed for this benchmark."""
    required = ["ID", *TARGET_COLUMNS, "mol%_IL", "mol%_HL", "mol%_Chol", "mol%_PEG", "mol%_Fifth"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    if frame["ID"].isna().any() or frame["ID"].astype(str).duplicated().any():
        raise ValueError(f"{label} requires a non-null, unique ID per row.")
    if frame[TARGET_COLUMNS].isna().any().any():
        raise ValueError(f"{label} requires observed values for all four targets.")
    for column in required[-5:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")


def run_feedback_baselines(
    output_dir: Path, feedback_path: Path, n_jobs: int,
) -> tuple[Path, pd.DataFrame]:
    """Fit the previously selected tabular models on benchmark-train rows only."""
    train_path = output_dir / "input_sanitized_utf8.csv"
    train_frame, _ = read_dataset(train_path)
    feedback_frame, feedback_encoding = read_dataset(feedback_path)
    train_frame = train_frame.copy()
    feedback_frame = feedback_frame.copy()
    _validate_regression_frame(train_frame, "training data")
    _validate_regression_frame(feedback_frame, "feedback data")
    feedback_sanitized_path = output_dir / "feedback_input_sanitized_utf8.csv"
    feedback_frame.to_csv(feedback_sanitized_path, index=False, encoding="utf-8")
    schema = discover_schema(train_csv=train_path, feedback_csv=feedback_sanitized_path)
    add_normalized_keys(train_frame, schema)
    add_normalized_keys(feedback_frame, schema)
    train_features, _, _ = build_stable_feature_sets(train_frame, schema)
    feedback_features, _, _ = build_stable_feature_sets(feedback_frame, schema)
    manifest = pd.read_csv(output_dir / "split_manifest.csv", dtype={"sample_id": str})
    indices = split_indices(manifest)
    selection = pd.read_csv(output_dir / "tabular_feature_selection.csv")
    models = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.8,
            random_state=42, n_jobs=n_jobs,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.7,
            random_state=42, n_jobs=n_jobs,
        ),
    }
    rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        y_train = train_frame.iloc[indices["train"]][target].astype(float)
        y_feedback = feedback_frame[target].astype(float)
        mean_prediction = np.full(len(feedback_frame), y_train.mean())
        mean_rows = pd.DataFrame({
            "sample_id": feedback_frame["ID"].astype(str), "evaluation_set": "feedback",
            "target": target, "model": "TrainMean", "feature_set": "none",
            "y_true": y_feedback, "y_pred": mean_prediction,
        })
        mean_rows["absolute_error"] = (mean_rows["y_true"] - mean_rows["y_pred"]).abs()
        rows.append(mean_rows)
        metric_rows.append({
            "evaluation_set": "feedback", "target": target, "model": "TrainMean",
            "feature_set": "none", "n": len(mean_rows),
            **metric_dict(mean_rows["y_true"], mean_rows["y_pred"]),
        })
        for model_name, estimator in models.items():
            selected = selection.loc[(selection["target"] == target) &
                                     (selection["model"] == model_name), "selected_feature_set"]
            if len(selected) != 1:
                raise ValueError(f"Cannot find one selected feature set for {target}/{model_name}.")
            feature_name = selected.iloc[0]
            fitted = make_pipeline(train_features[feature_name], clone(estimator)).fit(
                train_features[feature_name].iloc[indices["train"]], y_train,
            )
            prediction = fitted.predict(feedback_features[feature_name])
            model_rows = pd.DataFrame({
                "sample_id": feedback_frame["ID"].astype(str), "evaluation_set": "feedback",
                "target": target, "model": model_name, "feature_set": feature_name,
                "y_true": y_feedback, "y_pred": prediction,
            })
            model_rows["absolute_error"] = (model_rows["y_true"] - model_rows["y_pred"]).abs()
            rows.append(model_rows)
            metric_rows.append({
                "evaluation_set": "feedback", "target": target, "model": model_name,
                "feature_set": feature_name, "n": len(model_rows),
                **metric_dict(model_rows["y_true"], model_rows["y_pred"]),
            })
    metrics = pd.DataFrame(metric_rows).sort_values(["target", "model"])
    predictions = pd.concat(rows, ignore_index=True)
    metrics.to_csv(output_dir / "tabular_baseline_feedback_metrics.csv", index=False)
    predictions.to_csv(output_dir / "tabular_baseline_feedback_predictions.csv", index=False)
    provenance = {
        "feedback": str(feedback_path.resolve()), "feedback_sha256": sha256_file(feedback_path),
        "feedback_encoding": feedback_encoding, "feedback_rows": int(len(feedback_frame)),
        "fitting_rows": int(len(indices["train"])), "fitting_split": "benchmark fixed train partition",
        "id_overlap_with_training": int(len(set(train_frame["ID"].astype(str)) &
                                             set(feedback_frame["ID"].astype(str)))),
    }
    (output_dir / "feedback_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return feedback_sanitized_path, metrics


def write_feedback_prediction_config(output_dir: Path, feedback_path: Path) -> Path:
    """Expose the seed-42 training checkpoint as a one-member prediction ensemble."""
    checkpoint_root = output_dir / "graphgps_training" / "graphgps_standard"
    checkpoint_run = checkpoint_root / "42"
    if not (checkpoint_run / "ckpt").is_dir():
        raise FileNotFoundError(f"Missing GraphGPS checkpoint directory: {checkpoint_run / 'ckpt'}")
    prediction_view = output_dir / "feedback_graphgps_checkpoint_view"
    prediction_view.mkdir(exist_ok=True)
    seed_link = prediction_view / "0"
    if seed_link.exists() or seed_link.is_symlink():
        if not seed_link.is_symlink() or seed_link.resolve() != checkpoint_run.resolve():
            raise RuntimeError(f"Unexpected existing checkpoint view: {seed_link}")
    else:
        seed_link.symlink_to(checkpoint_run.resolve(), target_is_directory=True)
    with (ROOT / "configs/GPS/gps_predict.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.update({
        "accelerator": "cuda", "devices": 1, "gpu_serial": 0, "num_workers": 0,
        "seed": 0, "read_csv": str(feedback_path.resolve()), "property_num": 4,
        "fifth_component_delta_weight": 1.0, "use_component_aux_features": False,
        "use_mordred_features": False, "coarse_grain_enable": False, "result_out": False,
    })
    config["dataset"] = dict(config["dataset"])
    config["dataset"].update({
        "dir": str((ROOT / "datasets_lrx").resolve()), "cache_per_run": True,
        "cache_refresh": True, "cache_tag": "new_dataset_feedback_predict_seed_42",
    })
    config["pretrained"] = dict(config["pretrained"])
    config["pretrained"].update({
        "dir": str(prediction_view.resolve()), "freeze_main": False,
        "reset_prediction_head": False,
    })
    path = output_dir / "feedback_graphgps_predict.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def append_feedback_graphgps_comparison(
    output_dir: Path, graphgps_path: Path,
) -> None:
    """Align ordered GraphGPS feedback output to IDs and calculate external metrics."""
    feedback_frame, _ = read_dataset(output_dir / "feedback_input_sanitized_utf8.csv")
    prediction_frame = pd.read_csv(graphgps_path)
    if len(prediction_frame) != len(feedback_frame):
        raise ValueError(f"Feedback GraphGPS row mismatch: {len(prediction_frame)} != {len(feedback_frame)}")
    column_map = {
        "EE_before": ("true_EE_before", "pred_EE_before_average"),
        "EE_after": ("true_EE_after", "pred_EE_after_average"),
        "Aerosolization_Efficiency": ("true_Aero_Efficiency", "pred_Aero_Efficiency_average"),
        "mRNA_Recovery_Efficiency": ("true_Recovery_Efficiency", "pred_Recovery_Efficiency_average"),
    }
    rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for target, (true_column, prediction_column) in column_map.items():
        if true_column not in prediction_frame or prediction_column not in prediction_frame:
            raise ValueError(f"GraphGPS feedback output lacks columns for {target}.")
        true_values = prediction_frame[true_column].astype(float)
        expected_values = feedback_frame[target].astype(float)
        # main_predict's ensemble writer intentionally rounds both true and
        # predicted values to two decimals, so permit half a display unit.
        if not np.allclose(true_values, expected_values, atol=0.0051, rtol=0):
            raise ValueError(f"GraphGPS feedback row order/labels do not match source for {target}.")
        values = pd.DataFrame({
            "sample_id": feedback_frame["ID"].astype(str), "evaluation_set": "feedback",
            "target": target, "model": "GraphGPS_standard",
            "feature_set": "molecular_graphs_and_component_ratios",
            "y_true": true_values, "y_pred": prediction_frame[prediction_column].astype(float),
        })
        values["absolute_error"] = (values["y_true"] - values["y_pred"]).abs()
        rows.append(values)
        metric_rows.append({
            "evaluation_set": "feedback", "target": target, "model": "GraphGPS_standard",
            "feature_set": "molecular_graphs_and_component_ratios", "n": len(values),
            **metric_dict(values["y_true"], values["y_pred"]),
        })
    graphgps_predictions = pd.concat(rows, ignore_index=True)
    graphgps_metrics = pd.DataFrame(metric_rows)
    graphgps_predictions.to_csv(output_dir / "graphgps_standard_feedback_predictions.csv", index=False)
    graphgps_metrics.to_csv(output_dir / "graphgps_standard_feedback_metrics.csv", index=False)
    baseline_metrics = pd.read_csv(output_dir / "tabular_baseline_feedback_metrics.csv")
    comparison = pd.concat([baseline_metrics, graphgps_metrics], ignore_index=True)
    comparison = comparison.sort_values(["target", "mae", "model"])
    comparison.to_csv(output_dir / "model_comparison_feedback_metrics.csv", index=False)
    summary = comparison.groupby("model", as_index=False).agg(
        targets=("target", "nunique"), n_per_target=("n", "first"),
        mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"),
        mean_median_absolute_error=("median_absolute_error", "mean"),
        mean_pearson=("pearson", "mean"), mean_spearman=("spearman", "mean"),
    ).sort_values("mean_mae")
    summary.to_csv(output_dir / "model_comparison_feedback_summary.csv", index=False)


def write_onehot_fifth_configs(output_dir: Path, max_epochs: int) -> tuple[Path, Path]:
    """Materialize the one-hot-first-four / GraphGPS-fifth feedback workflow."""
    train_csv = output_dir / "input_sanitized_utf8.csv"
    feedback_csv = output_dir / "feedback_input_sanitized_utf8.csv"
    manifest = output_dir / "split_manifest.csv"
    for path in (train_csv, feedback_csv, manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Missing benchmark prerequisite: {path}")
    with (ROOT / "configs/GPS/onehot_fifth_mordred32_train.yaml").open(encoding="utf-8") as handle:
        train_config = yaml.safe_load(handle)
    training_root = output_dir / "onehot_fifth_graphgps_training"
    train_config.update({
        "out_dir": str(training_root.resolve()), "accelerator": "cuda", "devices": 1,
        "gpu_serial": 0, "num_workers": 0, "seed": 0, "read_csv": str(train_csv.resolve()),
        "use_mordred_features": False, "mordred_fifth_only": False,
    })
    train_config["dataset"] = dict(train_config["dataset"])
    train_config["dataset"].update({
        "dir": str((ROOT / "datasets_lrx").resolve()),
        "diagnostic_split_path": str(manifest.resolve()), "diagnostic_id_column": "ID",
        "diagnostic_manifest_id_column": "sample_id", "cache_per_run": True,
        "cache_refresh": True, "cache_tag": "onehot_fifth_graphgps_repeat1",
    })
    train_config["optim"] = dict(train_config["optim"])
    train_config["optim"].update({"max_epoch": max_epochs,
                                  "num_warmup_epochs": min(30, max(1, max_epochs // 5))})
    train_config["train"] = dict(train_config["train"])
    train_config["train"].update({"deterministic": True, "early_stop_patience": 50,
                                   "manifest_path": str(manifest.resolve()),
                                   "protocol": "onehot_first4_graphgps_fifth", "fold": "holdout"})
    train_path = output_dir / "onehot_fifth_graphgps_train.yaml"
    train_path.write_text(yaml.safe_dump(train_config, sort_keys=False), encoding="utf-8")

    with (ROOT / "configs/GPS/onehot_fifth_mordred32_predict.yaml").open(encoding="utf-8") as handle:
        predict_config = yaml.safe_load(handle)
    predict_config.update({
        "accelerator": "cuda", "devices": 1, "gpu_serial": 0, "num_workers": 0,
        "seed": 0, "read_csv": str(feedback_csv.resolve()), "property_num": 4,
        "use_mordred_features": False, "mordred_fifth_only": False, "result_out": False,
    })
    predict_config["dataset"] = dict(predict_config["dataset"])
    predict_config["dataset"].update({
        "dir": str((ROOT / "datasets_lrx").resolve()), "cache_per_run": True,
        "cache_refresh": True, "cache_tag": "onehot_fifth_graphgps_feedback_predict",
    })
    predict_config["pretrained"] = dict(predict_config["pretrained"])
    predict_config["pretrained"].update({
        "dir": str((training_root / train_path.stem).resolve()), "freeze_main": False,
        "reset_prediction_head": False,
    })
    predict_path = output_dir / "onehot_fifth_graphgps_feedback_predict.yaml"
    predict_path.write_text(yaml.safe_dump(predict_config, sort_keys=False), encoding="utf-8")
    return train_path, predict_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "datasets_lrx/raw/input/20260703_sum.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/new_dataset_benchmark_20260713")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--graphgps-predictions", type=Path, default=None)
    parser.add_argument("--summarize-only", action="store_true",
                        help="Merge an existing GraphGPS export with saved baseline metrics.")
    parser.add_argument("--feedback-csv", type=Path, default=None,
                        help="Evaluate saved benchmark models on this external feedback CSV.")
    parser.add_argument("--feedback-graphgps-predictions", type=Path, default=None,
                        help="main_predict.py predicted_average_6props.csv for the feedback CSV.")
    parser.add_argument("--feedback-summarize-only", action="store_true",
                        help="Merge existing feedback GraphGPS output without refitting tree baselines.")
    parser.add_argument("--prepare-repeat10", action="store_true",
                        help="Write a separate seed-0-to-9 GraphGPS training config only.")
    parser.add_argument("--prepare-onehot-fifth", action="store_true",
                        help="Write repeat-1 one-hot-first-four / GraphGPS-fifth configs.")
    arguments = parser.parse_args()

    dataset_path = arguments.dataset.resolve()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if arguments.feedback_summarize_only:
        if arguments.feedback_graphgps_predictions is None:
            raise ValueError("--feedback-summarize-only requires --feedback-graphgps-predictions.")
        append_feedback_graphgps_comparison(
            output_dir, arguments.feedback_graphgps_predictions.resolve(),
        )
        print(f"Wrote feedback comparison metrics in {output_dir}")
        return
    if arguments.prepare_repeat10:
        dataset_for_training = output_dir / "input_sanitized_utf8.csv"
        manifest_path = output_dir / "split_manifest.csv"
        if not dataset_for_training.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                "Missing benchmark input or manifest. Run the benchmark preparation first."
            )
        config_path = write_graphgps_config(
            output_dir, dataset_for_training, manifest_path, seed=0,
            max_epochs=arguments.max_epochs,
            config_stem="graphgps_standard_repeat10",
            training_dir_name="graphgps_repeat10_training",
            cache_tag="new_dataset_benchmark_repeat10",
        )
        print(f"Prepared repeat-10 GraphGPS config: {config_path}")
        print(f"  {sys.executable} main.py --cfg {config_path} --repeat 10")
        return
    if arguments.prepare_onehot_fifth:
        train_path, predict_path = write_onehot_fifth_configs(output_dir, arguments.max_epochs)
        print(f"Prepared one-hot fifth-GraphGPS configs in {output_dir}")
        print(f"  {sys.executable} main.py --cfg {train_path} --repeat 1")
        print(f"  {sys.executable} main_predict.py --cfg {predict_path} --repeat 1")
        return
    if arguments.feedback_csv is not None:
        feedback_path, _ = run_feedback_baselines(
            output_dir, arguments.feedback_csv.resolve(), arguments.n_jobs,
        )
        config_path = write_feedback_prediction_config(output_dir, feedback_path)
        print(f"Prepared feedback benchmark in {output_dir}")
        print("Run GraphGPS feedback prediction:")
        print(f"  {sys.executable} main_predict.py --cfg {config_path} --repeat 1")
        return
    if arguments.summarize_only:
        if arguments.graphgps_predictions is None:
            raise ValueError("--summarize-only requires --graphgps-predictions.")
        baseline_metrics = pd.read_csv(output_dir / "tabular_baseline_test_metrics.csv")
        append_graphgps_comparison(output_dir, baseline_metrics,
                                   arguments.graphgps_predictions.resolve())
        print(f"Wrote comparison metrics in {output_dir}")
        return
    frame, source_encoding = read_dataset(dataset_path)
    frame = frame.copy()
    _validate_regression_frame(frame, "Dataset")

    graphgps_dataset_path = output_dir / "input_sanitized_utf8.csv"
    frame.to_csv(graphgps_dataset_path, index=False, encoding="utf-8")
    schema = discover_schema(train_csv=graphgps_dataset_path,
                             feedback_csv=graphgps_dataset_path)
    add_normalized_keys(frame, schema)
    manifest = split_manifest(frame, arguments.seed)
    manifest_path = output_dir / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    feature_sets, _, feature_schema = build_stable_feature_sets(frame, schema)
    baseline_metrics = run_tabular_baselines(
        frame, manifest, feature_sets, output_dir, arguments.seed, arguments.n_jobs,
    )
    config_path = write_graphgps_config(
        output_dir, graphgps_dataset_path, manifest_path, arguments.seed, arguments.max_epochs,
    )
    split_counts = manifest["split"].value_counts().to_dict()
    provenance = {
        "dataset": str(dataset_path), "dataset_sha256": sha256_file(dataset_path),
        "source_encoding": source_encoding,
        "graphgps_input": str(graphgps_dataset_path),
        "graphgps_input_sha256": sha256_file(graphgps_dataset_path),
        "row_count": int(len(frame)), "target_columns": TARGET_COLUMNS,
        "split_counts": split_counts, "seed": arguments.seed,
        "graphgps_definition": "GPSDoubleModel_multi4_cat_v0 without Mordred, auxiliary, or coarse-grain features",
        "graphgps_config": str(config_path), "feature_schema": feature_schema,
    }
    (output_dir / "benchmark_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    if arguments.graphgps_predictions:
        append_graphgps_comparison(output_dir, baseline_metrics, arguments.graphgps_predictions.resolve())
    print(f"Prepared benchmark in {output_dir}")
    print(f"Split counts: {split_counts}")
    print("Run GraphGPS:")
    print(f"  {sys.executable} main.py --cfg {config_path} --repeat 1")


if __name__ == "__main__":
    main()
