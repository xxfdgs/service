#!/usr/bin/env python3
"""Use shared explicit GroupKFold manifests for tabular and partial full-budget GraphGPS CV."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import metric_dict  # noqa: E402
from stage2_common import (  # noqa: E402
    add_stage2_arguments, group_cv_manifests, load_manifest_frame, load_training_frame,
    record_execution, stage2_output,
)
from stable_formulation import build_stable_feature_sets  # noqa: E402


TARGET_PREDICTION_COLUMNS = {
    "EE_before": ("true_EE_before", "pred_EE_before_average"),
    "EE_after": ("true_EE_after", "pred_EE_after_average"),
    "Aerosolization_Efficiency": ("true_Aero_Efficiency", "pred_Aero_Efficiency_average"),
    "mRNA_Recovery_Efficiency": ("true_Recovery_Efficiency", "pred_Recovery_Efficiency_average"),
}


def make_pipeline(features: pd.DataFrame, estimator: object) -> Pipeline:
    """Apply all numeric and identity preprocessing inside each fold fit."""
    numeric_columns = features.select_dtypes(exclude="object").columns.tolist()
    categorical_columns = features.select_dtypes(include="object").columns.tolist()
    transformers = []
    if numeric_columns:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric_columns))
    if categorical_columns:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_columns))
    return Pipeline([("preprocess", ColumnTransformer(transformers, sparse_threshold=0.2)),
                     ("model", estimator)])


def models(seed: int, n_jobs: int) -> dict[str, object]:
    return {
        "Ridge": Ridge(alpha=1.0),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                            max_features=0.8, random_state=seed, n_jobs=n_jobs),
        "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                                max_features=0.7, random_state=seed, n_jobs=n_jobs),
    }


def bootstrap_ci(values: pd.Series, seed: int) -> tuple[float, float]:
    values_array = values.dropna().to_numpy(dtype=float)
    if len(values_array) == 0:
        return np.nan, np.nan
    generator = np.random.default_rng(seed)
    means = [generator.choice(values_array, len(values_array), replace=True).mean() for _ in range(4000)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def graph_train_config(output_dir: Path, protocol: str, fold: str, manifest_path: Path,
                       data_csv: Path | None = None) -> dict:
    """Keep all original GraphGPS training controls, replacing only split handling."""
    config = yaml.safe_load((ROOT / "configs/GPS/direct_train_coarse_noaux.yaml").read_text())
    config["out_dir"] = str(output_dir / "group_cv" / "graphgps_training")
    config.update({"accelerator": "cuda", "devices": 1, "seed": 0,
                   "use_mordred_features": True, "mordred_feature_dim": 11,
                   "mordred_feature_path": str(ROOT / "results/mordred_train_feedback/mordred_selected_features.csv")})
    config["dataset"] = dict(config["dataset"])
    config["dataset"].update({
        "diagnostic_split_path": str(manifest_path.resolve()), "diagnostic_id_column": "ID",
        "diagnostic_manifest_id_column": "sample_id", "cache_per_run": True,
        "cache_refresh": True, "cache_tag": f"stage2_cv_{protocol}_{fold}",
    })
    if data_csv is not None:
        config["read_csv"] = str(data_csv.resolve())
    return config


def graph_predict_config(training_dir: Path, protocol: str, fold: str, seed: int, input_csv: Path) -> dict:
    """Build a per-seed held-out prediction configuration."""
    config = yaml.safe_load((ROOT / "configs/GPS/gps_predict_coarse_noaux.yaml").read_text())
    config.update({"accelerator": "cuda", "devices": 1, "seed": seed,
                   "read_csv": str(input_csv.resolve()), "use_mordred_features": True,
                   "mordred_feature_dim": 11,
                   "mordred_feature_path": str(ROOT / "results/mordred_train_feedback/mordred_selected_features.csv"),
                   "pretrained": {"dir": str(training_dir.resolve()), "freeze_main": False,
                                 "reset_prediction_head": False}})
    config["dataset"] = dict(config["dataset"])
    config["dataset"].update({"cache_per_run": True, "cache_refresh": True,
                              "cache_tag": f"stage2_cv_prediction_{protocol}_{fold}_{seed}"})
    return config


def run_prediction(config_path: Path, repeat: int = 1) -> tuple[pd.DataFrame, Path]:
    """Run project inference and return its ensemble CSV plus timestamped directory."""
    before = {path.resolve() for path in (ROOT / "runs").iterdir() if path.is_dir()}
    subprocess.run([sys.executable, "main_predict.py", "--cfg", str(config_path), "--repeat", str(repeat)],
                   cwd=ROOT, check=True)
    created = [path for path in (ROOT / "runs").iterdir()
               if path.is_dir() and path.resolve() not in before]
    if len(created) != 1:
        raise RuntimeError(f"Expected one new prediction run for {config_path}, found {len(created)}")
    return pd.read_csv(created[0] / "predicted_average_6props.csv"), created[0]


def run_graph_fold(output_dir: Path, protocol: str, manifest_path: Path, train_frame: pd.DataFrame,
                   original_columns: list[str], config_dir: Path, input_dir: Path,
                   data_csv: Path | None = None) -> tuple[list[dict], list[pd.DataFrame]]:
    """Execute three GraphGPS seeds for a single explicit group fold and ensemble them."""
    fold = manifest_path.stem
    train_config_path = config_dir / f"{protocol}_{fold}_graphgps.yaml"
    train_config_path.write_text(yaml.safe_dump(
        graph_train_config(output_dir, protocol, fold, manifest_path, data_csv), sort_keys=False
    ), encoding="utf-8")
    training_dir = output_dir / "group_cv" / "graphgps_training" / f"{protocol}_{fold}_graphgps"
    checkpoint_ready = all((training_dir / str(seed) / "ckpt").is_dir() and
                           any((training_dir / str(seed) / "ckpt").glob("*.ckpt"))
                           for seed in (0, 1, 2))
    if not checkpoint_ready:
        subprocess.run([sys.executable, "main.py", "--cfg", str(train_config_path), "--repeat", "3"], cwd=ROOT, check=True)
    manifest_frame = load_manifest_frame(train_frame, manifest_path)
    test_frame = train_frame.loc[manifest_frame.index[manifest_frame["split"] == "test"]]
    input_csv = input_dir / f"{protocol}_{fold}_test.csv"
    test_frame[original_columns].to_csv(input_csv, index=False)
    seed_prediction_frames: dict[int, pd.DataFrame] = {}
    metric_records: list[dict] = []
    prediction_records: list[pd.DataFrame] = []
    ensemble_config_path = config_dir / f"{protocol}_{fold}_ensemble_predict.yaml"
    ensemble_config_path.write_text(yaml.safe_dump(
        graph_predict_config(training_dir, protocol, fold, 0, input_csv), sort_keys=False
    ), encoding="utf-8")
    ensemble_prediction, prediction_run_dir = run_prediction(ensemble_config_path, repeat=3)
    if len(ensemble_prediction) != len(test_frame):
        raise ValueError(f"GraphGPS ensemble row mismatch for {protocol}/{fold}")
    for seed in (0, 1, 2):
        seed_path = prediction_run_dir / f"{seed}test_true_pred_sum.csv"
        if not seed_path.is_file():
            raise FileNotFoundError(f"Missing per-seed prediction output: {seed_path}")
        seed_prediction_frames[seed] = pd.read_csv(seed_path, index_col=0)
    for seed, prediction in seed_prediction_frames.items():
        for target, (true_column, prediction_column) in TARGET_PREDICTION_COLUMNS.items():
            individual_prediction_column = prediction_column.replace("_average", "")
            record_frame = pd.DataFrame({
                "protocol": protocol, "fold": fold, "target": target,
                "model": "GraphGPS_coarse_mordred", "seed": seed,
                "sample_id": test_frame["sample_id"].astype(str).to_numpy(),
                "y_true": prediction[true_column].to_numpy(dtype=float),
                "y_pred": prediction[individual_prediction_column].to_numpy(dtype=float),
            })
            record_frame["absolute_error"] = (record_frame["y_true"] - record_frame["y_pred"]).abs()
            prediction_records.append(record_frame)
            metric_records.append({"protocol": protocol, "fold": fold, "target": target,
                                   "model": "GraphGPS_coarse_mordred", "seed": seed,
                                   "n_test": len(record_frame), **metric_dict(record_frame["y_true"], record_frame["y_pred"])})
    for target, (true_column, prediction_column) in TARGET_PREDICTION_COLUMNS.items():
        mean_prediction = ensemble_prediction[prediction_column].to_numpy(dtype=float)
        record_frame = pd.DataFrame({
            "protocol": protocol, "fold": fold, "target": target,
            "model": "GraphGPS_coarse_mordred_ensemble", "seed": "ensemble",
            "sample_id": test_frame["sample_id"].astype(str).to_numpy(),
            "y_true": ensemble_prediction[true_column].to_numpy(dtype=float), "y_pred": mean_prediction,
        })
        record_frame["absolute_error"] = (record_frame["y_true"] - record_frame["y_pred"]).abs()
        prediction_records.append(record_frame)
        metric_records.append({"protocol": protocol, "fold": fold, "target": target,
                               "model": "GraphGPS_coarse_mordred_ensemble", "seed": "ensemble",
                               "n_test": len(record_frame), **metric_dict(record_frame["y_true"], record_frame["y_pred"])})
    return metric_records, prediction_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_stage2_arguments(parser)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--graphgps-folds", type=int, default=1)
    parser.add_argument("--skip-graphgps", action="store_true")
    arguments = parser.parse_args()
    output_dir = stage2_output(arguments.output_dir)
    group_dir = output_dir / "group_cv"
    config_dir = group_dir / "graphgps_configs"
    input_dir = group_dir / "graphgps_inputs"
    for directory in (group_dir, config_dir, input_dir):
        directory.mkdir(parents=True, exist_ok=True)
    schema, train_frame, _ = load_training_frame(arguments.train_csv, arguments.feedback_csv)
    feature_sets, _, _ = build_stable_feature_sets(train_frame, schema)
    features = feature_sets["F2_identity_ratio"]
    protocols = {"fifth_component_group_cv": "fifth_component_key",
                 "formula_identity_group_cv": "formula_identity_key"}
    manifests = {name: group_cv_manifests(train_frame, column, name, output_dir, arguments.seed, arguments.n_splits)
                 for name, column in protocols.items()}
    metric_records: list[dict] = []
    prediction_records: list[pd.DataFrame] = []
    for protocol, paths in manifests.items():
        for manifest_path in paths:
            manifest_frame = load_manifest_frame(train_frame, manifest_path)
            train_indices = manifest_frame.index[manifest_frame["split"] == "train"]
            test_indices = manifest_frame.index[manifest_frame["split"] == "test"]
            for target in schema.targets:
                train_target = train_frame.loc[train_indices, target].astype(float)
                test_target = train_frame.loc[test_indices, target].astype(float)
                mean_prediction = np.full(len(test_indices), train_target.mean())
                baseline_mae = metric_dict(test_target, mean_prediction)["mae"]
                for model_name, prediction in (("TrainMean", mean_prediction), ("TrainMedian", np.full(len(test_indices), train_target.median()))):
                    metrics = metric_dict(test_target, prediction)
                    metric_records.append({"protocol": protocol, "fold": manifest_path.stem, "target": target,
                                           "model": model_name, "seed": "not_applicable", "n_test": len(test_indices),
                                           **metrics, "mae_improvement_vs_train_mean": baseline_mae - metrics["mae"]})
                    output = pd.DataFrame({"protocol": protocol, "fold": manifest_path.stem, "target": target,
                                           "model": model_name, "seed": "not_applicable",
                                           "sample_id": train_frame.loc[test_indices, "sample_id"].astype(str).to_numpy(),
                                           "y_true": test_target.to_numpy(), "y_pred": prediction})
                    output["absolute_error"] = (output["y_true"] - output["y_pred"]).abs()
                    prediction_records.append(output)
                for model_name, estimator in models(arguments.seed, arguments.n_jobs).items():
                    fitted = make_pipeline(features, clone(estimator)).fit(features.loc[train_indices], train_target)
                    prediction = fitted.predict(features.loc[test_indices])
                    metrics = metric_dict(test_target, prediction)
                    metric_records.append({"protocol": protocol, "fold": manifest_path.stem, "target": target,
                                           "model": model_name, "seed": "not_applicable", "n_test": len(test_indices),
                                           **metrics, "mae_improvement_vs_train_mean": baseline_mae - metrics["mae"]})
                    output = pd.DataFrame({"protocol": protocol, "fold": manifest_path.stem, "target": target,
                                           "model": model_name, "seed": "not_applicable",
                                           "sample_id": train_frame.loc[test_indices, "sample_id"].astype(str).to_numpy(),
                                           "y_true": test_target.to_numpy(), "y_pred": prediction})
                    output["absolute_error"] = (output["y_true"] - output["y_pred"]).abs()
                    prediction_records.append(output)
    seed_metric_records: list[dict] = []
    if not arguments.skip_graphgps:
        original_columns = pd.read_csv(schema.train_path, nrows=1).columns.tolist()
        for protocol, paths in manifests.items():
            for manifest_path in paths[:arguments.graphgps_folds]:
                graph_metrics, graph_predictions = run_graph_fold(output_dir, protocol, manifest_path,
                                                                  train_frame, original_columns, config_dir, input_dir)
                metric_records.extend([record for record in graph_metrics if record["seed"] == "ensemble"])
                seed_metric_records.extend([record for record in graph_metrics if record["seed"] != "ensemble"])
                prediction_records.extend(graph_predictions)
    fold_metrics = pd.DataFrame(metric_records)
    seed_metrics = pd.DataFrame(seed_metric_records)
    predictions = pd.concat(prediction_records, ignore_index=True)
    fold_metrics.to_csv(group_dir / "fold_metrics.csv", index=False)
    seed_metrics.to_csv(group_dir / "seed_metrics.csv", index=False)
    predictions.to_csv(group_dir / "oof_predictions.csv", index=False)
    summary_records = []
    for (protocol, target, model), group in fold_metrics.groupby(["protocol", "target", "model"]):
        ci_low, ci_high = bootstrap_ci(group["mae"], arguments.seed)
        summary_records.append({"protocol": protocol, "target": target, "model": model,
                                "completed_folds": group["fold"].nunique(), "mean_mae": group["mae"].mean(),
                                "std_mae": group["mae"].std(ddof=1), "mae_ci95_low": ci_low, "mae_ci95_high": ci_high,
                                "mean_rmse": group["rmse"].mean(), "mean_r2": group["r2"].mean(),
                                "mean_mae_improvement_vs_train_mean": group["mae_improvement_vs_train_mean"].mean()})
    pd.DataFrame(summary_records).to_csv(group_dir / "summary_metrics.csv", index=False)
    pair_records = []
    graph = fold_metrics.loc[fold_metrics["model"] == "GraphGPS_coarse_mordred_ensemble"]
    tree = fold_metrics.loc[fold_metrics["model"] == "ExtraTrees"]
    for (protocol, target), graph_group in graph.groupby(["protocol", "target"]):
        paired = graph_group.merge(tree.loc[(tree["protocol"] == protocol) & (tree["target"] == target)],
                                   on=["protocol", "fold", "target"], suffixes=("_graphgps", "_extratrees"))
        if not paired.empty:
            differences = paired["mae_graphgps"] - paired["mae_extratrees"]
            pair_records.append({"protocol": protocol, "target": target, "completed_paired_folds": len(paired),
                                 "mean_graphgps_minus_extratrees_mae": differences.mean(),
                                 "graphgps_win_fraction": (differences < 0).mean()})
    pd.DataFrame(pair_records).to_csv(group_dir / "paired_model_comparison.csv", index=False)
    (group_dir / "group_cv_report.md").write_text(
        "# 重复 Group CV 基准\n\n"
        "树模型完成所有五折。GraphGPS 使用原始完整训练预算、显式 manifest、三种子，"
        f"当前完成每协议前 {arguments.graphgps_folds} 个 fold；其余不伪造。\n", encoding="utf-8")
    record_execution(output_dir, Path(__file__).name, details={"seed": arguments.seed, "n_jobs": arguments.n_jobs,
                     "n_splits": arguments.n_splits, "graphgps_folds": arguments.graphgps_folds,
                     "graphgps_seeds": [0, 1, 2], "skip_graphgps": arguments.skip_graphgps})
    print(f"Wrote Group CV benchmark to {group_dir}")


if __name__ == "__main__":
    main()
