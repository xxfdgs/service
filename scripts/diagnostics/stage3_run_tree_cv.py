#!/usr/bin/env python3
"""Run fixed, validation-selected tree baselines on stage-three outer manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import metric_dict
from run_repeated_group_benchmark import make_pipeline
from stable_formulation import build_stable_feature_sets
from stage2_common import load_training_frame
from stage3_utils import append_execution


PROTOCOLS = ("fifth_component_group_cv", "formula_identity_group_cv")


def estimators(seed: int, n_jobs: int) -> dict[str, object]:
    """Return the fixed non-feedback-tuned tree comparisons."""
    return {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.8,
                                            random_state=seed, n_jobs=n_jobs),
        "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.7,
                                                random_state=seed, n_jobs=n_jobs),
    }


def candidate_sets(target: str) -> list[str]:
    """Return the prespecified feature candidates available to one target."""
    if target == "EE_before":
        return ["F2_identity_ratio", "F3_physchem_weighted"]
    if target == "EE_after":
        return ["F2_identity_ratio", "F4_physchem_interactions"]
    return ["F2_identity_ratio"]


def manifest_indexes(frame: pd.DataFrame, manifest_path: Path) -> dict[str, pd.Index]:
    """Map explicit sample IDs to source frame indexes without row-order assumptions."""
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    lookup = pd.Series(frame.index.to_numpy(), index=frame["sample_id"].astype(str))
    mapped = manifest["sample_id"].map(lookup)
    if mapped.isna().any() or manifest["sample_id"].duplicated().any():
        raise ValueError(f"Manifest cannot uniquely map source rows: {manifest_path}")
    return {name: pd.Index(mapped.loc[manifest["split"] == name].astype(int)) for name in ("train", "val", "test")}


def output_rows(protocol: str, fold: str, target: str, model: str, sample_ids: pd.Series,
                y_true: pd.Series, y_pred: np.ndarray, feature_set: str) -> pd.DataFrame:
    """Create strict sample-id prediction rows for one outer held-out set."""
    frame = pd.DataFrame({"protocol": protocol, "fold": fold, "target": target, "model": model,
                          "feature_set": feature_set, "sample_id": sample_ids.astype(str).to_numpy(),
                          "y_true": y_true.to_numpy(dtype=float), "y_pred": np.asarray(y_pred, dtype=float)})
    frame["absolute_error"] = (frame["y_true"] - frame["y_pred"]).abs()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generalization_stage3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    tree_dir = output_dir / "tree_cv"
    tree_dir.mkdir(parents=True, exist_ok=True)
    schema, train_frame, _ = load_training_frame()
    feature_sets, _, _ = build_stable_feature_sets(train_frame, schema)
    metric_records: list[dict[str, object]] = []
    prediction_records: list[pd.DataFrame] = []
    selection_records: list[dict[str, object]] = []
    importance_records: list[dict[str, object]] = []
    for protocol in PROTOCOLS:
        for manifest_path in sorted((output_dir / "manifests" / protocol / "raw_records").glob("fold_*.csv")):
            indexes = manifest_indexes(train_frame, manifest_path)
            fold = manifest_path.stem
            for target in schema.targets:
                train_y = train_frame.loc[indexes["train"], target].astype(float)
                val_y = train_frame.loc[indexes["val"], target].astype(float)
                test_y = train_frame.loc[indexes["test"], target].astype(float)
                for model_name, prediction in {
                    "TrainMean": np.full(len(indexes["test"]), train_y.mean()),
                    "TrainMedian": np.full(len(indexes["test"]), train_y.median()),
                }.items():
                    rows = output_rows(protocol, fold, target, model_name,
                                       train_frame.loc[indexes["test"], "sample_id"], test_y, prediction, "none")
                    prediction_records.append(rows)
                    metric_records.append({"protocol": protocol, "fold": fold, "target": target, "model": model_name,
                                           "feature_set": "none", "n_test": len(rows), **metric_dict(rows.y_true, rows.y_pred)})
                for model_name, estimator in estimators(arguments.seed, arguments.n_jobs).items():
                    candidates: list[tuple[str, float]] = []
                    for feature_set in candidate_sets(target):
                        fitted = make_pipeline(feature_sets[feature_set], clone(estimator)).fit(
                            feature_sets[feature_set].loc[indexes["train"]], train_y)
                        validation_prediction = fitted.predict(feature_sets[feature_set].loc[indexes["val"]])
                        candidates.append((feature_set, metric_dict(val_y, validation_prediction)["mae"]))
                    selected_feature_set, selected_validation_mae = min(candidates, key=lambda item: item[1])
                    selection_records.append({"protocol": protocol, "fold": fold, "target": target, "model": model_name,
                                              "selected_feature_set": selected_feature_set,
                                              "validation_mae": selected_validation_mae,
                                              "candidates": "|".join(f"{name}:{score:.8f}" for name, score in candidates)})
                    fitted = make_pipeline(feature_sets[selected_feature_set], clone(estimator)).fit(
                        feature_sets[selected_feature_set].loc[indexes["train"]], train_y)
                    test_prediction = fitted.predict(feature_sets[selected_feature_set].loc[indexes["test"]])
                    rows = output_rows(protocol, fold, target, model_name,
                                       train_frame.loc[indexes["test"], "sample_id"], test_y, test_prediction,
                                       selected_feature_set)
                    prediction_records.append(rows)
                    metric_records.append({"protocol": protocol, "fold": fold, "target": target, "model": model_name,
                                           "feature_set": selected_feature_set, "n_test": len(rows), **metric_dict(rows.y_true, rows.y_pred)})
                    transformed_names = fitted.named_steps["preprocess"].get_feature_names_out()
                    importances = fitted.named_steps["model"].feature_importances_
                    for feature_name, importance in zip(transformed_names, importances):
                        importance_records.append({"protocol": protocol, "fold": fold, "target": target, "model": model_name,
                                                   "feature_set": selected_feature_set, "feature": feature_name,
                                                   "importance": float(importance)})
    metrics = pd.DataFrame(metric_records)
    predictions = pd.concat(prediction_records, ignore_index=True)
    metrics.to_csv(tree_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(tree_dir / "pooled_oof_predictions.csv", index=False)
    pd.DataFrame(selection_records).to_csv(tree_dir / "model_selection_within_cv.csv", index=False)
    pd.DataFrame(importance_records).to_csv(tree_dir / "feature_importance_by_fold.csv", index=False)
    pooled_rows: list[dict[str, object]] = []
    for (protocol, target, model), group in predictions.groupby(["protocol", "target", "model"]):
        pooled_rows.append({"protocol": protocol, "target": target, "model": model, "n": len(group),
                            **metric_dict(group.y_true, group.y_pred)})
    pd.DataFrame(pooled_rows).to_csv(tree_dir / "pooled_oof_metrics.csv", index=False)
    summary = metrics.groupby(["protocol", "target", "model"], as_index=False).agg(
        completed_folds=("fold", "nunique"), mean_mae=("mae", "mean"), std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"), mean_spearman=("spearman", "mean"),
    )
    summary.to_csv(tree_dir / "summary_metrics.csv", index=False)
    append_execution(output_dir, command=[sys.executable, *sys.argv], protocol="both", fold="all", seed=arguments.seed,
                     data_version="raw_records", output=tree_dir)
    print(f"Wrote {tree_dir}")


if __name__ == "__main__":
    main()
