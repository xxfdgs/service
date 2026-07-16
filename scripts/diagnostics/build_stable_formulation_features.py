#!/usr/bin/env python3
"""Build and benchmark compact, interpretable formulation feature sets F1–F4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import metric_dict  # noqa: E402
from stage2_common import (  # noqa: E402
    add_stage2_arguments, group_cv_manifests, load_manifest_frame, load_training_frame,
    record_execution, stage2_output,
)
from stable_formulation import build_stable_feature_sets  # noqa: E402


def _pipeline(features: pd.DataFrame, estimator: object) -> Pipeline:
    """Build fold-local imputation, scaling, and one-hot identity encoding."""
    numeric_columns = features.select_dtypes(exclude="object").columns.tolist()
    categorical_columns = features.select_dtypes(include="object").columns.tolist()
    transformers = []
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


def _models(seed: int, n_jobs: int) -> dict[str, object]:
    """Return fixed low-dimensional interpretable-model specifications."""
    return {
        "Ridge": Ridge(alpha=1.0),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.8,
            random_state=seed, n_jobs=n_jobs,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.7,
            random_state=seed, n_jobs=n_jobs,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_stage2_arguments(parser)
    parser.add_argument("--n-splits", type=int, default=5)
    arguments = parser.parse_args()
    output_dir = stage2_output(arguments.output_dir)
    feature_dir = output_dir / "stable_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    schema, train_frame, _ = load_training_frame(arguments.train_csv, arguments.feedback_csv)
    feature_sets, component_feature_frame, schema_payload = build_stable_feature_sets(train_frame, schema)
    schema_payload["source_csv"] = str(schema.train_path)
    (feature_dir / "feature_schema.json").write_text(
        json.dumps(schema_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    exported = train_frame[["sample_id", "raw_index"]].copy()
    for feature_set_name, features in feature_sets.items():
        exported = pd.concat([exported, features.add_prefix(f"{feature_set_name}__")], axis=1)
    exported = pd.concat([exported, component_feature_frame.add_prefix("component_physchem__")], axis=1)
    exported.to_csv(feature_dir / "formulation_features.csv", index=False)

    protocol_columns = {
        "fifth_component_group_cv": "fifth_component_key",
        "formula_identity_group_cv": "formula_identity_key",
    }
    all_manifests: dict[str, list[Path]] = {}
    for protocol, group_column in protocol_columns.items():
        all_manifests[protocol] = group_cv_manifests(
            train_frame, group_column, protocol, output_dir, arguments.seed, arguments.n_splits
        )
    metric_records: list[dict[str, object]] = []
    importance_records: list[dict[str, object]] = []
    for protocol, manifest_paths in all_manifests.items():
        for manifest_path in manifest_paths:
            manifest_frame = load_manifest_frame(train_frame, manifest_path)
            train_indices = manifest_frame.index[manifest_frame["split"] == "train"]
            test_indices = manifest_frame.index[manifest_frame["split"] == "test"]
            for feature_set_name, features in feature_sets.items():
                for target in schema.targets:
                    train_target = train_frame.loc[train_indices, target].astype(float)
                    test_target = train_frame.loc[test_indices, target].astype(float)
                    train_mean_prediction = np.full(len(test_indices), train_target.mean())
                    baseline_mae = metric_dict(test_target, train_mean_prediction)["mae"]
                    for model_name, estimator in _models(arguments.seed, arguments.n_jobs).items():
                        fitted = _pipeline(features, clone(estimator)).fit(
                            features.loc[train_indices], train_target
                        )
                        prediction = fitted.predict(features.loc[test_indices])
                        metrics = metric_dict(test_target, prediction)
                        metric_records.append({
                            "protocol": protocol, "fold": manifest_path.stem,
                            "feature_set": feature_set_name, "target": target,
                            "model": model_name, "n_test": len(test_indices), **metrics,
                            "mae_improvement_vs_train_mean": baseline_mae - metrics["mae"],
                        })
                        if model_name in {"ExtraTrees", "RandomForest"}:
                            transformed_names = fitted.named_steps["preprocess"].get_feature_names_out()
                            importances = fitted.named_steps["model"].feature_importances_
                            for name, importance in zip(transformed_names, importances):
                                importance_records.append({
                                    "protocol": protocol, "fold": manifest_path.stem,
                                    "feature_set": feature_set_name, "target": target,
                                    "model": model_name, "feature": name,
                                    "importance": float(importance),
                                })
    metrics = pd.DataFrame(metric_records)
    metrics.to_csv(feature_dir / "metrics_by_feature_set.csv", index=False)
    pd.DataFrame(importance_records).to_csv(feature_dir / "feature_importance_by_fold.csv", index=False)
    summary = metrics.groupby(["protocol", "feature_set", "target", "model"], as_index=False).agg(
        mean_mae=("mae", "mean"), std_mae=("mae", "std"), mean_r2=("r2", "mean"),
        mean_improvement=("mae_improvement_vs_train_mean", "mean"),
    ).sort_values(["protocol", "target", "mean_mae"])
    report_lines = [
        "# 稳定低维配方特征", "",
        "- F1–F4 均只使用规定的比例、身份与 12 个稳定 RDKit 理化性质。",
        "- F4 维度受限于 100 以下；身份 one-hot 仅在训练 fold 拟合。",
        "- 所有结果来自与 Group CV 共用的显式 manifest；完整逐折结果见 CSV。", "",
        "## 汇总", "", summary.to_csv(index=False),
    ]
    (feature_dir / "stable_feature_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    record_execution(output_dir, Path(__file__).name, details={
        "seed": arguments.seed, "n_jobs": arguments.n_jobs,
        "n_splits": arguments.n_splits, "protocols": list(protocol_columns),
        "feature_dimensions": schema_payload["feature_dimensions"],
    })
    print(f"Wrote stable feature benchmark to {feature_dir}")


if __name__ == "__main__":
    main()
