#!/usr/bin/env python3
"""Validate non-saturated applicability scores on group-CV OOF predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import MinCovDet
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_common import (  # noqa: E402
    add_stage2_arguments, load_manifest_frame, load_training_frame, record_execution, stage2_output,
)
from stable_formulation import build_stable_feature_sets  # noqa: E402


def _fit_ad(train_features: pd.DataFrame) -> dict[str, object]:
    """Fit feature scaling, nearest-neighbor and robust covariance using train rows only."""
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    imputed = imputer.fit_transform(train_features)
    scaler = StandardScaler().fit(imputed)
    scaled = scaler.transform(imputed)
    neighbors = NearestNeighbors(n_neighbors=min(11, len(scaled))).fit(scaled)
    # Ratios and weighted summary features include exact linear relations. PCA
    # is fitted on training rows only to remove singular directions before the
    # required robust covariance calculation.
    pca = PCA(n_components=min(30, scaled.shape[1], len(scaled) - 1), random_state=42).fit(scaled)
    robust_train = pca.transform(scaled)
    covariance = MinCovDet(support_fraction=0.8, random_state=42).fit(robust_train)
    return {"imputer": imputer, "scaler": scaler, "scaled_train": scaled,
            "neighbors": neighbors, "pca": pca, "covariance": covariance,
            "lower": train_features.quantile(0.01), "upper": train_features.quantile(0.99)}


def _scores(fitted: dict[str, object], features: pd.DataFrame, same_training: bool = False) -> pd.DataFrame:
    """Calculate distances, density, range checks and robust covariance distance."""
    imputed = fitted["imputer"].transform(features)  # type: ignore[union-attr]
    scaled = fitted["scaler"].transform(imputed)  # type: ignore[union-attr]
    distances, _ = fitted["neighbors"].kneighbors(scaled)  # type: ignore[union-attr]
    start = 1 if same_training else 0
    nearest = distances[:, start]
    mean5 = distances[:, start:start + min(5, distances.shape[1] - start)].mean(axis=1)
    mean10 = distances[:, start:start + min(10, distances.shape[1] - start)].mean(axis=1)
    lower = fitted["lower"]  # type: ignore[assignment]
    upper = fitted["upper"]  # type: ignore[assignment]
    outside = ((features < lower) | (features > upper)).sum(axis=1)
    robust_distance = fitted["covariance"].mahalanobis(  # type: ignore[union-attr]
        fitted["pca"].transform(scaled)  # type: ignore[union-attr]
    )
    return pd.DataFrame({
        "nearest_training_distance": nearest, "mean_knn5_distance": mean5,
        "mean_knn10_distance": mean10, "local_density_knn5": 1.0 / np.maximum(mean5, 1e-12),
        "out_of_range_feature_count": outside.to_numpy(dtype=int),
        "robust_covariance_distance": robust_distance,
    }, index=features.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_stage2_arguments(parser)
    arguments = parser.parse_args()
    output_dir = stage2_output(arguments.output_dir)
    ad_dir = output_dir / "applicability_domain"
    ad_dir.mkdir(parents=True, exist_ok=True)
    group_dir = output_dir / "group_cv"
    oof_path = group_dir / "oof_predictions.csv"
    if not oof_path.is_file():
        raise FileNotFoundError(f"Run run_repeated_group_benchmark.py first: {oof_path}")
    schema, train_frame, feedback_frame = load_training_frame(arguments.train_csv, arguments.feedback_csv)
    train_feature_sets, _, _ = build_stable_feature_sets(train_frame, schema)
    feedback_feature_sets, _, _ = build_stable_feature_sets(feedback_frame, schema)
    train_features = train_feature_sets["F4_physchem_interactions"].replace([np.inf, -np.inf], np.nan)
    feedback_features = feedback_feature_sets["F4_physchem_interactions"].replace([np.inf, -np.inf], np.nan)
    manifests: list[Path] = []
    for protocol_name in ("fifth_component_group_cv", "formula_identity_group_cv"):
        manifests.extend(sorted((output_dir / "manifests" / protocol_name).glob("fold_*.csv")))
    oof_scores: list[pd.DataFrame] = []
    for manifest_path in manifests:
        protocol = manifest_path.parent.name
        manifest_frame = load_manifest_frame(train_frame, manifest_path)
        train_indices = manifest_frame.index[manifest_frame["split"] == "train"]
        test_indices = manifest_frame.index[manifest_frame["split"] == "test"]
        fitted = _fit_ad(train_features.loc[train_indices])
        scores = _scores(fitted, train_features.loc[test_indices])
        scores["protocol"] = protocol
        scores["fold"] = manifest_path.stem
        scores["sample_id"] = train_frame.loc[test_indices, "sample_id"].astype(str).to_numpy()
        scores["fifth_component_seen_in_train"] = train_frame.loc[test_indices, "fifth_component_key"].isin(
            set(train_frame.loc[train_indices, "fifth_component_key"])
        ).to_numpy()
        scores["formula_identity_seen_in_train"] = train_frame.loc[test_indices, "formula_identity_key"].isin(
            set(train_frame.loc[train_indices, "formula_identity_key"])
        ).to_numpy()
        for position in range(1, 6):
            scores[f"component_{position}_seen_in_train"] = train_frame.loc[
                test_indices, f"component_{position}_key"
            ].isin(set(train_frame.loc[train_indices, f"component_{position}_key"])).to_numpy()
        oof_scores.append(scores.reset_index(drop=True))
    oof_score_frame = pd.concat(oof_scores, ignore_index=True)
    oof_predictions = pd.read_csv(oof_path, dtype={"sample_id": str})
    merged = oof_predictions.merge(oof_score_frame, on=["protocol", "fold", "sample_id"], how="inner")
    correlation_records: list[dict[str, object]] = []
    rejection_records: list[dict[str, object]] = []
    score_columns = ["nearest_training_distance", "mean_knn5_distance", "mean_knn10_distance",
                     "robust_covariance_distance", "out_of_range_feature_count"]
    for (protocol, target, model), group in merged.groupby(["protocol", "target", "model"]):
        for score_name in score_columns:
            correlation = spearmanr(group[score_name], group["absolute_error"], nan_policy="omit")
            correlation_records.append({"protocol": protocol, "target": target, "model": model,
                                        "score": score_name, "spearman_r": correlation.statistic,
                                        "p_value": correlation.pvalue, "n_samples": len(group)})
            for rejection_fraction in (0.1, 0.2, 0.3):
                threshold = group[score_name].quantile(1 - rejection_fraction)
                retained = group.loc[group[score_name] <= threshold]
                rejection_records.append({"protocol": protocol, "target": target, "model": model,
                                          "score": score_name, "rejection_fraction": rejection_fraction,
                                          "retained_fraction": len(retained) / len(group),
                                          "retained_mae": retained["absolute_error"].mean(),
                                          "all_mae": group["absolute_error"].mean()})
    correlation_frame = pd.DataFrame(correlation_records)
    rejection_frame = pd.DataFrame(rejection_records)
    correlation_frame.to_csv(ad_dir / "ad_error_correlations.csv", index=False)
    rejection_frame.to_csv(ad_dir / "rejection_curves.csv", index=False)
    oof_score_frame.to_csv(ad_dir / "oof_ad_scores.csv", index=False)
    # Fit feedback AD only after score definitions/thresholds were determined from training OOF data.
    full_fitted = _fit_ad(train_features)
    feedback_scores = _scores(full_fitted, feedback_features)
    feedback_scores["sample_id"] = feedback_frame["sample_id"].astype(str).to_numpy()
    feedback_scores["fifth_component_seen_in_train"] = feedback_frame["fifth_component_key"].isin(
        set(train_frame["fifth_component_key"])
    ).to_numpy()
    feedback_scores["formula_identity_seen_in_train"] = feedback_frame["formula_identity_key"].isin(
        set(train_frame["formula_identity_key"])
    ).to_numpy()
    for position in range(1, 6):
        feedback_scores[f"component_{position}_seen_in_train"] = feedback_frame[
            f"component_{position}_key"
        ].isin(set(train_frame[f"component_{position}_key"])).to_numpy()
    domain_scores = pd.read_csv(ROOT / "results/generalization_diagnostics/feedback_ood_scores.csv", dtype={"diagnostic_sample_id": str})
    feedback_scores = feedback_scores.merge(domain_scores[["diagnostic_sample_id", "ood_score"]].rename(
        columns={"diagnostic_sample_id": "sample_id"}), on="sample_id", how="left")
    feedback_scores.to_csv(ad_dir / "feedback_ad_scores.csv", index=False)
    threshold_payload = {
        score_name: {"oof_p90": float(oof_score_frame[score_name].quantile(0.9)),
                     "oof_p95": float(oof_score_frame[score_name].quantile(0.95))}
        for score_name in score_columns
    }
    stable_positive = correlation_frame.groupby("score")["spearman_r"].median()
    threshold_payload["validated_primary_scores"] = stable_positive[stable_positive > 0].sort_values(
        ascending=False
    ).to_dict()
    (ad_dir / "training_ad_thresholds.json").write_text(
        json.dumps(threshold_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = [
        "# Applicability Domain", "",
        "- 所有阈值由内部 group-CV OOF 分数确定，未使用 feedback 误差。",
        "- domain classifier probability 仅作为 feedback 辅助列保留；主排序来自内部验证的距离/范围指标。",
        f"- 内部验证中中位 Spearman 为正的候选指标：{threshold_payload['validated_primary_scores']}。",
    ]
    (ad_dir / "applicability_domain_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    record_execution(output_dir, Path(__file__).name, details={"seed": arguments.seed,
                     "n_jobs": arguments.n_jobs, "feature_set": "F4_physchem_interactions"})
    print(f"Wrote applicability-domain analysis to {ad_dir}")


if __name__ == "__main__":
    main()
