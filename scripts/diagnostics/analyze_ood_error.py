#!/usr/bin/env python3
"""Relate feedback prediction errors to domain-classifier and NN OOD signals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    add_common_arguments, build_feature_frames, discover_schema, load_frames,
    load_mordred_table, metric_dict,
)


def _nearest_training_distances(
    train_features: pd.DataFrame, feedback_features: pd.DataFrame
) -> np.ndarray:
    """Calculate feedback-to-training nearest-neighbor distance using train-fit scaling."""
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    imputed_train = imputer.fit_transform(train_features)
    imputed_feedback = imputer.transform(feedback_features)
    scaler = StandardScaler().fit(imputed_train)
    return pairwise_distances(
        scaler.transform(imputed_feedback), scaler.transform(imputed_train),
        metric="euclidean", n_jobs=1,
    ).min(axis=1)


def _ood_bins(scores: pd.Series) -> pd.Series:
    """Create equally populated relative OOD strata when all probabilities are high."""
    lower_cutoff, upper_cutoff = scores.quantile([1 / 3, 2 / 3]).to_list()
    return pd.Series(np.select(
        [scores <= lower_cutoff, scores <= upper_cutoff],
        ["in-domain", "mild-OOD"], default="severe-OOD",
    ), index=scores.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-mordred-features", type=int, default=256)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    required_paths = {
        "ood": output_dir / "feedback_ood_scores.csv",
        "graphgps": output_dir / "graphgps_predictions.csv",
        "baseline": output_dir / "baseline_predictions.csv",
    }
    missing_paths = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing prerequisite diagnostics: " + ", ".join(missing_paths)
        )
    schema = discover_schema(arguments.train_csv, arguments.feedback_csv)
    train_frame, feedback_frame = load_frames(schema)
    mordred_frame = load_mordred_table(schema)
    train_numeric, _ = build_feature_frames(
        train_frame, schema, mordred_frame, arguments.max_mordred_features
    )
    feedback_numeric, _ = build_feature_frames(
        feedback_frame, schema, mordred_frame, arguments.max_mordred_features
    )
    nearest_distances = _nearest_training_distances(train_numeric, feedback_numeric)
    ood_frame = pd.read_csv(required_paths["ood"], dtype={"diagnostic_sample_id": str})
    ood_frame = ood_frame[["diagnostic_sample_id", "ood_score"]].copy()
    ood_frame["nearest_training_distance"] = nearest_distances
    ood_frame["ood_bin"] = _ood_bins(ood_frame["ood_score"])
    graphgps_frame = pd.read_csv(required_paths["graphgps"], dtype={"diagnostic_sample_id": str})
    graphgps_frame = graphgps_frame.loc[
        graphgps_frame["evaluation_set"] == "feedback",
        ["diagnostic_sample_id", "target", "y_true", "y_pred"],
    ].rename(columns={"y_pred": "graphgps_prediction"})
    tree_frame = pd.read_csv(required_paths["baseline"], dtype={"diagnostic_sample_id": str})
    tree_frame = tree_frame.loc[
        (tree_frame["split_name"] == "full_train") &
        (tree_frame["evaluation_set"] == "feedback") &
        (tree_frame["model"] == "ExtraTrees"),
        ["diagnostic_sample_id", "target", "y_pred"],
    ].rename(columns={"y_pred": "extra_trees_prediction"})
    merged = graphgps_frame.merge(
        tree_frame, on=["diagnostic_sample_id", "target"], how="inner", validate="one_to_one"
    ).merge(ood_frame, on="diagnostic_sample_id", how="inner", validate="many_to_one")
    expected_rows = len(feedback_frame) * len(schema.targets)
    if len(merged) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} feedback target rows after merge, found {len(merged)}."
        )
    merged["graphgps_absolute_error"] = (
        merged["y_true"] - merged["graphgps_prediction"]
    ).abs()
    merged["extra_trees_absolute_error"] = (
        merged["y_true"] - merged["extra_trees_prediction"]
    ).abs()
    merged.to_csv(output_dir / "feedback_error_analysis.csv", index=False)

    metric_records: list[dict[str, object]] = []
    correlation_records: list[dict[str, object]] = []
    for target in schema.targets:
        target_frame = merged.loc[merged["target"] == target]
        for model_name, prediction_column, error_column in (
            ("GraphGPS_coarse_mordred", "graphgps_prediction", "graphgps_absolute_error"),
            ("ExtraTrees", "extra_trees_prediction", "extra_trees_absolute_error"),
        ):
            for bin_name, bin_frame in target_frame.groupby("ood_bin", observed=True):
                metrics = metric_dict(bin_frame["y_true"], bin_frame[prediction_column])
                metric_records.append({
                    "target": target, "model": model_name, "ood_bin": bin_name,
                    "n_samples": int(len(bin_frame)), **metrics,
                })
            for signal_name in ("ood_score", "nearest_training_distance"):
                correlation = spearmanr(
                    target_frame[signal_name], target_frame[error_column], nan_policy="omit"
                )
                correlation_records.append({
                    "target": target,
                    "model": model_name,
                    "signal": signal_name,
                    "spearman_r": float(correlation.statistic),
                    "p_value": float(correlation.pvalue),
                    "n_samples": int(len(target_frame)),
                })
    pd.DataFrame(metric_records).to_csv(output_dir / "ood_error_by_bin.csv", index=False)
    pd.DataFrame(correlation_records).to_csv(
        output_dir / "ood_error_correlations.csv", index=False
    )
    print(f"Wrote OOD-error analysis for {len(merged)} feedback target rows.")


if __name__ == "__main__":
    main()
