#!/usr/bin/env python3
"""Measure label shift, feedback residual bias, and train-only group calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, linregress, pearsonr, spearmanr, wasserstein_distance
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import build_feature_frames, metric_dict  # noqa: E402
from stage2_common import (  # noqa: E402
    add_stage2_arguments, group_cv_manifests, load_manifest_frame, load_training_frame,
    record_execution, stage2_output,
)


def _distribution_stats(values: pd.Series) -> dict[str, float]:
    """Return stable distribution summaries for one numeric target."""
    numeric = values.dropna().astype(float)
    return {
        "mean": float(numeric.mean()), "median": float(numeric.median()),
        "std": float(numeric.std(ddof=1)),
        "iqr": float(numeric.quantile(0.75) - numeric.quantile(0.25)),
        "min": float(numeric.min()), "max": float(numeric.max()),
        "q01": float(numeric.quantile(0.01)), "q99": float(numeric.quantile(0.99)),
    }


def _simple_features(frame: pd.DataFrame, schema) -> pd.DataFrame:
    """Use only ratios, lightweight RDKit features, and raw identity categories."""
    numeric, categorical = build_feature_frames(frame, schema, mordred_frame=None)
    encoded_categories = pd.get_dummies(categorical.astype(str), prefix=categorical.columns, dtype=float)
    return pd.concat([numeric, encoded_categories], axis=1).replace([np.inf, -np.inf], np.nan)


def _fit_extra_trees(train_features: pd.DataFrame, train_targets: pd.Series, seed: int, n_jobs: int) -> ExtraTreesRegressor:
    """Fit a fixed non-feedback-tuned ExtraTrees model with train-only median imputation."""
    imputed = train_features.copy()
    medians = imputed.median(numeric_only=True)
    imputed = imputed.fillna(medians).fillna(0.0)
    model = ExtraTreesRegressor(
        n_estimators=400, min_samples_leaf=2, max_features=0.8,
        random_state=seed, n_jobs=n_jobs,
    )
    model.fit(imputed, train_targets)
    model._stage2_feature_medians = medians  # type: ignore[attr-defined]
    return model


def _predict_extra_trees(model: ExtraTreesRegressor, features: pd.DataFrame) -> np.ndarray:
    """Apply the persisted train-only imputation parameters before prediction."""
    medians = model._stage2_feature_medians  # type: ignore[attr-defined]
    return model.predict(features.fillna(medians).fillna(0.0))


def _calibrate_predictions(
    method: str, validation_true: np.ndarray, validation_pred: np.ndarray,
    test_pred: np.ndarray,
) -> np.ndarray:
    """Fit a calibration map on validation predictions only."""
    if method == "none":
        return test_pred
    if method == "intercept_only":
        return test_pred + np.mean(validation_true - validation_pred)
    if method == "affine":
        model = LinearRegression().fit(validation_pred.reshape(-1, 1), validation_true)
        return model.predict(test_pred.reshape(-1, 1))
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(validation_pred, validation_true)
        return model.predict(test_pred)
    raise ValueError(f"Unknown calibration method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_stage2_arguments(parser)
    parser.add_argument("--n-splits", type=int, default=5)
    arguments = parser.parse_args()
    output_dir = stage2_output(arguments.output_dir)
    label_dir = output_dir / "label_shift"
    label_dir.mkdir(parents=True, exist_ok=True)
    schema, train_frame, feedback_frame = load_training_frame(arguments.train_csv, arguments.feedback_csv)

    shift_records: list[dict[str, object]] = []
    curve_records: list[dict[str, object]] = []
    for target in schema.targets:
        train_values = train_frame[target].dropna().astype(float)
        feedback_values = feedback_frame[target].dropna().astype(float)
        train_stats = _distribution_stats(train_values)
        feedback_stats = _distribution_stats(feedback_values)
        outside_quantiles = ((feedback_values < train_stats["q01"]) | (feedback_values > train_stats["q99"])).mean()
        shift_records.append({
            "target": target,
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"feedback_{key}": value for key, value in feedback_stats.items()},
            "mean_difference_feedback_minus_train": feedback_stats["mean"] - train_stats["mean"],
            "median_difference_feedback_minus_train": feedback_stats["median"] - train_stats["median"],
            "standardized_mean_difference": (feedback_stats["mean"] - train_stats["mean"]) /
                                         max(train_stats["std"], 1e-12),
            "wasserstein_distance": float(wasserstein_distance(train_values, feedback_values)),
            "ks_statistic": float(ks_2samp(train_values, feedback_values).statistic),
            "feedback_outside_train_q01_q99_fraction": float(outside_quantiles),
            "train_range_covers_feedback": bool(
                feedback_stats["min"] >= train_stats["min"] and feedback_stats["max"] <= train_stats["max"]
            ),
        })
        edges = np.histogram_bin_edges(np.concatenate([train_values, feedback_values]), bins=30)
        for domain, values in (("train", train_values), ("feedback", feedback_values)):
            histogram, _ = np.histogram(values, bins=edges)
            sorted_values = np.sort(values.to_numpy())
            for bin_index, count in enumerate(histogram):
                curve_records.append({
                    "target": target, "domain": domain, "curve_type": "histogram",
                    "x_left": float(edges[bin_index]), "x_right": float(edges[bin_index + 1]),
                    "value": int(count),
                })
            for rank, value in enumerate(sorted_values, start=1):
                curve_records.append({
                    "target": target, "domain": domain, "curve_type": "ecdf",
                    "x_left": float(value), "x_right": float(value),
                    "value": rank / len(sorted_values),
                })
    pd.DataFrame(shift_records).to_csv(label_dir / "train_feedback_label_shift.csv", index=False)
    pd.DataFrame(curve_records).to_csv(label_dir / "label_distribution_curves.csv", index=False)

    stage1_dir = ROOT / "results/generalization_diagnostics"
    graph_predictions = pd.read_csv(stage1_dir / "graphgps_predictions.csv")
    graph_predictions = graph_predictions.loc[
        graph_predictions["evaluation_set"] == "feedback",
        ["target", "y_true", "y_pred"],
    ].assign(model="GraphGPS_coarse_mordred")
    baseline_predictions = pd.read_csv(stage1_dir / "baseline_predictions.csv")
    baseline_predictions = baseline_predictions.loc[
        (baseline_predictions["split_name"] == "full_train") &
        (baseline_predictions["evaluation_set"] == "feedback") &
        baseline_predictions["model"].isin(["ExtraTrees", "TrainMean"]),
        ["target", "y_true", "y_pred", "model"],
    ]
    feedback_predictions = pd.concat([graph_predictions, baseline_predictions], ignore_index=True)
    residual_records: list[dict[str, object]] = []
    for (model_name, target), group in feedback_predictions.groupby(["model", "target"]):
        residual = group["y_true"].astype(float) - group["y_pred"].astype(float)
        true_values = group["y_true"].astype(float)
        predicted_values = group["y_pred"].astype(float)
        prediction_std = float(predicted_values.std(ddof=1))
        if prediction_std <= 1e-10:
            regression_intercept = np.nan
            regression_slope = np.nan
            pearson = np.nan
            spearman = np.nan
        else:
            regression = linregress(predicted_values, true_values)
            regression_intercept = float(regression.intercept)
            regression_slope = float(regression.slope)
            pearson = float(pearsonr(predicted_values, true_values).statistic)
            spearman = float(spearmanr(predicted_values, true_values).statistic)
        prediction_range = float(predicted_values.max() - predicted_values.min())
        true_range = float(true_values.max() - true_values.min())
        issue_flags = []
        if abs(residual.mean()) > 0.5 * max(residual.std(ddof=1), 1e-12):
            issue_flags.append("global_bias")
        if predicted_values.std(ddof=1) < 0.6 * true_values.std(ddof=1):
            issue_flags.append("variance_compression")
        if pd.isna(spearman) or spearman < 0.3:
            issue_flags.append("weak_ranking")
        extreme_mask = (true_values < true_values.quantile(0.05)) | (true_values > true_values.quantile(0.95))
        if residual[extreme_mask].abs().mean() > residual.abs().mean() * 1.25:
            issue_flags.append("extreme_samples")
        residual_records.append({
            "model": model_name, "target": target,
            "mean_residual": float(residual.mean()), "median_residual": float(residual.median()),
            "residual_std": float(residual.std(ddof=1)),
            "prediction_mean": float(predicted_values.mean()), "prediction_std": prediction_std,
            "true_std": float(true_values.std(ddof=1)),
            "true_std_over_prediction_std": float(true_values.std(ddof=1) / prediction_std)
            if prediction_std > 1e-10 else np.inf,
            "pearson": float(pearson), "spearman": float(spearman),
            "regression_intercept": regression_intercept, "regression_slope": regression_slope,
            "prediction_range": prediction_range, "true_range": true_range,
            "prediction_range_over_true_range": prediction_range / max(true_range, 1e-12),
            "dominant_error_factors": "|".join(issue_flags) or "none",
        })
    residual_frame = pd.DataFrame(residual_records)
    residual_frame.to_csv(label_dir / "residual_analysis.csv", index=False)
    residual_frame.to_csv(label_dir / "model_calibration_metrics.csv", index=False)

    manifests = group_cv_manifests(
        train_frame, "formula_identity_key", "formula_identity_calibration_cv", output_dir,
        seed=arguments.seed, n_splits=arguments.n_splits,
    )
    feature_frame = _simple_features(train_frame, schema)
    calibration_records: list[dict[str, object]] = []
    for manifest_path in manifests:
        fold_frame = load_manifest_frame(train_frame, manifest_path)
        split_indexes = {
            split_name: fold_frame.index[fold_frame["split"] == split_name]
            for split_name in ("train", "val", "test")
        }
        for target in schema.targets:
            train_target = train_frame.loc[split_indexes["train"], target].astype(float)
            extra_trees = _fit_extra_trees(
                feature_frame.loc[split_indexes["train"]], train_target, arguments.seed, arguments.n_jobs
            )
            model_predictions = {
                "TrainMean": (
                    np.full(len(split_indexes["val"]), train_target.mean()),
                    np.full(len(split_indexes["test"]), train_target.mean()),
                ),
                "ExtraTrees": (
                    _predict_extra_trees(extra_trees, feature_frame.loc[split_indexes["val"]]),
                    _predict_extra_trees(extra_trees, feature_frame.loc[split_indexes["test"]]),
                ),
            }
            validation_true = train_frame.loc[split_indexes["val"], target].to_numpy(dtype=float)
            test_true = train_frame.loc[split_indexes["test"], target].to_numpy(dtype=float)
            for model_name, (validation_pred, test_pred) in model_predictions.items():
                for method in ("none", "intercept_only", "affine", "isotonic"):
                    calibrated = _calibrate_predictions(method, validation_true, validation_pred, test_pred)
                    calibration_records.append({
                        "fold": manifest_path.stem, "target": target, "model": model_name,
                        "calibration": method, "n_validation": len(validation_true),
                        "n_test": len(test_true), **metric_dict(test_true, calibrated),
                    })
    calibration_frame = pd.DataFrame(calibration_records)
    calibration_frame.to_csv(label_dir / "group_cv_calibration_results.csv", index=False)
    report_lines = [
        "# 标签偏移与系统预测偏差", "",
        "- 训练/feedback 标签分布仅用于外部评估和偏差分析，未用于训练、特征选择或校准。",
        "- 内部校准在 formula identity group CV 中仅使用每折 validation 拟合，随后只在 test 评估。",
        "- feedback 上的残差、范围压缩和排序结果见 `residual_analysis.csv`。",
    ]
    (label_dir / "label_shift_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    record_execution(output_dir, Path(__file__).name, details={
        "seed": arguments.seed, "n_jobs": arguments.n_jobs, "n_splits": arguments.n_splits,
        "calibration_protocol": "formula_identity_calibration_cv",
    })
    print(f"Wrote label-shift and calibration analysis to {label_dir}")


if __name__ == "__main__":
    main()
