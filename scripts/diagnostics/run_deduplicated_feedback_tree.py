#!/usr/bin/env python3
"""Score the formula-CV-selected tree models on labelled feedback data.

Fits each target's preselected tree/feature pair on all 700 audited new-data
rows, never on feedback labels, then writes aligned predictions, metrics, and
a 2x2 actual-vs-predicted R² figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from audit_deduplicated_dataset import COMPONENTS, TARGETS, enrich, sha256_file  # noqa: E402
from run_deduplicated_tree_baselines import map_unknown_categories, pipeline_for  # noqa: E402
from stable_formulation import build_stable_feature_sets  # noqa: E402


SELECTED = {
    "EE_before": ("ExtraTrees", "F3_physchem_weighted"),
    "EE_after": ("ExtraTrees", "F2_identity_ratio"),
    "Aerosolization_Efficiency": ("ExtraTrees", "F2_identity_ratio"),
    "mRNA_Recovery_Efficiency": ("RandomForest", "F2_identity_ratio"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/deduplicated_rebaseline")
    parser.add_argument("--feedback-csv", type=Path, default=ROOT / "datasets_lrx/raw/feedback/20260703_validation.csv")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    root = arguments.output_dir.resolve()
    source = json.loads((root / "data_source.json").read_text(encoding="utf-8"))
    train_path = root / "data_audit/dataset_with_sample_id.csv"
    if source.get("audit_status") not in {"PASS", "PASS_WITH_WARNINGS"} or not train_path.is_file():
        raise RuntimeError("Audited new-dataset inputs are required.")
    train = pd.read_csv(train_path, dtype={"sample_id": str})
    feedback_path = arguments.feedback_csv.resolve()
    feedback = pd.read_csv(feedback_path, dtype={"ID": str})
    if any(column not in feedback for column in ["ID", *TARGETS]):
        raise ValueError("Feedback CSV must include ID and all four target columns for R² evaluation.")
    if feedback.ID.isna().any() or feedback.ID.duplicated().any() or feedback[TARGETS].isna().any().any():
        raise ValueError("Feedback IDs and target labels must be non-null and unique.")
    # Reproduce the new-dataset feature definition; feedback labels are not used
    # during fitting or category construction.
    feedback = enrich(feedback)
    schema = SimpleNamespace(components=[{"name_column": name, "smiles_column": smiles, "ratio_column": ratio}
                                         for name, smiles, ratio in COMPONENTS])
    train_features, _, _ = build_stable_feature_sets(train, schema)
    feedback_features, _, _ = build_stable_feature_sets(feedback, schema)
    output = root / "feedback_tree_best"
    output.mkdir(parents=True, exist_ok=True)
    prediction_frames, metric_rows = [], []
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, target in zip(axes.flat, TARGETS):
        model_name, feature_name = SELECTED[target]
        x_train, x_feedback = map_unknown_categories(train_features[feature_name], feedback_features[feature_name])
        fitted = pipeline_for(x_train, model_name, arguments.seed, arguments.n_jobs).fit(x_train, train[target].astype(float))
        y_true = feedback[target].to_numpy(dtype=float)
        y_pred = fitted.predict(x_feedback)
        frame = pd.DataFrame({"sample_id": feedback.ID.astype(str), "target": target, "model": model_name,
                              "feature_set": feature_name, "y_true": y_true, "y_pred": y_pred})
        frame["absolute_error"] = (frame.y_true - frame.y_pred).abs()
        prediction_frames.append(frame)
        mae, rmse, r2 = mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred) ** 0.5, r2_score(y_true, y_pred)
        metric_rows.append({"target": target, "model": model_name, "feature_set": feature_name, "n": len(frame),
                            "mae": mae, "rmse": rmse, "r2": r2})
        low, high = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        margin = max((high - low) * .05, 1e-6)
        axis.scatter(y_true, y_pred, s=34, alpha=.75, color="#2878b5", edgecolors="white", linewidths=.45)
        axis.plot([low - margin, high + margin], [low - margin, high + margin], "--", color="black", linewidth=1)
        axis.set(xlim=(low - margin, high + margin), ylim=(low - margin, high + margin), xlabel="Measured", ylabel="Predicted",
                 title=f"{target}\n{model_name} / {feature_name}")
        axis.text(.04, .96, f"R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}", transform=axis.transAxes, va="top",
                  bbox={"boxstyle": "round", "facecolor": "white", "alpha": .85})
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    predictions.to_csv(output / "feedback_predictions.csv", index=False)
    metrics.to_csv(output / "feedback_metrics.csv", index=False)
    figure.suptitle("Best Formula-CV Tree Models: Feedback R²", fontsize=14)
    figure.savefig(output / "feedback_r2_scatter.png", dpi=220)
    plt.close(figure)
    (output / "provenance.json").write_text(json.dumps({"train_csv": str(train_path), "train_sha256": source["dataset_sha256"],
        "feedback_csv": str(feedback_path), "feedback_sha256": sha256_file(feedback_path), "fitting": "all 700 audited rows; no feedback labels used for fitting",
        "selected_models": SELECTED}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
