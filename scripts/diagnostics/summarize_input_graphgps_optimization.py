#!/usr/bin/env python3
"""Summarize completed input-only GraphGPS optimization experiments.

Every completed experiment is represented by its validation-selected checkpoint.
The script never opens a feedback file and refuses to summarize a run whose
effective configuration or provenance mentions feedback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGETS = [
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
]
SPLITS = ["train", "val", "test"]


def correlation(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.std(truth) == 0 or np.std(prediction) == 0:
        return float("nan")
    return float(function(truth, prediction).statistic)


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame.y_true.to_numpy(float)
    prediction = frame.y_pred.to_numpy(float)
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)) if len(truth) > 1 and np.std(truth) else float("nan"),
        "pearson": correlation(pearsonr, truth, prediction),
        "spearman": correlation(spearmanr, truth, prediction),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    experiment_root = root / "experiments"
    rows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for run_dir in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
        summary_path = run_dir / "summary.json"
        prediction_path = run_dir / "predictions.csv"
        settings_path = run_dir / "run_settings.json"
        if not (summary_path.is_file() and prediction_path.is_file() and settings_path.is_file()):
            inventory.append({"experiment": run_dir.name, "status": "incomplete"})
            continue
        text = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                         for path in (summary_path, settings_path, run_dir / "effective_config.yaml")
                         if path.is_file()).lower()
        if "feedback" in text:
            raise RuntimeError(f"Input-only guard rejected feedback reference in {run_dir}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        prediction = pd.read_csv(prediction_path)
        if set(prediction.split) != set(SPLITS) or set(prediction.target) != set(TARGETS):
            raise RuntimeError(f"Incomplete split/target predictions in {run_dir}")
        if prediction.duplicated(["sample_id", "split", "target"]).any():
            raise RuntimeError(f"Duplicate sample/split/target predictions in {run_dir}")
        for (split, target), group in prediction.groupby(["split", "target"], sort=False):
            rows.append({
                "experiment": run_dir.name,
                "architecture": summary["architecture_name"],
                "candidate": summary["candidate"],
                "fusion_type": summary["fusion_type"],
                "head_type": summary["head_type"],
                "model_type": settings.get("model_type"),
                "loss_targets": ",".join(summary.get("loss_targets", TARGETS)),
                "use_mordred_features": bool(settings.get("use_mordred_features", False)),
                "use_component_aux_features": bool(settings.get("use_component_aux_features", False)),
                "use_fifth_identity_embedding": bool(settings.get("use_fifth_identity_embedding", False)),
                "use_fifth_ratio_modulation": bool(settings.get("use_fifth_ratio_modulation", False)),
                "best_epoch": int(summary["best_epoch"]),
                "best_validation_loss_normalized": float(summary["best_validation_loss_normalized"]),
                "last_epoch": int(summary["last_epoch"]),
                "split": split,
                "target": target,
                "n": len(group),
                **metrics(group),
            })
        inventory.append({
            "experiment": run_dir.name, "status": "completed",
            "best_epoch": int(summary["best_epoch"]),
            "last_epoch": int(summary["last_epoch"]),
            "architecture": summary["architecture_name"],
            "candidate": summary["candidate"],
            "fusion_type": summary["fusion_type"], "head_type": summary["head_type"],
            "model_type": settings.get("model_type"),
            "loss_targets": ",".join(summary.get("loss_targets", TARGETS)),
            "use_mordred_features": bool(settings.get("use_mordred_features", False)),
            "use_component_aux_features": bool(settings.get("use_component_aux_features", False)),
            "use_fifth_identity_embedding": bool(settings.get("use_fifth_identity_embedding", False)),
            "use_fifth_ratio_modulation": bool(settings.get("use_fifth_ratio_modulation", False)),
            "base_lr": settings.get("base_lr"), "weight_decay": settings.get("weight_decay"),
            "batch_size": settings.get("batch_size"), "head_hidden_dim": settings.get("head_hidden_dim"),
            "head_dropout": settings.get("head_dropout"),
        })
    result = pd.DataFrame(rows)
    result.to_csv(root / "best_epoch_metrics.csv", index=False)
    pd.DataFrame(inventory).to_csv(root / "experiment_inventory.csv", index=False)
    if not result.empty:
        summary = result.groupby(["experiment", "architecture", "candidate", "fusion_type", "head_type", "model_type", "loss_targets", "use_mordred_features", "use_component_aux_features", "use_fifth_identity_embedding", "use_fifth_ratio_modulation", "best_epoch", "split"], as_index=False).agg(
            n_per_target=("n", "first"), mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"),
            mean_r2=("r2", "mean"), mean_pearson=("pearson", "mean"), mean_spearman=("spearman", "mean"),
        )
        summary.to_csv(root / "best_epoch_metrics_summary.csv", index=False)
    print(json.dumps({"completed_experiments": int(sum(item["status"] == "completed" for item in inventory)),
                      "incomplete_experiments": int(sum(item["status"] == "incomplete" for item in inventory))}))


if __name__ == "__main__":
    main()
