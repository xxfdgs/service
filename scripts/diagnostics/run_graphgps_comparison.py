#!/usr/bin/env python3
"""Evaluate the existing coarse+Mordred GraphGPS and prepare group retraining.

The script always records the already-completed external-feedback ensemble. When
CUDA is available, ``--retrain`` additionally trains three fresh seeds for each
explicit split manifest, preserving the network while replacing only the loader
split. Without CUDA it writes an explicit blocked status instead of presenting
post-hoc predictions as held-out results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import add_common_arguments, discover_schema, load_frames, metric_dict  # noqa: E402


BEST_TRAIN_DIR = ROOT / "results/coarse_mordred/direct_train_coarse_noaux"
BEST_PREDICTION_RUN = ROOT / "runs/20260710_232954_163135_gps_predict_coarse_noaux"
PREDICTION_COLUMNS = {
    "EE_before": ("true_EE_before", "pred_EE_before_average"),
    "EE_after": ("true_EE_after", "pred_EE_after_average"),
    "Aerosolization_Efficiency": (
        "true_Aero_Efficiency", "pred_Aero_Efficiency_average",
    ),
    "mRNA_Recovery_Efficiency": (
        "true_Recovery_Efficiency", "pred_Recovery_Efficiency_average",
    ),
}


def _prediction_rows(
    split_name: str, evaluation_set: str, source: str, sample_ids: pd.Series,
    prediction_frame: pd.DataFrame,
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    """Convert the project-wide prediction CSV into standard long diagnostics."""
    rows: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    for target, (true_column, prediction_column) in PREDICTION_COLUMNS.items():
        true_values = prediction_frame[true_column].astype(float)
        predictions = prediction_frame[prediction_column].astype(float)
        output = pd.DataFrame({
            "split_name": split_name,
            "evaluation_set": evaluation_set,
            "target": target,
            "model": "GraphGPS_coarse_mordred",
            "source": source,
            "diagnostic_sample_id": sample_ids.astype(str).to_numpy(),
            "y_true": true_values,
            "y_pred": predictions,
        })
        output["absolute_error"] = (output["y_true"] - output["y_pred"]).abs()
        rows.append(output)
        metrics.append({
            "split_name": split_name,
            "evaluation_set": evaluation_set,
            "target": target,
            "model": "GraphGPS_coarse_mordred",
            "source": source,
            "n_samples": int(len(output)),
            **metric_dict(true_values, predictions),
            "status": "ok",
        })
    return rows, metrics


def _best_logged_random_metrics() -> list[dict[str, object]]:
    """Extract independent random-test MAE values at the selected checkpoint epoch."""
    property_names = {
        "EE_before": "EE_before_mae",
        "EE_after": "EE_after_mae",
        "Aerosolization_Efficiency": "Aero_Efficiency_mae",
        "mRNA_Recovery_Efficiency": "Recovery_Efficiency_mae",
    }
    records: list[dict[str, object]] = []
    for seed in range(3):
        checkpoint_dir = BEST_TRAIN_DIR / str(seed) / "ckpt"
        test_stats_path = BEST_TRAIN_DIR / str(seed) / "test/stats.json"
        if not checkpoint_dir.is_dir() or not test_stats_path.is_file():
            continue
        checkpoint_epochs = sorted(
            int(path.stem) for path in checkpoint_dir.glob("*.ckpt") if path.stem.isdigit()
        )
        if not checkpoint_epochs:
            continue
        selected_epoch = checkpoint_epochs[-1]
        stats = [json.loads(line) for line in test_stats_path.read_text().splitlines() if line]
        matching = [entry for entry in stats if entry.get("epoch") == selected_epoch]
        selected = matching[-1] if matching else min(stats, key=lambda entry: entry["mae_sum"])
        for target, metric_name in property_names.items():
            # GraphGPS logger evaluates targets scaled to [0, 1]; convert MAE
            # back to the source CSV's percentage scale for comparability.
            records.append({
                "split_name": "random_split",
                "evaluation_set": "test",
                "target": target,
                "model": "GraphGPS_coarse_mordred",
                "source": f"existing_seed_{seed}_heldout_log",
                "seed": seed,
                "n_samples": np.nan,
                "mae": float(selected["mae_per_property"][metric_name]) * 100.0,
                "rmse": np.nan,
                "r2": np.nan,
                "median_absolute_error": np.nan,
                "pearson": np.nan,
                "spearman": np.nan,
                "status": "ok_logged_heldout",
            })
    return records


def _write_retraining_configs(output_dir: Path, max_epochs: int) -> list[Path]:
    """Create one diagnostic config per manifest without mutating model configs."""
    base_config_path = ROOT / "configs/GPS/direct_train_coarse_noaux.yaml"
    base_config = yaml.safe_load(base_config_path.read_text())
    config_dir = output_dir / "graphgps_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    requested_splits = {
        "random_split", "fifth_component_group_split",
        "formula_identity_group_split", "feedback_like_split",
    }
    for split_path in sorted((output_dir / "splits").glob("*_split.csv")):
        split_name = split_path.stem
        if split_name not in requested_splits:
            continue
        config = dict(base_config)
        config["out_dir"] = str(output_dir / "graphgps_retrained")
        config["accelerator"] = "cuda"
        config["devices"] = 1
        config["dataset"] = dict(base_config["dataset"])
        config["dataset"].update({
            "diagnostic_split_path": str(split_path.resolve()),
            "diagnostic_id_column": "ID",
            "cache_tag": f"diagnostic_{split_name}",
            "cache_per_run": True,
            "cache_refresh": True,
        })
        config["optim"] = dict(base_config["optim"])
        config["optim"]["max_epoch"] = max_epochs
        config["optim"]["num_warmup_epochs"] = min(20, max(1, max_epochs // 5))
        config["train"] = dict(base_config["train"])
        config["train"].update({"early_stop_patience": min(30, max_epochs // 3)})
        # These settings identify the established best coarse+Mordred model;
        # the original training YAML predates the Mordred integration.
        config["use_mordred_features"] = True
        config["mordred_feature_dim"] = 11
        config["mordred_feature_path"] = str(
            ROOT / "results/mordred_train_feedback/mordred_selected_features.csv"
        )
        config_path = config_dir / f"{split_name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths.append(config_path)
    return paths


def _write_evaluation_input(
    split_name: str, output_dir: Path, train_frame: pd.DataFrame
) -> tuple[str, pd.DataFrame]:
    """Persist one generated evaluation CSV while leaving source data unchanged."""
    split_frame = pd.read_csv(output_dir / "splits" / f"{split_name}.csv")
    evaluation_label = "val" if split_name == "random_split" else "test"
    selected_ids = split_frame.loc[
        split_frame["split"] == evaluation_label, "diagnostic_sample_id"
    ].astype(str)
    indexed_frame = train_frame.set_index("diagnostic_sample_id", drop=False)
    evaluation_frame = indexed_frame.loc[selected_ids].reset_index(drop=True)
    input_dir = output_dir / "graphgps_evaluation_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / f"{split_name}_{evaluation_label}.csv"
    evaluation_frame.drop(columns=[
        column for column in evaluation_frame.columns
        if column.startswith("component_") or column.startswith("formula_") or
        column in {"fifth_component_key", "diagnostic_sample_id"}
    ]).to_csv(input_path, index=False)
    return evaluation_label, evaluation_frame


def _run_retraining(
    config_paths: list[Path], output_dir: Path, train_frame: pd.DataFrame, seeds: int,
) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[dict[str, object]]]:
    """Train explicit splits, then generate held-out per-sample GraphGPS outputs."""
    status_records: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    base_prediction_config = yaml.safe_load(
        (ROOT / "configs/GPS/gps_predict_coarse_noaux.yaml").read_text()
    )
    for config_path in config_paths:
        command = [sys.executable, "main.py", "--cfg", str(config_path), "--repeat", str(seeds)]
        subprocess.run(command, cwd=ROOT, check=True)
        split_name = config_path.stem
        train_dir = output_dir / "graphgps_retrained" / split_name
        if not train_dir.is_dir():
            raise FileNotFoundError(f"Expected trained GraphGPS directory: {train_dir}")
        evaluation_label, evaluation_frame = _write_evaluation_input(
            split_name, output_dir, train_frame
        )
        prediction_config = dict(base_prediction_config)
        prediction_config["read_csv"] = str(
            (output_dir / "graphgps_evaluation_inputs" /
             f"{split_name}_{evaluation_label}.csv").resolve()
        )
        prediction_config["pretrained"] = dict(base_prediction_config["pretrained"])
        prediction_config["pretrained"].update({"dir": str(train_dir), "reset_prediction_head": False})
        prediction_config["dataset"] = dict(base_prediction_config["dataset"])
        prediction_config["dataset"].update({
            "cache_tag": f"diagnostic_predict_{split_name}",
            "cache_per_run": True,
            "cache_refresh": True,
        })
        prediction_config["use_mordred_features"] = True
        prediction_config["mordred_feature_dim"] = 11
        prediction_config["mordred_feature_path"] = str(
            ROOT / "results/mordred_train_feedback/mordred_selected_features.csv"
        )
        prediction_config_path = output_dir / "graphgps_configs" / f"{split_name}_predict.yaml"
        prediction_config_path.write_text(
            yaml.safe_dump(prediction_config, sort_keys=False), encoding="utf-8"
        )
        before_runs = {path.resolve() for path in (ROOT / "runs").iterdir() if path.is_dir()}
        prediction_command = [
            sys.executable, "main_predict.py", "--cfg", str(prediction_config_path),
            "--repeat", str(seeds),
        ]
        subprocess.run(prediction_command, cwd=ROOT, check=True)
        new_runs = [
            path for path in (ROOT / "runs").iterdir()
            if path.is_dir() and path.resolve() not in before_runs
        ]
        if len(new_runs) != 1:
            raise RuntimeError(
                f"Expected one new prediction run for {split_name}, found {len(new_runs)}."
            )
        prediction_path = new_runs[0] / "predicted_average_6props.csv"
        prediction_frame = pd.read_csv(prediction_path)
        if len(prediction_frame) != len(evaluation_frame):
            raise ValueError(
                f"Prediction row count mismatch for {split_name}: {len(prediction_frame)} != "
                f"{len(evaluation_frame)}"
            )
        rows, metrics = _prediction_rows(
            split_name, evaluation_label, "retrained_3_seed_explicit_split",
            evaluation_frame["diagnostic_sample_id"], prediction_frame,
        )
        prediction_rows.extend(rows)
        metric_rows.extend(metrics)
        status_records.append({
            "split_name": split_name,
            "evaluation_set": evaluation_label,
            "target": "all",
            "model": "GraphGPS_coarse_mordred",
            "source": "retrained_explicit_manifest",
            "status": "ok",
        })
    return status_records, prediction_rows, metric_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--retrain", action="store_true", help="Run three-seed explicit-split training when CUDA is available.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=150)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = discover_schema(arguments.train_csv, arguments.feedback_csv)
    train_frame, feedback_frame = load_frames(schema)
    existing_prediction_path = BEST_PREDICTION_RUN / "predicted_average_6props.csv"
    if not existing_prediction_path.is_file():
        raise FileNotFoundError(f"Existing GraphGPS feedback output not found: {existing_prediction_path}")
    existing_predictions = pd.read_csv(existing_prediction_path)
    if len(existing_predictions) != len(feedback_frame):
        raise ValueError("Existing GraphGPS feedback output does not match feedback CSV row count.")
    prediction_rows, metric_rows = _prediction_rows(
        "existing_external_feedback", "feedback", "existing_10_seed_ensemble",
        feedback_frame["diagnostic_sample_id"], existing_predictions,
    )
    metric_rows.extend(_best_logged_random_metrics())
    config_paths = _write_retraining_configs(output_dir, arguments.max_epochs)
    cuda_available = bool(torch.cuda.is_available())
    if arguments.retrain and cuda_available:
        statuses, retrained_predictions, retrained_metrics = _run_retraining(
            config_paths, output_dir, train_frame, arguments.seeds
        )
        metric_rows.extend(statuses)
        metric_rows.extend(retrained_metrics)
        prediction_rows.extend(retrained_predictions)
    else:
        status = "blocked_no_cuda" if arguments.retrain else "configs_prepared_not_requested"
        for config_path in config_paths:
            metric_rows.append({
                "split_name": config_path.stem,
                "evaluation_set": "test",
                "target": "all",
                "model": "GraphGPS_coarse_mordred",
                "source": "explicit_manifest_retraining",
                "status": status,
                "detail": "CUDA is not available in the active environment; no CPU fallback is run for a 12-model diagnostic." if not cuda_available else "Use --retrain to execute.",
            })
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output_dir / "graphgps_predictions.csv", index=False
    )
    pd.DataFrame(metric_rows).to_csv(output_dir / "graphgps_comparison.csv", index=False)
    print(f"CUDA available: {cuda_available}")
    print(f"Prepared {len(config_paths)} explicit-split GraphGPS configs.")
    print(f"Wrote {output_dir / 'graphgps_comparison.csv'}")


if __name__ == "__main__":
    main()
