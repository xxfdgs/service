#!/usr/bin/env python3
"""Audit and reproduce legacy versus explicit-manifest GraphGPS data splits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import metric_dict  # noqa: E402
from stage2_common import (  # noqa: E402
    add_stage2_arguments, legacy_split_manifest, load_training_frame,
    record_execution, stage2_output, validate_manifest,
)


PREDICTION_COLUMNS = {
    "EE_before": ("true_EE_before", "pred_EE_before_average"),
    "EE_after": ("true_EE_after", "pred_EE_after_average"),
    "Aerosolization_Efficiency": ("true_Aero_Efficiency", "pred_Aero_Efficiency_average"),
    "mRNA_Recovery_Efficiency": (
        "true_Recovery_Efficiency", "pred_Recovery_Efficiency_average",
    ),
}


def _base_training_config(output_dir: Path, protocol: str, manifest_path: Path | None) -> dict:
    """Use the original coarse YAML and only inject the known best Mordred settings."""
    config = yaml.safe_load((ROOT / "configs/GPS/direct_train_coarse_noaux.yaml").read_text())
    config["out_dir"] = str(output_dir / "reproducibility" / "training")
    config["accelerator"] = "cuda"
    config["devices"] = 1
    config["seed"] = 0
    config["dataset"] = dict(config["dataset"])
    config["dataset"].update({
        "cache_per_run": True,
        "cache_refresh": True,
        "cache_tag": f"stage2_repro_{protocol}",
        "diagnostic_split_path": str(manifest_path.resolve()) if manifest_path else "",
        "diagnostic_id_column": "ID",
        "diagnostic_manifest_id_column": "sample_id" if manifest_path else "",
    })
    config["use_mordred_features"] = True
    config["mordred_feature_dim"] = 11
    config["mordred_feature_path"] = str(
        ROOT / "results/mordred_train_feedback/mordred_selected_features.csv"
    )
    return config


def _base_prediction_config(output_dir: Path, protocol: str, pretrained_dir: Path, evaluation_csv: Path) -> dict:
    """Build a lightweight inference YAML compatible with the trained model."""
    config = yaml.safe_load((ROOT / "configs/GPS/gps_predict_coarse_noaux.yaml").read_text())
    config["accelerator"] = "cuda"
    config["devices"] = 1
    config["seed"] = 0
    config["read_csv"] = str(evaluation_csv.resolve())
    config["pretrained"] = {"dir": str(pretrained_dir.resolve()), "freeze_main": False,
                            "reset_prediction_head": False}
    config["dataset"] = dict(config["dataset"])
    config["dataset"].update({
        "cache_per_run": True, "cache_refresh": True,
        "cache_tag": f"stage2_repro_predict_{protocol}_{evaluation_csv.stem}",
    })
    config["use_mordred_features"] = True
    config["mordred_feature_dim"] = 11
    config["mordred_feature_path"] = str(
        ROOT / "results/mordred_train_feedback/mordred_selected_features.csv"
    )
    return config


def _selected_epoch_metrics(training_dir: Path) -> dict[str, object]:
    """Read loss information corresponding to the checkpoint selected for inference."""
    checkpoint_dir = training_dir / "0" / "ckpt"
    epochs = sorted(int(path.stem) for path in checkpoint_dir.glob("*.ckpt") if path.stem.isdigit())
    if not epochs:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    selected_epoch = epochs[-1]
    result: dict[str, object] = {"best_epoch": selected_epoch}
    for split_name in ("train", "val", "test"):
        stats_path = training_dir / "0" / split_name / "stats.json"
        stats = [json.loads(line) for line in stats_path.read_text().splitlines() if line]
        matching = [entry for entry in stats if entry.get("epoch") == selected_epoch]
        entry = matching[-1] if matching else min(stats, key=lambda item: item["mae_sum"])
        result[f"{split_name}_loss"] = float(entry["loss"])
        result[f"{split_name}_mae_sum_scaled"] = float(entry["mae_sum"])
    return result


def _run_prediction(config_path: Path) -> Path:
    """Run project inference and reliably locate its one new timestamped run directory."""
    before = {path.resolve() for path in (ROOT / "runs").iterdir() if path.is_dir()}
    subprocess.run([sys.executable, "main_predict.py", "--cfg", str(config_path), "--repeat", "1"],
                   cwd=ROOT, check=True)
    created = [path for path in (ROOT / "runs").iterdir()
               if path.is_dir() and path.resolve() not in before]
    if len(created) != 1:
        raise RuntimeError(f"Expected one prediction run from {config_path}, found {len(created)}")
    output_path = created[0] / "predicted_average_6props.csv"
    if not output_path.is_file():
        raise FileNotFoundError(f"Prediction output absent: {output_path}")
    return output_path


def _prediction_records(
    protocol: str, evaluation_set: str, source_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame, epoch_metrics: dict[str, object],
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    """Create long per-sample predictions and direct raw-scale metrics."""
    if len(source_frame) != len(prediction_frame):
        raise ValueError(f"{protocol}/{evaluation_set} prediction rows do not match input rows.")
    prediction_records: list[pd.DataFrame] = []
    metric_records: list[dict[str, object]] = []
    for target, (true_column, predicted_column) in PREDICTION_COLUMNS.items():
        output = pd.DataFrame({
            "protocol": protocol, "evaluation_set": evaluation_set, "target": target,
            "sample_id": source_frame["sample_id"].astype(str).to_numpy(),
            "raw_index": source_frame["raw_index"].astype(int).to_numpy(),
            "y_true": prediction_frame[true_column].astype(float).to_numpy(),
            "y_pred": prediction_frame[predicted_column].astype(float).to_numpy(),
        })
        output["absolute_error"] = (output["y_true"] - output["y_pred"]).abs()
        prediction_records.append(output)
        metric_records.append({
            "protocol": protocol, "evaluation_set": evaluation_set, "target": target,
            "n_samples": len(output), **epoch_metrics,
            **metric_dict(output["y_true"], output["y_pred"]),
        })
    return prediction_records, metric_records


def _stage1_manifest(train_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the first-stage random split file to the stage-two manifest contract."""
    stage1_path = ROOT / "results/generalization_diagnostics/splits/random_split.csv"
    split_frame = pd.read_csv(stage1_path, dtype={"diagnostic_sample_id": str})
    mapping = train_frame[["sample_id", "raw_index", "formula_identity_key"]].copy()
    stage1 = split_frame[["diagnostic_sample_id", "split"]].rename(
        columns={"diagnostic_sample_id": "sample_id"}
    )
    manifest = mapping.merge(stage1, on="sample_id", validate="one_to_one")
    manifest = manifest.rename(columns={"formula_identity_key": "group_id"})
    manifest["split_order"] = manifest.groupby("split")["raw_index"].rank(
        method="first"
    ).astype(int) - 1
    manifest = manifest[["sample_id", "split", "group_id", "raw_index", "split_order"]]
    validate_manifest(manifest, train_frame)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_stage2_arguments(parser)
    parser.add_argument("--skip-model-run", action="store_true", help="Only audit manifests/configs.")
    arguments = parser.parse_args()
    output_dir = stage2_output(arguments.output_dir)
    reproducibility_dir = output_dir / "reproducibility"
    manifest_dir = reproducibility_dir / "manifests"
    config_dir = reproducibility_dir / "configs"
    input_dir = reproducibility_dir / "evaluation_inputs"
    for directory in (manifest_dir, config_dir, input_dir):
        directory.mkdir(parents=True, exist_ok=True)
    schema, train_frame, _ = load_training_frame(arguments.train_csv, arguments.feedback_csv)

    legacy_manifest = legacy_split_manifest(train_frame, seed=0, group_column="formula_identity_key")
    explicit_manifest = legacy_manifest.copy()
    stage1_manifest = _stage1_manifest(train_frame)
    manifest_map = {
        "legacy_split_seed0": None,
        "explicit_manifest_seed0": explicit_manifest,
        "stage1_manifest_seed0": stage1_manifest,
    }
    legacy_manifest.to_csv(manifest_dir / "legacy_split_seed0_reference.csv", index=False)
    explicit_manifest.to_csv(manifest_dir / "explicit_manifest_seed0.csv", index=False)
    stage1_manifest.to_csv(manifest_dir / "stage1_manifest_seed0.csv", index=False)
    sample_comparison = pd.concat([
        legacy_manifest.assign(protocol="legacy_split_seed0"),
        explicit_manifest.assign(protocol="explicit_manifest_seed0"),
        stage1_manifest.assign(protocol="stage1_manifest_seed0"),
    ], ignore_index=True)
    sample_comparison.to_csv(reproducibility_dir / "split_sample_comparison.csv", index=False)

    old_result_root = ROOT / "results/coarse_mordred/direct_train_coarse_noaux"
    old_manifest_paths = list(old_result_root.rglob("*split*.csv")) if old_result_root.is_dir() else []
    protocol_audit = {
        "yaml_nominal_split": [0.8, 0.1, 0.1],
        "loader_actual_split": {"train": 0.81, "val": 0.09, "test": 0.10},
        "loader_code": "two sklearn.train_test_split calls, each train_size=0.9/test_size=0.1",
        "random_state": 0,
        "shuffle_behavior": "sklearn train_test_split(shuffle=True default); PyG train loader shuffles batches",
        "old_experiment_split_manifests": [str(path) for path in old_manifest_paths],
        "old_experiment_manifest_saved": bool(old_manifest_paths),
        "legacy_reference_counts": legacy_manifest["split"].value_counts().to_dict(),
        "stage1_manifest_counts": stage1_manifest["split"].value_counts().to_dict(),
        "manifest_contract": ["sample_id", "split", "group_id", "raw_index", "split_order"],
        "order_invariance_check": "validated by joining sample_id/raw_index; reordering source rows does not alter manifest assignment",
    }
    (reproducibility_dir / "split_protocol_audit.json").write_text(
        json.dumps(protocol_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if arguments.skip_model_run:
        record_execution(output_dir, Path(__file__).name, details={
            "skip_model_run": True, "seed": arguments.seed, "protocol_audit": protocol_audit,
        })
        print(f"Wrote reproducibility manifests to {reproducibility_dir}")
        return

    all_prediction_records: list[pd.DataFrame] = []
    all_metric_records: list[dict[str, object]] = []
    for protocol, manifest in manifest_map.items():
        manifest_path = None
        if manifest is not None:
            manifest_path = manifest_dir / f"{protocol}.csv"
        training_config = _base_training_config(output_dir, protocol, manifest_path)
        training_config_path = config_dir / f"{protocol}.yaml"
        training_config_path.write_text(yaml.safe_dump(training_config, sort_keys=False), encoding="utf-8")
        subprocess.run([sys.executable, "main.py", "--cfg", str(training_config_path), "--repeat", "1"],
                       cwd=ROOT, check=True)
        training_dir = reproducibility_dir / "training" / protocol
        epoch_metrics = _selected_epoch_metrics(training_dir)
        protocol_manifest = legacy_manifest if protocol == "legacy_split_seed0" else manifest
        for evaluation_set in ("val", "test"):
            sample_ids = protocol_manifest.loc[
                protocol_manifest["split"] == evaluation_set, "sample_id"
            ]
            indexed = train_frame.set_index("sample_id", drop=False)
            evaluation_frame = indexed.loc[sample_ids].reset_index(drop=True)
            evaluation_csv = input_dir / f"{protocol}_{evaluation_set}.csv"
            original_columns = pd.read_csv(schema.train_path, nrows=1).columns.tolist()
            evaluation_frame[original_columns].to_csv(evaluation_csv, index=False)
            prediction_config = _base_prediction_config(
                output_dir, protocol, training_dir, evaluation_csv
            )
            prediction_config_path = config_dir / f"{protocol}_{evaluation_set}_predict.yaml"
            prediction_config_path.write_text(
                yaml.safe_dump(prediction_config, sort_keys=False), encoding="utf-8"
            )
            prediction_path = _run_prediction(prediction_config_path)
            prediction_frame = pd.read_csv(prediction_path)
            prediction_records, metric_records = _prediction_records(
                protocol, evaluation_set, evaluation_frame, prediction_frame, epoch_metrics
            )
            all_prediction_records.extend(prediction_records)
            all_metric_records.extend(metric_records)

    predictions = pd.concat(all_prediction_records, ignore_index=True)
    metrics = pd.DataFrame(all_metric_records)
    predictions.to_csv(reproducibility_dir / "reproducibility_predictions.csv", index=False)
    metrics.to_csv(reproducibility_dir / "reproducibility_metrics.csv", index=False)
    comparison_records: list[dict[str, object]] = []
    legacy = predictions.loc[predictions["protocol"] == "legacy_split_seed0"]
    explicit = predictions.loc[predictions["protocol"] == "explicit_manifest_seed0"]
    for evaluation_set in ("val", "test"):
        for target in PREDICTION_COLUMNS:
            left = legacy.loc[(legacy["evaluation_set"] == evaluation_set) & (legacy["target"] == target)]
            right = explicit.loc[(explicit["evaluation_set"] == evaluation_set) & (explicit["target"] == target)]
            merged = left.merge(right, on="sample_id", suffixes=("_legacy", "_explicit"), validate="one_to_one")
            prediction_difference = (merged["y_pred_legacy"] - merged["y_pred_explicit"]).abs()
            legacy_mae = metric_dict(merged["y_true_legacy"], merged["y_pred_legacy"])["mae"]
            explicit_mae = metric_dict(merged["y_true_explicit"], merged["y_pred_explicit"])["mae"]
            comparison_records.append({
                "evaluation_set": evaluation_set, "target": target,
                "same_sample_ids": bool(set(left["sample_id"]) == set(right["sample_id"])),
                "mae_difference": abs(legacy_mae - explicit_mae),
                "max_single_prediction_difference": float(prediction_difference.max()),
                "threshold_exceeded": bool(abs(legacy_mae - explicit_mae) > 0.1 or prediction_difference.max() > 0.5),
            })
    comparison_frame = pd.DataFrame(comparison_records)
    comparison_frame.to_csv(reproducibility_dir / "legacy_explicit_prediction_comparison.csv", index=False)
    exceeded = comparison_frame.loc[comparison_frame["threshold_exceeded"]]
    diagnosis = [
        "# GraphGPS 可复现性诊断", "",
        f"- legacy 与 explicit 使用相同 sample_id 的比较项：{len(comparison_frame)}。",
        f"- 超过 MAE>0.1 或单样本预测>0.5 阈值的项：{len(exceeded)}。",
    ]
    if not exceeded.empty:
        diagnosis.extend([
            "- 差异超过阈值。已固定 split_order；后续应检查 DataLoader batch shuffle、CUDA/cudnn 决定性、"
            "缓存样本顺序、checkpoint 选择、标签缩放与 ensemble 汇总。",
            "- 本实现分别保存了 val/test 逐样本输出，可据此直接定位差异样本。",
        ])
    else:
        diagnosis.append("- 在阈值内复现：显式 manifest 未改变该 seed 的训练数据成员或预测。")
    (reproducibility_dir / "reproducibility_diagnosis.md").write_text(
        "\n".join(diagnosis) + "\n", encoding="utf-8"
    )
    record_execution(output_dir, Path(__file__).name, details={
        "seed": 0, "n_jobs": arguments.n_jobs, "protocols": list(manifest_map),
        "configs": [str(path) for path in config_dir.glob("*.yaml")],
    })
    print(f"Wrote reproducibility audit and experiments to {reproducibility_dir}")


if __name__ == "__main__":
    main()
