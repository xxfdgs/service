#!/usr/bin/env python3
"""Run resumable deterministic GraphGPS outer-CV with sample-id ensembles."""

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

from common import TARGET_COLUMNS, metric_dict
from stage3_utils import append_execution, read_best_checkpoint, sha256_file


PROTOCOLS = ("fifth_component_group_cv", "formula_identity_group_cv")


def parse_list(value: str, valid: tuple[str, ...] | None = None) -> list[str]:
    """Parse a comma-separated CLI list and optionally validate every member."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if valid and any(item not in valid for item in items):
        raise ValueError(f"Unsupported value in {items}; expected one of {valid}.")
    return items


def make_config(output_dir: Path, manifest: Path, protocol: str, fold: str, seed: int,
                data_version: str) -> Path:
    """Materialize the unmodified coarse+11-Mordred model under deterministic controls."""
    config = yaml.safe_load((ROOT / "configs/GPS/direct_train_coarse_noaux.yaml").read_text(encoding="utf-8"))
    config["out_dir"] = str((output_dir / "graphgps_raw_cv" / "training").resolve())
    config.update({"accelerator": "cuda", "devices": 1, "gpu_serial": 0, "seed": seed,
                   "use_mordred_features": True, "mordred_feature_dim": 11,
                   "mordred_feature_path": str(ROOT / "results/mordred_train_feedback/mordred_selected_features.csv")})
    config["dataset"] = dict(config["dataset"])
    config["dataset"].update({
        "diagnostic_split_path": str(manifest.resolve()), "diagnostic_id_column": "ID",
        "diagnostic_manifest_id_column": "sample_id", "cache_per_run": True,
        "cache_refresh": True, "cache_tag": f"stage3_{data_version}_{protocol}_{fold}",
    })
    config["train"] = dict(config["train"])
    config["train"].update({"deterministic": True, "manifest_path": str(manifest.resolve()),
                            "fold": fold, "protocol": protocol})
    config_dir = output_dir / "graphgps_raw_cv" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{protocol}_{data_version}_{fold}_seed_{seed}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_dir_for(config_path: Path, output_dir: Path, seed: int) -> Path:
    """Match main.py's config-stem output layout exactly."""
    return output_dir / "graphgps_raw_cv" / "training" / config_path.stem / str(seed)


def checkpoint_is_reusable(checkpoint: Path, manifest: Path, seed: int) -> bool:
    """Allow only stage-three checkpoints with complete manifest provenance."""
    if not checkpoint.is_file():
        return False
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return bool(payload.get("stage3_checkpoint_metadata")) and payload.get("seed") == seed \
        and payload.get("manifest_hash") == sha256_file(manifest) and bool(payload.get("sample_id_hash")) \
        and bool(payload.get("feature_hash")) and bool(payload.get("config_hash")) \
        and payload.get("target_scaler", {}).get("scale") == 100.0


def prediction_path(output_dir: Path, protocol: str, fold: str, seed: int, split: str) -> Path:
    """Return a non-ambiguous stage-three per-seed prediction path."""
    return output_dir / "graphgps_raw_cv" / "seed_predictions" / protocol / f"{fold}_seed_{seed}_{split}.csv"


def run_fold_seed(output_dir: Path, protocol: str, fold: str, seed: int,
                  data_version: str) -> dict[str, object]:
    """Train (if needed), reload, and export one fold/seed's validation and test predictions."""
    manifest = output_dir / "manifests" / protocol / data_version / f"{fold}.csv"
    config = make_config(output_dir, manifest, protocol, fold, seed, data_version)
    run_dir = run_dir_for(config, output_dir, seed)
    log_path = output_dir / "graphgps_raw_cv" / "logs" / protocol / f"{fold}_seed_{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: Path | None = None
    try:
        if run_dir.exists():
            try:
                candidate = read_best_checkpoint(run_dir)
                checkpoint = candidate if checkpoint_is_reusable(candidate, manifest, seed) else None
            except RuntimeError:
                checkpoint = None
        if checkpoint is None:
            command = [sys.executable, "main.py", "--cfg", str(config), "--repeat", "1"]
            with log_path.open("w", encoding="utf-8") as handle:
                subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            checkpoint = read_best_checkpoint(run_dir)
            if not checkpoint_is_reusable(checkpoint, manifest, seed):
                raise RuntimeError(f"Checkpoint metadata audit failed: {checkpoint}")
            append_execution(output_dir, command=command, protocol=protocol, fold=fold, seed=seed,
                             data_version=data_version, manifest_path=manifest, config_path=config,
                             checkpoint=checkpoint, output=run_dir)
        for split in ("val", "test"):
            output = prediction_path(output_dir, protocol, fold, seed, split)
            if output.is_file():
                continue
            command = [sys.executable, "scripts/diagnostics/stage3_export_predictions.py", "--config", str(config),
                       "--checkpoint", str(checkpoint), "--manifest", str(manifest), "--output", str(output),
                       "--split", split, "--seed", str(seed), "--fold", fold, "--protocol", protocol]
            with log_path.open("a", encoding="utf-8") as handle:
                subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            append_execution(output_dir, command=command, protocol=protocol, fold=fold, seed=seed,
                             data_version=data_version, manifest_path=manifest, config_path=config,
                             checkpoint=checkpoint, output=output)
        return {"protocol": protocol, "fold": fold, "seed": seed, "data_version": data_version,
                "manifest": str(manifest), "manifest_hash": sha256_file(manifest), "config": str(config),
                "config_hash": sha256_file(config), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                "status": "completed", "error_message": ""}
    except Exception as error:
        append_execution(output_dir, command=[sys.executable, *sys.argv], protocol=protocol, fold=fold, seed=seed,
                         data_version=data_version, manifest_path=manifest, config_path=config,
                         checkpoint=checkpoint, output=log_path, status="failed", error_message=repr(error))
        return {"protocol": protocol, "fold": fold, "seed": seed, "data_version": data_version,
                "manifest": str(manifest), "manifest_hash": sha256_file(manifest), "config": str(config),
                "config_hash": sha256_file(config), "checkpoint": str(checkpoint or ""), "checkpoint_sha256": "",
                "status": "failed", "error_message": repr(error)}


def update_aggregates(output_dir: Path, protocols: list[str], data_version: str) -> None:
    """Build strict three-seed fold ensembles and complete pooled OOF artifacts when available."""
    base = output_dir / "graphgps_raw_cv"
    seed_metrics: list[dict[str, object]] = []
    ensemble_metrics: list[dict[str, object]] = []
    ensemble_frames: list[pd.DataFrame] = []
    for protocol in protocols:
        for fold in [f"fold_{index}" for index in range(5)]:
            frames: list[pd.DataFrame] = []
            for seed in (0, 1, 2):
                path = prediction_path(output_dir, protocol, fold, seed, "test")
                if not path.is_file():
                    continue
                frame = pd.read_csv(path, dtype={"sample_id": str})
                if frame.duplicated(["sample_id", "target"]).any():
                    raise ValueError(f"Duplicate prediction key: {path}")
                for target, group in frame.groupby("target"):
                    seed_metrics.append({"protocol": protocol, "fold": fold, "seed": seed, "target": target,
                                         "n_test": len(group), **metric_dict(group.y_true, group.y_pred)})
                frames.append(frame)
            if len(frames) != 3:
                continue
            merged = frames[0][["sample_id", "split", "target", "y_true", "y_pred"]].rename(columns={"y_pred": "y_pred_0"})
            for seed, frame in enumerate(frames[1:], start=1):
                merged = merged.merge(frame[["sample_id", "target", "y_true", "y_pred"]].rename(
                    columns={"y_true": f"y_true_{seed}", "y_pred": f"y_pred_{seed}"}),
                    on=["sample_id", "target"], how="inner", validate="one_to_one")
                if not np.allclose(merged["y_true"], merged[f"y_true_{seed}"], atol=1e-6, rtol=0):
                    raise ValueError(f"Seed label mismatch for {protocol}/{fold}")
            merged["y_pred"] = merged[["y_pred_0", "y_pred_1", "y_pred_2"]].mean(axis=1)
            merged["seed"] = "ensemble"
            merged["fold"] = fold
            merged["protocol"] = protocol
            merged["data_version"] = data_version
            merged["absolute_error"] = (merged.y_true - merged.y_pred).abs()
            output = base / "fold_ensemble_predictions" / protocol / f"{fold}.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            merged[["sample_id", "split", "target", "y_true", "y_pred", "seed", "fold", "protocol", "data_version", "absolute_error"]].to_csv(output, index=False)
            ensemble_frames.append(merged)
            for target, group in merged.groupby("target"):
                ensemble_metrics.append({"protocol": protocol, "fold": fold, "target": target,
                                         "n_test": len(group), **metric_dict(group.y_true, group.y_pred)})
    pd.DataFrame(seed_metrics).to_csv(base / "fold_seed_metrics.csv", index=False)
    pd.DataFrame(ensemble_metrics).to_csv(base / "fold_ensemble_metrics.csv", index=False)
    if ensemble_frames:
        pooled = pd.concat(ensemble_frames, ignore_index=True)
        pooled.to_csv(base / "pooled_oof_predictions.csv", index=False)
        pooled_rows: list[dict[str, object]] = []
        for (protocol, target), group in pooled.groupby(["protocol", "target"]):
            pooled_rows.append({"protocol": protocol, "target": target, "n": len(group),
                                **metric_dict(group.y_true, group.y_pred)})
        pd.DataFrame(pooled_rows).to_csv(base / "pooled_oof_metrics.csv", index=False)
        summary = pd.DataFrame(ensemble_metrics).groupby(["protocol", "target"], as_index=False).agg(
            completed_folds=("fold", "nunique"), mean_mae=("mae", "mean"), std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"), mean_spearman=("spearman", "mean"),
        )
        summary.to_csv(base / "summary_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generalization_stage3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds; run 0 before 1,2.")
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument("--data-version", choices=["raw_records", "replicate_median"], default="raw_records")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    protocols = parse_list(arguments.protocols, PROTOCOLS)
    seeds = [int(value) for value in parse_list(arguments.seeds)]
    inventory_rows: list[dict[str, object]] = []
    for seed in seeds:
        for protocol in protocols:
            for fold_index in range(5):
                inventory_rows.append(run_fold_seed(output_dir, protocol, f"fold_{fold_index}", seed, arguments.data_version))
    base = output_dir / "graphgps_raw_cv"
    base.mkdir(parents=True, exist_ok=True)
    inventory_path = base / "run_inventory.csv"
    previous = pd.read_csv(inventory_path) if inventory_path.is_file() else pd.DataFrame()
    updated = pd.concat([previous, pd.DataFrame(inventory_rows)], ignore_index=True)
    updated = updated.drop_duplicates(["protocol", "fold", "seed", "data_version"], keep="last")
    updated.to_csv(inventory_path, index=False)
    pd.DataFrame([row for row in inventory_rows if row["status"] == "failed"]).to_csv(base / "failed_tasks.csv", index=False)
    update_aggregates(output_dir, protocols, arguments.data_version)
    print(f"Updated {base}")


if __name__ == "__main__":
    main()
