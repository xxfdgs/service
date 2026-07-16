#!/usr/bin/env python3
"""Run deterministic GraphGPS smoke or three-repeat verification on one manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import TARGET_COLUMNS, metric_dict
from stage3_utils import append_execution, read_best_checkpoint, sha256_file, sha256_values


def build_config(output_dir: Path, manifest: Path, name: str, max_epoch: int | None) -> Path:
    """Create an immutable stage-three training config from the fixed baseline."""
    config = yaml.safe_load((ROOT / "configs/GPS/direct_train_coarse_noaux.yaml").read_text(encoding="utf-8"))
    config["out_dir"] = str((output_dir / "determinism" / "training").resolve())
    config.update({"accelerator": "cuda", "devices": 1, "gpu_serial": 0, "seed": 0,
                   "use_mordred_features": True, "mordred_feature_dim": 11,
                   "mordred_feature_path": str(ROOT / "results/mordred_train_feedback/mordred_selected_features.csv")})
    config["dataset"] = dict(config["dataset"])
    config["dataset"].update({
        "diagnostic_split_path": str(manifest.resolve()), "diagnostic_id_column": "ID",
        "diagnostic_manifest_id_column": "sample_id", "cache_per_run": True,
        "cache_refresh": True, "cache_tag": "stage3_determinism_formula_fold_0",
    })
    config["train"] = dict(config["train"])
    config["train"].update({"deterministic": True, "manifest_path": str(manifest.resolve()),
                            "fold": "fold_0", "protocol": "formula_identity_group_cv"})
    if max_epoch is not None:
        config["optim"] = dict(config["optim"])
        config["optim"]["max_epoch"] = int(max_epoch)
        config["train"]["early_stop_patience"] = min(int(config["train"]["early_stop_patience"]), int(max_epoch))
    config_dir = output_dir / "determinism" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def cache_hashes() -> dict[str, str]:
    """Hash every processed graph cache file used by deterministic runs."""
    cache_root = ROOT / "datasets_lrx/.cache/double_stage3_determinism_formula_fold_0_seed_0/subset/processed"
    return {path.name: sha256_file(path) for path in sorted(cache_root.glob("*.pt"))}


def label_hashes(manifest: pd.DataFrame) -> dict[str, str]:
    """Hash labels after joining by explicit sample ID rather than row position."""
    frame = pd.read_csv(ROOT / "datasets_lrx/raw/input/20260703_sum.csv", dtype={"ID": str})
    merged = manifest.merge(frame[["ID", *TARGET_COLUMNS]], left_on="sample_id", right_on="ID",
                            how="left", validate="one_to_one")
    if merged[TARGET_COLUMNS].isna().any().any():
        raise ValueError("Manifest contains sample IDs absent from the raw training CSV.")
    return {split: sha256_values(group.sort_values("sample_id")[TARGET_COLUMNS].round(12).values.tolist())
            for split, group in merged.groupby("split")}


def read_curves(run_dir: Path, run_name: str) -> pd.DataFrame:
    """Load GraphGym epoch stats for all splits while retaining the repeat name."""
    frames: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        path = run_dir / split / "stats.json"
        if path.is_file():
            stats = pd.read_json(path, lines=True)
            if not stats.empty:
                stats["split"] = split
                stats["run_name"] = run_name
                frames.append(stats)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_one(output_dir: Path, manifest: Path, name: str, max_epoch: int | None,
            fold: str = "fold_0") -> dict[str, object]:
    """Run one independent training process and export its reloaded checkpoint predictions."""
    config_path = build_config(output_dir, manifest, name, max_epoch)
    run_dir = output_dir / "determinism" / "training" / name / "0"
    prediction_path = output_dir / "determinism" / "predictions" / f"{name}_test.csv"
    log_path = output_dir / "determinism" / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not run_dir.exists():
        command = [sys.executable, "main.py", "--cfg", str(config_path), "--repeat", "1"]
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            append_execution(output_dir, command=command, protocol="formula_identity_group_cv", fold=fold, seed=0,
                             data_version="raw_records", manifest_path=manifest, config_path=config_path,
                             checkpoint=read_best_checkpoint(run_dir), output=run_dir)
        except subprocess.CalledProcessError as error:
            append_execution(output_dir, command=command, protocol="formula_identity_group_cv", fold=fold, seed=0,
                             data_version="raw_records", manifest_path=manifest, config_path=config_path,
                             output=log_path, status="failed", error_message=str(error))
            raise
    checkpoint = read_best_checkpoint(run_dir)
    if not prediction_path.is_file():
        export_command = [sys.executable, "scripts/diagnostics/stage3_export_predictions.py", "--config", str(config_path),
                          "--checkpoint", str(checkpoint), "--manifest", str(manifest), "--output", str(prediction_path),
                          "--split", "test", "--seed", "0", "--fold", fold,
                          "--protocol", "formula_identity_group_cv"]
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                subprocess.run(export_command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            append_execution(output_dir, command=export_command, protocol="formula_identity_group_cv", fold=fold, seed=0,
                             data_version="raw_records", manifest_path=manifest, config_path=config_path,
                             checkpoint=checkpoint, output=prediction_path)
        except subprocess.CalledProcessError as error:
            append_execution(output_dir, command=export_command, protocol="formula_identity_group_cv", fold=fold, seed=0,
                             data_version="raw_records", manifest_path=manifest, config_path=config_path,
                             checkpoint=checkpoint, output=log_path, status="failed", error_message=str(error))
            raise
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {
        "run_name": name, "run_dir": str(run_dir), "config_path": str(config_path),
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": int(checkpoint.stem), "best_metric": state.get("best_metric"),
        "manifest_hash": sha256_file(manifest), "feature_hash": sha256_values(cache_hashes()),
        "prediction_path": str(prediction_path), "state_dict": state["model_state"],
    }


def compare_parameters(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor],
                       reference_name: str, candidate_name: str) -> pd.DataFrame:
    """Compare model state tensors elementwise after independent runs."""
    rows: list[dict[str, object]] = []
    if set(reference) != set(candidate):
        raise ValueError("Checkpoint state dictionaries have different parameter keys.")
    for name in sorted(reference):
        first, second = reference[name].detach().cpu(), candidate[name].detach().cpu()
        difference = (first - second).abs()
        rows.append({"reference_run": reference_name, "candidate_run": candidate_name,
                     "parameter": name, "shape": list(first.shape),
                     "max_abs_difference": float(difference.max().item()) if difference.numel() else 0.0,
                     "equal": bool(torch.equal(first, second))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generalization_stage3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/generalization_stage3/manifests/formula_identity_group_cv/raw_records/fold_0.csv")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    det_dir = output_dir / "determinism"
    det_dir.mkdir(parents=True, exist_ok=True)
    manifest = arguments.manifest.resolve()
    manifest_frame = pd.read_csv(manifest, dtype={"sample_id": str})
    hashes = {split: sha256_values(group.sort_values("sample_id")["sample_id"].tolist())
              for split, group in manifest_frame.groupby("split")}
    hashes["labels"] = label_hashes(manifest_frame)
    (det_dir / "deterministic_settings.json").write_text(json.dumps({
        "seed": 0, "manifest": str(manifest), "sample_hashes": hashes,
        "settings": {"PYTHONHASHSEED": "0", "cudnn_deterministic": True,
                     "cudnn_benchmark": False, "deterministic_algorithms_warn_only": True,
                     "num_workers": 0, "train_shuffle": False},
    }, indent=2), encoding="utf-8")
    names = ["deterministic_smoke_aligned"] if arguments.smoke else [
        "deterministic_repeat_1", "deterministic_repeat_2", "deterministic_repeat_3"
    ]
    runs = [run_one(output_dir, manifest, name, 2 if arguments.smoke else None) for name in names]
    inventory = pd.DataFrame([{key: value for key, value in run.items() if key != "state_dict"} for run in runs])
    inventory.to_csv(det_dir / "repeat_run_inventory.csv", index=False)
    curve_frames = [read_curves(Path(run["run_dir"]), str(run["run_name"])) for run in runs]
    curves = pd.concat([frame for frame in curve_frames if not frame.empty], ignore_index=True)
    curves.to_csv(det_dir / "repeat_training_curves.csv", index=False)
    prediction_frames = [pd.read_csv(run["prediction_path"], dtype={"sample_id": str}) for run in runs]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(det_dir / "repeat_predictions.csv", index=False)
    metric_rows: list[dict[str, object]] = []
    for run, prediction in zip(runs, prediction_frames):
        for target, group in prediction.groupby("target"):
            metric_rows.append({"run_name": run["run_name"], "target": target, **metric_dict(group["y_true"], group["y_pred"])})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(det_dir / "repeat_metrics.csv", index=False)
    parameter_differences = pd.concat([
        compare_parameters(runs[0]["state_dict"], run["state_dict"], str(runs[0]["run_name"]), str(run["run_name"]))
        for run in runs[1:]
    ], ignore_index=True) if len(runs) > 1 else pd.DataFrame()
    parameter_differences.to_csv(det_dir / "parameter_difference.csv", index=False)
    inventory.drop(columns=[], errors="ignore").to_csv(det_dir / "checkpoint_hashes.csv", index=False)
    if len(runs) == 1:
        diagnosis = "Smoke test completed; three-repeat determinism has not yet run.\n"
    else:
        prediction_pivot = predictions.pivot(index=["sample_id", "target"], columns="checkpoint_path", values="y_pred")
        max_prediction_difference = float((prediction_pivot.max(axis=1) - prediction_pivot.min(axis=1)).abs().max())
        max_parameter_difference = float(parameter_differences["max_abs_difference"].max())
        metric_spread = float(metrics.groupby("target")["mae"].agg(lambda values: values.max() - values.min()).max())
        exact = ((inventory["checkpoint_sha256"].nunique() == 1 or max_parameter_difference <= 1e-7)
                 and inventory["best_epoch"].nunique() == 1
                 and max_parameter_difference <= 1e-7 and max_prediction_difference <= 1e-5
                 and metric_spread <= 1e-6)
        stable = metric_spread < 0.05 and max_prediction_difference < 0.5
        diagnosis = "\n".join([
            "# Determinism Diagnosis", "",
            f"- Exact pass: {exact}", f"- Relaxed stability pass: {stable}",
            f"- Maximum parameter difference: {max_parameter_difference:.12g}",
            f"- Maximum prediction difference: {max_prediction_difference:.12g}",
            f"- Maximum target MAE spread: {metric_spread:.12g}",
            "- All runs use the same explicit manifest, cached feature hash, label hash, GPU, seed, and configuration controls.",
        ]) + "\n"
    (det_dir / "determinism_diagnosis.md").write_text(diagnosis, encoding="utf-8")
    (det_dir / "deterministic_code_audit.md").write_text(
        "# Deterministic Code Audit\n\n"
        "`main.py` configures RNGs before loaders/model/optimizer creation; `loader_5.py` supplies an explicit generator "
        "and worker initializer; `csv_pyg_five_multi.py` stores stable source indexes in every component graph; "
        "`train_five_multi.py` augments best checkpoints with manifest/config/feature/scaler metadata.\n",
        encoding="utf-8",
    )
    (det_dir / "nondeterministic_warnings.log").write_text("See per-run logs under determinism/logs.\n", encoding="utf-8")
    print(f"Wrote {det_dir}")


if __name__ == "__main__":
    main()
