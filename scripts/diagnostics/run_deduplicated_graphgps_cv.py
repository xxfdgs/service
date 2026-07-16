#!/usr/bin/env python3
"""Run the gated, seed-0 formula-identity GraphGPS outer CV on new data only.

This is intentionally limited to ``formula_identity_group_cv`` and seed 0.
It refuses to start unless the two-process deterministic smoke gate passed.  A
future decision may add seeds 1/2 or Fifth-component grouping only after the
pooled seed-0 result is reviewed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from audit_deduplicated_dataset import TARGETS, append_execution, sha256_file, sha256_text  # noqa: E402


PROTOCOL = "formula_identity_group_cv"
SEED = 0
LABEL_EXPORT_ATOL = 1e-4


def json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def require_gate_and_inputs(output_dir: Path) -> dict[str, object]:
    """Bind full CV to the passed smoke, exact new source, features, and labels."""
    smoke_report = output_dir / "graphgps_smoke" / "determinism_report.md"
    source_path = output_dir / "data_source.json"
    profile_path = output_dir / "data_audit" / "dataset_profile.json"
    dataset_path = output_dir / "data_audit" / "dataset_with_sample_id.csv"
    lookup_path = output_dir / "artifacts" / "mordred_11_lookup.csv"
    inventory_path = output_dir / "artifacts" / "artifact_inventory.csv"
    integrity_path = output_dir / "artifacts" / "cache_integrity.csv"
    required = [smoke_report, source_path, profile_path, dataset_path, lookup_path, inventory_path, integrity_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Full GraphGPS CV input is missing: {missing}")
    if "Gate result: **PASS**" not in smoke_report.read_text(encoding="utf-8"):
        raise RuntimeError("Full GraphGPS CV is blocked: deterministic smoke gate is not PASS.")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if source.get("audit_status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise RuntimeError("Full GraphGPS CV is blocked by the data-audit status.")
    if sha256_file(Path(source["dataset_path"])) != source["dataset_sha256"]:
        raise RuntimeError("The audited new raw input changed after the data audit.")
    data = pd.read_csv(dataset_path, dtype={"sample_id": str})
    if len(data) != 700 or data.sample_id.isna().any() or data.sample_id.duplicated().any():
        raise ValueError("Audited data must have 700 unique sample_id values.")
    sample_id_hash = sha256_text("\n".join(sorted(data.sample_id)))
    if sample_id_hash != profile["sample_id_hash"]:
        raise RuntimeError("Audited sample IDs no longer match their profile hash.")
    if data[TARGETS].isna().any().any() or not np.isfinite(data[TARGETS].to_numpy(dtype=float)).all():
        raise ValueError("All four audited targets must be finite.")
    integrity = pd.read_csv(integrity_path)
    if not (integrity.status == "PASS").all():
        raise RuntimeError("Full GraphGPS CV is blocked by failed artifact integrity.")
    inventory = pd.read_csv(inventory_path)
    lookup_record = inventory.loc[inventory.artifact == "mordred_11_lookup"]
    if len(lookup_record) != 1 or lookup_record.iloc[0].sha256 != sha256_file(lookup_path):
        raise RuntimeError("Mordred lookup inventory/hash audit failed.")
    if lookup_record.iloc[0].dataset_sha256 != source["dataset_sha256"]:
        raise RuntimeError("Mordred lookup was not generated from the selected new dataset.")
    labels = data[["sample_id", *TARGETS]].copy()
    return {
        "source": source, "profile": profile, "dataset": dataset_path.resolve(), "lookup": lookup_path.resolve(),
        "labels": labels, "dataset_hash": source["dataset_sha256"], "sample_id_hash": sample_id_hash,
        "labels_hash": sha256_text(labels.sort_values("sample_id").to_csv(index=False, float_format="%.12g")),
        "feature_hash": sha256_file(lookup_path),
    }


def manifest_for(output_dir: Path, fold: str, inputs: dict[str, object]) -> tuple[Path, pd.DataFrame]:
    path = output_dir / "manifests" / PROTOCOL / f"{fold}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing required manifest: {path}")
    frame = pd.read_csv(path, dtype={"sample_id": str})
    data = inputs["labels"]
    assert isinstance(data, pd.DataFrame)
    if len(frame) != len(data) or frame.sample_id.duplicated().any() or set(frame.sample_id) != set(data.sample_id):
        raise ValueError(f"{fold} does not cover audited sample_id values exactly once.")
    if set(frame.dataset_sha256) != {inputs["dataset_hash"]}:
        raise RuntimeError(f"{fold} is bound to a different dataset SHA256.")
    if set(frame.split) != {"train", "val", "test"} or not all((frame.split == split).any() for split in ("train", "val", "test")):
        raise ValueError(f"{fold} has invalid or empty split membership.")
    return path.resolve(), frame


def build_config(output_dir: Path, inputs: dict[str, object], manifest: Path, fold: str) -> Path:
    """Use the unchanged full fixed model/training budget on one outer fold."""
    config = yaml.safe_load((ROOT / "configs/GPS/direct_train_coarse_noaux.yaml").read_text(encoding="utf-8"))
    if int(config["optim"]["max_epoch"]) != 1500:
        raise RuntimeError("Reference GraphGPS full training budget is expected to be 1500 epochs.")
    cache_root = (output_dir / "graphgps_cv" / "isolated_cache_roots" / f"{fold}_seed_{SEED}").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    config["out_dir"] = str((output_dir / "graphgps_cv" / "training").resolve())
    config["read_csv"] = str(inputs["dataset"])
    config.update({
        "accelerator": "cuda", "devices": 1, "gpu_serial": 0, "num_workers": 0, "seed": SEED,
        "use_mordred_features": True, "mordred_feature_dim": 11, "mordred_feature_path": str(inputs["lookup"]),
    })
    config["dataset"] = dict(config["dataset"])
    config["dataset"].update({
        "dir": str(cache_root), "diagnostic_split_path": str(manifest), "diagnostic_id_column": "sample_id",
        "diagnostic_manifest_id_column": "sample_id", "cache_per_run": True, "cache_refresh": True,
        "cache_tag": f"deduplicated_graphgps_cv_{PROTOCOL}_{fold}_seed_{SEED}",
    })
    config["train"] = dict(config["train"])
    config["train"].update({"deterministic": True, "manifest_path": str(manifest), "fold": fold, "protocol": PROTOCOL})
    path = output_dir / "graphgps_cv" / "configs" / f"{PROTOCOL}_{fold}_seed_{SEED}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_dir_for(output_dir: Path, config: Path) -> Path:
    return output_dir / "graphgps_cv" / "training" / config.stem / str(SEED)


def best_checkpoint(run_dir: Path) -> Path:
    files = sorted((run_dir / "ckpt").glob("*.ckpt"), key=lambda path: int(path.stem))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one best checkpoint in {run_dir / 'ckpt'}, found {len(files)}")
    return files[0]


def checkpoint_reusable(checkpoint: Path, manifest: Path, inputs: dict[str, object], fold: str) -> bool:
    """Resume only this pipeline's own fully provenance-bound checkpoint."""
    if not checkpoint.is_file():
        return False
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    manifest_frame = pd.read_csv(manifest, dtype={"sample_id": str})
    expected_manifest_sample_hash = sha256_text(json.dumps(manifest_frame.sample_id.astype(str).tolist(), separators=(",", ":")))
    return bool(state.get("stage3_checkpoint_metadata")) and state.get("seed") == SEED and state.get("fold") == fold \
        and state.get("protocol") == PROTOCOL and state.get("manifest_hash") == sha256_file(manifest) \
        and state.get("feature_hash") == inputs["feature_hash"] and state.get("sample_id_hash") == expected_manifest_sample_hash \
        and state.get("target_scaler", {}).get("scale") == 100.0


def training_completed(log_path: Path) -> bool:
    """Do not mistake an interrupted best checkpoint for a completed fold."""
    return log_path.is_file() and "[*] All done:" in log_path.read_text(encoding="utf-8", errors="replace")


def prediction_path(output_dir: Path, fold: str, split: str) -> Path:
    return output_dir / "graphgps_cv" / "seed_predictions" / PROTOCOL / f"{fold}_seed_{SEED}_{split}.csv"


def run_fold(output_dir: Path, inputs: dict[str, object], fold: str) -> dict[str, object]:
    manifest, _ = manifest_for(output_dir, fold, inputs)
    config = build_config(output_dir, inputs, manifest, fold)
    run_dir = run_dir_for(output_dir, config)
    log_path = output_dir / "graphgps_cv" / "logs" / PROTOCOL / f"{fold}_seed_{SEED}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: Path | None = None
    try:
        if run_dir.exists():
            try:
                candidate = best_checkpoint(run_dir)
                checkpoint = candidate if checkpoint_reusable(candidate, manifest, inputs, fold) and training_completed(log_path) else None
            except RuntimeError:
                checkpoint = None
        if checkpoint is None:
            command = [sys.executable, "main.py", "--cfg", str(config), "--repeat", "1"]
            with log_path.open("w", encoding="utf-8") as handle:
                subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            checkpoint = best_checkpoint(run_dir)
            if not checkpoint_reusable(checkpoint, manifest, inputs, fold):
                raise RuntimeError(f"Checkpoint provenance audit failed: {checkpoint}")
        for split in ("val", "test"):
            output = prediction_path(output_dir, fold, split)
            if output.is_file():
                continue
            command = [
                sys.executable, "scripts/diagnostics/stage3_export_predictions.py", "--config", str(config),
                "--checkpoint", str(checkpoint), "--manifest", str(manifest), "--output", str(output),
                "--split", split, "--seed", str(SEED), "--fold", fold, "--protocol", PROTOCOL,
            ]
            with log_path.open("a", encoding="utf-8") as handle:
                subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        return {"fold": fold, "seed": SEED, "protocol": PROTOCOL, "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest), "config": str(config), "config_sha256": sha256_file(config),
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "status": "PASS", "error": ""}
    except Exception as error:
        return {"fold": fold, "seed": SEED, "protocol": PROTOCOL, "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest), "config": str(config), "config_sha256": sha256_file(config),
                "checkpoint": str(checkpoint or ""), "checkpoint_sha256": "", "status": "FAIL", "error": repr(error)}


def metric_row(values: pd.DataFrame) -> dict[str, float]:
    true, predicted = values.y_true.to_numpy(), values.y_pred.to_numpy()
    correlation = spearmanr(true, predicted)
    return {"mae": float(mean_absolute_error(true, predicted)), "rmse": float(mean_squared_error(true, predicted) ** 0.5),
            "r2": float(r2_score(true, predicted)), "spearman": float(correlation.statistic) if np.isfinite(correlation.statistic) else np.nan}


def aggregate(output_dir: Path, inputs: dict[str, object], completed_folds: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate and pool OOF predictions solely by their stable sample IDs."""
    all_predictions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for fold in completed_folds:
        path = prediction_path(output_dir, fold, "test")
        manifest, manifest_frame = manifest_for(output_dir, fold, inputs)
        prediction = pd.read_csv(path, dtype={"sample_id": str})
        test_ids = set(manifest_frame.loc[manifest_frame.split == "test", "sample_id"])
        labels = inputs["labels"]
        assert isinstance(labels, pd.DataFrame)
        expected = labels.loc[labels.sample_id.isin(test_ids)].melt(id_vars="sample_id", value_vars=TARGETS,
                                                                     var_name="target", value_name="expected_y_true")
        merged = prediction.merge(expected, on=["sample_id", "target"], how="outer", indicator=True, validate="one_to_one")
        maximum_label_error = float((merged.loc[merged["_merge"] == "both", "y_true"] -
                                     merged.loc[merged["_merge"] == "both", "expected_y_true"]).abs().max())
        passed = len(prediction) == len(expected) and (merged["_merge"] == "both").all() \
            and not prediction.duplicated(["sample_id", "target"]).any() and set(prediction.sample_id) == test_ids \
            and set(prediction.target) == set(TARGETS) \
            and np.allclose(merged.y_true, merged.expected_y_true, atol=LABEL_EXPORT_ATOL, rtol=0)
        audits.append({"fold": fold, "expected_samples": len(test_ids), "prediction_rows": len(prediction),
                       "expected_rows": len(expected), "missing_or_extra_rows": int((merged["_merge"] != "both").sum()),
                       "duplicate_sample_target": int(prediction.duplicated(["sample_id", "target"]).sum()),
                       "max_y_true_abs_error": maximum_label_error, "label_export_atol": LABEL_EXPORT_ATOL,
                       "dataset_sha256": inputs["dataset_hash"], "labels_sha256": inputs["labels_hash"],
                       "feature_sha256": inputs["feature_hash"], "manifest_sha256": sha256_file(manifest),
                       "status": "PASS" if passed else "FAIL"})
        if not passed:
            raise RuntimeError(f"Prediction alignment audit failed for {fold}.")
        all_predictions.append(prediction)
    pooled = pd.concat(all_predictions, ignore_index=True)
    if pooled.duplicated(["sample_id", "target"]).any():
        raise RuntimeError("OOF prediction duplicate detected across outer folds.")
    if len(completed_folds) == 5 and len(pooled) != 700 * len(TARGETS):
        raise RuntimeError("Completed full CV does not have exactly one OOF prediction per sample/target.")
    fold_metrics = pd.DataFrame([
        {"fold": fold, "target": target, "n": len(group), **metric_row(group)}
        for (fold, target), group in pooled.groupby(["fold", "target"], sort=True)
    ])
    pooled_metrics = pd.DataFrame([
        {"protocol": PROTOCOL, "seed": SEED, "target": target, "n": len(group), **metric_row(group)}
        for target, group in pooled.groupby("target", sort=True)
    ])
    return pooled, fold_metrics, pd.DataFrame(audits), pooled_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "deduplicated_rebaseline")
    parser.add_argument("--folds", default="fold_0,fold_1,fold_2,fold_3,fold_4")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    folds = [fold.strip() for fold in arguments.folds.split(",") if fold.strip()]
    expected_folds = {f"fold_{index}" for index in range(5)}
    if not folds or any(fold not in expected_folds for fold in folds):
        raise SystemExit(f"--folds must be a comma-separated subset of {sorted(expected_folds)}")
    try:
        inputs = require_gate_and_inputs(output_dir)
    except Exception as error:
        append_execution(output_dir, {"timestamp": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
                                      "dataset_path": None, "dataset_sha256": None, "protocol": PROTOCOL, "fold": None,
                                      "seed": SEED, "manifest_sha256": None, "feature_hash": None, "config_hash": None,
                                      "checkpoint": None, "status": "BLOCKED_SMOKE_GATE", "error": repr(error),
                                      "output_path": str(output_dir / "graphgps_cv")})
        raise SystemExit(str(error))
    cv_dir = output_dir / "graphgps_cv"
    json_dump({"protocol": PROTOCOL, "seed": SEED, "folds_requested": folds, "full_training_budget_epochs": 1500,
               "dataset_path": str(inputs["dataset"]), "dataset_sha256": inputs["dataset_hash"],
               "sample_id_hash": inputs["sample_id_hash"], "labels_sha256": inputs["labels_hash"],
               "feature_path": str(inputs["lookup"]), "feature_sha256": inputs["feature_hash"],
               "smoke_gate": str((output_dir / "graphgps_smoke" / "determinism_report.md"))}, cv_dir / "cv_settings.json")
    results = [run_fold(output_dir, inputs, fold) for fold in folds]
    inventory_path = cv_dir / "run_inventory.csv"
    current = pd.DataFrame(results)
    if inventory_path.is_file():
        current = pd.concat([pd.read_csv(inventory_path), current], ignore_index=True)
    current = current.drop_duplicates(["protocol", "fold", "seed"], keep="last")
    current.to_csv(inventory_path, index=False)
    # Include previously completed folds when this invocation resumes the
    # remaining work; otherwise their OOF predictions would be overwritten by
    # a partial aggregate.
    successful_folds = current.loc[current.status == "PASS", "fold"].astype(str).tolist()
    if successful_folds:
        pooled, fold_metrics, audits, pooled_metrics = aggregate(output_dir, inputs, successful_folds)
        pooled.to_csv(cv_dir / "pooled_oof_predictions.csv", index=False)
        fold_metrics.to_csv(cv_dir / "fold_metrics.csv", index=False)
        audits.to_csv(cv_dir / "alignment_audit.csv", index=False)
        pooled_metrics.to_csv(cv_dir / "pooled_oof_metrics.csv", index=False)
    failed = current.loc[current.status != "PASS"].to_dict("records")
    report = ["# Deduplicated GraphGPS Seed-0 Formula CV", "",
              f"- Protocol: `{PROTOCOL}`; seed: `{SEED}`; full budget: `1500` epochs.",
              f"- Requested folds: `{', '.join(folds)}`.",
              f"- Completed folds: `{', '.join(successful_folds) or 'none'}`.",
              f"- Failed folds: `{', '.join(record['fold'] for record in failed) or 'none'}`.",
              f"- Dataset SHA256: `{inputs['dataset_hash']}`.", f"- Feature SHA256: `{inputs['feature_hash']}`.",
              "- Full evaluation uses only the passed deterministic smoke gate and newly generated data/artifacts.",
              "- Seeds 1/2 and Fifth-component grouping remain intentionally unstarted pending review of this seed-0 result."]
    (cv_dir / "cv_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    append_execution(output_dir, {"timestamp": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
                                  "dataset_path": str(inputs["dataset"]), "dataset_sha256": inputs["dataset_hash"],
                                  "protocol": PROTOCOL, "fold": ",".join(folds), "seed": SEED,
                                  "manifest_sha256": None, "feature_hash": inputs["feature_hash"], "config_hash": None,
                                  "checkpoint": [record["checkpoint"] for record in results],
                                  "status": "PASS" if not failed else "FAIL", "error": [record["error"] for record in failed],
                                  "output_path": str(cv_dir)})
    if failed:
        raise SystemExit("One or more GraphGPS folds failed; see graphgps_cv/run_inventory.csv.")
    print(f"Completed GraphGPS CV folds: {', '.join(successful_folds)}")


if __name__ == "__main__":
    main()
