#!/usr/bin/env python3
"""Gate full deduplicated GraphGPS CV on two independent seed-0 smoke runs.

The only data source is the audited ``dataset_with_sample_id.csv``.  Each
process gets a distinct, disposable graph-cache tag and reloads its best
checkpoint to produce test predictions aligned through the manifest's
``sample_id`` values.  A failed gate deliberately exits non-zero, so a caller
cannot accidentally launch the full CV after an unstable smoke result.
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
from sklearn.metrics import mean_absolute_error


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from audit_deduplicated_dataset import TARGETS, append_execution, sha256_file, sha256_text  # noqa: E402


PROTOCOL = "formula_identity_group_cv"
FOLD = "fold_0"
SEED = 0
LABEL_EXPORT_ATOL = 1e-4  # Graph tensors store labels as float32 scaled by 100.


def json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def sample_hash(sample_ids: pd.Series) -> str:
    return sha256_text("\n".join(sorted(sample_ids.astype(str))))


def require_inputs(output_dir: Path) -> dict[str, object]:
    """Validate all new-dataset provenance before a GPU process is spawned."""
    source_path = output_dir / "data_source.json"
    profile_path = output_dir / "data_audit" / "dataset_profile.json"
    dataset_path = output_dir / "data_audit" / "dataset_with_sample_id.csv"
    lookup_path = output_dir / "artifacts" / "mordred_11_lookup.csv"
    inventory_path = output_dir / "artifacts" / "artifact_inventory.csv"
    integrity_path = output_dir / "artifacts" / "cache_integrity.csv"
    manifest = output_dir / "manifests" / PROTOCOL / f"{FOLD}.csv"
    required = [source_path, profile_path, dataset_path, lookup_path, inventory_path, integrity_path, manifest]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"GraphGPS smoke input is missing: {missing}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if source.get("audit_status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise RuntimeError(f"Audit gate is {source.get('audit_status')!r}, not passing.")
    if sha256_file(Path(source["dataset_path"])) != source["dataset_sha256"]:
        raise RuntimeError("The selected new raw input changed after audit.")
    data = pd.read_csv(dataset_path, dtype={"sample_id": str})
    if len(data) != 700 or data.sample_id.isna().any() or data.sample_id.duplicated().any():
        raise ValueError("Audited new dataset must have exactly 700 unique non-null sample_id values.")
    if sample_hash(data.sample_id) != profile["sample_id_hash"]:
        raise RuntimeError("Audited data sample_id hash does not match dataset_profile.json.")
    labels = data[["sample_id", *TARGETS]].copy()
    if labels[TARGETS].isna().any().any() or not np.isfinite(labels[TARGETS].to_numpy(dtype=float)).all():
        raise ValueError("Audited labels must be finite.")
    manifest_frame = pd.read_csv(manifest, dtype={"sample_id": str})
    expected_columns = {"sample_id", "split", "original_row_index", "dataset_sha256", "manifest_sha256"}
    if not expected_columns.issubset(manifest_frame.columns):
        raise ValueError(f"Manifest lacks columns: {sorted(expected_columns - set(manifest_frame.columns))}")
    if len(manifest_frame) != len(data) or manifest_frame.sample_id.duplicated().any():
        raise ValueError("Manifest must cover every audited sample exactly once.")
    if set(manifest_frame.sample_id) != set(data.sample_id):
        raise ValueError("Manifest sample_id set differs from audited data.")
    if set(manifest_frame.dataset_sha256) != {source["dataset_sha256"]}:
        raise RuntimeError("Manifest dataset SHA256 is not the audited new source SHA256.")
    if set(manifest_frame["split"]) != {"train", "val", "test"}:
        raise ValueError("Manifest must include train, val, and test splits.")
    test = manifest_frame.loc[manifest_frame["split"] == "test", "sample_id"]
    if test.empty:
        raise ValueError("Smoke fold has no test samples.")
    inventory = pd.read_csv(inventory_path)
    lookup_record = inventory.loc[inventory.artifact == "mordred_11_lookup"]
    if len(lookup_record) != 1 or lookup_record.iloc[0].sha256 != sha256_file(lookup_path):
        raise RuntimeError("The new Mordred lookup is absent from, or mismatches, its inventory.")
    if lookup_record.iloc[0].dataset_sha256 != source["dataset_sha256"]:
        raise RuntimeError("Mordred lookup is bound to a different dataset.")
    integrity = pd.read_csv(integrity_path)
    if not (integrity.status == "PASS").all():
        raise RuntimeError("Artifact cache-integrity audit is not fully passing.")
    return {
        "source": source, "profile": profile, "dataset": dataset_path.resolve(), "labels": labels,
        "lookup": lookup_path.resolve(), "manifest": manifest.resolve(), "manifest_frame": manifest_frame,
        "dataset_hash": source["dataset_sha256"], "sample_id_hash": profile["sample_id_hash"],
        "labels_hash": sha256_text(labels.sort_values("sample_id").to_csv(index=False, float_format="%.12g")),
        "feature_hash": sha256_file(lookup_path), "manifest_hash": sha256_file(manifest),
    }


def build_config(output_dir: Path, inputs: dict[str, object], run_name: str, smoke_epochs: int) -> Path:
    """Materialize the fixed coarse+11D model with only data/provenance changes."""
    config = yaml.safe_load((ROOT / "configs/GPS/direct_train_coarse_noaux.yaml").read_text(encoding="utf-8"))
    cache_root = (output_dir / "graphgps_smoke" / "isolated_cache_roots" / run_name).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    config["out_dir"] = str((output_dir / "graphgps_smoke" / "training").resolve())
    config["read_csv"] = str(inputs["dataset"])
    config.update({
        "accelerator": "cuda", "devices": 1, "gpu_serial": 0, "num_workers": 0, "seed": SEED,
        "use_mordred_features": True, "mordred_feature_dim": 11,
        "mordred_feature_path": str(inputs["lookup"]),
    })
    config["dataset"] = dict(config["dataset"])
    config["dataset"].update({
        # Keep generated graph caches beneath the required result root.  The
        # loader's framework raw symlink is never used as a tabular data source:
        # `read_csv` above is the audited absolute CSV path.
        "dir": str(cache_root),
        "diagnostic_split_path": str(inputs["manifest"]),
        "diagnostic_id_column": "sample_id", "diagnostic_manifest_id_column": "sample_id",
        "cache_per_run": True, "cache_refresh": True,
        "cache_tag": f"deduplicated_graphgps_smoke_{PROTOCOL}_{FOLD}_{run_name}",
    })
    config["train"] = dict(config["train"])
    config["train"].update({
        "deterministic": True, "manifest_path": str(inputs["manifest"]),
        "fold": FOLD, "protocol": PROTOCOL,
        "early_stop_patience": min(int(config["train"]["early_stop_patience"]), smoke_epochs),
    })
    config["optim"] = dict(config["optim"])
    config["optim"]["max_epoch"] = int(smoke_epochs)
    config_dir = output_dir / "graphgps_smoke" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{run_name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def expected_run_dir(output_dir: Path, config_path: Path) -> Path:
    return output_dir / "graphgps_smoke" / "training" / config_path.stem / str(SEED)


def best_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "ckpt").glob("*.ckpt"), key=lambda path: int(path.stem))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one retained best checkpoint in {run_dir / 'ckpt'}, found {len(checkpoints)}")
    return checkpoints[0]


def run_one(output_dir: Path, inputs: dict[str, object], run_name: str, smoke_epochs: int) -> dict[str, object]:
    """Run a fresh process, then evaluate by reloading its selected checkpoint."""
    config_path = build_config(output_dir, inputs, run_name, smoke_epochs)
    run_dir = expected_run_dir(output_dir, config_path)
    prediction_path = output_dir / "graphgps_smoke" / "predictions" / f"{run_name}_test.csv"
    log_path = output_dir / "graphgps_smoke" / "logs" / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "main.py", "--cfg", str(config_path), "--repeat", "1"]
    try:
        # Never resume: independent process/cache/run directories are a part of
        # the determinism test.  Existing material is retained as evidence.
        if run_dir.exists() or prediction_path.exists():
            raise RuntimeError(f"Smoke run path already exists for {run_name}; choose a clean output directory.")
        with log_path.open("w", encoding="utf-8") as handle:
            subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        checkpoint = best_checkpoint(run_dir)
        export_command = [
            sys.executable, "scripts/diagnostics/stage3_export_predictions.py", "--config", str(config_path),
            "--checkpoint", str(checkpoint), "--manifest", str(inputs["manifest"]), "--output", str(prediction_path),
            "--split", "test", "--seed", str(SEED), "--fold", FOLD, "--protocol", PROTOCOL,
        ]
        with log_path.open("a", encoding="utf-8") as handle:
            subprocess.run(export_command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return {
            "run_name": run_name, "run_dir": str(run_dir.resolve()), "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path), "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint), "best_epoch": int(checkpoint.stem),
            "best_metric": checkpoint_payload.get("best_metric"), "checkpoint_metadata": checkpoint_payload,
            "prediction_path": str(prediction_path.resolve()), "log_path": str(log_path.resolve()), "status": "PASS", "error": "",
        }
    except Exception as error:
        return {
            "run_name": run_name, "run_dir": str(run_dir.resolve()), "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path), "checkpoint_path": "", "checkpoint_sha256": "",
            "best_epoch": None, "best_metric": None, "checkpoint_metadata": {},
            "prediction_path": str(prediction_path.resolve()), "log_path": str(log_path.resolve()), "status": "FAIL", "error": repr(error),
        }


def load_completed_run(output_dir: Path, run_name: str) -> dict[str, object]:
    """Load an already completed independent smoke run for post-hoc auditing."""
    config_path = output_dir / "graphgps_smoke" / "configs" / f"{run_name}.yaml"
    run_dir = expected_run_dir(output_dir, config_path)
    prediction_path = output_dir / "graphgps_smoke" / "predictions" / f"{run_name}_test.csv"
    log_path = output_dir / "graphgps_smoke" / "logs" / f"{run_name}.log"
    checkpoint = best_checkpoint(run_dir)
    if not config_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f"Completed smoke material is incomplete for {run_name}.")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {
        "run_name": run_name, "run_dir": str(run_dir.resolve()), "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path), "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint), "best_epoch": int(checkpoint.stem),
        "best_metric": checkpoint_payload.get("best_metric"), "checkpoint_metadata": checkpoint_payload,
        "prediction_path": str(prediction_path.resolve()), "log_path": str(log_path.resolve()), "status": "PASS", "error": "",
    }


def alignment_rows(run: dict[str, object], inputs: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate prediction-to-label alignment via ``sample_id``, never row order."""
    prediction = pd.read_csv(run["prediction_path"], dtype={"sample_id": str})
    manifest = inputs["manifest_frame"]
    assert isinstance(manifest, pd.DataFrame)
    test_ids = set(manifest.loc[manifest["split"] == "test", "sample_id"])
    labels = inputs["labels"]
    assert isinstance(labels, pd.DataFrame)
    expected = labels.loc[labels.sample_id.isin(test_ids)].melt(id_vars="sample_id", value_vars=TARGETS,
                                                                  var_name="target", value_name="expected_y_true")
    merged = prediction.merge(expected, on=["sample_id", "target"], how="outer", indicator=True, validate="one_to_one")
    metadata = run.get("checkpoint_metadata", {})
    metadata_ok = (
        metadata.get("stage3_checkpoint_metadata") is True
        and metadata.get("seed") == SEED
        and metadata.get("fold") == FOLD
        and metadata.get("protocol") == PROTOCOL
        and metadata.get("manifest_hash") == inputs["manifest_hash"]
        and metadata.get("feature_hash") == inputs["feature_hash"]
        and metadata.get("sample_id_hash") == sha256_text(json.dumps(manifest.sample_id.astype(str).tolist(), separators=(",", ":")))
        and metadata.get("target_scaler", {}).get("scale") == 100.0
    )
    audit = pd.DataFrame([{
        "run_name": run["run_name"], "expected_samples": len(test_ids), "predicted_samples": prediction.sample_id.nunique(),
        "expected_rows": len(expected), "predicted_rows": len(prediction),
        "missing_or_extra_rows": int((merged["_merge"] != "both").sum()),
        "duplicate_sample_target": int(prediction.duplicated(["sample_id", "target"]).sum()),
        "target_set_exact": set(prediction.target) == set(TARGETS),
        "sample_id_set_exact": set(prediction.sample_id) == test_ids,
        "max_y_true_abs_error": float((merged.loc[merged["_merge"] == "both", "y_true"] -
                                         merged.loc[merged["_merge"] == "both", "expected_y_true"]).abs().max()),
        "label_export_atol": LABEL_EXPORT_ATOL,
        "y_true_matches_audited_dataset": bool(np.allclose(merged.loc[merged["_merge"] == "both", "y_true"],
                                                              merged.loc[merged["_merge"] == "both", "expected_y_true"],
                                                              atol=LABEL_EXPORT_ATOL, rtol=0)),
        "dataset_sha256": inputs["dataset_hash"], "labels_sha256": inputs["labels_hash"],
        "feature_sha256": inputs["feature_hash"], "manifest_sha256": inputs["manifest_hash"],
        "checkpoint_metadata_valid": metadata_ok,
        "status": "PASS" if len(prediction) == len(expected) and (merged["_merge"] == "both").all()
        and not prediction.duplicated(["sample_id", "target"]).any() and set(prediction.sample_id) == test_ids
        and set(prediction.target) == set(TARGETS) and np.allclose(merged["y_true"], merged["expected_y_true"],
                                                                     atol=LABEL_EXPORT_ATOL, rtol=0)
        and metadata_ok else "FAIL",
    }])
    return prediction, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "deduplicated_rebaseline")
    parser.add_argument("--smoke-epochs", type=int, default=20)
    parser.add_argument("--attempt-tag", default="", help="Suffix for a clean retry while retaining earlier failed evidence.")
    parser.add_argument("--verify-existing-attempt-tag", default="",
                        help="Re-audit an already completed run_a/run_b suffix without retraining.")
    arguments = parser.parse_args()
    if arguments.smoke_epochs < 1:
        raise SystemExit("--smoke-epochs must be positive.")
    output_dir = arguments.output_dir.resolve()
    smoke_dir = output_dir / "graphgps_smoke"
    try:
        inputs = require_inputs(output_dir)
    except Exception as error:
        append_execution(output_dir, {"timestamp": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
                                      "dataset_path": None, "dataset_sha256": None, "protocol": PROTOCOL, "fold": FOLD,
                                      "seed": SEED, "manifest_sha256": None, "feature_hash": None, "config_hash": None,
                                      "checkpoint": None, "status": "BLOCKED_AUDIT_GATE", "error": repr(error),
                                      "output_path": str(smoke_dir)})
        raise SystemExit(str(error))
    json_dump({
        "purpose": "two independent deterministic GraphGPS seed-0 smoke runs before full CV",
        "protocol": PROTOCOL, "fold": FOLD, "seed": SEED, "smoke_epochs": arguments.smoke_epochs,
        "dataset_path": str(inputs["dataset"]), "dataset_sha256": inputs["dataset_hash"],
        "sample_id_hash": inputs["sample_id_hash"], "labels_sha256": inputs["labels_hash"],
        "mordred_lookup": str(inputs["lookup"]), "feature_sha256": inputs["feature_hash"],
        "manifest": str(inputs["manifest"]), "manifest_sha256": inputs["manifest_hash"],
        "settings": {"num_workers": 0, "train_shuffle": False, "cudnn_deterministic": True,
                     "cudnn_benchmark": False, "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
    }, smoke_dir / "deterministic_settings.json")
    suffix = ("_" + arguments.attempt_tag.strip()) if arguments.attempt_tag.strip() else ""
    existing_suffix = ("_" + arguments.verify_existing_attempt_tag.strip()) if arguments.verify_existing_attempt_tag.strip() else ""
    if suffix and existing_suffix:
        raise SystemExit("Use either --attempt-tag or --verify-existing-attempt-tag, not both.")
    if existing_suffix:
        runs = [load_completed_run(output_dir, name + existing_suffix) for name in ("run_a", "run_b")]
    else:
        runs = [run_one(output_dir, inputs, name + suffix, arguments.smoke_epochs) for name in ("run_a", "run_b")]
    inventory = pd.DataFrame([{key: value for key, value in run.items() if key != "checkpoint_metadata"} for run in runs])
    inventory_path = smoke_dir / "run_inventory.csv"
    if inventory_path.is_file():
        previous_inventory = pd.read_csv(inventory_path)
        inventory = pd.concat([previous_inventory, inventory], ignore_index=True)
    inventory = inventory.drop_duplicates(["run_name"], keep="last")
    inventory.to_csv(inventory_path, index=False)
    failed_runs = inventory.loc[inventory.run_name.isin([run["run_name"] for run in runs]) & (inventory.status != "PASS")]
    if not failed_runs.empty:
        report = "# GraphGPS Determinism Smoke\n\n- Gate result: **FAIL** (training/export process failed).\n"
        (smoke_dir / "determinism_report.md").write_text(report, encoding="utf-8")
        append_execution(output_dir, {"timestamp": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
                                      "dataset_path": str(inputs["dataset"]), "dataset_sha256": inputs["dataset_hash"],
                                      "protocol": PROTOCOL, "fold": FOLD, "seed": SEED,
                                      "manifest_sha256": inputs["manifest_hash"], "feature_hash": inputs["feature_hash"],
                                      "config_hash": None, "checkpoint": None, "status": "FAIL", "error": failed_runs.error.tolist(),
                                      "output_path": str(smoke_dir)})
        raise SystemExit("GraphGPS smoke run failed; inspect graphgps_smoke/run_inventory.csv and logs.")
    predictions: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    for run in runs:
        prediction, audit = alignment_rows(run, inputs)
        predictions.append(prediction)
        audits.append(audit)
    alignment = pd.concat(audits, ignore_index=True)
    alignment.to_csv(smoke_dir / "alignment_audit.csv", index=False)
    metrics_rows: list[dict[str, object]] = []
    for run, prediction in zip(runs, predictions):
        for target, group in prediction.groupby("target", sort=True):
            metrics_rows.append({"run_name": run["run_name"], "target": target, "n": len(group),
                                 "mae": mean_absolute_error(group.y_true, group.y_pred)})
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(smoke_dir / "repeat_metrics.csv", index=False)
    comparison = predictions[0][["sample_id", "target", "y_true", "y_pred"]].rename(columns={"y_pred": "y_pred_run_a"}).merge(
        predictions[1][["sample_id", "target", "y_true", "y_pred"]].rename(columns={"y_true": "y_true_run_b", "y_pred": "y_pred_run_b"}),
        on=["sample_id", "target"], how="outer", validate="one_to_one")
    comparison["absolute_prediction_difference"] = (comparison.y_pred_run_a - comparison.y_pred_run_b).abs()
    comparison["y_true_difference"] = (comparison.y_true - comparison.y_true_run_b).abs()
    comparison.to_csv(smoke_dir / "prediction_comparison.csv", index=False)
    mae_spread = float(metrics.groupby("target").mae.agg(lambda values: values.max() - values.min()).max())
    max_prediction_difference = float(comparison.absolute_prediction_difference.max())
    current_inventory = inventory.loc[inventory.run_name.isin([run["run_name"] for run in runs])].copy()
    best_epoch_equal = current_inventory.best_epoch.nunique() == 1
    label_equal = bool(np.allclose(comparison.y_true, comparison.y_true_run_b, atol=1e-6, rtol=0))
    passed = bool((alignment.status == "PASS").all() and label_equal and best_epoch_equal
                  and mae_spread <= 0.05 and max_prediction_difference <= 0.5)
    report = "\n".join([
        "# GraphGPS Determinism Smoke", "",
        f"- Gate result: **{'PASS' if passed else 'FAIL'}**",
        f"- Protocol/fold/seed: `{PROTOCOL}` / `{FOLD}` / `{SEED}`",
        f"- Independent runs: `{runs[0]['run_name']}`, `{runs[1]['run_name']}`; each used a distinct disposable cache tag.",
        f"- Dataset SHA256: `{inputs['dataset_hash']}`; sample_id hash: `{inputs['sample_id_hash']}`.",
        f"- Feature SHA256: `{inputs['feature_hash']}`; labels SHA256: `{inputs['labels_hash']}`.",
        f"- Same best epoch: `{best_epoch_equal}` ({current_inventory.best_epoch.tolist()}).",
        f"- Maximum per-target MAE spread: `{mae_spread:.8g}` (threshold ≤ 0.05).",
        f"- Maximum absolute prediction difference: `{max_prediction_difference:.8g}` (threshold ≤ 0.5).",
        f"- Sample-aligned exported-label tolerance: `{LABEL_EXPORT_ATOL:.0e}` (float32 conversion); observed maxima are in `alignment_audit.csv`.",
        f"- Reloaded prediction labels identical: `{label_equal}`.",
        f"- Checkpoint SHA256 values identical: `{current_inventory.checkpoint_sha256.nunique() == 1}` (recorded, not required for relaxed gate).",
        "- Full GraphGPS CV must not start unless this gate is PASS.",
    ]) + "\n"
    (smoke_dir / "determinism_report.md").write_text(report, encoding="utf-8")
    append_execution(output_dir, {"timestamp": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
                                  "dataset_path": str(inputs["dataset"]), "dataset_sha256": inputs["dataset_hash"],
                                  "protocol": PROTOCOL, "fold": FOLD, "seed": SEED,
                                  "manifest_sha256": inputs["manifest_hash"], "feature_hash": inputs["feature_hash"],
                                  "config_hash": None, "checkpoint": current_inventory.checkpoint_path.tolist(),
                                  "status": "PASS" if passed else "FAIL", "error": None if passed else "determinism thresholds not met",
                                  "output_path": str(smoke_dir)})
    if not passed:
        raise SystemExit("GraphGPS determinism gate failed; full CV remains blocked.")
    print(f"GraphGPS determinism gate PASS: wrote {smoke_dir}")


if __name__ == "__main__":
    main()
