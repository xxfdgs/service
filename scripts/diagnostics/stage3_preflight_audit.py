#!/usr/bin/env python3
"""Audit the executable GraphGPS environment before stage-three experiments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import safe_json_dump


def file_sha256(path: Path) -> str:
    """Return a deterministic SHA256 hash for a manifest or configuration file."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version_or_missing(package: str) -> str:
    """Read an installed package version without failing the preflight audit."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def git_state() -> dict[str, str]:
    """Record git identity when this checkout exposes a repository."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        return {"commit": commit, "worktree_status": status or "clean"}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": "unavailable", "worktree_status": str(error)}


def audit_manifest(path: Path) -> dict[str, object]:
    """Validate an existing manifest without changing it."""
    required = {"sample_id", "split"}
    try:
        frame = pd.read_csv(path, dtype={"sample_id": str})
        missing = sorted(required - set(frame.columns))
        valid_splits = set(frame.get("split", pd.Series(dtype=str)).astype(str)) <= {"train", "val", "test"}
        return {
            "path": str(path), "rows": len(frame), "columns": "|".join(frame.columns),
            "sha256": file_sha256(path), "duplicate_sample_id": bool(frame.get("sample_id", pd.Series(dtype=str)).duplicated().any()),
            "missing_required": "|".join(missing), "valid_split_values": valid_splits,
            "status": "valid" if not missing and valid_splits and not frame["sample_id"].duplicated().any() else "invalid",
        }
    except Exception as error:
        return {"path": str(path), "rows": 0, "columns": "", "sha256": "", "duplicate_sample_id": None,
                "missing_required": "", "valid_split_values": False, "status": f"error: {error}"}


def inventory_runs() -> pd.DataFrame:
    """List GraphGPS stage-two/25 run directories and checkpoint availability."""
    rows: list[dict[str, object]] = []
    for root in (ROOT / "results" / "generalization_stage2", ROOT / "results" / "generalization_stage25"):
        if not root.exists():
            continue
        for checkpoint in root.rglob("*.ckpt"):
            run_dir = checkpoint.parent.parent
            rows.append({"result_root": str(root), "run_dir": str(run_dir), "checkpoint": str(checkpoint),
                         "seed": checkpoint.parent.parent.name, "status": "checkpoint_found"})
        for log_path in root.rglob("logging.log"):
            run_dir = log_path.parent
            if not any(row["run_dir"] == str(run_dir) for row in rows):
                rows.append({"result_root": str(root), "run_dir": str(run_dir), "checkpoint": "",
                             "seed": run_dir.name, "status": "log_without_checkpoint"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "generalization_stage3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    preflight_dir = output_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": sys.version,
        "pytorch": torch.__version__,
        "torch_geometric": version_or_missing("torch-geometric"),
        "rdkit": version_or_missing("rdkit"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
        "git": git_state(),
        "seed": arguments.seed,
    }
    safe_json_dump(environment, preflight_dir / "environment.json")
    code_paths = {
        "training_entry": "main.py",
        "prediction_entry": "main_predict.py",
        "five_component_loader": "loader_5.py",
        "dataset_class": "graphgps/lrx_add/csv_pyg_five_multi.py:LRX_five_multi",
        "checkpoint_training": "graphgps/train/train_five_multi.py",
        "checkpoint_library": "torch_geometric.graphgym.checkpoint",
        "prediction_ensemble": "graphgps/lrx_add/predict_average_multi.py",
        "metric_logger": "graphgps/logger.py",
        "determinism_helper": "graphgps/determinism.py",
    }
    safe_json_dump(code_paths, preflight_dir / "code_path_inventory.json")
    inventory = inventory_runs()
    inventory.to_csv(preflight_dir / "existing_run_inventory.csv", index=False)
    manifest_paths = [*ROOT.glob("results/generalization_stage2/manifests/**/fold_*.csv"),
                      *ROOT.glob("results/generalization_stage25/manifests/**/fold_*.csv"),
                      *ROOT.glob("results/generalization_stage3/manifests/**/fold_*.csv")]
    manifest_inventory = pd.DataFrame([audit_manifest(path) for path in sorted(set(manifest_paths))])
    manifest_inventory.to_csv(preflight_dir / "manifest_inventory.csv", index=False)
    report = ["# Stage 3 Preflight Audit", "", "## Environment", "",
              "```json", json.dumps(environment, indent=2, ensure_ascii=False), "```", "",
              "## Inventory", "",
              f"- Existing checkpoint files: {int((inventory['status'] == 'checkpoint_found').sum()) if not inventory.empty else 0}",
              f"- Inspected manifests: {len(manifest_inventory)}",
              f"- Valid manifests: {int((manifest_inventory['status'] == 'valid').sum()) if not manifest_inventory.empty else 0}", "",
              "## Entrypoints", ""]
    report.extend(f"- `{name}`: `{path}`" for name, path in code_paths.items())
    (preflight_dir / "preflight_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {preflight_dir}")


if __name__ == "__main__":
    main()
