"""Shared hashing and execution-manifest helpers for stage-three diagnostics."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from common import safe_json_dump


def sha256_file(path: Path) -> str:
    """Hash an on-disk file without loading it into memory all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_values(values: object) -> str:
    """Hash a JSON-compatible value using stable ordering and UTF-8 encoding."""
    payload = json.dumps(values, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_execution(output_dir: Path, *, command: list[str], protocol: str = "",
                     fold: str = "", seed: int | str = "", data_version: str = "",
                     manifest_path: Path | None = None, config_path: Path | None = None,
                     checkpoint: Path | None = None, output: Path | None = None,
                     status: str = "completed", error_message: str = "") -> None:
    """Append a complete, failure-tolerant execution record for stage three."""
    manifest_hash = sha256_file(manifest_path) if manifest_path and manifest_path.is_file() else ""
    config_hash = sha256_file(config_path) if config_path and config_path.is_file() else ""
    try:
        environment = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"],
                                     capture_output=True, text=True, check=False).stdout.strip().splitlines()
    except OSError:
        environment = []
    manifest_file = output_dir / "execution_manifest.json"
    payload: dict[str, list[dict[str, Any]]] = {"executions": []}
    if manifest_file.is_file():
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    payload.setdefault("executions", []).append({
        "timestamp": datetime.now(timezone.utc).isoformat(), "command": command,
        "protocol": protocol, "fold": fold, "seed": seed, "data_version": data_version,
        "manifest_hash": manifest_hash, "config_hash": config_hash,
        "environment": environment, "checkpoint": str(checkpoint or ""),
        "output": str(output or ""), "status": status, "error_message": error_message,
    })
    safe_json_dump(payload, manifest_file)


def read_best_checkpoint(run_dir: Path) -> Path:
    """Return the single retained best checkpoint from a checkpoint-clean run."""
    checkpoints = sorted((run_dir / "ckpt").glob("*.ckpt"), key=lambda path: int(path.stem))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected exactly one best checkpoint in {run_dir / 'ckpt'}, found {len(checkpoints)}")
    return checkpoints[0]


def metric_frame(predictions: pd.DataFrame, metrics: dict[str, float]) -> pd.DataFrame:
    """Create a metric table indexed by split and target from long predictions."""
    rows: list[dict[str, object]] = []
    for (split, target), group in predictions.groupby(["split", "target"], sort=True):
        from common import metric_dict

        rows.append({"split": split, "target": target, "n": len(group),
                     **metric_dict(group["y_true"], group["y_pred"]), **metrics})
    return pd.DataFrame(rows)
