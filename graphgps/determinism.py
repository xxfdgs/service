"""Deterministic runtime and checkpoint metadata helpers for GraphGPS runs."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.checkpoint import get_ckpt_path, save_ckpt


def stable_file_hash(path: str | Path) -> str:
    """Return a SHA256 hash for an existing file or an empty marker."""
    resolved = Path(path)
    if not resolved.is_file():
        return ""
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    """Hash JSON-compatible metadata using a canonical representation."""
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def configure_determinism(seed: int, enabled: bool = True) -> dict[str, Any]:
    """Seed all supported RNGs before loaders, models, and optimizers exist."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(enabled)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(bool(enabled), warn_only=True)
    return {
        "seed": int(seed),
        "deterministic": bool(enabled),
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_available": torch.cuda.is_available(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def dataloader_generator(seed: int, offset: int = 0) -> torch.Generator:
    """Create an explicit generator for a deterministic DataLoader sequence."""
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator


def dataloader_worker_init(worker_id: int) -> None:
    """Synchronize Python and NumPy worker RNGs with PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def checkpoint_metadata(epoch: int, best_metric: float | None) -> dict[str, Any]:
    """Create immutable provenance fields required to audit a saved checkpoint."""
    manifest_path = str(getattr(cfg.train, "manifest_path", "") or cfg.dataset.diagnostic_split_path)
    config_path = Path(cfg.run_dir) / "config.yaml"
    if not config_path.is_file():
        config_path = Path(cfg.out_dir) / "config.yaml"
    feature_path = str(getattr(cfg, "mordred_feature_path", ""))
    sample_hash = ""
    if manifest_path and Path(manifest_path).is_file():
        import pandas as pd

        manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
        sample_hash = stable_json_hash(manifest["sample_id"].astype(str).tolist())
    return {
        "stage3_checkpoint_metadata": True,
        "epoch": int(epoch),
        "best_metric": None if best_metric is None else float(best_metric),
        "seed": int(cfg.seed),
        "fold": str(getattr(cfg.train, "fold", "")),
        "protocol": str(getattr(cfg.train, "protocol", "")),
        "manifest_path": manifest_path,
        "manifest_hash": stable_file_hash(manifest_path),
        "config_hash": stable_file_hash(config_path),
        "feature_hash": stable_file_hash(feature_path),
        "sample_id_hash": sample_hash,
        "target_scaler": {"type": "fixed_percent", "scale": 100.0},
    }


def save_checkpoint_with_metadata(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None, epoch: int, best_metric: float | None = None,
) -> None:
    """Save GraphGym state and augment it with stage-three provenance metadata."""
    save_ckpt(model, optimizer, scheduler, epoch)
    if not bool(getattr(cfg.train, "deterministic", False)):
        return
    checkpoint_path = Path(get_ckpt_path(epoch))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint.update(checkpoint_metadata(epoch, best_metric))
    torch.save(checkpoint, checkpoint_path)
