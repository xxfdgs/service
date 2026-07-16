#!/usr/bin/env python3
"""Audit and export frozen representations from the legacy GraphGPS model.

The legacy model fuses *predictions*, not three embeddings.  This utility
therefore records the historical head input as ``fused_embedding`` and keeps
the prediction-level softmax output separately as ``final_prediction``.  It
never calls backward or modifies a model parameter.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphgps  # noqa: F401,E402
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from graphgps.determinism import configure_determinism  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402


TARGETS = [
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
]
EMBEDDINGS = [
    "graph_branch_raw",
    "descriptor_branch_raw",
    "formula_branch_raw",
    "graph_branch_projected",
    "descriptor_branch_projected",
    "formula_branch_projected",
    "fused_embedding",
    "head_hidden",
    "final_prediction",
]


@dataclass(frozen=True)
class CheckpointSpec:
    fold: str
    epoch_label: str
    epoch_number: int
    path: Path
    provenance: str
    notes: str


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(np.asarray(tensor.numpy()).tobytes())
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Cannot encode {type(value)!r}")


def append_manifest(root: Path, **record: Any) -> None:
    path = root / "execution_manifest.json"
    existing: list[dict[str, Any]] = []
    if path.is_file():
        existing = json.loads(path.read_text())
    base = {
        "timestamp": now(), "command": " ".join(sys.argv), "stage": None,
        "fold": None, "split": None, "epoch": None, "checkpoint": None,
        "embedding_name": None, "probe": None, "seed": 0,
        "dataset_hash": None, "manifest_hash": None, "feature_hash": None,
        "config_hash": None, "checkpoint_hash": None, "embedding_hash": None,
        "status": "completed", "error": None, "output_path": None,
    }
    base.update(record)
    existing.append(base)
    path.write_text(json.dumps(existing, indent=2, default=json_default) + "\n")


def prepare_batches(items: list[Any], split: str, device: torch.device) -> list[Any]:
    for batch, suffix in zip(items, ("", "_2", "_3", "_4", "_5")):
        batch.split = split + suffix
        batch.to(device)
    return items


class RepresentationHooks:
    """Non-invasive hooks for representations absent from legacy diagnostics."""

    def __init__(self, core: torch.nn.Module):
        self.core = core
        self.graph_raw: list[torch.Tensor] = []
        self.fusion_input: torch.Tensor | None = None
        self.head_hidden: torch.Tensor | None = None
        self.handles = [
            core.gnn.register_forward_hook(self._graph_hook),
            core.FC_layers[0].register_forward_pre_hook(self._fusion_hook),
            core.FC_layers[2].register_forward_pre_hook(self._head_hook),
        ]

    def _graph_hook(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        self.graph_raw.append(self.core.pooling_fun(output.x, output.batch).detach().cpu())

    def _fusion_hook(self, _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        self.fusion_input = inputs[0].detach().cpu()

    def _head_hook(self, _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        self.head_hidden = inputs[0].detach().cpu()

    def start(self) -> None:
        self.graph_raw = []
        self.fusion_input = None
        self.head_hidden = None

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.graph_raw) != 5 or self.fusion_input is None or self.head_hidden is None:
            raise RuntimeError(
                "Representation hooks did not observe the expected five graph calls, "
                f"fusion input, and head input ({len(self.graph_raw)}, "
                f"{self.fusion_input is not None}, {self.head_hidden is not None}).")
        return (
            torch.cat(self.graph_raw, dim=1).numpy(),
            self.fusion_input.numpy(),
            self.head_hidden.numpy(),
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def checkpoint_specs() -> list[CheckpointSpec]:
    """Frozen source inventory; no source checkpoint is copied or overwritten."""
    p = ROOT / "results"
    f0 = p / "fold4_collapse_audit/diagnostic_ablations/controls/fold0_baseline_original_scheduler_60"
    f4 = p / "fold4_collapse_audit/reproduction/reproduction_a_exact"
    cv = p / "deduplicated_rebaseline/graphgps_cv/training"
    return [
        CheckpointSpec("fold_0", "epoch_initial", 0, f0 / "checkpoints/best_candidate_epoch_0.pt",
                       "reproducible_rerun", "First valid validation checkpoint after epoch 0."),
        CheckpointSpec("fold_0", "epoch_precollapse", 30, f0 / "checkpoints/best_candidate_epoch_30.pt",
                       "reproducible_rerun", "Requested epoch-30 development timepoint."),
        CheckpointSpec("fold_0", "epoch_collapse", 40, f0 / "checkpoints/best_candidate_epoch_40.pt",
                       "reproducible_rerun", "Comparable progress point; fold 0 has no documented collapse."),
        CheckpointSpec("fold_0", "epoch_best", 106,
                       cv / "formula_identity_group_cv_fold_0_seed_0/0/ckpt/106.ckpt",
                       "historical_cv", "Current historical validation-selected checkpoint."),
        CheckpointSpec("fold_0", "epoch_last", 59, f0 / "resume_state.pt",
                       "reproducible_rerun", "Last available state from same-mathematics diagnostic rerun."),
        CheckpointSpec("fold_4", "epoch_initial", 0, f4 / "checkpoints/best_candidate_epoch_0.pt",
                       "reproducible_rerun", "First valid validation checkpoint after epoch 0."),
        CheckpointSpec("fold_4", "epoch_precollapse", 31, f4 / "checkpoints/best_candidate_epoch_31.pt",
                       "reproducible_rerun", "Nearest preserved timepoint to requested epoch 30."),
        CheckpointSpec("fold_4", "epoch_collapse", 42, f4 / "checkpoints/best_candidate_epoch_42.pt",
                       "reproducible_rerun", "Documented post-saturation collapse timepoint."),
        CheckpointSpec("fold_4", "epoch_best", 49,
                       cv / "formula_identity_group_cv_fold_4_seed_0/0/ckpt/49.ckpt",
                       "historical_cv", "Current historical validation-selected checkpoint."),
        CheckpointSpec("fold_4", "epoch_last", 100, f4 / "resume_state.pt",
                       "reproducible_rerun", "Last saved state from exact reproduction."),
        CheckpointSpec("fold_1", "epoch_best", 110,
                       cv / "formula_identity_group_cv_fold_1_seed_0/0/ckpt/110.ckpt",
                       "historical_cv", "Historical validation-selected checkpoint used only after development candidate lock."),
        CheckpointSpec("fold_2", "epoch_best", 110,
                       cv / "formula_identity_group_cv_fold_2_seed_0/0/ckpt/110.ckpt",
                       "historical_cv", "Untouched confirmation checkpoint under the locked epoch-best rule."),
        CheckpointSpec("fold_3", "epoch_best", 154,
                       cv / "formula_identity_group_cv_fold_3_seed_0/0/ckpt/154.ckpt",
                       "historical_cv", "Untouched confirmation checkpoint under the locked epoch-best rule."),
    ]


def read_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in payload:
        raise KeyError(f"{path} does not contain model_state")
    return payload["model_state"], payload


def config_for_fold(fold: str) -> Path:
    return ROOT / "results/deduplicated_rebaseline/graphgps_cv/configs" / f"formula_identity_group_cv_{fold}_seed_0.yaml"


def load_runtime(spec: CheckpointSpec, work_root: Path):
    config_path = config_for_fold(spec.fold)
    set_cfg_gps(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(config_path), opts=[]))
    cfg.run_dir = str(work_root / "cache" / spec.fold)
    cfg.out_dir = cfg.run_dir
    # Preserve the fold-specific isolated cache named by the original config.
    # Rebuilding a fresh cache could silently change a molecular featurization,
    # while refreshing this source cache would be destructive.  It is read
    # only during frozen inference.
    cfg.dataset.cache_refresh = False
    configure_determinism(int(cfg.seed), bool(cfg.train.deterministic))
    cache_log = work_root / "cache" / f"{spec.fold}_cache_build.log"
    cache_log.parent.mkdir(parents=True, exist_ok=True)
    with cache_log.open("a") as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            loaders = create_loader_5()
    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model = create_model_gps().to(device)
    state, payload = read_state(spec.path)
    model.load_state_dict(state, strict=True)
    return model, model.model, loaders, device, payload


def manifest_lookup(manifest_path: Path) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str, "group_id": str})
    if manifest.sample_id.duplicated().any() or manifest.original_row_index.duplicated().any():
        raise ValueError(f"Manifest has duplicate alignment identifiers: {manifest_path}")
    return manifest, {int(row.original_row_index): row.to_dict() for _, row in manifest.iterrows()}


def extract_split(model: torch.nn.Module, core: torch.nn.Module, loaders: list[Any], split: str,
                  device: torch.device) -> dict[str, np.ndarray]:
    hooks = RepresentationHooks(core)
    outputs: dict[str, list[np.ndarray]] = {name: [] for name in EMBEDDINGS}
    metadata: dict[str, list[np.ndarray]] = {
        "source_index": [], "labels": [], "branch_weights": [], "fusion_entropy": [],
        "component_count": [], "ratio_sum": [], "fifth_component_mask": [],
    }
    split_index = {"train": 0, "val": 1, "test": 2}[split]
    try:
        model.eval()
        with torch.no_grad():
            for batches in zip(*[group[split_index] for group in loaders]):
                batches = prepare_batches(list(batches), split, device)
                ratio_raw = torch.cat([batch.ratio.view(-1, 1).float() for batch in batches], dim=1)
                descriptor_raw = torch.cat([
                    batch.mordred_feat.view(batch.num_graphs, -1).float() for batch in batches
                ], dim=1)
                hooks.start()
                prediction, labels = model(*batches)
                graph_raw, fusion_input, head_hidden = hooks.result()
                diag = core.last_diagnostics
                expected = {"graph_input", "descriptor_input", "formula_input", "legacy_branch_weights"}
                missing = expected.difference(diag)
                if missing:
                    raise KeyError(f"Legacy diagnostics missing {sorted(missing)}")
                weights = diag["legacy_branch_weights"].detach().cpu()
                entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=-1)
                batch_size = int(ratio_raw.size(0))
                outputs["graph_branch_raw"].append(graph_raw)
                outputs["descriptor_branch_raw"].append(descriptor_raw.detach().cpu().numpy())
                outputs["formula_branch_raw"].append(ratio_raw.detach().cpu().numpy())
                outputs["graph_branch_projected"].append(diag["graph_input"].detach().cpu().numpy())
                # The historical model has no descriptor projection.  This
                # identity alias is deliberate and recorded in the schema.
                outputs["descriptor_branch_projected"].append(diag["descriptor_input"].detach().cpu().numpy())
                outputs["formula_branch_projected"].append(diag["formula_input"].detach().cpu().numpy())
                outputs["fused_embedding"].append(fusion_input)
                outputs["head_hidden"].append(head_hidden)
                outputs["final_prediction"].append(prediction.detach().cpu().reshape(batch_size, 4).numpy())
                metadata["source_index"].append(batches[0].sample_uid.detach().cpu().view(-1).numpy())
                metadata["labels"].append(labels.detach().cpu().reshape(batch_size, 4).numpy())
                metadata["branch_weights"].append(weights.numpy())
                metadata["fusion_entropy"].append(entropy.numpy())
                metadata["component_count"].append((ratio_raw > 0).sum(dim=1).detach().cpu().numpy())
                metadata["ratio_sum"].append(ratio_raw.sum(dim=1).detach().cpu().numpy())
                metadata["fifth_component_mask"].append((ratio_raw[:, 4] > 0).long().detach().cpu().numpy())
    finally:
        hooks.close()
    merged = {key: np.vstack(value).astype(np.float32, copy=False) for key, value in outputs.items()}
    merged.update({
        "source_index": np.concatenate(metadata["source_index"]).astype(np.int64, copy=False),
        "labels": np.vstack(metadata["labels"]).astype(np.float32, copy=False),
        "branch_weights": np.vstack(metadata["branch_weights"]).astype(np.float32, copy=False),
        "fusion_entropy": np.vstack(metadata["fusion_entropy"]).astype(np.float32, copy=False),
        "component_count": np.concatenate(metadata["component_count"]).astype(np.int64, copy=False),
        "ratio_sum": np.concatenate(metadata["ratio_sum"]).astype(np.float32, copy=False),
        "fifth_component_mask": np.concatenate(metadata["fifth_component_mask"]).astype(np.int64, copy=False),
    })
    return merged


def validate_split_arrays(arrays: dict[str, np.ndarray], split: str, manifest: pd.DataFrame,
                          lookup: dict[int, dict[str, Any]], spec: CheckpointSpec) -> list[dict[str, Any]]:
    source = arrays["source_index"]
    if len(source) != len(np.unique(source)):
        raise ValueError(f"Duplicate source_index in {spec.fold}/{spec.epoch_label}/{split}")
    observed = {lookup[int(index)]["sample_id"] for index in source}
    expected = set(manifest.loc[manifest.split.eq(split), "sample_id"])
    if observed != expected:
        raise ValueError(
            f"Sample alignment failure for {spec.fold}/{spec.epoch_label}/{split}: "
            f"observed={len(observed)}, expected={len(expected)}, missing={len(expected-observed)}, extra={len(observed-expected)}")
    rows = []
    for name in EMBEDDINGS:
        vector = arrays[name]
        finite = bool(np.isfinite(vector).all())
        if not finite:
            raise ValueError(f"Non-finite values in {name} for {spec.fold}/{spec.epoch_label}/{split}")
        rows.append({
            "fold": spec.fold, "epoch_label": spec.epoch_label, "epoch": spec.epoch_number,
            "split": split, "embedding_name": name, "n_samples": int(vector.shape[0]),
            "embedding_dim": int(vector.shape[1]), "finite": finite,
            "unique_sample_ids": int(len(observed)), "manifest_sample_ids": int(len(expected)),
            "alignment_pass": True,
        })
    return rows


def write_embedding_archives(root: Path, spec: CheckpointSpec, arrays_by_split: dict[str, dict[str, np.ndarray]],
                             manifest: pd.DataFrame, lookup: dict[int, dict[str, Any]], checkpoint_hash: str,
                             state_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, Any]] = []
    for split, arrays in arrays_by_split.items():
        audit_rows.extend(validate_split_arrays(arrays, split, manifest, lookup, spec))
        directory = root / "embeddings" / spec.fold / spec.epoch_label
        directory.mkdir(parents=True, exist_ok=True)
        source = arrays["source_index"]
        sample_ids = np.asarray([lookup[int(index)]["sample_id"] for index in source], dtype=str)
        group_ids = np.asarray([lookup[int(index)]["group_id"] for index in source], dtype=str)
        labels_raw = arrays["labels"] * 100.0
        target_mask = np.isfinite(labels_raw).astype(np.uint8)
        for name in EMBEDDINGS:
            vector = arrays[name]
            output_path = directory / f"{split}_{name}.npz"
            np.savez_compressed(
                output_path, embedding=vector, sample_id=sample_ids, source_index=source,
                group_id=group_ids, labels=labels_raw, target_valid_mask=target_mask,
                branch_weights=arrays["branch_weights"], fusion_entropy=arrays["fusion_entropy"],
                component_count=arrays["component_count"], ratio_sum=arrays["ratio_sum"],
                fifth_component_mask=arrays["fifth_component_mask"],
            )
            embedding_hash = sha256_path(output_path)
            numeric_rows.append({
                "fold": spec.fold, "epoch_label": spec.epoch_label, "epoch": spec.epoch_number,
                "split": split, "embedding_name": name, "dtype": str(vector.dtype),
                "embedding_dim": int(vector.shape[1]), "n_samples": int(vector.shape[0]),
                "nan_count": int(np.isnan(vector).sum()), "inf_count": int(np.isinf(vector).sum()),
                "min": float(vector.min()), "max": float(vector.max()), "embedding_hash": embedding_hash,
                "path": str(output_path.relative_to(root)),
            })
            for index, sample_id, group_id, label, mask, weights, entropy, count, ratio_sum, fifth in zip(
                    source, sample_ids, group_ids, labels_raw, target_mask, arrays["branch_weights"],
                    arrays["fusion_entropy"], arrays["component_count"], arrays["ratio_sum"], arrays["fifth_component_mask"]):
                index_rows.append({
                    "sample_id": sample_id, "source_index": int(index), "fold": spec.fold, "split": split,
                    "group_id": group_id, "epoch": spec.epoch_number, "epoch_label": spec.epoch_label,
                    "checkpoint_path": str(spec.path), "checkpoint_hash": checkpoint_hash,
                    "model_state_hash": state_hash, "embedding_name": name,
                    "embedding_path": str(output_path.relative_to(root)),
                    "y_EE_before": float(label[0]), "y_EE_after": float(label[1]),
                    "y_Aerosolization_Efficiency": float(label[2]),
                    "y_mRNA_Recovery_Efficiency": float(label[3]),
                    "target_valid_mask": json.dumps(mask.tolist()),
                    "branch_weights": json.dumps(weights.tolist()),
                    "fusion_entropy": json.dumps(entropy.tolist()),
                    "component_count": int(count), "ratio_sum": float(ratio_sum),
                    "fifth_component_mask": int(fifth),
                })
    return audit_rows, numeric_rows, index_rows


def write_audit(root: Path) -> None:
    audit = root / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    locations = """# Frozen embedding locations\n\nThe audited checkpoint is the legacy `GPSDoubleModel_multi4_cat_v0`. It has three prediction branches, not three embedding branches. `branch_weight_mlp` receives only the 20-dimensional ratio features and uses target-specific softmax weights over `pred_main`, `pred_direct`, and `pred_middle`. The fifth-component additive delta is then added after this prediction-level fusion.\n\n| requested name | actual source | dim | transformations / caveat |\n| --- | --- | ---: | --- |\n| graph_branch_raw | five `gnn` outputs pooled by `add` | 5×64=320 | GraphGPS encoder pooled representation; before ratio/type/LayerNorm composition. |\n| descriptor_branch_raw | concatenated five Mordred/RDKit vectors | 5×11=55 | Raw cached descriptors; legacy model has no descriptor encoder. |\n| formula_branch_raw | five raw ratios | 5 | Raw mixture proportions. |\n| graph_branch_projected | `component_norm(graph + ratio_encoder + type_embedding)` | 320 | LayerNorm then absent-component zero mask; this is a composition embedding, not a learned fusion projection. |\n| descriptor_branch_projected | identity alias of descriptor raw | 55 | No descriptor projection exists in this legacy architecture. |\n| formula_branch_projected | `_ratio_features` for five components | 20 | `[r, sqrt(r), log1p(100r)/log(101), present]` each. |\n| fused_embedding | pre-hook input to `FC_layers[0]` | 395 | Concatenation of graph projected, formula projected, and descriptor raw. Historical model has no embedding-level softmax fusion. |\n| head_hidden | pre-hook input to `FC_layers[2]` | 256 | `ReLU(FC_layers[1](ReLU(FC_layers[0](fused_embedding))))`. |\n| final_prediction | model forward return | 4 | Prediction-level branch softmax plus fifth-component additive delta; target-specific. |\n\nAll exports run `model.eval()` and `torch.no_grad()`. GraphGPS/attention dropout and the additive delta dropout are therefore disabled. No representation contains `sample_id` or `group_id` as a direct input. Component position embeddings are legitimate component-slot identifiers, not sample/group shortcuts.\n"""
    (audit / "embedding_locations.md").write_text(locations)
    schema = {
        "architecture": "GPSDoubleModel_multi4_cat_v0 legacy_baseline",
        "eval_mode": True,
        "dropout_disabled": True,
        "target_specific": {
            "graph_branch_raw": False, "descriptor_branch_raw": False, "formula_branch_raw": False,
            "graph_branch_projected": False, "descriptor_branch_projected": False,
            "formula_branch_projected": False, "fused_embedding": False, "head_hidden": False,
            "final_prediction": True,
        },
        "identity_shortcut": "No sample_id/group_id field is fed to the model; component_type_emb encodes only component position.",
        "embeddings": {
            "graph_branch_raw": {"shape": "[N,320]", "dtype": "float32", "layernorm": False, "activation": "inside GraphGPS layers", "dropout_eval": False},
            "descriptor_branch_raw": {"shape": "[N,55]", "dtype": "float32", "layernorm": False, "activation": None, "dropout_eval": False},
            "formula_branch_raw": {"shape": "[N,5]", "dtype": "float32", "layernorm": False, "activation": None, "dropout_eval": False},
            "graph_branch_projected": {"shape": "[N,320]", "dtype": "float32", "layernorm": True, "activation": "ratio_encoder ReLU before addition", "dropout_eval": False},
            "descriptor_branch_projected": {"shape": "[N,55]", "dtype": "float32", "layernorm": False, "activation": "identity alias", "dropout_eval": False},
            "formula_branch_projected": {"shape": "[N,20]", "dtype": "float32", "layernorm": False, "activation": "deterministic ratio transforms", "dropout_eval": False},
            "fused_embedding": {"shape": "[N,395]", "dtype": "float32", "layernorm": False, "activation": "concatenation only", "dropout_eval": False},
            "head_hidden": {"shape": "[N,256]", "dtype": "float32", "layernorm": False, "activation": "ReLU", "dropout_eval": False},
            "final_prediction": {"shape": "[N,4]", "dtype": "float32", "layernorm": False, "activation": None, "dropout_eval": False},
        },
    }
    (audit / "embedding_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    (audit / "hook_implementation.md").write_text(
        "# Hook implementation\n\n`RepresentationHooks` uses detached forward hooks only: `gnn` output is add-pooled for `graph_branch_raw`; the pre-hook of `FC_layers[0]` yields `fused_embedding`; and the pre-hook of `FC_layers[2]` yields the activated final hidden layer. Other branch tensors are read from the model's existing detached `last_diagnostics`. Hooks are removed in `finally`; no source parameter, checkpoint, optimizer, gradient, or model forward branch is changed.\n")
    (audit / "architecture_map.md").write_text("""# Architecture map

```
five component graphs -> shared GraphGPS -> add pool -> graph_branch_raw
                     + ratio encoder + slot embedding -> LayerNorm/mask -> graph_branch_projected
five 11D descriptors -----------------------------------------------> descriptor_branch_raw
five ratios -> deterministic 4D expansion --------------------------> formula_branch_projected
graph_projected || formula_projected || descriptor_raw -> fused_embedding
  -> Linear(395,256)+ReLU -> Linear(256,256)+ReLU -> head_hidden -> main prediction
  -> direct prediction; middle prediction
ratio features -> target-specific 3-way softmax -> prediction-level weighted sum
matrix/fifth branch -> additive delta -> final_prediction
```

There is no separate descriptor encoder, formula encoder, or embedding-level softmax fusion in the historical checkpoint.
""")


def write_checkpoint_metadata(root: Path, specs: list[CheckpointSpec]) -> None:
    directory = root / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    selected: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if not spec.path.is_file():
            raise FileNotFoundError(spec.path)
        state, payload = read_state(spec.path)
        row = {
            "fold": spec.fold, "epoch_label": spec.epoch_label, "epoch": spec.epoch_number,
            "checkpoint_path": str(spec.path), "checkpoint_hash": sha256_path(spec.path),
            "model_state_hash": model_state_hash(state), "payload_epoch": payload.get("epoch"),
            "provenance": spec.provenance, "notes": spec.notes, "exists": True,
        }
        rows.append(row)
        selected.setdefault(spec.fold, {})[spec.epoch_label] = {
            "epoch": spec.epoch_number, "checkpoint_path": str(spec.path),
            "provenance": spec.provenance, "notes": spec.notes,
        }
    pd.DataFrame(rows).to_csv(directory / "checkpoint_inventory.csv", index=False)
    pd.DataFrame(rows).to_csv(directory / "checkpoint_provenance.csv", index=False)
    (directory / "selected_epochs.json").write_text(json.dumps(selected, indent=2) + "\n")


def run_export(root: Path, folds: set[str], labels: set[str]) -> None:
    specs = [item for item in checkpoint_specs() if item.fold in folds and item.epoch_label in labels]
    if not specs:
        raise ValueError("No checkpoint specs selected")
    write_checkpoint_metadata(root, checkpoint_specs())
    all_audit: list[dict[str, Any]] = []
    all_numeric: list[dict[str, Any]] = []
    all_index: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"exports": []}
    for spec in specs:
        model, core, loaders, device, _payload = load_runtime(spec, root)
        manifest, lookup = manifest_lookup(Path(cfg.train.manifest_path))
        dataset_hash = manifest.dataset_sha256.iloc[0]
        manifest_hash = manifest.manifest_sha256.iloc[0]
        checkpoint_hash = sha256_path(spec.path)
        state, _ = read_state(spec.path)
        state_hash = model_state_hash(state)
        arrays_by_split = {split: extract_split(model, core, loaders, split, device) for split in ("train", "val", "test")}
        audit_rows, numeric_rows, index_rows = write_embedding_archives(
            root, spec, arrays_by_split, manifest, lookup, checkpoint_hash, state_hash)
        all_audit.extend(audit_rows)
        all_numeric.extend(numeric_rows)
        all_index.extend(index_rows)
        metadata["exports"].append({
            "fold": spec.fold, "epoch_label": spec.epoch_label, "epoch": spec.epoch_number,
            "checkpoint": str(spec.path), "checkpoint_hash": checkpoint_hash, "model_state_hash": state_hash,
            "dataset_hash": dataset_hash, "manifest_hash": manifest_hash,
            "config_hash": sha256_path(config_for_fold(spec.fold)),
            "feature_hash": sha256_path(ROOT / "results/deduplicated_rebaseline/artifacts/mordred_11_lookup.csv"),
        })
        append_manifest(root, stage="embedding_export", fold=spec.fold, epoch=spec.epoch_number,
                        checkpoint=str(spec.path), dataset_hash=dataset_hash, manifest_hash=manifest_hash,
                        feature_hash=metadata["exports"][-1]["feature_hash"], config_hash=metadata["exports"][-1]["config_hash"],
                        checkpoint_hash=checkpoint_hash, status="completed",
                        output_path=str(root / "embeddings" / spec.fold / spec.epoch_label))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(all_audit).to_csv(root / "audit" / "embedding_alignment_audit.csv", index=False)
    pd.DataFrame(all_numeric).to_csv(root / "audit" / "embedding_numerical_audit.csv", index=False)
    pd.DataFrame(all_index).to_csv(root / "embeddings" / "embedding_index.csv", index=False,
                                   quoting=csv.QUOTE_MINIMAL)
    (root / "embeddings" / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run_determinism(root: Path) -> None:
    spec = next(item for item in checkpoint_specs() if item.fold == "fold_4" and item.epoch_label == "epoch_best")
    model, core, loaders, device, _ = load_runtime(spec, root)
    first = {split: extract_split(model, core, loaders, split, device) for split in ("train", "val", "test")}
    second = {split: extract_split(model, core, loaders, split, device) for split in ("train", "val", "test")}
    rows = []
    for split in first:
        if not np.array_equal(first[split]["source_index"], second[split]["source_index"]):
            raise ValueError("Determinism comparison source ordering mismatch")
        for name in EMBEDDINGS:
            delta = np.abs(first[split][name].astype(np.float64) - second[split][name].astype(np.float64))
            rows.append({"fold": spec.fold, "epoch_label": spec.epoch_label, "split": split,
                         "embedding_name": name, "max_abs_difference": float(delta.max()),
                         "deterministic_pass": bool(delta.max() < 1e-6)})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "audit" / "embedding_determinism_audit.csv", index=False)
    if not frame.deterministic_pass.all():
        raise RuntimeError("Eval-mode representation export was not deterministic within 1e-6")
    append_manifest(root, stage="determinism_audit", fold=spec.fold, epoch=spec.epoch_number,
                    checkpoint=str(spec.path), checkpoint_hash=sha256_path(spec.path), status="completed",
                    output_path=str(root / "audit" / "embedding_determinism_audit.csv"))


def consolidate_archives(root: Path) -> None:
    """Rebuild tabular audits from durable NPZ archives after an interrupted batch.

    Each archive is self-contained, so this operation is read-only with
    respect to both checkpoints and vectors.  It makes export batches safely
    resumable when an external runtime limit ends a long GPU invocation.
    """
    specs = {(item.fold, item.epoch_label): item for item in checkpoint_specs()}
    audit_rows: list[dict[str, Any]] = []
    numerical_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    seen_archives: set[Path] = set()
    for path in sorted((root / "embeddings").glob("fold_*/*/*.npz")):
        relative = path.relative_to(root)
        fold, epoch_label = path.parts[-3], path.parts[-2]
        split, embedding_tail = path.stem.split("_", 1)
        spec = specs.get((fold, epoch_label))
        if spec is None or embedding_tail not in EMBEDDINGS:
            raise ValueError(f"Unexpected embedding archive path: {path}")
        archive = np.load(path, allow_pickle=False)
        vector = archive["embedding"]
        source = archive["source_index"].astype(np.int64)
        sample_ids = archive["sample_id"].astype(str)
        group_ids = archive["group_id"].astype(str)
        labels = archive["labels"]
        mask = archive["target_valid_mask"]
        weights = archive["branch_weights"]
        entropy = archive["fusion_entropy"]
        component_count = archive["component_count"]
        ratio_sum = archive["ratio_sum"]
        fifth_mask = archive["fifth_component_mask"]
        manifest, _lookup = manifest_lookup(
            ROOT / "results/deduplicated_rebaseline/manifests/formula_identity_group_cv" / f"{fold}.csv")
        expected = manifest.loc[manifest.split.eq(split)].sort_values("original_row_index")
        expected_map = {int(row.original_row_index): str(row.sample_id) for _, row in expected.iterrows()}
        observed_map = {int(index): sample_id for index, sample_id in zip(source, sample_ids)}
        if expected_map != observed_map or len(source) != len(np.unique(source)):
            raise ValueError(f"Archive alignment failure: {path}")
        checkpoint_hash = sha256_path(spec.path)
        state, _ = read_state(spec.path)
        state_hash = model_state_hash(state)
        archive_hash = sha256_path(path)
        audit_rows.append({
            "fold": fold, "epoch_label": epoch_label, "epoch": spec.epoch_number, "split": split,
            "embedding_name": embedding_tail, "n_samples": int(vector.shape[0]),
            "embedding_dim": int(vector.shape[1]), "finite": bool(np.isfinite(vector).all()),
            "unique_sample_ids": int(len(set(sample_ids))), "manifest_sample_ids": int(len(expected)),
            "alignment_pass": True,
        })
        numerical_rows.append({
            "fold": fold, "epoch_label": epoch_label, "epoch": spec.epoch_number, "split": split,
            "embedding_name": embedding_tail, "dtype": str(vector.dtype), "embedding_dim": int(vector.shape[1]),
            "n_samples": int(vector.shape[0]), "nan_count": int(np.isnan(vector).sum()),
            "inf_count": int(np.isinf(vector).sum()), "min": float(vector.min()), "max": float(vector.max()),
            "embedding_hash": archive_hash, "path": str(relative),
        })
        for values in zip(source, sample_ids, group_ids, labels, mask, weights, entropy, component_count, ratio_sum, fifth_mask):
            index, sample_id, group_id, label, valid_mask, branch_weight, entropy_row, count, ratio, fifth = values
            index_rows.append({
                "sample_id": sample_id, "source_index": int(index), "fold": fold, "split": split,
                "group_id": group_id, "epoch": spec.epoch_number, "epoch_label": epoch_label,
                "checkpoint_path": str(spec.path), "checkpoint_hash": checkpoint_hash,
                "model_state_hash": state_hash, "embedding_name": embedding_tail,
                "embedding_path": str(relative), "y_EE_before": float(label[0]), "y_EE_after": float(label[1]),
                "y_Aerosolization_Efficiency": float(label[2]), "y_mRNA_Recovery_Efficiency": float(label[3]),
                "target_valid_mask": json.dumps(valid_mask.tolist()), "branch_weights": json.dumps(branch_weight.tolist()),
                "fusion_entropy": json.dumps(entropy_row.tolist()), "component_count": int(count),
                "ratio_sum": float(ratio), "fifth_component_mask": int(fifth),
            })
        seen_archives.add(path)
    pd.DataFrame(audit_rows).sort_values(["fold", "epoch", "split", "embedding_name"]).to_csv(
        root / "audit" / "embedding_alignment_audit.csv", index=False)
    pd.DataFrame(numerical_rows).sort_values(["fold", "epoch", "split", "embedding_name"]).to_csv(
        root / "audit" / "embedding_numerical_audit.csv", index=False)
    pd.DataFrame(index_rows).sort_values(["fold", "epoch", "split", "embedding_name", "source_index"]).to_csv(
        root / "embeddings" / "embedding_index.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    exported = sorted({(row["fold"], row["epoch_label"]) for row in audit_rows})
    (root / "embeddings" / "metadata.json").write_text(json.dumps({
        "archive_count": len(seen_archives), "exported_fold_epoch_labels": exported,
        "note": "Rebuilt from self-contained NPZ archives; no test labels are used by development probes.",
    }, indent=2) + "\n")
    append_manifest(root, stage="embedding_consolidation", status="completed",
                    output_path=str(root / "embeddings" / "embedding_index.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/frozen_embedding_signal_exp")
    parser.add_argument("--stage", choices=("audit", "export", "determinism", "consolidate"), required=True)
    parser.add_argument("--folds", nargs="*", default=["fold_0", "fold_4"])
    parser.add_argument("--epoch-labels", nargs="*",
                        default=["epoch_initial", "epoch_precollapse", "epoch_collapse", "epoch_best", "epoch_last"])
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit":
        write_audit(root)
        write_checkpoint_metadata(root, checkpoint_specs())
        append_manifest(root, stage="architecture_audit", status="completed", output_path=str(root / "audit"))
    elif args.stage == "export":
        write_audit(root)
        run_export(root, set(args.folds), set(args.epoch_labels))
    elif args.stage == "determinism":
        run_determinism(root)
    else:
        consolidate_archives(root)


if __name__ == "__main__":
    main()
