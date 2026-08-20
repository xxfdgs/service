#!/usr/bin/env python3
"""
Stage 4 — pretrain the exact local OneHotEmbedGPS Fifth GraphGPS encoder.

This runner is intentionally coupled to the current biology_prediction codebase.

Exact downstream interface reused
---------------------------------
1. Graph construction:
       graph_feature.smiles2graph(...)
   with the same coarse-grain flags used by csv_pyg_five_multi.py.

2. Encoder class:
       Comp5GraphEncoder
   imported from the project's existing onehot_embed_gps.py.
   The class is NOT copied or reimplemented here.

3. Pooling:
       torch_geometric.graphgym.register.pooling_dict[cfg.model.graph_pooling]

4. Architecture/config:
   loaded from the same GraphGPS YAML used by the downstream O12/O13 runs,
   with explicit O13-D-compatible defaults:
       graph_pooling = mean
       gt.layers = 2
       gt.dropout = 0.1
       gt.attn_dropout = 0.2
   Hidden width, layer_type, node/edge encoders, heads, normalization,
   positional encodings, etc. remain sourced from the supplied YAML.

Pretraining ablations
---------------------
PT-D:
    --task descriptor_only

PT-DF:
    --task descriptor_plus_morgan

Targets
-------
Stage 3:
    descriptor_targets_scaled.npz
    morgan_fp_1024.npz
    morgan_fp_train_statistics.npz
    descriptor_train_statistics.csv
    pretraining_split.csv

Loss masks
----------
- Descriptor targets constant on pretraining train are excluded from loss.
  In the current Stage 3 output this removes FormalCharge.
- Morgan bits constant on pretraining train are excluded from loss.
  In the current Stage 3 output this retains the informative 316 bits.

Checkpoint transfer contract
----------------------------
The best checkpoint stores:
    encoder_state_dict

This state dict is EXACTLY the state dict of the imported Comp5GraphEncoder.

It additionally exports:
    best_comp5_encoder_state_dict.pt
        raw encoder state dict, suitable for:
            model.comp5_encoder.load_state_dict(state, strict=True)

    best_downstream_prefixed_state_dict.pt
        same tensors with "comp5_encoder." prefix, suitable for audited
        partial loading into OneHotEmbedGPSModel.

No Fifth_class embedding, ratio modulation, Mordred features, component aux
features, fusion MLP, or downstream property head is pretrained here. Those
belong outside Comp5GraphEncoder in OneHotEmbedGPSModel.

Positional encodings
--------------------
If the supplied GraphGPS config enables positional/structural encodings,
Stage 4 attempts to use the project's standard
graphgps.transform.posenc_stats.compute_posenc_stats before batching.
Failure is fatal rather than silently training a mismatched encoder input.

Outputs
-------
run_settings.json
encoder_interface.json
history.csv
descriptor_metrics.csv
fp_metrics.json
checkpoints/best.pt
checkpoints/last.pt
checkpoints/best_comp5_encoder_state_dict.pt
checkpoints/best_downstream_prefixed_state_dict.pt
summary.json
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg, load_cfg


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphgps  # noqa: E402,F401
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graph_feature import smiles2graph  # noqa: E402


# =============================================================================
# Utilities
# =============================================================================

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(
        True,
        warn_only=False,
        )
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def scalar_cfg(path: str, default: Any = None) -> Any:
    obj: Any = cfg
    for key in path.split("."):
        if not hasattr(obj, key):
            return default
        obj = getattr(obj, key)
    return obj


def jsonable_cfg_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable_cfg_value(v) for v in value]
    return str(value)


# =============================================================================
# Exact local Comp5GraphEncoder import
# =============================================================================

def load_local_comp5_encoder_class():
    """
    Resolve Comp5GraphEncoder from the module that is ALREADY registered as
    OneHotEmbedGPS by ``import graphgps``. This avoids re-executing
    onehot_embed_gps.py and therefore avoids duplicate GraphGym registration.

    The registered OneHotEmbedGPSModel and Comp5GraphEncoder must live in the
    same module; this is the current local downstream implementation.
    """
    import importlib

    if "OneHotEmbedGPS" not in register.network_dict:
        raise RuntimeError(
            "OneHotEmbedGPS is not registered after import graphgps; "
            "Stage 4 cannot prove that it is using the downstream encoder."
        )

    downstream_model_cls = register.network_dict["OneHotEmbedGPS"]
    module_name = downstream_model_cls.__module__
    module = importlib.import_module(module_name)

    if not hasattr(module, "Comp5GraphEncoder"):
        raise AttributeError(
            f"Registered OneHotEmbedGPS module {module_name!r} does not "
            "expose Comp5GraphEncoder"
        )

    encoder_cls = module.Comp5GraphEncoder
    source_path = Path(inspect.getfile(encoder_cls)).resolve()

    if ROOT not in source_path.parents:
        raise RuntimeError(
            "Resolved Comp5GraphEncoder is not from this repository: "
            f"{source_path}"
        )

    return encoder_cls, source_path


# =============================================================================
# Config
# =============================================================================

def load_stage4_config(args) -> dict[str, Any]:
    set_cfg_gps(cfg)
    load_cfg(
        cfg,
        SimpleNamespace(
            cfg_file=str(args.config.resolve()),
            opts=[],
        ),
    )

    # Current O13-D Fifth encoder contract.
    cfg.model.graph_pooling = args.graph_pooling
    cfg.gt.layers = int(args.gps_layers)
    cfg.gt.dropout = float(args.gt_dropout)
    cfg.gt.attn_dropout = float(args.gt_attn_dropout)

    # OneHotEmbedGPS requires these to match.
    cfg.gt.dim_hidden = int(cfg.gnn.dim_inner)

    if args.graph_hidden_dim is not None:
        hidden = int(args.graph_hidden_dim)
        if hidden <= 0:
            raise ValueError("--graph-hidden-dim must be positive")
        if hidden % int(cfg.gt.n_heads) != 0:
            raise ValueError(
                "--graph-hidden-dim must be divisible by cfg.gt.n_heads="
                f"{cfg.gt.n_heads}"
            )
        cfg.gnn.dim_inner = hidden
        cfg.gt.dim_hidden = hidden

    if int(cfg.gt.dim_hidden) != int(cfg.gnn.dim_inner):
        raise RuntimeError(
            "cfg.gt.dim_hidden and cfg.gnn.dim_inner must match"
        )

    return {
        "source_config": str(args.config.resolve()),
        "source_config_sha256": file_sha256(args.config.resolve()),
        "graph_pooling": str(cfg.model.graph_pooling),
        "gnn_dim_inner": int(cfg.gnn.dim_inner),
        "gt_dim_hidden": int(cfg.gt.dim_hidden),
        "gt_layers": int(cfg.gt.layers),
        "gt_layer_type": str(cfg.gt.layer_type),
        "gt_n_heads": int(cfg.gt.n_heads),
        "gt_dropout": float(cfg.gt.dropout),
        "gt_attn_dropout": float(cfg.gt.attn_dropout),
        "gt_layer_norm": bool(cfg.gt.layer_norm),
        "gt_batch_norm": bool(cfg.gt.batch_norm),
        "node_encoder": bool(cfg.dataset.node_encoder),
        "node_encoder_name": str(cfg.dataset.node_encoder_name),
        "edge_encoder": bool(cfg.dataset.edge_encoder),
        "edge_encoder_name": str(cfg.dataset.edge_encoder_name),
        "layers_pre_mp": int(cfg.gnn.layers_pre_mp),
        "gnn_act": str(cfg.gnn.act),
        "coarse_grain_enable": bool(
            getattr(cfg, "coarse_grain_enable", False)
        ),
        "coarse_grain_min_chain_length": int(
            getattr(cfg, "coarse_grain_min_chain_length", 0)
        ),
    }


# =============================================================================
# Positional encoding preprocessing
# =============================================================================

PE_NAMES = (
    "LapPE",
    "EquivStableLapPE",
    "SignNet",
    "RWSE",
    "HKdiagSE",
    "ElstaticSE",
)


def enabled_pe_types() -> list[str]:
    enabled = []
    for name in PE_NAMES:
        config_name = f"posenc_{name}"
        if hasattr(cfg, config_name):
            pe_cfg = getattr(cfg, config_name)
            if bool(getattr(pe_cfg, "enable", False)):
                enabled.append(name)
    return enabled


def materialize_posenc_kernel_times(pe_types: list[str]) -> dict[str, list]:
    """
    Match GraphGPS master_loader preprocessing.

    GraphGPS configs commonly store e.g.
        posenc_RWSE.kernel.times_func = "range(1,21)"
    while compute_posenc_stats() consumes
        posenc_RWSE.kernel.times

    The standard GraphGPS master loader materializes `times` from
    `times_func` before applying compute_posenc_stats. Stage 4 constructs
    Data objects directly, so it must perform that same config step itself.

    We intentionally support the range/list/tuple expressions used by the
    project's GraphGPS configs without exposing unrestricted Python eval.
    """
    resolved: dict[str, list] = {}

    safe_globals = {"__builtins__": {}}
    safe_locals = {
        "range": range,
        "list": list,
        "tuple": tuple,
    }

    for pe_name in pe_types:
        config_name = f"posenc_{pe_name}"

        if not hasattr(cfg, config_name):
            raise RuntimeError(
                f"Enabled PE {pe_name} has no cfg.{config_name}"
            )

        pe_cfg = getattr(cfg, config_name)

        if not hasattr(pe_cfg, "kernel"):
            continue

        kernel = pe_cfg.kernel
        times_func = str(
            getattr(kernel, "times_func", "")
        ).strip()

        current_times = list(
            getattr(kernel, "times", [])
        )

        if times_func:
            try:
                evaluated = eval(
                    times_func,
                    safe_globals,
                    safe_locals,
                )
                times = list(evaluated)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not materialize {config_name}.kernel.times from "
                    f"times_func={times_func!r}"
                ) from exc

            if not times:
                raise RuntimeError(
                    f"{config_name}.kernel.times_func produced an empty list: "
                    f"{times_func!r}"
                )

            kernel.times = times
            current_times = list(kernel.times)

        if not current_times:
            raise RuntimeError(
                f"{config_name} requires kernel times, but neither "
                "kernel.times nor a usable kernel.times_func is configured."
            )

        resolved[pe_name] = current_times

    return resolved


def apply_posenc_if_needed(data: Data, pe_types: list[str]) -> Data:
    if not pe_types:
        return data

    try:
        from graphgps.transform.posenc_stats import compute_posenc_stats
    except Exception as exc:
        raise RuntimeError(
            f"Config enables positional encodings {pe_types}, but "
            "graphgps.transform.posenc_stats.compute_posenc_stats could not "
            "be imported. Refusing a mismatched Stage-4 graph pipeline."
        ) from exc

    signature = inspect.signature(compute_posenc_stats)
    parameters = list(signature.parameters)

    try:
        if len(parameters) >= 4:
            result = compute_posenc_stats(
                data,
                pe_types,
                True,   # molecular graph is undirected
                cfg,
            )
        elif len(parameters) == 3:
            result = compute_posenc_stats(
                data,
                pe_types,
                True,
            )
        else:
            raise RuntimeError(
                "Unsupported compute_posenc_stats signature: "
                f"{signature}"
            )
    except Exception as exc:
        raise RuntimeError(
            f"Failed computing positional encodings {pe_types} for a Stage-4 "
            "molecule. Refusing to continue with an encoder-input mismatch."
        ) from exc

    return data if result is None else result


# =============================================================================
# Target loading
# =============================================================================

def load_npz_indexed(path: Path, value_key: str):
    payload = np.load(path, allow_pickle=False)
    ids = payload["stage2c_id"].astype(str)
    values = payload[value_key]

    if len(ids) != len(values):
        raise ValueError(
            f"{path}: stage2c_id and {value_key} length mismatch"
        )

    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{path}: duplicate stage2c_id")

    return ids, values, payload


def align_targets(
    library: pd.DataFrame,
    stage3_dir: Path,
) -> dict[str, Any]:
    split_df = pd.read_csv(
        stage3_dir / "pretraining_split.csv",
        dtype={"stage2c_id": str},
    )

    if split_df["stage2c_id"].duplicated().any():
        raise ValueError("pretraining_split.csv has duplicate stage2c_id")

    desc_ids, desc_targets, desc_npz = load_npz_indexed(
        stage3_dir / "descriptor_targets_scaled.npz",
        "targets",
    )
    fp_ids, fp_targets, fp_npz = load_npz_indexed(
        stage3_dir / "morgan_fp_1024.npz",
        "fingerprints",
    )

    library_ids = library["stage2c_id"].astype(str).tolist()

    if set(library_ids) != set(split_df["stage2c_id"].astype(str)):
        raise ValueError(
            "Stage-2C library IDs and Stage-3 split IDs differ"
        )
    if set(library_ids) != set(desc_ids.tolist()):
        raise ValueError(
            "Stage-2C library IDs and descriptor target IDs differ"
        )
    if set(library_ids) != set(fp_ids.tolist()):
        raise ValueError(
            "Stage-2C library IDs and Morgan target IDs differ"
        )

    desc_map = {sid: i for i, sid in enumerate(desc_ids)}
    fp_map = {sid: i for i, sid in enumerate(fp_ids)}
    split_map = dict(
        zip(
            split_df["stage2c_id"].astype(str),
            split_df["split"].astype(str),
        )
    )

    desc_aligned = np.stack(
        [desc_targets[desc_map[sid]] for sid in library_ids],
        axis=0,
    ).astype(np.float32)

    fp_aligned = np.stack(
        [fp_targets[fp_map[sid]] for sid in library_ids],
        axis=0,
    ).astype(np.float32)

    splits = np.asarray(
        [split_map[sid] for sid in library_ids],
        dtype=str,
    )

    descriptor_names = desc_npz["descriptor_names"].astype(str).tolist()

    desc_stats = pd.read_csv(
        stage3_dir / "descriptor_train_statistics.csv"
    )

    stats_map = {
        str(row.descriptor): bool(row.constant_train)
        for row in desc_stats.itertuples(index=False)
    }

    desc_mask = np.asarray(
        [
            not stats_map.get(name, False)
            for name in descriptor_names
        ],
        dtype=bool,
    )

    fp_stats = np.load(
        stage3_dir / "morgan_fp_train_statistics.npz",
        allow_pickle=False,
    )

    fp_mask = fp_stats["nonconstant_mask"].astype(bool)

    if fp_mask.shape != (fp_aligned.shape[1],):
        raise ValueError(
            "Morgan nonconstant_mask shape does not match fingerprint width"
        )

    pos_weight = fp_stats[
        "pos_weight_clipped_1_20"
    ].astype(np.float32)

    return {
        "descriptor_targets": desc_aligned,
        "fingerprint_targets": fp_aligned,
        "splits": splits,
        "descriptor_names": descriptor_names,
        "descriptor_mask": desc_mask,
        "fp_mask": fp_mask,
        "fp_pos_weight": pos_weight,
        "descriptor_npz_sha256": file_sha256(
            stage3_dir / "descriptor_targets_scaled.npz"
        ),
        "fingerprint_npz_sha256": file_sha256(
            stage3_dir / "morgan_fp_1024.npz"
        ),
        "split_sha256": file_sha256(
            stage3_dir / "pretraining_split.csv"
        ),
    }


# =============================================================================
# Exact graph dataset
# =============================================================================

def build_graph_data(
    library: pd.DataFrame,
    targets: dict[str, Any],
    pe_types: list[str],
) -> list[Data]:
    data_list: list[Data] = []

    coarse_enable = bool(
        getattr(cfg, "coarse_grain_enable", False)
    )
    coarse_min = int(
        getattr(cfg, "coarse_grain_min_chain_length", 0)
    )

    for row_index, row in enumerate(
        library.itertuples(index=False)
    ):
        sid = str(row.stage2c_id)
        smiles = clean(row.Fifth_SMILE)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(
                f"RDKit failed on Stage-2C molecule {sid}: {smiles}"
            )

        # EXACT same custom graph constructor used by csv_pyg_five_multi.py.
        graph = smiles2graph(
            mol,
            coarse_enable,
            coarse_min,
        )

        if len(graph["edge_feat"]) != graph["edge_index"].shape[1]:
            raise ValueError(f"{sid}: edge feature/index mismatch")
        if len(graph["node_feat"]) != graph["num_nodes"]:
            raise ValueError(f"{sid}: node feature/count mismatch")

        x = torch.from_numpy(
            np.asarray(graph["node_feat"])
        ).to(torch.int64)

        edge_index = torch.from_numpy(
            np.asarray(graph["edge_index"])
        ).to(torch.int64)

        edge_attr = torch.from_numpy(
            np.asarray(graph["edge_feat"]).flatten()
        ).to(torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            descriptor_target=torch.from_numpy(
                targets["descriptor_targets"][row_index]
            ).float().view(1, -1),
            fp_target=torch.from_numpy(
                targets["fingerprint_targets"][row_index]
            ).float().view(1, -1),
            sample_index=torch.tensor(
                [row_index],
                dtype=torch.long,
            ),
        )

        data = apply_posenc_if_needed(
            data,
            pe_types,
        )

        data_list.append(data)

        if (
            (row_index + 1) % 500 == 0
            or row_index + 1 == len(library)
        ):
            print(
                f"[Stage4] Graph preprocessing: "
                f"{row_index + 1}/{len(library)}"
            )

    if not data_list:
        raise ValueError("No Stage-4 graphs were built")

    return data_list


def graph_interface_summary(
    first_data: Data,
    pe_types: list[str],
) -> dict[str, Any]:
    return {
        "raw_x_shape_example": list(first_data.x.shape),
        "raw_x_dtype": str(first_data.x.dtype),
        "edge_index_shape_example": list(first_data.edge_index.shape),
        "edge_attr_shape_example": list(first_data.edge_attr.shape),
        "edge_attr_dtype": str(first_data.edge_attr.dtype),
        "dim_in_passed_to_Comp5GraphEncoder": int(
            first_data.x.shape[-1]
        ),
        "enabled_positional_encodings": pe_types,
        "resolved_posenc_kernel_times": {
            pe_name: list(getattr(
                getattr(cfg, f"posenc_{pe_name}").kernel,
                "times",
                [],
            ))
            for pe_name in pe_types
            if hasattr(getattr(cfg, f"posenc_{pe_name}"), "kernel")
        },
        "graph_constructor": "graph_feature.smiles2graph",
        "coarse_grain_enable": bool(
            getattr(cfg, "coarse_grain_enable", False)
        ),
        "coarse_grain_min_chain_length": int(
            getattr(cfg, "coarse_grain_min_chain_length", 0)
        ),
    }


# =============================================================================
# Model
# =============================================================================

class PretrainHeads(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        descriptor_dim: int,
        fp_dim: int,
        dropout: float,
    ):
        super().__init__()
        head_hidden = max(hidden_dim, 128)

        self.descriptor = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, descriptor_dim),
        )

        self.morgan = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, fp_dim),
        )


class Stage4Model(nn.Module):
    def __init__(
        self,
        encoder_cls,
        dim_in: int,
        descriptor_dim: int,
        fp_dim: int,
        head_dropout: float,
    ):
        super().__init__()

        # This is the exact imported local class.
        self.comp5_encoder = encoder_cls(dim_in)
        self.pooling_fun = register.pooling_dict[
            cfg.model.graph_pooling
        ]

        hidden = int(cfg.gt.dim_hidden)

        self.pretrain_heads = PretrainHeads(
            hidden_dim=hidden,
            descriptor_dim=descriptor_dim,
            fp_dim=fp_dim,
            dropout=head_dropout,
        )

    def forward(self, batch):
        encoded = self.comp5_encoder(batch)
        graph_emb = self.pooling_fun(
            encoded.x,
            encoded.batch,
        )
        return (
            self.pretrain_heads.descriptor(graph_emb),
            self.pretrain_heads.morgan(graph_emb),
            graph_emb,
        )


# =============================================================================
# Loss / metrics
# =============================================================================

def masked_descriptor_loss(
    prediction,
    target,
    mask,
    beta: float,
):
    return F.smooth_l1_loss(
        prediction[:, mask],
        target[:, mask],
        beta=beta,
        reduction="mean",
    )


def masked_fp_loss(
    prediction,
    target,
    mask,
    pos_weight=None,
):
    logits = prediction[:, mask]
    labels = target[:, mask]

    kwargs = {}
    if pos_weight is not None:
        kwargs["pos_weight"] = pos_weight[mask]

    return F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="mean",
        **kwargs,
    )


def fp_f1_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    pred = (torch.sigmoid(logits) >= 0.5)
    truth = target >= 0.5

    tp = (pred & truth).sum().item()
    fp = (pred & ~truth).sum().item()
    fn = (~pred & truth).sum().item()

    micro_den = 2 * tp + fp + fn
    micro_f1 = (
        2 * tp / micro_den
        if micro_den > 0
        else math.nan
    )

    tp_b = (pred & truth).sum(dim=0).float()
    fp_b = (pred & ~truth).sum(dim=0).float()
    fn_b = (~pred & truth).sum(dim=0).float()
    den_b = 2 * tp_b + fp_b + fn_b

    valid = den_b > 0
    if valid.any():
        macro = (
            2 * tp_b[valid] / den_b[valid]
        ).mean().item()
    else:
        macro = math.nan

    return float(micro_f1), float(macro)


def evaluate(
    model,
    loader,
    device,
    task,
    descriptor_mask,
    fp_mask,
    fp_lambda,
    huber_beta,
    fp_pos_weight,
):
    model.eval()

    desc_losses = []
    fp_losses = []
    all_desc_pred = []
    all_desc_true = []
    all_fp_logits = []
    all_fp_true = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            desc_pred, fp_pred, _ = model(batch)

            desc_true = batch.descriptor_target.view(
                desc_pred.shape[0],
                -1,
            )
            fp_true = batch.fp_target.view(
                fp_pred.shape[0],
                -1,
            )

            desc_loss = masked_descriptor_loss(
                desc_pred,
                desc_true,
                descriptor_mask,
                huber_beta,
            )

            desc_losses.append(
                float(desc_loss.item())
            )

            all_desc_pred.append(
                desc_pred.detach().cpu()
            )
            all_desc_true.append(
                desc_true.detach().cpu()
            )

            if task == "descriptor_plus_morgan":
                fp_loss = masked_fp_loss(
                    fp_pred,
                    fp_true,
                    fp_mask,
                    fp_pos_weight,
                )
                fp_losses.append(
                    float(fp_loss.item())
                )
                all_fp_logits.append(
                    fp_pred[:, fp_mask].detach().cpu()
                )
                all_fp_true.append(
                    fp_true[:, fp_mask].detach().cpu()
                )

    desc_pred_all = torch.cat(
        all_desc_pred,
        dim=0,
    )
    desc_true_all = torch.cat(
        all_desc_true,
        dim=0,
    )

    normalized_mae_by_desc = (
        torch.abs(
            desc_pred_all - desc_true_all
        )
        .mean(dim=0)
        .numpy()
    )

    desc_loss_mean = float(
        np.mean(desc_losses)
    )

    result = {
        "descriptor_loss": desc_loss_mean,
        "descriptor_mae_normalized": float(
            np.mean(
                normalized_mae_by_desc[
                    descriptor_mask.cpu().numpy()
                ]
            )
        ),
        "descriptor_mae_by_target": (
            normalized_mae_by_desc
        ),
        "fp_loss": math.nan,
        "fp_micro_f1": math.nan,
        "fp_macro_f1": math.nan,
        "total_loss": desc_loss_mean,
    }

    if task == "descriptor_plus_morgan":
        fp_logits_all = torch.cat(
            all_fp_logits,
            dim=0,
        )
        fp_true_all = torch.cat(
            all_fp_true,
            dim=0,
        )

        micro, macro = fp_f1_metrics(
            fp_logits_all,
            fp_true_all,
        )

        fp_loss_mean = float(
            np.mean(fp_losses)
        )

        result.update(
            {
                "fp_loss": fp_loss_mean,
                "fp_micro_f1": micro,
                "fp_macro_f1": macro,
                "total_loss": (
                    desc_loss_mean
                    + fp_lambda * fp_loss_mean
                ),
            }
        )

    return result


# =============================================================================
# Checkpoints
# =============================================================================

def state_shape_signature(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, list[int]]:
    return {
        key: list(value.shape)
        for key, value in state_dict.items()
    }


def save_checkpoint(
    path: Path,
    model: Stage4Model,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict[str, Any],
    run_settings: dict[str, Any],
):
    encoder_state = {
        key: value.detach().cpu()
        for key, value
        in model.comp5_encoder.state_dict().items()
    }

    payload = {
        "format": "biology_prediction_stage4_comp5_v1",
        "epoch": int(epoch),
        "metrics": {
            key: (
                float(value)
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                else value
            )
            for key, value in metrics.items()
            if key != "descriptor_mae_by_target"
        },
        "run_settings": run_settings,
        "encoder_state_dict": encoder_state,
        "encoder_state_shapes": state_shape_signature(
            encoder_state
        ),
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value
            in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    torch.save(payload, path)


def export_transfer_artifacts(
    checkpoint_path: Path,
    checkpoint_dir: Path,
):
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state = ckpt["encoder_state_dict"]

    torch.save(
        state,
        checkpoint_dir
        / "best_comp5_encoder_state_dict.pt",
    )

    prefixed = {
        f"comp5_encoder.{key}": value
        for key, value in state.items()
    }

    torch.save(
        prefixed,
        checkpoint_dir
        / "best_downstream_prefixed_state_dict.pt",
    )


# =============================================================================
# Scheduler
# =============================================================================

def make_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    max_epochs: int,
):
    def factor(epoch_index: int) -> float:
        epoch = epoch_index + 1

        if warmup_epochs > 0 and epoch <= warmup_epochs:
            return epoch / warmup_epochs

        if max_epochs <= warmup_epochs:
            return 1.0

        progress = (
            epoch - warmup_epochs
        ) / (
            max_epochs - warmup_epochs
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        return 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(
        optimizer,
        lr_lambda=factor,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 4 pretraining of the exact local "
            "OneHotEmbedGPS Comp5GraphEncoder."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Same source_config.yaml used by O12/O13 downstream."
        ),
    )
    parser.add_argument(
        "--library",
        type=Path,
        required=True,
        help=(
            "Stage-2C stage2c_pretraining_molecular_library.csv"
        ),
    )
    parser.add_argument(
        "--stage3-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--task",
        choices=[
            "descriptor_only",
            "descriptor_plus_morgan",
        ],
        required=True,
    )

    # Locked current O13-D encoder-facing overrides.
    parser.add_argument(
        "--graph-pooling",
        default="mean",
    )
    parser.add_argument(
        "--gps-layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--graph-hidden-dim",
        type=int,
        default=None,
        help=(
            "Default: preserve source config. "
            "Current O12/O13 source config is 64."
        ),
    )
    parser.add_argument(
        "--gt-dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--gt-attn-dropout",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--base-lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--huber-beta",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--fp-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--use-fp-pos-weight",
        action="store_true",
        help=(
            "Use Stage-3 clipped train-only Morgan pos_weight. "
            "Default off for the first clean ablation."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Default 0 avoids duplicating large PyG objects in workers."
        ),
    )

    args = parser.parse_args()

    for path in (
        args.config,
        args.library,
        args.stage3_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(
            f"Stage-4 run directory already exists: {run_dir}"
        )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    configure_determinism(
        int(args.seed)
    )

    encoder_cfg = load_stage4_config(
        args
    )

    EncoderClass, encoder_source_path = (
        load_local_comp5_encoder_class()
    )

    library = pd.read_csv(
        args.library,
        dtype={"stage2c_id": str},
    )

    required = {
        "stage2c_id",
        "Fifth_SMILE",
        "canonical_connectivity",
    }
    missing = required.difference(
        library.columns
    )
    if missing:
        raise ValueError(
            "Stage-2C library missing columns: "
            + ", ".join(sorted(missing))
        )

    if library["stage2c_id"].duplicated().any():
        raise ValueError(
            "Stage-2C stage2c_id is not unique"
        )

    stage3_targets = align_targets(
        library,
        args.stage3_dir.resolve(),
    )

    pe_types = enabled_pe_types()
    pe_kernel_times = materialize_posenc_kernel_times(
        pe_types
    )

    if pe_types:
        print(
            "[Stage4] Positional encodings: "
            + ", ".join(pe_types)
        )
        for pe_name, times in pe_kernel_times.items():
            print(
                f"[Stage4] {pe_name} kernel times: {times}"
            )

    print(
        f"[Stage4] Building {len(library)} exact downstream-format graphs..."
    )

    data_list = build_graph_data(
        library,
        stage3_targets,
        pe_types,
    )

    interface = graph_interface_summary(
        data_list[0],
        pe_types,
    )

    dim_in = int(
        data_list[0].x.shape[-1]
    )

    # Hard raw-feature dimensionality consistency.
    for i, data in enumerate(data_list):
        if data.x.ndim != 2:
            raise ValueError(
                f"graph {i}: expected x rank 2, got {data.x.shape}"
            )
        if int(data.x.shape[1]) != dim_in:
            raise ValueError(
                f"graph {i}: raw node feature dim changed "
                f"{data.x.shape[1]} != {dim_in}"
            )

    split_array = stage3_targets["splits"]

    indices = {
        split: np.flatnonzero(
            split_array == split
        ).tolist()
        for split in ("train", "val", "test")
    }

    if any(
        len(indices[split]) == 0
        for split in indices
    ):
        raise ValueError(
            f"Empty Stage-4 split: "
            f"{ {k: len(v) for k, v in indices.items()} }"
        )

    generator = torch.Generator()
    generator.manual_seed(
        int(args.seed)
    )

    train_loader = DataLoader(
        [data_list[i] for i in indices["train"]],
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        [data_list[i] for i in indices["val"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        [data_list[i] for i in indices["test"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = Stage4Model(
        encoder_cls=EncoderClass,
        dim_in=dim_in,
        descriptor_dim=stage3_targets[
            "descriptor_targets"
        ].shape[1],
        fp_dim=stage3_targets[
            "fingerprint_targets"
        ].shape[1],
        head_dropout=args.head_dropout,
    )

    # The trained state must exactly match a fresh instance of the SAME
    # local Comp5GraphEncoder class.
    fresh_encoder = EncoderClass(dim_in)

    model_shapes = state_shape_signature(
        model.comp5_encoder.state_dict()
    )
    fresh_shapes = state_shape_signature(
        fresh_encoder.state_dict()
    )

    if model_shapes != fresh_shapes:
        raise RuntimeError(
            "Fresh local Comp5GraphEncoder state signature differs "
            "from Stage-4 encoder instance."
        )

    del fresh_encoder

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    descriptor_mask = torch.from_numpy(
        stage3_targets["descriptor_mask"]
    ).bool().to(device)

    fp_mask = torch.from_numpy(
        stage3_targets["fp_mask"]
    ).bool().to(device)

    fp_pos_weight = None
    if (
        args.task == "descriptor_plus_morgan"
        and args.use_fp_pos_weight
    ):
        fp_pos_weight = torch.from_numpy(
            stage3_targets["fp_pos_weight"]
        ).float().to(device)

    if int(descriptor_mask.sum()) <= 0:
        raise ValueError(
            "No informative descriptor targets"
        )

    if (
        args.task == "descriptor_plus_morgan"
        and int(fp_mask.sum()) <= 0
    ):
        raise ValueError(
            "No informative Morgan bits"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
    )

    run_settings = {
        "task": args.task,
        "seed": int(args.seed),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "base_lr": float(args.base_lr),
        "weight_decay": float(args.weight_decay),
        "warmup_epochs": int(args.warmup_epochs),
        "early_stop_patience": int(
            args.early_stop_patience
        ),
        "huber_beta": float(args.huber_beta),
        "fp_loss_weight": float(
            args.fp_loss_weight
        ),
        "use_fp_pos_weight": bool(
            args.use_fp_pos_weight
        ),
        "informative_descriptors": int(
            descriptor_mask.sum().item()
        ),
        "descriptor_dim_total": int(
            descriptor_mask.numel()
        ),
        "informative_morgan_bits": int(
            fp_mask.sum().item()
        ),
        "morgan_dim_total": int(
            fp_mask.numel()
        ),
        "split_counts": {
            key: len(value)
            for key, value in indices.items()
        },
        "library": str(
            args.library.resolve()
        ),
        "library_sha256": file_sha256(
            args.library.resolve()
        ),
        "stage3_dir": str(
            args.stage3_dir.resolve()
        ),
        "stage3_split_sha256": (
            stage3_targets["split_sha256"]
        ),
        "stage3_descriptor_npz_sha256": (
            stage3_targets[
                "descriptor_npz_sha256"
            ]
        ),
        "stage3_fingerprint_npz_sha256": (
            stage3_targets[
                "fingerprint_npz_sha256"
            ]
        ),
        "encoder_source": str(
            encoder_source_path
        ),
        "encoder_source_sha256": file_sha256(
            encoder_source_path
        ),
        "encoder_config": encoder_cfg,
        "graph_interface": interface,
    }

    (
        run_dir / "run_settings.json"
    ).write_text(
        json.dumps(
            run_settings,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    encoder_interface = {
        "class_name": "Comp5GraphEncoder",
        "source_file": str(
            encoder_source_path
        ),
        "source_sha256": file_sha256(
            encoder_source_path
        ),
        "dim_in": dim_in,
        "state_shapes": model_shapes,
        "pooling": str(
            cfg.model.graph_pooling
        ),
        "transfer_target": (
            "OneHotEmbedGPSModel.comp5_encoder"
        ),
        "strict_transfer_required": True,
    }

    (
        run_dir / "encoder_interface.json"
    ).write_text(
        json.dumps(
            encoder_interface,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    history_path = (
        run_dir / "history.csv"
    )
    history_fields = [
        "epoch",
        "lr",
        "train_total_loss",
        "train_descriptor_loss",
        "train_fp_loss",
        "val_total_loss",
        "val_descriptor_loss",
        "val_descriptor_mae_normalized",
        "val_fp_loss",
        "val_fp_micro_f1",
        "val_fp_macro_f1",
        "best_epoch",
        "best_val_total_loss",
        "early_stop_counter",
    ]

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        csv.DictWriter(
            f,
            fieldnames=history_fields,
        ).writeheader()

    best_loss = math.inf
    best_epoch = -1
    early_counter = 0

    best_path = (
        checkpoint_dir / "best.pt"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        train_total = []
        train_desc = []
        train_fp = []

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            desc_pred, fp_pred, _ = model(
                batch
            )

            desc_true = (
                batch.descriptor_target.view(
                    desc_pred.shape[0],
                    -1,
                )
            )
            fp_true = batch.fp_target.view(
                fp_pred.shape[0],
                -1,
            )

            desc_loss = (
                masked_descriptor_loss(
                    desc_pred,
                    desc_true,
                    descriptor_mask,
                    args.huber_beta,
                )
            )

            if (
                args.task
                == "descriptor_plus_morgan"
            ):
                fp_loss = masked_fp_loss(
                    fp_pred,
                    fp_true,
                    fp_mask,
                    fp_pos_weight,
                )
                total_loss = (
                    desc_loss
                    + args.fp_loss_weight
                    * fp_loss
                )
                train_fp.append(
                    float(fp_loss.item())
                )
            else:
                fp_loss = None
                total_loss = desc_loss

            if not torch.isfinite(
                total_loss
            ):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}"
                )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            train_total.append(
                float(total_loss.item())
            )
            train_desc.append(
                float(desc_loss.item())
            )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            args.task,
            descriptor_mask,
            fp_mask,
            args.fp_loss_weight,
            args.huber_beta,
            fp_pos_weight,
        )

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        improved = (
            val_metrics["total_loss"]
            < best_loss - 1e-8
        )

        if improved:
            best_loss = float(
                val_metrics["total_loss"]
            )
            best_epoch = int(epoch)
            early_counter = 0

            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics,
                run_settings,
            )
        else:
            early_counter += 1

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_total_loss": float(
                np.mean(train_total)
            ),
            "train_descriptor_loss": float(
                np.mean(train_desc)
            ),
            "train_fp_loss": (
                float(np.mean(train_fp))
                if train_fp
                else math.nan
            ),
            "val_total_loss": val_metrics[
                "total_loss"
            ],
            "val_descriptor_loss": (
                val_metrics["descriptor_loss"]
            ),
            "val_descriptor_mae_normalized": (
                val_metrics[
                    "descriptor_mae_normalized"
                ]
            ),
            "val_fp_loss": val_metrics[
                "fp_loss"
            ],
            "val_fp_micro_f1": val_metrics[
                "fp_micro_f1"
            ],
            "val_fp_macro_f1": val_metrics[
                "fp_macro_f1"
            ],
            "best_epoch": best_epoch,
            "best_val_total_loss": best_loss,
            "early_stop_counter": early_counter,
        }

        with history_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            csv.DictWriter(
                f,
                fieldnames=history_fields,
            ).writerow(row)

        print(
            f"epoch={epoch:03d} "
            f"train={row['train_total_loss']:.5f} "
            f"val={row['val_total_loss']:.5f} "
            f"desc_mae_z={row['val_descriptor_mae_normalized']:.4f} "
            + (
                f"fp_f1={row['val_fp_micro_f1']:.4f} "
                if args.task
                == "descriptor_plus_morgan"
                else ""
            )
            + f"lr={current_lr:.3e} "
            f"best={best_epoch}"
        )

        scheduler.step()

        if (
            args.early_stop_patience > 0
            and early_counter
            >= args.early_stop_patience
        ):
            print(
                f"[Stage4] Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}"
            )
            break

    if not best_path.is_file():
        raise RuntimeError(
            "No best checkpoint was created"
        )

    # Save last after training.
    final_val = evaluate(
        model,
        val_loader,
        device,
        args.task,
        descriptor_mask,
        fp_mask,
        args.fp_loss_weight,
        args.huber_beta,
        fp_pos_weight,
    )

    save_checkpoint(
        checkpoint_dir / "last.pt",
        model,
        optimizer,
        scheduler,
        epoch,
        final_val,
        run_settings,
    )

    # Restore best for final validation/test.
    best = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best["model_state_dict"],
        strict=True,
    )

    # Final strict encoder-interface self-check.
    loaded_shapes = state_shape_signature(
        model.comp5_encoder.state_dict()
    )

    if loaded_shapes != encoder_interface[
        "state_shapes"
    ]:
        raise RuntimeError(
            "Best checkpoint encoder signature changed"
        )

    val_best = evaluate(
        model,
        val_loader,
        device,
        args.task,
        descriptor_mask,
        fp_mask,
        args.fp_loss_weight,
        args.huber_beta,
        fp_pos_weight,
    )
    test_best = evaluate(
        model,
        test_loader,
        device,
        args.task,
        descriptor_mask,
        fp_mask,
        args.fp_loss_weight,
        args.huber_beta,
        fp_pos_weight,
    )

    export_transfer_artifacts(
        best_path,
        checkpoint_dir,
    )

    # Descriptor per-target normalized MAE.
    descriptor_rows = []
    for split_name, metrics in (
        ("val", val_best),
        ("test", test_best),
    ):
        values = metrics[
            "descriptor_mae_by_target"
        ]
        for name, value, informative in zip(
            stage3_targets["descriptor_names"],
            values,
            stage3_targets["descriptor_mask"],
        ):
            descriptor_rows.append(
                {
                    "split": split_name,
                    "descriptor": name,
                    "informative_train": bool(
                        informative
                    ),
                    "normalized_mae": float(
                        value
                    ),
                }
            )

    pd.DataFrame(
        descriptor_rows
    ).to_csv(
        run_dir / "descriptor_metrics.csv",
        index=False,
    )

    fp_metrics = {
        "val": {
            key: val_best[key]
            for key in (
                "fp_loss",
                "fp_micro_f1",
                "fp_macro_f1",
            )
        },
        "test": {
            key: test_best[key]
            for key in (
                "fp_loss",
                "fp_micro_f1",
                "fp_macro_f1",
            )
        },
    }

    (
        run_dir / "fp_metrics.json"
    ).write_text(
        json.dumps(
            fp_metrics,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "task": args.task,
        "best_epoch": int(
            best["epoch"]
        ),
        "best_validation": {
            key: (
                float(value)
                if isinstance(
                    value,
                    (int, float),
                )
                else value
            )
            for key, value in val_best.items()
            if key != "descriptor_mae_by_target"
        },
        "test_at_best": {
            key: (
                float(value)
                if isinstance(
                    value,
                    (int, float),
                )
                else value
            )
            for key, value in test_best.items()
            if key != "descriptor_mae_by_target"
        },
        "checkpoint": str(
            best_path.resolve()
        ),
        "raw_encoder_state_dict": str(
            (
                checkpoint_dir
                / "best_comp5_encoder_state_dict.pt"
            ).resolve()
        ),
        "downstream_prefixed_state_dict": str(
            (
                checkpoint_dir
                / "best_downstream_prefixed_state_dict.pt"
            ).resolve()
        ),
        "encoder_source_sha256": (
            encoder_interface[
                "source_sha256"
            ]
        ),
        "source_config_sha256": (
            encoder_cfg[
                "source_config_sha256"
            ]
        ),
        "strict_interface_signature_passed": True,
    }

    (
        run_dir / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("STAGE 4 COMPLETE")
    print("=" * 88)
    print(f"task:                  {args.task}")
    print(f"best epoch:            {best['epoch']}")
    print(
        f"val total loss:        {val_best['total_loss']:.6f}"
    )
    print(
        f"val descriptor MAE(z): {val_best['descriptor_mae_normalized']:.6f}"
    )
    if args.task == "descriptor_plus_morgan":
        print(
            f"val Morgan micro-F1:   {val_best['fp_micro_f1']:.6f}"
        )
    print(
        f"test total loss:       {test_best['total_loss']:.6f}"
    )
    print()
    print(
        "Strict transfer artifact:\n  "
        + str(
            checkpoint_dir
            / "best_comp5_encoder_state_dict.pt"
        )
    )
    print(
        "Prefixed downstream artifact:\n  "
        + str(
            checkpoint_dir
            / "best_downstream_prefixed_state_dict.pt"
        )
    )
    print()
    print(
        "Encoder class and state-shape signature match the "
        "local Comp5GraphEncoder interface."
    )


if __name__ == "__main__":
    main()
