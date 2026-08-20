#!/usr/bin/env python3
"""Audit Comp5GraphEncoder drift after Stage-5 downstream fine-tuning.

This revised version deliberately separates learnable-weight-like tensors from
BatchNorm running statistics and counters.

Why this matters
----------------
A raw ``state_dict`` contains more than trainable parameters.  In particular,
BatchNorm ``num_batches_tracked`` can change by thousands during downstream
training and completely dominate naive L2/RMS drift calculations.

Primary drift statistics therefore use only:

    - floating-point tensors
    - excluding ``running_mean``
    - excluding ``running_var``
    - excluding ``num_batches_tracked``

These tensors are reported as ``parameter_like`` because a bare state_dict does
not itself encode ``requires_grad`` provenance.  In the current
Comp5GraphEncoder this is the correct practical proxy for trainable weights.

BatchNorm running statistics and counters are audited separately.

Inputs
------
Stage-4 initial checkpoints:
    PT_D/checkpoints/best_comp5_encoder_state_dict.pt
    PT_DF/checkpoints/best_comp5_encoder_state_dict.pt

Stage-5 selected-best checkpoints:
    P1_PT_D/split{seed}/checkpoints/selected_best.pt
    P2_PT_DF/split{seed}/checkpoints/selected_best.pt

Outputs
-------
stage5_comp5_parameter_drift.csv
    Primary parameter-like drift, one row per model/split.

stage5_comp5_parameter_drift_by_block.csv
    Parameter-like drift grouped by top-level Comp5GraphEncoder block.

stage5_comp5_parameter_drift_by_tensor.csv
    Per-tensor parameter-like drift.

stage5_comp5_bn_running_drift.csv
    BatchNorm running_mean / running_var drift.

stage5_comp5_counter_drift.csv
    num_batches_tracked changes.

Interpretation
--------------
Primary quantities:

    relative_l2
        ||theta_final - theta_init||_2 / ||theta_init||_2

    relative_rms
        RMS(theta_final - theta_init) / RMS(theta_init)

    cosine_similarity
        cosine(theta_init, theta_final)

Do NOT use BatchNorm counters to infer catastrophic forgetting.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch


BN_RUNNING_SUFFIXES = ("running_mean", "running_var")
COUNTER_SUFFIX = "num_batches_tracked"


def load_state(path: Path) -> dict[str, torch.Tensor]:
    """Load a raw state_dict or a checkpoint containing one."""
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            return obj["model_state"]
        if obj and all(torch.is_tensor(value) for value in obj.values()):
            return obj

    raise TypeError(f"Unsupported checkpoint/state-dict format: {path}")


def extract_comp5_from_full(
    full_state: dict[str, torch.Tensor],
    reference_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Map each Stage-4 Comp5 key to the matching key in a full Stage-5 model."""
    extracted: dict[str, torch.Tensor] = {}
    mapping: dict[str, str] = {}

    for key, reference in reference_state.items():
        exact_candidates = (
            key,
            f"comp5_encoder.{key}",
            f"model.comp5_encoder.{key}",
        )
        hits = [candidate for candidate in exact_candidates if candidate in full_state]

        if len(hits) != 1:
            suffix = f"comp5_encoder.{key}"
            hits = [candidate for candidate in full_state if candidate.endswith(suffix)]

        if len(hits) != 1:
            raise KeyError(
                f"Could not uniquely map Stage-4 Comp5 key {key!r}. "
                f"Found candidates: {hits[:20]}"
            )

        value = full_state[hits[0]]
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f"Shape mismatch for {key!r}: "
                f"Stage4={tuple(reference.shape)} vs "
                f"Stage5={tuple(value.shape)} via {hits[0]!r}"
            )

        extracted[key] = value.detach().cpu()
        mapping[key] = hits[0]

    if set(extracted) != set(reference_state):
        raise RuntimeError("Extracted Comp5 state does not exactly match Stage-4 key set.")

    return extracted, mapping


def classify_keys(state: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    """Classify state_dict tensors into parameter-like / BN-running / counters."""
    classes = {
        "parameter_like": [],
        "bn_running": [],
        "counters": [],
        "other": [],
    }

    for key, value in state.items():
        if key.endswith(COUNTER_SUFFIX):
            classes["counters"].append(key)
        elif key.endswith(BN_RUNNING_SUFFIXES):
            classes["bn_running"].append(key)
        elif torch.is_floating_point(value):
            classes["parameter_like"].append(key)
        else:
            classes["other"].append(key)

    return classes


def flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat([tensor.detach().float().reshape(-1) for tensor in tensors])


def drift_metrics(
    initial_tensors: list[torch.Tensor],
    final_tensors: list[torch.Tensor],
) -> dict[str, float | int]:
    """Compute scale-aware drift metrics for aligned tensors."""
    if len(initial_tensors) != len(final_tensors):
        raise ValueError("Initial/final tensor list lengths differ.")

    a = flatten(initial_tensors)
    b = flatten(final_tensors)

    if a.numel() != b.numel():
        raise ValueError("Initial/final flattened tensor sizes differ.")

    if not a.numel():
        return {
            "parameter_count": 0,
            "init_l2": math.nan,
            "final_l2": math.nan,
            "delta_l2": math.nan,
            "relative_l2": math.nan,
            "init_rms": math.nan,
            "delta_rms": math.nan,
            "relative_rms": math.nan,
            "cosine_similarity": math.nan,
            "mean_abs_delta": math.nan,
            "max_abs_delta": math.nan,
        }

    delta = b - a
    init_l2 = torch.linalg.vector_norm(a)
    final_l2 = torch.linalg.vector_norm(b)
    delta_l2 = torch.linalg.vector_norm(delta)

    init_rms = torch.sqrt(torch.mean(a.square()))
    delta_rms = torch.sqrt(torch.mean(delta.square()))

    relative_l2 = (
        float(delta_l2 / init_l2)
        if float(init_l2) > 0
        else math.nan
    )
    relative_rms = (
        float(delta_rms / init_rms)
        if float(init_rms) > 0
        else math.nan
    )

    cosine = (
        float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        if a.numel()
        else math.nan
    )

    return {
        "parameter_count": int(a.numel()),
        "init_l2": float(init_l2),
        "final_l2": float(final_l2),
        "delta_l2": float(delta_l2),
        "relative_l2": relative_l2,
        "init_rms": float(init_rms),
        "delta_rms": float(delta_rms),
        "relative_rms": relative_rms,
        "cosine_similarity": cosine,
        "mean_abs_delta": float(delta.abs().mean()),
        "max_abs_delta": float(delta.abs().max()),
    }


def tensor_drift_metrics(
    initial: torch.Tensor,
    final: torch.Tensor,
) -> dict[str, float | int]:
    return drift_metrics([initial], [final])


def top_level_block(key: str) -> str:
    """Coarse block name for readable aggregation."""
    return key.split(".", 1)[0]


def counter_value(tensor: torch.Tensor) -> float:
    if tensor.numel() != 1:
        raise ValueError("Expected scalar counter tensor.")
    return float(tensor.detach().cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage5-root",
        type=Path,
        default=Path("results/fifth_pretraining/stage5_downstream_transfer"),
    )
    parser.add_argument(
        "--stage4-root",
        type=Path,
        default=Path("results/fifth_pretraining/stage4_graphgps_pretraining"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        default=[100, 101, 102],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    stage5_root = args.stage5_root.resolve()
    stage4_root = args.stage4_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else stage5_root / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "P1_PT_D": (
            stage4_root
            / "PT_D"
            / "checkpoints"
            / "best_comp5_encoder_state_dict.pt"
        ),
        "P2_PT_DF": (
            stage4_root
            / "PT_DF"
            / "checkpoints"
            / "best_comp5_encoder_state_dict.pt"
        ),
    }

    overall_rows = []
    block_rows = []
    tensor_rows = []
    bn_running_rows = []
    counter_rows = []

    for label, initial_checkpoint in variants.items():
        if not initial_checkpoint.is_file():
            raise FileNotFoundError(initial_checkpoint)

        initial_state = load_state(initial_checkpoint)
        key_classes = classify_keys(initial_state)

        if not key_classes["parameter_like"]:
            raise RuntimeError(
                f"{label}: no parameter-like floating tensors found in Stage-4 state."
            )

        print(
            f"[{label}] Stage-4 key classes: "
            f"parameter_like={len(key_classes['parameter_like'])}, "
            f"bn_running={len(key_classes['bn_running'])}, "
            f"counters={len(key_classes['counters'])}, "
            f"other={len(key_classes['other'])}"
        )

        for split_seed in args.splits:
            downstream_checkpoint = (
                stage5_root
                / label
                / f"split{split_seed}"
                / "checkpoints"
                / "selected_best.pt"
            )
            if not downstream_checkpoint.is_file():
                raise FileNotFoundError(downstream_checkpoint)

            full_state = load_state(downstream_checkpoint)
            final_state, mapping = extract_comp5_from_full(
                full_state,
                initial_state,
            )

            # --------------------------------------------------------------
            # 1) PRIMARY: parameter-like tensors only.
            # --------------------------------------------------------------
            parameter_keys = sorted(key_classes["parameter_like"])

            overall = drift_metrics(
                [initial_state[key] for key in parameter_keys],
                [final_state[key] for key in parameter_keys],
            )
            overall_rows.append({
                "label": label,
                "split_seed": int(split_seed),
                "tensor_class": "parameter_like",
                "tensor_count": len(parameter_keys),
                "stage4_checkpoint": str(initial_checkpoint),
                "stage5_checkpoint": str(downstream_checkpoint),
                **overall,
            })

            # Per-block parameter-like drift.
            block_to_keys: dict[str, list[str]] = {}
            for key in parameter_keys:
                block_to_keys.setdefault(top_level_block(key), []).append(key)

            for block, keys in sorted(block_to_keys.items()):
                block_metrics = drift_metrics(
                    [initial_state[key] for key in keys],
                    [final_state[key] for key in keys],
                )
                block_rows.append({
                    "label": label,
                    "split_seed": int(split_seed),
                    "block": block,
                    "tensor_count": len(keys),
                    **block_metrics,
                })

            # Per-tensor parameter-like drift.
            for key in parameter_keys:
                metrics = tensor_drift_metrics(
                    initial_state[key],
                    final_state[key],
                )
                tensor_rows.append({
                    "label": label,
                    "split_seed": int(split_seed),
                    "key": key,
                    "stage5_key": mapping[key],
                    "shape": "x".join(map(str, initial_state[key].shape)),
                    **metrics,
                })

            # --------------------------------------------------------------
            # 2) BatchNorm running statistics: report separately.
            # --------------------------------------------------------------
            for key in sorted(key_classes["bn_running"]):
                metrics = tensor_drift_metrics(
                    initial_state[key],
                    final_state[key],
                )
                bn_running_rows.append({
                    "label": label,
                    "split_seed": int(split_seed),
                    "key": key,
                    "stage5_key": mapping[key],
                    "shape": "x".join(map(str, initial_state[key].shape)),
                    **metrics,
                })

            # --------------------------------------------------------------
            # 3) num_batches_tracked: scalar delta only.
            # --------------------------------------------------------------
            for key in sorted(key_classes["counters"]):
                initial_value = counter_value(initial_state[key])
                final_value = counter_value(final_state[key])
                counter_rows.append({
                    "label": label,
                    "split_seed": int(split_seed),
                    "key": key,
                    "stage5_key": mapping[key],
                    "initial_value": initial_value,
                    "final_value": final_value,
                    "delta": final_value - initial_value,
                })

    overall_df = pd.DataFrame(overall_rows)
    block_df = pd.DataFrame(block_rows)
    tensor_df = pd.DataFrame(tensor_rows)
    bn_df = pd.DataFrame(bn_running_rows)
    counter_df = pd.DataFrame(counter_rows)

    overall_path = output_dir / "stage5_comp5_parameter_drift.csv"
    block_path = output_dir / "stage5_comp5_parameter_drift_by_block.csv"
    tensor_path = output_dir / "stage5_comp5_parameter_drift_by_tensor.csv"
    bn_path = output_dir / "stage5_comp5_bn_running_drift.csv"
    counter_path = output_dir / "stage5_comp5_counter_drift.csv"

    overall_df.to_csv(overall_path, index=False)
    block_df.to_csv(block_path, index=False)
    tensor_df.to_csv(tensor_path, index=False)
    bn_df.to_csv(bn_path, index=False)
    counter_df.to_csv(counter_path, index=False)

    print()
    print("=" * 104)
    print("STAGE-5 Comp5 PARAMETER-LIKE DRIFT (BN RUNNING STATS / COUNTERS EXCLUDED)")
    print("=" * 104)
    display_cols = [
        "label",
        "split_seed",
        "parameter_count",
        "relative_l2",
        "relative_rms",
        "cosine_similarity",
        "mean_abs_delta",
        "max_abs_delta",
    ]
    print(overall_df[display_cols].to_string(index=False))

    print()
    print("Mean parameter-like drift by initialization:")
    mean_cols = [
        "relative_l2",
        "relative_rms",
        "cosine_similarity",
        "mean_abs_delta",
        "max_abs_delta",
    ]
    print(
        overall_df.groupby("label")[mean_cols]
        .mean()
        .to_string()
    )

    if not counter_df.empty:
        print()
        print("BatchNorm num_batches_tracked deltas (excluded from primary drift):")
        counter_summary = (
            counter_df.groupby(["label", "split_seed"])["delta"]
            .agg(["count", "min", "max", "mean"])
            .reset_index()
        )
        print(counter_summary.to_string(index=False))

    print()
    print("Top parameter-like blocks by relative_l2:")
    if not block_df.empty:
        top_blocks = (
            block_df.sort_values(
                ["label", "split_seed", "relative_l2"],
                ascending=[True, True, False],
            )
            .groupby(["label", "split_seed"], as_index=False)
            .head(5)
        )
        print(
            top_blocks[
                [
                    "label",
                    "split_seed",
                    "block",
                    "parameter_count",
                    "relative_l2",
                    "relative_rms",
                    "cosine_similarity",
                ]
            ].to_string(index=False)
        )

    print()
    print("Outputs:")
    for path in (
        overall_path,
        block_path,
        tensor_path,
        bn_path,
        counter_path,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
