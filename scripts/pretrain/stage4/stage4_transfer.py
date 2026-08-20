#!/usr/bin/env python3
"""
Strict Stage-4 -> OneHotEmbedGPS Fifth-encoder transfer helper.

Use from the downstream runner AFTER constructing OneHotEmbedGPSModel:

    from scripts.pretrain.stage4.stage4_transfer import (
        load_stage4_comp5_encoder,
    )

    report = load_stage4_comp5_encoder(
        model,
        checkpoint_path,
    )

The function only touches:
    model.comp5_encoder

It does NOT load:
    component embeddings
    Fifth identity/class embedding
    ratio modulators
    aux/Mordred encoders
    fusion/head
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _extract_encoder_state(payload: Any):
    if not isinstance(payload, dict):
        raise TypeError(
            "Stage-4 checkpoint/state must be a dict"
        )

    if "encoder_state_dict" in payload:
        state = payload["encoder_state_dict"]
    else:
        # Accept the raw best_comp5_encoder_state_dict.pt artifact.
        state = payload

    if not isinstance(state, dict):
        raise TypeError(
            "encoder state is not a state_dict mapping"
        )

    # Also accept an accidentally supplied prefixed transfer artifact.
    if state and all(
        str(key).startswith("comp5_encoder.")
        for key in state
    ):
        state = {
            str(key)[len("comp5_encoder."):]: value
            for key, value in state.items()
        }

    return state


def _shape_map(state):
    return {
        str(key): tuple(value.shape)
        for key, value in state.items()
    }


def load_stage4_comp5_encoder(
    model,
    checkpoint_path,
    *,
    map_location="cpu",
):
    """
    Strictly load a Stage-4 encoder into OneHotEmbedGPSModel.comp5_encoder.

    Raises on ANY key or shape mismatch.
    """
    if not hasattr(model, "comp5_encoder"):
        raise AttributeError(
            "Target model has no comp5_encoder; expected OneHotEmbedGPSModel"
        )

    return load_stage4_encoder_into(
        model.comp5_encoder,
        checkpoint_path,
        map_location=map_location,
    )


def load_stage4_encoder_into(
    encoder,
    checkpoint_path,
    *,
    map_location="cpu",
):
    """Strictly load a Stage-4 state into a compatible encoder module.

    This is the shared primitive for the trainable Stage-5 encoder and the
    Stage-8 frozen auxiliary encoder.  It deliberately validates every key
    and tensor shape before modifying ``encoder``.
    """
    if not isinstance(encoder, torch.nn.Module):
        raise TypeError("encoder must be a torch.nn.Module")

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    source_state = _extract_encoder_state(payload)
    target_state = encoder.state_dict()

    source_shapes = _shape_map(
        source_state
    )
    target_shapes = _shape_map(
        target_state
    )

    source_keys = set(source_shapes)
    target_keys = set(target_shapes)

    missing = sorted(
        target_keys - source_keys
    )
    unexpected = sorted(
        source_keys - target_keys
    )
    shape_mismatches = {
        key: {
            "source": source_shapes[key],
            "target": target_shapes[key],
        }
        for key in sorted(
            source_keys & target_keys
        )
        if source_shapes[key]
        != target_shapes[key]
    }

    if (
        missing
        or unexpected
        or shape_mismatches
    ):
        raise RuntimeError(
            "Stage-4 encoder is NOT interface-compatible with "
            "the current downstream comp5_encoder.\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}\n"
            f"shape_mismatches={shape_mismatches}"
        )

    encoder.load_state_dict(source_state, strict=True)

    return {
        "checkpoint": str(
            checkpoint_path
        ),
        "loaded_parameter_tensors": len(
            source_state
        ),
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": {},
        "strict": True,
    }
