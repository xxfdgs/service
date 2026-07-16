#!/usr/bin/env python3
"""Reload one GraphGPS checkpoint and export sample-id-aligned predictions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphgps  # noqa: F401,E402
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from graphgps.determinism import configure_determinism  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from stage3_utils import metric_frame  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402
from torch_geometric.graphgym.checkpoint import MODEL_STATE  # noqa: E402


TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]


def setup_config(config_path: Path, seed: int) -> None:
    """Load a training configuration without invoking the legacy CLI entrypoint."""
    set_cfg_gps(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(config_path), opts=[]))
    cfg.seed = int(seed)
    configure_determinism(cfg.seed, bool(cfg.train.deterministic))


def collect_predictions(loaders: tuple[list, list, list, list, list], model: torch.nn.Module,
                        split: str) -> pd.DataFrame:
    """Evaluate one split and retain each graph's stable source index."""
    split_to_loader = {"train": 0, "val": 1, "test": 2}
    if split not in split_to_loader:
        raise ValueError(f"Unsupported split: {split}")
    loader_index = split_to_loader[split]
    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batches in zip(*(loader[loader_index] for loader in loaders)):
            batch, batch_2, batch_3, batch_4, batch_5 = batches
            sample_indices = batch.sample_uid.detach().cpu().view(-1).tolist()
            for current, suffix in zip((batch, batch_2, batch_3, batch_4, batch_5), ("", "_2", "_3", "_4", "_5")):
                current.split = split + suffix
                current.to(device)
            prediction, label = model(batch, batch_2, batch_3, batch_4, batch_5)
            predicted_normalized = prediction.detach().cpu().view(-1, len(TARGETS)).numpy()
            true_normalized = label.detach().cpu().view(-1, len(TARGETS)).numpy()
            # The current five-component loader applies the fixed /100
            # transform to labels.  Preserve both model-space and restored
            # units so diagnostics can prove there is no double inverse-scale.
            predicted = predicted_normalized * 100.0
            true = true_normalized * 100.0
            if not (len(sample_indices) == len(predicted) == len(true)):
                raise RuntimeError("Batch source index and target dimensions differ.")
            for sample_index, true_row, predicted_row, raw_true_row, raw_predicted_row in zip(
                sample_indices, true, predicted, true_normalized, predicted_normalized
            ):
                for target_index, target in enumerate(TARGETS):
                    rows.append({"source_index": int(sample_index), "split": split, "target": target,
                                 "y_true": float(true_row[target_index]), "y_pred": float(predicted_row[target_index]),
                                 "label_before_inverse_transform": float(raw_true_row[target_index]),
                                 "prediction_before_inverse_transform": float(raw_predicted_row[target_index]),
                                 "prediction_after_inverse_transform": float(predicted_row[target_index])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--protocol", required=True)
    arguments = parser.parse_args()
    setup_config(arguments.config.resolve(), arguments.seed)
    manifest = pd.read_csv(arguments.manifest, dtype={"sample_id": str})
    subset = manifest.loc[manifest["split"] == arguments.split, ["sample_id", "split", "original_row_index"]].copy()
    if subset["sample_id"].duplicated().any() or subset["original_row_index"].duplicated().any():
        raise ValueError("Manifest split does not uniquely map source indexes to sample IDs.")
    loaders = create_loader_5()
    model = create_model_gps()
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint[MODEL_STATE], strict=True)
    predictions = collect_predictions(loaders, model, arguments.split)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    predictions = predictions.merge(subset, left_on="source_index", right_on="original_row_index",
                                    how="left", validate="many_to_one", suffixes=("", "_manifest"))
    if predictions["sample_id"].isna().any() or predictions.duplicated(["sample_id", "target"]).any():
        debug_path = arguments.output.with_name(arguments.output.stem + "_alignment_debug.csv")
        predictions.to_csv(debug_path, index=False)
        raise RuntimeError(
            "Prediction export failed sample_id alignment validation: "
            f"predictions={len(predictions)}, expected={len(subset) * len(TARGETS)}, "
            f"unmapped={int(predictions['sample_id'].isna().sum())}, "
            f"duplicate_sample_target={int(predictions.duplicated(['sample_id', 'target']).sum())}, "
            f"debug={debug_path}"
        )
    if set(predictions["sample_id"]) != set(subset["sample_id"]):
        raise RuntimeError("Predicted sample_id set differs from the manifest split.")
    predictions["seed"] = arguments.seed
    predictions["fold"] = arguments.fold
    predictions["protocol"] = arguments.protocol
    predictions["checkpoint"] = str(arguments.checkpoint.resolve())
    predictions["checkpoint_path"] = str(arguments.checkpoint.resolve())
    scaler = checkpoint.get("target_scaler", {})
    if scaler.get("type") == "fixed_percent" and scaler.get("scale") == 100.0:
        predictions["scaler_type"] = "fixed_percent"
        predictions["scaler_mean"] = 0.0
        predictions["scaler_std"] = 100.0
    else:
        predictions["scaler_type"] = str(scaler.get("type", "not_saved"))
        predictions["scaler_mean"] = float("nan")
        predictions["scaler_std"] = float("nan")
    predictions["absolute_error"] = (predictions["y_true"] - predictions["y_pred"]).abs()
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions = predictions[["sample_id", "split", "target", "y_true", "y_pred",
                               "label_before_inverse_transform", "prediction_before_inverse_transform",
                               "prediction_after_inverse_transform", "scaler_type", "scaler_mean", "scaler_std",
                               "seed", "fold", "protocol", "checkpoint", "checkpoint_path", "source_index", "absolute_error"]]
    predictions.to_csv(output, index=False)
    metric_frame(predictions, {"seed": arguments.seed, "fold": arguments.fold,
                               "protocol": arguments.protocol}).to_csv(output.with_name(output.stem + "_metrics.csv"), index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
