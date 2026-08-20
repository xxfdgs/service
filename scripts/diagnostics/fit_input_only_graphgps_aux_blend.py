#!/usr/bin/env python3
"""Freeze continuous input-validation weights for GraphGPS + molecular aux."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ("Norm_before", "Norm_after")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphgps-root", type=Path, required=True)
    parser.add_argument("--graphgps-prefix", required=True)
    parser.add_argument("--aux-predictions", type=Path, required=True)
    parser.add_argument("--aux-protocol", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=200)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    graphgps_root = args.graphgps_root.resolve()
    aux_path = args.aux_predictions.resolve()
    aux_protocol_path = args.aux_protocol.resolve()
    aux_protocol = json.loads(
        aux_protocol_path.read_text(encoding="utf-8"))
    if (
        aux_protocol.get("external_feedback_read") is not False
        or aux_protocol.get("threshold_or_side_criterion_used") is not False
    ):
        raise RuntimeError("Auxiliary predictions are not certified input-only.")
    seeds = list(range(args.first_seed, args.first_seed + args.seed_count))
    graph_frames = []
    for seed in seeds:
        run_dir = graphgps_root / f"{args.graphgps_prefix}{seed}"
        settings = json.loads(
            (run_dir / "run_settings.json").read_text(encoding="utf-8"))
        if (
            settings.get("model_type") != "OneHotEmbedGPS"
            or settings.get("outer_test_read_during_selection") is not False
        ):
            raise RuntimeError(f"Invalid GraphGPS run: {run_dir}")
        predictions = pd.read_csv(run_dir / "predictions.csv")
        part = predictions.loc[
            predictions["split"].eq("val")
            & predictions["target"].isin(TARGETS),
            ["sample_id", "target", "y_true", "y_pred"],
        ].copy()
        part["split_seed"] = seed
        graph_frames.append(part.rename(columns={
            "y_true": "graphgps_y_true",
            "y_pred": "graphgps_prediction",
        }))
    graphgps = pd.concat(graph_frames, ignore_index=True)
    auxiliary = pd.read_csv(aux_path)
    auxiliary = auxiliary.loc[
        auxiliary["split"].eq("val")
        & auxiliary["target"].isin(TARGETS),
        ["split_seed", "sample_id", "target", "y_true", "y_pred"],
    ].rename(columns={
        "y_true": "aux_y_true",
        "y_pred": "aux_prediction",
    })
    keys = ["split_seed", "sample_id", "target"]
    paired = graphgps.merge(auxiliary, on=keys, validate="one_to_one")
    if not np.allclose(
        paired["graphgps_y_true"], paired["aux_y_true"], rtol=0, atol=1e-6
    ):
        raise RuntimeError("GraphGPS and auxiliary validation labels differ.")
    paired = paired.rename(
        columns={"graphgps_y_true": "y_true"}).drop(
            columns="aux_y_true")

    grid_rows = []
    selected_rows = []
    for target in TARGETS:
        part = paired.loc[paired["target"].eq(target)]
        truth = part["y_true"].to_numpy(float)
        graph_prediction = part["graphgps_prediction"].to_numpy(float)
        aux_prediction = part["aux_prediction"].to_numpy(float)
        candidates = []
        for aux_weight in np.linspace(0.0, 1.0, 101):
            prediction = (
                (1.0 - aux_weight) * graph_prediction
                + aux_weight * aux_prediction
            )
            row = {
                "target": target,
                "first_family": "GraphGPS",
                "second_family": "MorganAux",
                "first_weight": float(1.0 - aux_weight),
                "second_weight": float(aux_weight),
                "pooled_validation_mae": float(
                    np.mean(np.abs(truth - prediction))),
            }
            candidates.append(row)
            grid_rows.append(row)
        selected_rows.append(min(
            candidates,
            key=lambda row: (
                row["pooled_validation_mae"],
                row["second_weight"],
            ),
        ))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "paired_input_validation_predictions.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(
        output / "pair_blend_weight_grid.csv", index=False)
    weights_path = output / "pair_blend_weights.csv"
    pd.DataFrame(selected_rows).to_csv(weights_path, index=False)
    protocol = {
        "frozen": True,
        "model_family": "O12 GraphGPS plus molecular auxiliary continuous blend",
        "graphgps_root": str(graphgps_root),
        "graphgps_prefix": args.graphgps_prefix,
        "aux_predictions": str(aux_path),
        "aux_predictions_sha256": sha256(aux_path),
        "aux_protocol": str(aux_protocol_path),
        "aux_protocol_sha256": sha256(aux_protocol_path),
        "split_seeds": seeds,
        "selection_metric": "pooled continuous raw-scale input validation MAE",
        "external_feedback_read": False,
        "threshold_or_side_criterion_used": False,
        "weights": str(weights_path),
        "weights_sha256": sha256(weights_path),
    }
    (output / "pair_blend_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(pd.DataFrame(selected_rows).to_string(index=False))


if __name__ == "__main__":
    main()
