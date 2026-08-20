#!/usr/bin/env python3
"""Freeze a continuous-MAE convex blend between paired O12 run families."""

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


def read_validation(root: Path, prefix: str, first_seed: int, seed_count: int,
                    expected_transform: str, family: str) -> pd.DataFrame:
    frames = []
    for seed in range(first_seed, first_seed + seed_count):
        run_dir = root / f"{prefix}{seed}"
        settings_path = run_dir / "run_settings.json"
        predictions_path = run_dir / "predictions.csv"
        checkpoint_path = run_dir / "checkpoints" / "selected_best.pt"
        if not all(
            path.is_file()
            for path in (settings_path, predictions_path, checkpoint_path)
        ):
            raise FileNotFoundError(f"Incomplete paired O12 run: {run_dir}")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if settings.get("target_transform", "identity") != expected_transform:
            raise RuntimeError(f"Target transform mismatch in {run_dir}")
        if (
            settings.get("model_type") != "OneHotEmbedGPS"
            or settings.get("outer_test_read_during_selection") is not False
        ):
            raise RuntimeError(f"Invalid O12 selection protocol: {run_dir}")
        predictions = pd.read_csv(predictions_path)
        validation = predictions.loc[
            predictions["split"].eq("val")
            & predictions["target"].isin(TARGETS),
            ["sample_id", "target", "y_true", "y_pred"],
        ].copy()
        if validation.empty:
            raise RuntimeError(f"No validation predictions in {run_dir}")
        validation["split_seed"] = seed
        validation = validation.rename(columns={
            "y_true": f"{family}_y_true",
            "y_pred": f"{family}_prediction",
        })
        frames.append(validation)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-root", type=Path, required=True)
    parser.add_argument("--first-prefix", required=True)
    parser.add_argument("--first-name", default="first")
    parser.add_argument("--first-transform", choices=("identity", "log1p"),
                        default="log1p")
    parser.add_argument("--second-root", type=Path, required=True)
    parser.add_argument("--second-prefix", required=True)
    parser.add_argument("--second-name", default="second")
    parser.add_argument("--second-transform", choices=("identity", "log1p"),
                        default="log1p")
    parser.add_argument("--first-seed", type=int, default=200)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.first_name == args.second_name:
        raise ValueError("The two family names must differ.")

    first_root = args.first_root.resolve()
    second_root = args.second_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    first = read_validation(
        first_root, args.first_prefix, args.first_seed, args.seed_count,
        args.first_transform, "first")
    second = read_validation(
        second_root, args.second_prefix, args.first_seed, args.seed_count,
        args.second_transform, "second")
    keys = ["split_seed", "sample_id", "target"]
    paired = first.merge(second, on=keys, validate="one_to_one")
    if not np.allclose(
        paired["first_y_true"], paired["second_y_true"], rtol=0, atol=1e-7
    ):
        raise RuntimeError("Paired validation labels differ.")
    paired = paired.rename(columns={"first_y_true": "y_true"}).drop(
        columns="second_y_true")

    grid_rows = []
    chosen_rows = []
    for target in TARGETS:
        part = paired.loc[paired["target"].eq(target)]
        truth = part["y_true"].to_numpy(float)
        first_prediction = part["first_prediction"].to_numpy(float)
        second_prediction = part["second_prediction"].to_numpy(float)
        candidates = []
        for second_weight in np.linspace(0.0, 1.0, 101):
            prediction = (
                (1.0 - second_weight) * first_prediction
                + second_weight * second_prediction
            )
            row = {
                "target": target,
                "first_family": args.first_name,
                "second_family": args.second_name,
                "first_weight": float(1.0 - second_weight),
                "second_weight": float(second_weight),
                "pooled_validation_mae": float(
                    np.mean(np.abs(truth - prediction))),
            }
            candidates.append(row)
            grid_rows.append(row)
        chosen_rows.append(min(
            candidates,
            key=lambda row: (
                row["pooled_validation_mae"],
                -row["second_weight"],
            ),
        ))
    paired.to_csv(output / "paired_input_validation_predictions.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(
        output / "pair_blend_weight_grid.csv", index=False)
    weights_path = output / "pair_blend_weights.csv"
    pd.DataFrame(chosen_rows).to_csv(weights_path, index=False)
    protocol = {
        "frozen": True,
        "model_family": "paired O12 GraphGPS continuous convex blend",
        "first": {
            "name": args.first_name,
            "root": str(first_root),
            "prefix": args.first_prefix,
            "target_transform": args.first_transform,
        },
        "second": {
            "name": args.second_name,
            "root": str(second_root),
            "prefix": args.second_prefix,
            "target_transform": args.second_transform,
        },
        "split_seeds": list(
            range(args.first_seed, args.first_seed + args.seed_count)),
        "selection_metric": "pooled continuous raw-scale input validation MAE",
        "external_feedback_read": False,
        "threshold_or_side_criterion_used": False,
        "weights": str(weights_path),
        "weights_sha256": sha256(weights_path),
    }
    (output / "pair_blend_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(pd.DataFrame(chosen_rows).to_string(index=False))


if __name__ == "__main__":
    main()
