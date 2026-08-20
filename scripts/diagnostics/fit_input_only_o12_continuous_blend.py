#!/usr/bin/env python3
"""Freeze input-validation MAE weights for identity/log1p O12 predictions.

The candidate grid is a convex blend of paired O12 GraphGPS checkpoints.
Selection uses only continuous raw-scale MAE on the ten input validation
splits.  No external table or threshold-derived quantity is read.
"""

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


def read_validation(root: Path, prefix: str, expected_transform: str) -> pd.DataFrame:
    frames = []
    for seed in range(100, 110):
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
        transform = settings.get("target_transform", "identity")
        if transform != expected_transform:
            raise RuntimeError(
                f"Expected {expected_transform} targets in {run_dir}, got {transform}")
        if settings.get("model_type") != "OneHotEmbedGPS":
            raise RuntimeError(f"Run is not OneHotEmbedGPS: {run_dir}")
        if settings.get("outer_test_read_during_selection") is not False:
            raise RuntimeError(f"Outer test was not isolated in {run_dir}")
        predictions = pd.read_csv(predictions_path)
        validation = predictions.loc[
            predictions["split"].eq("val")
            & predictions["target"].isin(TARGETS),
            ["sample_id", "target", "y_true", "y_pred"],
        ].copy()
        if len(validation) != 70 * len(TARGETS):
            raise RuntimeError(
                f"Expected 140 validation rows in {run_dir}, "
                f"found {len(validation)}")
        validation["split_seed"] = seed
        frames.append(validation)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-runs-root", type=Path, required=True)
    parser.add_argument("--log1p-runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    identity_root = args.identity_runs_root.resolve()
    log1p_root = args.log1p_runs_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = read_validation(identity_root, "O12_split", "identity").rename(
        columns={"y_true": "identity_y_true", "y_pred": "identity_prediction"})
    log1p = read_validation(log1p_root, "O12Log_split", "log1p").rename(
        columns={"y_true": "log1p_y_true", "y_pred": "log1p_prediction"})
    keys = ["split_seed", "sample_id", "target"]
    paired = identity.merge(log1p, on=keys, validate="one_to_one")
    if not np.allclose(
        paired["identity_y_true"], paired["log1p_y_true"], rtol=0, atol=1e-7
    ):
        raise RuntimeError("Paired validation labels do not match.")
    paired = paired.rename(columns={"identity_y_true": "y_true"}).drop(
        columns="log1p_y_true")

    grid_rows: list[dict[str, object]] = []
    weights_rows: list[dict[str, object]] = []
    weights = np.linspace(0.0, 1.0, 101)
    for target in TARGETS:
        target_rows = paired.loc[paired["target"].eq(target)]
        truth = target_rows["y_true"].to_numpy(float)
        identity_prediction = target_rows["identity_prediction"].to_numpy(float)
        log1p_prediction = target_rows["log1p_prediction"].to_numpy(float)
        target_grid = []
        for log1p_weight in weights:
            prediction = (
                (1.0 - log1p_weight) * identity_prediction
                + log1p_weight * log1p_prediction
            )
            seed_maes = []
            for seed in range(100, 110):
                mask = target_rows["split_seed"].to_numpy(int) == seed
                seed_maes.append(float(np.mean(np.abs(
                    truth[mask] - prediction[mask]))))
            row = {
                "target": target,
                "identity_weight": float(1.0 - log1p_weight),
                "log1p_weight": float(log1p_weight),
                "pooled_validation_mae": float(
                    np.mean(np.abs(truth - prediction))),
                "mean_seed_validation_mae": float(np.mean(seed_maes)),
                "std_seed_validation_mae": float(
                    np.std(seed_maes, ddof=1)),
            }
            target_grid.append(row)
            grid_rows.append(row)
        chosen = min(
            target_grid,
            key=lambda row: (
                row["pooled_validation_mae"],
                -row["log1p_weight"],
            ),
        )
        weights_rows.append(chosen)

    weights_frame = pd.DataFrame(weights_rows)
    grid_frame = pd.DataFrame(grid_rows)
    paired.to_csv(output / "paired_input_validation_predictions.csv", index=False)
    grid_frame.to_csv(output / "continuous_blend_weight_grid.csv", index=False)
    weights_path = output / "continuous_blend_weights.csv"
    weights_frame.to_csv(weights_path, index=False)
    protocol = {
        "frozen": True,
        "model_family": "paired ten-seed O12 GraphGPS continuous convex blend",
        "identity_runs_root": str(identity_root),
        "log1p_runs_root": str(log1p_root),
        "paired_split_seeds": list(range(100, 110)),
        "targets": list(TARGETS),
        "candidate_log1p_weights": [
            float(value) for value in weights],
        "selection_metric": "pooled continuous raw-scale input validation MAE",
        "tie_breaker": "larger log1p weight",
        "external_feedback_read": False,
        "threshold_or_side_criterion_used": False,
        "weights_file": str(weights_path),
        "weights_sha256": sha256(weights_path),
    }
    (output / "continuous_blend_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(weights_frame.to_string(index=False))


if __name__ == "__main__":
    main()
