#!/usr/bin/env python3
"""Apply frozen input-only O12 blend weights without reading target labels."""

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
    parser.add_argument("--identity-ensemble", type=Path, required=True)
    parser.add_argument("--identity-long", type=Path, required=True)
    parser.add_argument("--log1p-ensemble", type=Path, required=True)
    parser.add_argument("--log1p-long", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    identity_ensemble_path = args.identity_ensemble.resolve()
    identity_long_path = args.identity_long.resolve()
    log1p_ensemble_path = args.log1p_ensemble.resolve()
    log1p_long_path = args.log1p_long.resolve()
    weights_path = args.weights.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # The labelled columns may be carried through for later evaluation, but
    # only these prediction tables and keys participate in the blend.
    identity_long = pd.read_csv(
        identity_long_path, dtype={"ID": str}).rename(
        columns={"prediction": "identity_prediction"})
    log1p_long = pd.read_csv(
        log1p_long_path, dtype={"ID": str}).rename(
        columns={"prediction": "log1p_prediction"})
    keys = ["ID", "target_group", "split_seed", "target"]
    paired = identity_long[keys + ["identity_prediction"]].merge(
        log1p_long[keys + ["log1p_prediction"]],
        on=keys,
        validate="one_to_one",
    )
    if sorted(paired["split_seed"].unique()) != list(range(100, 110)):
        raise RuntimeError("Blend requires paired split seeds 100 through 109.")
    weights = pd.read_csv(weights_path).set_index("target")
    if set(weights.index) != set(TARGETS):
        raise RuntimeError("Frozen blend weights do not cover both Norm targets.")
    paired["prediction"] = np.nan
    for target in TARGETS:
        mask = paired["target"].eq(target)
        identity_weight = float(weights.loc[target, "identity_weight"])
        log1p_weight = float(weights.loc[target, "log1p_weight"])
        if not np.isclose(identity_weight + log1p_weight, 1.0):
            raise RuntimeError(f"Blend weights do not sum to one for {target}.")
        paired.loc[mask, "prediction"] = (
            identity_weight * paired.loc[mask, "identity_prediction"]
            + log1p_weight * paired.loc[mask, "log1p_prediction"]
        )
    if not np.isfinite(paired["prediction"]).all():
        raise RuntimeError("Non-finite blended prediction encountered.")

    aggregates = (
        paired.groupby(["ID", "target"], as_index=False)
        .agg(
            prediction_mean=("prediction", "mean"),
            prediction_std=("prediction", lambda values: np.std(values, ddof=0)),
        )
    )
    log1p_ensemble = pd.read_csv(
        log1p_ensemble_path, dtype={"ID": str})
    identity_ensemble = pd.read_csv(
        identity_ensemble_path, dtype={"ID": str})
    if set(log1p_ensemble["ID"]) != set(identity_ensemble["ID"]):
        raise RuntimeError("Identity and log1p ensemble IDs differ.")
    if log1p_ensemble["ID"].duplicated().any():
        raise RuntimeError("Ensemble IDs must be unique.")
    blended = log1p_ensemble.copy()
    for target in TARGETS:
        values = aggregates.loc[
            aggregates["target"].eq(target)].set_index("ID")
        values = values.loc[blended["ID"]]
        blended[f"pred_{target}_mean"] = values[
            "prediction_mean"].to_numpy(float)
        blended[f"pred_{target}_std_10models"] = values[
            "prediction_std"].to_numpy(float)

    ensemble_output = output / "ensemble_mean_predictions_norm2.csv"
    long_output = output / "predictions_by_model_long_norm2.csv"
    blended.to_csv(ensemble_output, index=False)
    paired.to_csv(long_output, index=False)
    protocol_path = weights_path.parent / "continuous_blend_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen weight-selection protocol: {protocol_path}")
    weight_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        weight_protocol.get("external_feedback_read") is not False
        or weight_protocol.get("threshold_or_side_criterion_used") is not False
    ):
        raise RuntimeError("Blend weights are not certified input-only.")
    provenance = {
        "model_family": "O12_continuous_blend_10_paired_seeds",
        "evaluation_model_name": "O12_continuous_blend_10seed",
        "model_type": "OneHotEmbedGPS",
        "paired_split_seeds": list(range(100, 110)),
        "target_transforms": ["identity", "log1p"],
        "blend_scope": "pair checkpoints by split seed, then average ten blended predictions",
        "weights": weights.reset_index().to_dict(orient="records"),
        "weights_path": str(weights_path),
        "weights_sha256": sha256(weights_path),
        "weight_selection_protocol": str(protocol_path),
        "weight_selection_protocol_sha256": sha256(protocol_path),
        "labels_used_for_model_input": False,
        "labels_used_for_blending": False,
        "identity_ensemble": str(identity_ensemble_path),
        "identity_ensemble_sha256": sha256(identity_ensemble_path),
        "identity_long_predictions": str(identity_long_path),
        "identity_long_predictions_sha256": sha256(identity_long_path),
        "log1p_ensemble": str(log1p_ensemble_path),
        "log1p_ensemble_sha256": sha256(log1p_ensemble_path),
        "log1p_long_predictions": str(log1p_long_path),
        "log1p_long_predictions_sha256": sha256(log1p_long_path),
        "output": str(ensemble_output),
        "output_sha256": sha256(ensemble_output),
    }
    (output / "provenance_norm2.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(weights.reset_index().to_string(index=False))
    print(f"\nWrote {ensemble_output}")


if __name__ == "__main__":
    main()
