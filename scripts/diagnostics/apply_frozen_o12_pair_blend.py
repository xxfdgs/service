#!/usr/bin/env python3
"""Apply an input-validation-frozen blend to two paired O12 ensembles."""

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
    parser.add_argument("--first-ensemble", type=Path, required=True)
    parser.add_argument("--first-long", type=Path, required=True)
    parser.add_argument("--second-ensemble", type=Path, required=True)
    parser.add_argument("--second-long", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name", default="O12_group_pair_blend_10seed")
    args = parser.parse_args()

    paths = {
        "first_ensemble": args.first_ensemble.resolve(),
        "first_long": args.first_long.resolve(),
        "second_ensemble": args.second_ensemble.resolve(),
        "second_long": args.second_long.resolve(),
    }
    weights_path = args.weights.resolve()
    protocol_path = weights_path.parent / "pair_blend_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing pair-blend protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("external_feedback_read") is not False
        or protocol.get("threshold_or_side_criterion_used") is not False
    ):
        raise RuntimeError("Blend weights are not certified input-only.")

    first_long = pd.read_csv(paths["first_long"], dtype={"ID": str}).rename(
        columns={"prediction": "first_prediction"})
    second_long = pd.read_csv(paths["second_long"], dtype={"ID": str}).rename(
        columns={"prediction": "second_prediction"})
    keys = ["ID", "target_group", "split_seed", "target"]
    paired = first_long[keys + ["first_prediction"]].merge(
        second_long[keys + ["second_prediction"]],
        on=keys,
        validate="one_to_one",
    )
    expected_seeds = protocol["split_seeds"]
    if sorted(paired["split_seed"].unique()) != expected_seeds:
        raise RuntimeError("External prediction seeds do not match frozen weights.")
    weights = pd.read_csv(weights_path).set_index("target")
    paired["prediction"] = np.nan
    for target in TARGETS:
        mask = paired["target"].eq(target)
        first_weight = float(weights.loc[target, "first_weight"])
        second_weight = float(weights.loc[target, "second_weight"])
        paired.loc[mask, "prediction"] = (
            first_weight * paired.loc[mask, "first_prediction"]
            + second_weight * paired.loc[mask, "second_prediction"]
        )
    if not np.isfinite(paired["prediction"]).all():
        raise RuntimeError("Non-finite paired blend prediction.")

    aggregates = (
        paired.groupby(["ID", "target"], as_index=False)
        .agg(
            prediction_mean=("prediction", "mean"),
            prediction_std=("prediction", lambda values: np.std(values, ddof=0)),
        )
    )
    first_ensemble = pd.read_csv(paths["first_ensemble"], dtype={"ID": str})
    second_ensemble = pd.read_csv(paths["second_ensemble"], dtype={"ID": str})
    if set(first_ensemble["ID"]) != set(second_ensemble["ID"]):
        raise RuntimeError("The two ensemble ID sets differ.")
    blended = second_ensemble.copy()
    for target in TARGETS:
        values = aggregates.loc[
            aggregates["target"].eq(target)].set_index("ID").loc[blended["ID"]]
        blended[f"pred_{target}_mean"] = values[
            "prediction_mean"].to_numpy(float)
        blended[f"pred_{target}_std_10models"] = values[
            "prediction_std"].to_numpy(float)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ensemble_path = output / "ensemble_mean_predictions_norm2.csv"
    long_path = output / "predictions_by_model_long_norm2.csv"
    blended.to_csv(ensemble_path, index=False)
    paired.to_csv(long_path, index=False)
    provenance = {
        "model_family": "O12_continuous_pair_blend_10_paired_seeds",
        "evaluation_model_name": args.model_name,
        "model_type": "OneHotEmbedGPS",
        "paired_split_seeds": expected_seeds,
        "weights": weights.reset_index().to_dict(orient="records"),
        "weights_path": str(weights_path),
        "weights_sha256": sha256(weights_path),
        "weight_selection_protocol": str(protocol_path),
        "weight_selection_protocol_sha256": sha256(protocol_path),
        "labels_used_for_model_input": False,
        "labels_used_for_blending": False,
        "components": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "output": str(ensemble_path),
    }
    (output / "provenance_norm2.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(weights.reset_index().to_string(index=False))
    print(f"\nWrote {ensemble_path}")


if __name__ == "__main__":
    main()
