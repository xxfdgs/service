#!/usr/bin/env python3
"""Predict Norm targets with a frozen input-only log-RF ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.build_input_only_o12_residual_tree_head import feature_frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    model_path = args.model.resolve()
    input_csv = args.input_csv.resolve()
    artifact = joblib.load(model_path)
    frame = pd.read_csv(input_csv, dtype={"ID": str})
    if "ID" not in frame or frame.ID.isna().any() or frame.ID.duplicated().any():
        raise ValueError("Prediction input requires unique, non-null ID values.")
    features = feature_frame(frame)
    if features.columns.tolist() != artifact["feature_columns"]:
        raise RuntimeError("Prediction feature schema differs from the frozen model.")
    transformed = artifact["preprocessor"].transform(features)

    output = frame.copy()
    for target in artifact["targets"]:
        predictions = np.stack([
            np.maximum(np.expm1(estimator.predict(transformed)), 0.0)
            for estimator in artifact["models"][target]
        ])
        output[f"pred_{target}_input_only_mean"] = predictions.mean(axis=0)
        output[f"pred_{target}_input_only_std_10models"] = predictions.std(axis=0, ddof=0)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    provenance = {
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "input": str(input_csv),
        "input_sha256": sha256(input_csv),
        "rows": len(frame),
        "labels_read_by_model": False,
        "targets": list(artifact["targets"]),
        "target_transform": artifact["target_transform"],
        "selected": artifact["selected"],
    }
    args.output_csv.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output[[
        "ID",
        *[f"pred_{target}_input_only_mean" for target in artifact["targets"]],
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
