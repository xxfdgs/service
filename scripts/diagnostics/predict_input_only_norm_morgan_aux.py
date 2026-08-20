#!/usr/bin/env python3
"""Predict with the frozen grouped-input molecular auxiliary ensemble."""

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

from scripts.diagnostics.train_input_only_norm_morgan_aux import (
    feature_matrix,
    predict_raw,
)


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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model_path = args.model.resolve()
    input_path = args.input_csv.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(model_path)
    frame = pd.read_csv(input_path, dtype={"ID": str})
    if frame["ID"].isna().any() or frame["ID"].duplicated().any():
        raise ValueError("Prediction IDs must be complete and unique.")
    features = feature_matrix(frame)
    long_rows = []
    result = frame.copy()
    for target in artifact["targets"]:
        predictions = []
        for seed in artifact["split_seeds"]:
            values = predict_raw(
                artifact["models"][target][seed], features)
            predictions.append(values)
            long_rows.extend({
                "ID": str(sample_id),
                "target_group": "norm2",
                "split_seed": int(seed),
                "target": target,
                "prediction": float(value),
            } for sample_id, value in zip(frame["ID"], values))
        stacked = np.stack(predictions)
        result[f"pred_{target}_mean"] = stacked.mean(axis=0)
        result[f"pred_{target}_std_10models"] = stacked.std(axis=0, ddof=0)
    ensemble_path = output / "ensemble_mean_predictions_norm2.csv"
    long_path = output / "predictions_by_model_long_norm2.csv"
    result.to_csv(ensemble_path, index=False)
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    protocol_path = model_path.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("external_feedback_read") is not False
        or protocol.get("threshold_or_side_criterion_used") is not False
    ):
        raise RuntimeError("Auxiliary model is not certified input-only.")
    (output / "provenance_norm2.json").write_text(json.dumps({
        "model_family": "input_only_norm_morgan_aux_10seed",
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "training_protocol": str(protocol_path),
        "training_protocol_sha256": sha256(protocol_path),
        "source": str(input_path),
        "source_sha256": sha256(input_path),
        "models": 10,
        "labels_used_for_model_input": False,
        "output": str(ensemble_path),
        "output_sha256": sha256(ensemble_path),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {ensemble_path}")


if __name__ == "__main__":
    main()
