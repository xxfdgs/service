#!/usr/bin/env python3
"""Repeat the legacy GraphGPS split once to distinguish manifest effects from run variance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_graphgps_reproducibility import (  # noqa: E402
    _base_prediction_config, _base_training_config, _prediction_records,
    _run_prediction, _selected_epoch_metrics,
)
from stage2_common import add_stage2_arguments, load_training_frame, record_execution, stage2_output  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_stage2_arguments(parser)
    parser.add_argument("--update-audit-only", action="store_true")
    arguments = parser.parse_args()
    output_dir = stage2_output(arguments.output_dir)
    reproduce_dir = output_dir / "reproducibility"
    config_dir = reproduce_dir / "configs"
    input_dir = reproduce_dir / "evaluation_inputs"
    if arguments.update_audit_only:
        audit_path = reproduce_dir / "split_protocol_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["cuda_determinism_runtime"] = {
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        }
        audit["legacy_repeat_completed"] = True
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        record_execution(output_dir, Path(__file__).name, details={"update_audit_only": True, "seed": 0})
        print(f"Updated {audit_path}")
        return
    schema, train_frame, _ = load_training_frame(arguments.train_csv, arguments.feedback_csv)
    protocol = "legacy_repeat_seed0"
    config_path = config_dir / f"{protocol}.yaml"
    config_path.write_text(yaml.safe_dump(_base_training_config(output_dir, protocol, None), sort_keys=False), encoding="utf-8")
    import subprocess
    subprocess.run([sys.executable, "main.py", "--cfg", str(config_path), "--repeat", "1"], cwd=ROOT, check=True)
    training_dir = reproduce_dir / "training" / protocol
    epoch_metrics = _selected_epoch_metrics(training_dir)
    manifest = pd.read_csv(reproduce_dir / "manifests/legacy_split_seed0_reference.csv", dtype={"sample_id": str})
    all_predictions = []
    all_metrics = []
    original_columns = pd.read_csv(schema.train_path, nrows=1).columns.tolist()
    indexed = train_frame.set_index("sample_id", drop=False)
    for split_name in ("val", "test"):
        frame = indexed.loc[manifest.loc[manifest["split"] == split_name, "sample_id"]].reset_index(drop=True)
        input_path = input_dir / f"{protocol}_{split_name}.csv"
        frame[original_columns].to_csv(input_path, index=False)
        prediction_config_path = config_dir / f"{protocol}_{split_name}_predict.yaml"
        prediction_config_path.write_text(yaml.safe_dump(
            _base_prediction_config(output_dir, protocol, training_dir, input_path), sort_keys=False
        ), encoding="utf-8")
        prediction = pd.read_csv(_run_prediction(prediction_config_path))
        records, metrics = _prediction_records(protocol, split_name, frame, prediction, epoch_metrics)
        all_predictions.extend(records)
        all_metrics.extend(metrics)
    repeated_predictions = pd.concat(all_predictions, ignore_index=True)
    existing_predictions = pd.read_csv(reproduce_dir / "reproducibility_predictions.csv")
    legacy_predictions = existing_predictions.loc[existing_predictions["protocol"] == "legacy_split_seed0"]
    comparisons = []
    for (split_name, target), group in repeated_predictions.groupby(["evaluation_set", "target"]):
        original = legacy_predictions.loc[(legacy_predictions["evaluation_set"] == split_name) &
                                          (legacy_predictions["target"] == target)]
        merged = original.merge(group, on="sample_id", suffixes=("_first", "_repeat"), validate="one_to_one")
        comparisons.append({
            "comparison": "legacy_first_vs_legacy_repeat", "evaluation_set": split_name, "target": target,
            "mae_difference": abs(group["absolute_error"].mean() - original["absolute_error"].mean()),
            "max_single_prediction_difference": (merged["y_pred_first"] - merged["y_pred_repeat"]).abs().max(),
        })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(reproduce_dir / "legacy_repeat_comparison.csv", index=False)
    repeated_predictions.to_csv(reproduce_dir / "legacy_repeat_predictions.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(reproduce_dir / "legacy_repeat_metrics.csv", index=False)
    audit_path = reproduce_dir / "split_protocol_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["cuda_determinism_runtime"] = {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
    }
    audit["legacy_repeat_completed"] = True
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    record_execution(output_dir, Path(__file__).name, details={"seed": 0, "protocol": protocol})
    print(f"Wrote legacy repeat variance diagnosis to {reproduce_dir}")


if __name__ == "__main__":
    main()
