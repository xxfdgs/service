#!/usr/bin/env python3
"""Run resource-limited three-seed GraphGPS folds on audited replicate-median data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import metric_dict  # noqa: E402
from run_repeated_group_benchmark import run_graph_fold  # noqa: E402
from stage2_common import add_stage2_arguments, load_training_frame, record_execution, stage2_output  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_stage2_arguments(parser)
    parser.add_argument("--fold-count", type=int, default=1)
    arguments = parser.parse_args()
    output_dir = stage2_output(arguments.output_dir)
    audit_dir = output_dir / "data_audit"
    experiment_dir = output_dir / "replicate_experiments"
    config_dir = experiment_dir / "graphgps_configs"
    input_dir = experiment_dir / "graphgps_inputs"
    for directory in (config_dir, input_dir):
        directory.mkdir(parents=True, exist_ok=True)
    median_csv = audit_dir / "dataset_replicate_median.csv"
    schema, median_frame, _ = load_training_frame(median_csv, arguments.feedback_csv)
    original_columns = pd.read_csv(ROOT / "datasets_lrx/raw/input/20260703_sum.csv", nrows=1).columns.tolist()
    metrics = pd.read_csv(experiment_dir / "replicate_treatment_metrics.csv")
    predictions = pd.read_csv(experiment_dir / "replicate_treatment_predictions.csv")
    for base_protocol in ("fifth_component_group_cv", "formula_identity_group_cv"):
        manifest_dir = output_dir / "manifests" / f"replicate_replicate_median_{base_protocol}"
        for manifest_path in sorted(manifest_dir.glob("fold_*.csv"))[:arguments.fold_count]:
            graph_metrics, graph_predictions = run_graph_fold(
                experiment_dir, f"replicate_median_{base_protocol}", manifest_path, median_frame,
                original_columns, config_dir, input_dir, data_csv=median_csv,
            )
            for metric in graph_metrics:
                if metric["seed"] == "ensemble":
                    metric["version"] = "replicate_median"
                    metric["protocol"] = base_protocol
                    metric["used_sample_weight"] = False
                    metrics = pd.concat([metrics, pd.DataFrame([metric])], ignore_index=True)
            for prediction in graph_predictions:
                if prediction["seed"].iloc[0] == "ensemble":
                    prediction["version"] = "replicate_median"
                    prediction["protocol"] = base_protocol
                    predictions = pd.concat([predictions, prediction], ignore_index=True)
    # Reuse matching raw-data GraphGPS group-CV OOF results as the raw arm.
    group_oof = pd.read_csv(output_dir / "group_cv" / "oof_predictions.csv")
    raw_graph = group_oof.loc[group_oof["model"] == "GraphGPS_coarse_mordred_ensemble"].copy()
    for keys, group in raw_graph.groupby(["protocol", "fold", "target"]):
        metrics = pd.concat([metrics, pd.DataFrame([{
            "version": "raw_records", "protocol": keys[0], "fold": keys[1], "target": keys[2],
            "model": "GraphGPS_coarse_mordred_ensemble", "n_test": len(group),
            "used_sample_weight": False, **metric_dict(group["y_true"], group["y_pred"]),
        }])], ignore_index=True)
    raw_graph["version"] = "raw_records"
    predictions = pd.concat([predictions, raw_graph], ignore_index=True)
    metrics.to_csv(experiment_dir / "replicate_treatment_metrics.csv", index=False)
    predictions.to_csv(experiment_dir / "replicate_treatment_predictions.csv", index=False)
    record_execution(output_dir, Path(__file__).name, details={"fold_count": arguments.fold_count,
                     "seeds": [0, 1, 2], "graphgps_sample_weight": "unsupported"})
    print(f"Appended replicate-median GraphGPS results to {experiment_dir}")


if __name__ == "__main__":
    main()
