#!/usr/bin/env python3
"""Compare audited record treatments with ExtraTrees and compatible GraphGPS outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import metric_dict  # noqa: E402
from run_repeated_group_benchmark import make_pipeline, run_graph_fold  # noqa: E402
from stage2_common import (  # noqa: E402
    add_stage2_arguments, group_cv_manifests, load_manifest_frame, load_training_frame,
    record_execution, stage2_output,
)
from stable_formulation import build_stable_feature_sets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_stage2_arguments(parser)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--skip-graphgps", action="store_true")
    parser.add_argument("--run-median-graphgps", action="store_true",
                        help="Run one full-budget three-seed GraphGPS fold per protocol for replicate_median.")
    arguments = parser.parse_args()
    output_dir = stage2_output(arguments.output_dir)
    audit_dir = output_dir / "data_audit"
    experiment_dir = output_dir / "replicate_experiments"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    versions = {
        "raw_records": audit_dir / "dataset_raw_records.csv",
        "replicate_median": audit_dir / "dataset_replicate_median.csv",
        "replicate_weighted": audit_dir / "dataset_replicate_weighted.csv",
    }
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    protocols = {
        "fifth_component_group_cv": "fifth_component_key",
        "formula_identity_group_cv": "formula_identity_key",
    }
    for version, data_path in versions.items():
        schema, frame, _ = load_training_frame(data_path, arguments.feedback_csv)
        feature_sets, _, _ = build_stable_feature_sets(frame, schema)
        features = feature_sets["F2_identity_ratio"]
        for protocol, group_column in protocols.items():
            manifest_paths = group_cv_manifests(
                frame, group_column, f"replicate_{version}_{protocol}", output_dir,
                arguments.seed, arguments.n_splits,
            )
            for manifest_path in manifest_paths:
                manifest = load_manifest_frame(frame, manifest_path)
                train_indices = manifest.index[manifest["split"] == "train"]
                test_indices = manifest.index[manifest["split"] == "test"]
                for target in schema.targets:
                    model = make_pipeline(features, clone(ExtraTreesRegressor(
                        n_estimators=500, min_samples_leaf=2, max_features=0.8,
                        random_state=arguments.seed, n_jobs=arguments.n_jobs,
                    )))
                    fit_arguments = {}
                    if version == "replicate_weighted":
                        fit_arguments["model__sample_weight"] = frame.loc[
                            train_indices, f"sample_weight_{target}"
                        ].astype(float).to_numpy()
                    model.fit(features.loc[train_indices], frame.loc[train_indices, target], **fit_arguments)
                    prediction = model.predict(features.loc[test_indices])
                    true_values = frame.loc[test_indices, target]
                    metric_rows.append({
                        "version": version, "protocol": protocol, "fold": manifest_path.stem,
                        "target": target, "model": "ExtraTrees", "n_test": len(test_indices),
                        "used_sample_weight": version == "replicate_weighted",
                        **metric_dict(true_values, prediction),
                    })
                    prediction_frame = pd.DataFrame({
                        "version": version, "protocol": protocol, "fold": manifest_path.stem,
                        "target": target, "model": "ExtraTrees",
                        "sample_id": frame.loc[test_indices, "sample_id"].astype(str).to_numpy(),
                        "y_true": true_values.to_numpy(), "y_pred": prediction,
                    })
                    prediction_frame["absolute_error"] = (prediction_frame["y_true"] - prediction_frame["y_pred"]).abs()
                    prediction_frames.append(prediction_frame)
            if version == "replicate_median" and arguments.run_median_graphgps and not arguments.skip_graphgps:
                original_columns = pd.read_csv(
                    ROOT / "datasets_lrx/raw/input/20260703_sum.csv", nrows=1
                ).columns.tolist()
                graph_metrics, graph_predictions = run_graph_fold(
                    experiment_dir, f"replicate_median_{protocol}", manifest_paths[0], frame,
                    original_columns, experiment_dir / "graphgps_configs", experiment_dir / "graphgps_inputs",
                    data_csv=data_path,
                )
                for graph_metric in graph_metrics:
                    if graph_metric["seed"] == "ensemble":
                        graph_metric["version"] = version
                        graph_metric["protocol"] = protocol
                        graph_metric["used_sample_weight"] = False
                        metric_rows.append(graph_metric)
                for graph_prediction in graph_predictions:
                    if graph_prediction["seed"].iloc[0] == "ensemble":
                        graph_prediction["version"] = version
                        graph_prediction["protocol"] = protocol
                        prediction_frames.append(graph_prediction)
    graph_path = output_dir / "group_cv" / "oof_predictions.csv"
    if graph_path.is_file():
        graph = pd.read_csv(graph_path)
        graph = graph.loc[graph["model"] == "GraphGPS_coarse_mordred_ensemble"].copy()
        if not graph.empty:
            graph["version"] = "raw_records"
            prediction_frames.append(graph)
            for keys, group in graph.groupby(["protocol", "fold", "target"]):
                metric_rows.append({
                    "version": "raw_records", "protocol": keys[0], "fold": keys[1],
                    "target": keys[2], "model": "GraphGPS_coarse_mordred_ensemble",
                    "n_test": len(group), "used_sample_weight": False,
                    **metric_dict(group["y_true"], group["y_pred"]),
                })
    pd.DataFrame(metric_rows).to_csv(experiment_dir / "replicate_treatment_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        experiment_dir / "replicate_treatment_predictions.csv", index=False
    )
    report = [
        "# 重复样本处理对照", "",
        "- ExtraTrees 对 replicate_weighted 使用目标专属 sample_weight。",
        "- GraphGPS 当前不支持 sample_weight，未以复制或删除记录模拟权重。",
        "- raw GraphGPS 复用完全相同 raw manifest 的已完成 Group CV OOF 结果；"
        "replicate_median GraphGPS 完整训练须在下一 GPU 队列运行。",
    ]
    (experiment_dir / "replicate_treatment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    record_execution(output_dir, Path(__file__).name, details={
        "seed": arguments.seed, "n_jobs": arguments.n_jobs, "n_splits": arguments.n_splits,
        "versions": list(versions), "graphgps_weight_support": False,
        "run_median_graphgps": arguments.run_median_graphgps,
    })
    print(f"Wrote replicate treatment comparisons to {experiment_dir}")


if __name__ == "__main__":
    main()
