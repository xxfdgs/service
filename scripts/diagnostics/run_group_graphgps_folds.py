#!/usr/bin/env python3
"""Append full-budget three-seed GraphGPS results to existing group-CV artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_repeated_group_benchmark import bootstrap_ci, run_graph_fold  # noqa: E402
from stage2_common import add_stage2_arguments, load_training_frame, record_execution, stage2_output  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_stage2_arguments(parser)
    parser.add_argument("--fold-count", type=int, default=1)
    arguments = parser.parse_args()
    output_dir = stage2_output(arguments.output_dir)
    group_dir = output_dir / "group_cv"
    config_dir = group_dir / "graphgps_configs"
    input_dir = group_dir / "graphgps_inputs"
    schema, train_frame, _ = load_training_frame(arguments.train_csv, arguments.feedback_csv)
    metrics = pd.read_csv(group_dir / "fold_metrics.csv")
    predictions = pd.read_csv(group_dir / "oof_predictions.csv")
    seed_path = group_dir / "seed_metrics.csv"
    try:
        seed_metrics = pd.read_csv(seed_path) if seed_path.is_file() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        seed_metrics = pd.DataFrame()
    original_columns = pd.read_csv(schema.train_path, nrows=1).columns.tolist()
    for protocol in ("fifth_component_group_cv", "formula_identity_group_cv"):
        manifest_paths = sorted((output_dir / "manifests" / protocol).glob("fold_*.csv"))[:arguments.fold_count]
        for manifest_path in manifest_paths:
            fold = manifest_path.stem
            if ((metrics["protocol"] == protocol) & (metrics["fold"] == fold) &
                    (metrics["model"] == "GraphGPS_coarse_mordred_ensemble")).any():
                continue
            graph_metrics, graph_predictions = run_graph_fold(
                output_dir, protocol, manifest_path, train_frame, original_columns, config_dir, input_dir
            )
            ensemble = pd.DataFrame([row for row in graph_metrics if row["seed"] == "ensemble"])
            individual = pd.DataFrame([row for row in graph_metrics if row["seed"] != "ensemble"])
            metrics = pd.concat([metrics, ensemble], ignore_index=True)
            seed_metrics = pd.concat([seed_metrics, individual], ignore_index=True)
            predictions = pd.concat([predictions, *graph_predictions], ignore_index=True)
    metrics.to_csv(group_dir / "fold_metrics.csv", index=False)
    seed_metrics.to_csv(seed_path, index=False)
    predictions.to_csv(group_dir / "oof_predictions.csv", index=False)
    summary_rows = []
    for (protocol, target, model), group in metrics.groupby(["protocol", "target", "model"]):
        lower, upper = bootstrap_ci(group["mae"], arguments.seed)
        summary_rows.append({"protocol": protocol, "target": target, "model": model,
                             "completed_folds": group["fold"].nunique(), "mean_mae": group["mae"].mean(),
                             "std_mae": group["mae"].std(ddof=1), "mae_ci95_low": lower, "mae_ci95_high": upper,
                             "mean_rmse": group["rmse"].mean(), "mean_r2": group["r2"].mean()})
    pd.DataFrame(summary_rows).to_csv(group_dir / "summary_metrics.csv", index=False)
    graph = metrics.loc[metrics["model"] == "GraphGPS_coarse_mordred_ensemble"]
    trees = metrics.loc[metrics["model"] == "ExtraTrees"]
    paired_rows = []
    for (protocol, target), group in graph.groupby(["protocol", "target"]):
        paired = group.merge(trees.loc[(trees["protocol"] == protocol) & (trees["target"] == target)],
                             on=["protocol", "fold", "target"], suffixes=("_graphgps", "_extratrees"))
        if not paired.empty:
            difference = paired["mae_graphgps"] - paired["mae_extratrees"]
            paired_rows.append({"protocol": protocol, "target": target,
                                "completed_paired_folds": len(paired),
                                "mean_graphgps_minus_extratrees_mae": difference.mean(),
                                "graphgps_win_fraction": (difference < 0).mean()})
    pd.DataFrame(paired_rows).to_csv(group_dir / "paired_model_comparison.csv", index=False)
    record_execution(output_dir, Path(__file__).name, details={"fold_count": arguments.fold_count,
                     "seeds": [0, 1, 2], "full_original_budget": True})
    print(f"Appended GraphGPS folds to {group_dir}")


if __name__ == "__main__":
    main()
