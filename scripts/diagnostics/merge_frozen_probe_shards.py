#!/usr/bin/env python3
"""Merge isolated frozen-probe shards and verify complete development coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FOLDS = ["fold_0", "fold_4"]
EPOCHS = ["epoch_initial", "epoch_precollapse", "epoch_collapse", "epoch_best", "epoch_last"]
EMBEDDINGS = [
    "graph_branch_raw", "descriptor_branch_raw", "formula_branch_raw",
    "graph_branch_projected", "descriptor_branch_projected", "formula_branch_projected",
    "fused_embedding", "head_hidden", "final_prediction",
]
TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]
PROBES = ["P0_TrainMean", "P1_Ridge", "P2_ElasticNet", "P3_PLS", "P4_ExtraTrees", "P5_RandomForest"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/frozen_embedding_signal_exp")
    args = parser.parse_args()
    root = args.output_root.resolve()
    shards = sorted(path for path in (root / "probes_shards").iterdir() if path.is_dir())
    if not shards:
        raise FileNotFoundError("No probe shards found")
    for shard in shards:
        for name in ("probe_metrics.csv", "probe_predictions.csv", "inner_cv_selection.csv", "protocol.json"):
            if not (shard / name).is_file():
                raise RuntimeError(f"Shard is incomplete: {shard / name}")
    frames = {name: pd.concat([pd.read_csv(shard / name) for shard in shards], ignore_index=True)
              for name in ("probe_metrics.csv", "probe_predictions.csv", "inner_cv_selection.csv")}
    metrics = frames["probe_metrics.csv"].drop_duplicates(
        subset=["fold", "epoch_label", "embedding_name", "target", "probe", "split"], keep="first")
    predictions = frames["probe_predictions.csv"].drop_duplicates(
        subset=["fold", "epoch_label", "embedding_name", "target", "probe", "split", "sample_id"], keep="first")
    selection_source = frames["inner_cv_selection.csv"].copy()
    selection_source["_status_priority"] = selection_source.status.eq("ok").astype(int)
    selection = selection_source.sort_values("_status_priority", ascending=False).drop_duplicates(
        subset=["fold", "epoch_label", "embedding_name", "target", "probe", "params"], keep="first").drop(columns="_status_priority")
    expected = pd.MultiIndex.from_product([FOLDS, EPOCHS, EMBEDDINGS, TARGETS, PROBES, ["train", "validation"]],
                                          names=["fold", "epoch_label", "embedding_name", "target", "probe", "split"])
    observed = pd.MultiIndex.from_frame(metrics.loc[metrics.probe.ne("GraphGPS_final"), expected.names])
    missing = expected.difference(observed)
    if len(missing):
        raise RuntimeError(f"Missing nested-probe coverage for {len(missing)} required rows; first={list(missing[:5])}")
    destination = root / "probes"
    destination.mkdir(parents=True, exist_ok=True)
    metrics.sort_values(["fold", "epoch_label", "embedding_name", "target", "probe", "split"]).to_csv(destination / "probe_metrics.csv", index=False)
    predictions.sort_values(["fold", "epoch_label", "embedding_name", "target", "probe", "split", "sample_id"]).to_csv(destination / "probe_predictions.csv", index=False)
    selection.sort_values(["fold", "epoch_label", "embedding_name", "target", "probe", "params"]).to_csv(destination / "inner_cv_selection.csv", index=False)
    (destination / "protocol.json").write_text(json.dumps({
        "outer_test_opened": False,
        "inner_cv": "GroupKFold formula identity within outer-train",
        "shards": [str(path.relative_to(root)) for path in shards],
        "required_metrics_rows": int(len(expected)), "merged_metrics_rows": int(len(metrics)),
    }, indent=2) + "\n")
    manifest = root / "execution_manifest.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    for shard in shards:
        protocol = json.loads((shard / "protocol.json").read_text())
        records.append({"timestamp": pd.Timestamp.now("UTC").isoformat(),
                        "command": f"run_frozen_embedding_probes.py --probes-output-dir {shard}",
                        "stage": "nested_frozen_probe_shard", "fold": "fold_0,fold_4",
                        "split": "outer-train,validation", "epoch": "all selected", "checkpoint": None,
                        "embedding_name": "shard:" + shard.name, "probe": "P0-P5", "seed": 0,
                        "dataset_hash": None, "manifest_hash": None, "feature_hash": None, "config_hash": None,
                        "checkpoint_hash": None, "embedding_hash": None, "status": "completed", "error": None,
                        "output_path": str(shard), "notes": protocol.get("cached_duplicate_embeddings")})
    records.append({"timestamp": pd.Timestamp.now("UTC").isoformat(), "command": " ".join(sys.argv), "stage": "nested_probe_shard_merge",
                    "fold": "fold_0,fold_4", "split": "outer-train,validation", "epoch": "all selected", "checkpoint": None,
                    "embedding_name": "all", "probe": "P0-P5", "seed": 0, "dataset_hash": None, "manifest_hash": None,
                    "feature_hash": None, "config_hash": None, "checkpoint_hash": None, "embedding_hash": None,
                    "status": "completed", "error": None, "output_path": str(destination)})
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"shards": len(shards), "metrics": len(metrics), "predictions": len(predictions),
                      "inner_selection": len(selection), "missing_required": len(missing)}))


if __name__ == "__main__":
    main()
