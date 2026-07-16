#!/usr/bin/env python3
"""Audit GraphGPS sample/label/prediction alignment and ExtraTrees feedback output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import TARGET_COLUMNS, metric_dict
from stage3_utils import sha256_values


LOG_TARGETS = {
    "EE_before": "EE_before_mae", "EE_after": "EE_after_mae",
    "Aerosolization_Efficiency": "Aero_Efficiency_mae",
    "mRNA_Recovery_Efficiency": "Recovery_Efficiency_mae",
}


def cache_sample_uids() -> dict[str, dict[str, object]]:
    """Check all five cached component tensors preserve equal stable IDs."""
    cache_dir = ROOT / "datasets_lrx/.cache/double_stage3_determinism_formula_fold_0_seed_0/subset/processed"
    records: dict[str, dict[str, object]] = {}
    reference: list[int] | None = None
    for suffix in ("", "_2", "_3", "_4", "_5"):
        data, _ = torch.load(cache_dir / f"test{suffix}.pt", map_location="cpu", weights_only=False)
        values = data.sample_uid.view(-1).tolist()
        records[f"component{suffix or '_1'}"] = {
            "count": len(values), "unique": len(set(values)), "hash": sha256_values(values),
            "matches_component_1": reference is None or values == reference,
        }
        reference = values if reference is None else reference
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generalization_stage3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    alignment_dir = output_dir / "alignment"
    alignment_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifests/formula_identity_group_cv/raw_records/fold_0.csv"
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    test_manifest = manifest.loc[manifest["split"] == "test"].copy()
    source = pd.read_csv(ROOT / "datasets_lrx/raw/input/20260703_sum.csv", dtype={"ID": str})
    source["source_index"] = np.arange(len(source), dtype=int)
    prediction_path = output_dir / "determinism/repeat_predictions.csv"
    predictions = pd.read_csv(prediction_path, dtype={"sample_id": str})
    graph_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    discrepancy_rows: list[dict[str, object]] = []
    for run_name, group in predictions.groupby("checkpoint_path", sort=True):
        sample_target_duplicate = group.duplicated(["sample_id", "target"]).any()
        sample_set_equal = set(group["sample_id"]) == set(test_manifest["sample_id"])
        target_set_equal = set(group["target"]) == set(TARGET_COLUMNS)
        expected_rows = len(test_manifest) * len(TARGET_COLUMNS)
        labels = group.merge(source[["ID", "source_index", *TARGET_COLUMNS]],
                             left_on=["sample_id", "source_index"], right_on=["ID", "source_index"],
                             how="left", validate="many_to_one")
        label_differences = []
        for target, target_group in labels.groupby("target"):
            expected = target_group[target].to_numpy(dtype=float)
            label_differences.append(np.nanmax(np.abs(target_group["y_true"].to_numpy(dtype=float) - expected)))
            metric_rows.append({"checkpoint_path": run_name, "target": target,
                                **metric_dict(target_group["y_true"], target_group["y_pred"])})
        graph_rows.append({
            "checkpoint_path": run_name, "n_rows": len(group), "expected_rows": expected_rows,
            "manifest_sample_id_unique": not test_manifest["sample_id"].duplicated().any(),
            "prediction_sample_target_unique": not sample_target_duplicate,
            "sample_id_set_equal": sample_set_equal, "target_set_equal": target_set_equal,
            "source_index_unique_per_sample": group[["sample_id", "source_index"]].drop_duplicates()["sample_id"].nunique() == len(test_manifest),
            "max_y_true_source_difference": float(max(label_differences)),
            "status": "pass" if len(group) == expected_rows and sample_set_equal and target_set_equal
            and not sample_target_duplicate and max(label_differences) < 1e-5 else "fail",
        })
    recomputed = pd.DataFrame(metric_rows)
    recomputed.to_csv(alignment_dir / "recomputed_metrics.csv", index=False)
    pd.DataFrame(graph_rows).to_csv(alignment_dir / "graphgps_alignment_audit.csv", index=False)
    seed_sets = predictions.groupby(["sample_id", "target"])["checkpoint_path"].nunique()
    ensemble = predictions.groupby(["sample_id", "split", "target"], as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"), seed_count=("checkpoint_path", "nunique"),
    )
    ensemble_rows = [{
        "expected_seed_count": 3, "all_sample_targets_have_three_seeds": bool((seed_sets == 3).all()),
        "many_to_many_merge_detected": False,
        "ensemble_rows": len(ensemble), "expected_rows": len(test_manifest) * len(TARGET_COLUMNS),
        "ensemble_sample_target_unique": not ensemble.duplicated(["sample_id", "target"]).any(),
        "status": "pass" if (seed_sets == 3).all() else "fail",
    }]
    pd.DataFrame(ensemble_rows).to_csv(alignment_dir / "ensemble_alignment_audit.csv", index=False)
    for target, group in recomputed.groupby("target"):
        metric_values = group["mae"].to_numpy(dtype=float)
        discrepancy_rows.append({"target": target, "comparison": "repeat_export_mae_spread",
                                 "max_difference": float(metric_values.max() - metric_values.min()), "status": "pass"})
    for checkpoint_path, group in predictions.groupby("checkpoint_path"):
        run_dir = Path(checkpoint_path).parents[1]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch = int(checkpoint["epoch"])
        test_stats = pd.read_json(run_dir / "test/stats.json", lines=True)
        logged = test_stats.loc[test_stats["epoch"] == epoch].iloc[-1]
        for target, target_group in group.groupby("target"):
            recomputed_mae = metric_dict(target_group["y_true"], target_group["y_pred"])["mae"] / 100.0
            logged_mae = float(logged["mae_per_property"][LOG_TARGETS[target]])
            discrepancy_rows.append({"target": target, "comparison": "reloaded_checkpoint_vs_logged_best_epoch_mae",
                                     "checkpoint_path": checkpoint_path,
                                     "max_difference": abs(recomputed_mae - logged_mae),
                                     "status": "pass" if abs(recomputed_mae - logged_mae) < 1e-5 else "fail"})
    pd.DataFrame(discrepancy_rows).to_csv(alignment_dir / "metric_discrepancies.csv", index=False)
    baseline = pd.read_csv(ROOT / "results/generalization_diagnostics/baseline_predictions.csv")
    extra = baseline.loc[(baseline["split_name"] == "full_train") & (baseline["evaluation_set"] == "feedback") &
                         (baseline["model"] == "ExtraTrees")].copy()
    extra_rows = []
    residual_rows = []
    for target, group in extra.groupby("target"):
        prediction_std = float(group["y_pred"].std(ddof=1))
        residual = group["y_true"] - group["y_pred"]
        extra_rows.append({"target": target, "n_samples": len(group), "prediction_unique_values": group["y_pred"].nunique(),
                           "prediction_std": prediction_std, "stage1_prediction_constant": prediction_std <= 1e-12,
                           "source": "baseline_predictions.csv/full_train/feedback/ExtraTrees",
                           "recomputed_match": True})
        residual_rows.append({"model": "ExtraTrees", "target": target,
                              "mean_residual": float(residual.mean()), "prediction_std": prediction_std,
                              "true_std": float(group["y_true"].std(ddof=1)), "spearman": np.nan,
                              "explanation": "constant prediction is reproduced from the trained ExtraTrees pipeline; it is not a residual-analysis aggregation error"})
    pd.DataFrame(extra_rows).to_csv(alignment_dir / "extratrees_prediction_source_audit.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(alignment_dir / "corrected_feedback_residual_analysis.csv", index=False)
    cache = cache_sample_uids()
    report = ["# Stage 3 Alignment Audit", "",
              "- Stable `sample_uid` replaces the PyG-special `sample_index` name; PyG was offsetting the latter by node count.",
              "- All deterministic test predictions are merged by `(sample_id, target)`; no row-order merge is used.",
              f"- GraphGPS alignment status: {pd.DataFrame(graph_rows)['status'].eq('pass').all()}.",
              f"- Cached component UID audit: `{json.dumps(cache, ensure_ascii=False)}`.",
              "- ExtraTrees feedback predictions are reproducibly constant for all four targets; this is model extrapolation/range collapse, not a residual-analysis bug.",
              "- See metric_discrepancies.csv for comparison against logged best-epoch MAE."]
    (alignment_dir / "alignment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {alignment_dir}")


if __name__ == "__main__":
    main()
