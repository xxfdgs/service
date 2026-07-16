#!/usr/bin/env python3
"""Lock development candidates from completed hybrid GroupKFold shards.

This script consumes only development-fold *inner OOF* outputs.  It does not
open any outer-test prediction or label.  It is intentionally a separate
operation so a partial or interrupted grid cannot silently become a lock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_hybrid_embedding_tree_experiment import BASE, ROOT, TARGETS, append_execution


OUTPUT = ROOT / "results/hybrid_embedding_tree_exp"
STAGE = OUTPUT / "stage1"
EMBEDDING_FAMILIES = {"A5", "A6", "A7", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"}
TREE_ONLY_FAMILIES = {"A1", "A2", "A3", "A4"}
MODEL_COMPLEXITY = {"M0": 0, "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 6}
# M3 is deliberately deferred by the user while its long-running independent
# queue continues.  It must not be compared partially or silently selected.
ACTIVE_MODELS = {"M0", "M1", "M2", "M4", "M5", "M6"}
DEFERRED_MODELS = {"M3"}


def load_required_shards() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Shards are deliberately model/family granular so an interrupted full
    # grid can resume without recomputing a completed random-forest block.
    # Ignore one-off smoke directories and discover only the canonical f0_/f4_
    # development namespaces.
    metric_paths = []
    prediction_paths = []
    for fold in (0, 4):
        directories = sorted(
            path for path in (STAGE / "shards").glob(f"f{fold}_*")
            if path.is_dir() and not any(path.name.endswith(f"_{model.lower()}") for model in DEFERRED_MODELS)
        )
        metric_paths.extend(path / f"fold_{fold}_metrics.csv" for path in directories)
        prediction_paths.extend(path / f"fold_{fold}_selected_inner_oof_predictions.csv" for path in directories)
    missing = [str(path) for path in metric_paths + prediction_paths if not path.is_file()]
    if missing:
        raise RuntimeError("STAGE1_INCOMPLETE: " + "; ".join(missing))
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    predictions = pd.concat([pd.read_csv(path) for path in prediction_paths], ignore_index=True)
    metrics = metrics.loc[metrics.model.isin(ACTIVE_MODELS)].copy()
    predictions = predictions.loc[predictions.model.isin(ACTIVE_MODELS)].copy()
    expected = set(TARGETS)
    required_families = {"A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"}
    required_models = ACTIVE_MODELS
    if set(metrics.target) != expected or set(metrics.outer_fold) != {0, 4} or not required_families.issubset(set(metrics.feature_family)):
        raise RuntimeError("STAGE1_INCOMPLETE_OR_MALFORMED")
    # M6 is legitimately absent from categorical F2-containing families; all
    # other grid combinations must have an explicit completed or N/A record.
    absent = required_models - set(metrics.model)
    if absent:
        raise RuntimeError(f"STAGE1_INCOMPLETE_MISSING_MODEL_FAMILIES: {sorted(absent)}")
    return metrics, predictions


def current_development_baseline() -> pd.DataFrame:
    path = BASE / "tree_baselines" / "validation_feature_selection.csv"
    values = pd.read_csv(path)
    values = values.loc[(values.protocol == "formula_identity_group_cv") & values.outer_fold.isin([0, 4])].copy()
    best = values.loc[values.groupby(["outer_fold", "target"]).validation_mae.idxmin()].copy()
    return best.rename(columns={"validation_mae": "existing_tree_baseline_validation_mae", "model": "existing_tree_model", "feature_set": "existing_tree_feature"})[
        ["outer_fold", "target", "existing_tree_baseline_validation_mae", "existing_tree_model", "existing_tree_feature"]]


def choose_params(metrics: pd.DataFrame) -> pd.DataFrame:
    # Hyperparameters are chosen inside each outer train by inner-group OOF MAE.
    keyed = ["outer_fold", "target", "feature_family", "b11_base", "model"]
    # Include NaN b11_base in groups consistently by making the representation explicit.
    rows = metrics.copy()
    rows["b11_base_key"] = rows.b11_base.fillna("__NONE__")
    keyed[-2] = "b11_base_key"
    rows = rows.sort_values(["mae", "r2", "params_json"], ascending=[True, False, True])
    return rows.groupby(keyed, as_index=False, dropna=False).first().drop(columns="b11_base_key")


def gate_candidates(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tree = selected.loc[selected.feature_family.isin(TREE_ONLY_FAMILIES)].copy()
    tree = tree.sort_values(["outer_fold", "target", "mae", "r2"], ascending=[True, True, True, False])
    tree_best = tree.groupby(["outer_fold", "target"], as_index=False).first()
    tree_best = tree_best.rename(columns={column: f"tree_{column}" for column in tree_best.columns if column not in {"outer_fold", "target"}})
    hybrid = selected.loc[(selected.feature_family.isin(EMBEDDING_FAMILIES)) & (selected.model != "M0")].copy()
    merged = hybrid.merge(tree_best, on=["outer_fold", "target"], how="left", validate="many_to_one")
    merged["mae_change_vs_inner_tree"] = merged.mae - merged.tree_mae
    merged["mae_change_pct_vs_inner_tree"] = merged.mae_change_vs_inner_tree / merged.tree_mae
    merged["spearman_change_vs_inner_tree"] = merged.spearman - merged.tree_spearman
    merged["std_distance_change_vs_inner_tree"] = (merged.std_ratio - 1).abs() - (merged.tree_std_ratio - 1).abs()
    merged["prediction_range_ratio"] = (merged.prediction_std * 6) / np.maximum(merged.target_std * 6, 1e-12)
    per_pipeline = []
    group_cols = ["target", "feature_family", "b11_base", "model"]
    merged["b11_base_key"] = merged.b11_base.fillna("__NONE__")
    group_cols[2] = "b11_base_key"
    for keys, part in merged.groupby(group_cols, dropna=False):
        part = part.sort_values("outer_fold")
        changes = part.mae_change_pct_vs_inner_tree.to_numpy(float)
        conditions = {
            "mean_mae_not_worse_than_1pct": bool(np.mean(changes) <= .01),
            "at_least_one_fold_improves": bool(np.any(changes < 0)),
            "other_fold_not_worse_than_3pct": bool(np.max(changes) <= .03),
            "spearman_not_clearly_worse": bool(np.nanmean(part.spearman_change_vs_inner_tree) >= -.03),
            "std_ratio_not_clearly_worse": bool(np.nanmean(part.std_distance_change_vs_inner_tree) <= .05),
            "no_extreme_prediction_explosion": bool(np.nanmax(part.prediction_range_ratio) <= 3.0),
        }
        status = all(conditions.values())
        first = part.iloc[0].to_dict()
        per_pipeline.append({
            "target": keys[0], "feature_family": keys[1], "b11_base": None if keys[2] == "__NONE__" else keys[2],
            "model": keys[3], "fold_selected_params_json": json.dumps({str(int(row.outer_fold)): row.params_json for _, row in part.iterrows()}, sort_keys=True),
            "development_folds": ",".join(map(str, part.outer_fold)),
            "mean_inner_oof_mae": float(part.mae.mean()), "mean_inner_oof_r2": float(part.r2.mean()),
            "mean_inner_oof_spearman": float(part.spearman.mean()), "mean_inner_oof_std_ratio": float(part.std_ratio.mean()),
            "mean_mae_change_pct_vs_inner_tree": float(np.mean(changes)),
            "fold_0_mae_change_pct_vs_inner_tree": float(part.loc[part.outer_fold.eq(0), "mae_change_pct_vs_inner_tree"].iloc[0]),
            "fold_4_mae_change_pct_vs_inner_tree": float(part.loc[part.outer_fold.eq(4), "mae_change_pct_vs_inner_tree"].iloc[0]),
            "stage1_gate_pass": status, **conditions,
            "feature_dim_raw": int(first["feature_dim_raw"]), "feature_blocks_json": first["feature_blocks_json"],
            "embedding_locked": first.get("embedding_locked"),
        })
    candidates = pd.DataFrame(per_pipeline)
    return merged, candidates


def choose_max_two(candidates: pd.DataFrame) -> pd.DataFrame:
    passed = candidates.loc[candidates.stage1_gate_pass].copy()
    if passed.empty:
        return passed
    passed["complexity"] = passed.model.map(MODEL_COMPLEXITY)
    passed = passed.sort_values(["target", "mean_inner_oof_mae", "mean_inner_oof_r2", "feature_dim_raw", "complexity"],
                                ascending=[True, True, False, True, True])
    return passed.groupby("target", as_index=False, group_keys=False).head(2).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    output = STAGE / "selected_candidates.csv"
    if output.exists() and not arguments.force:
        print("ALREADY_COMPLETE", output)
        return
    metrics, predictions = load_required_shards()
    selected = choose_params(metrics)
    details, candidate_table = gate_candidates(selected)
    baseline = current_development_baseline()
    candidate_table = candidate_table.merge(
        baseline.groupby("target", as_index=False).agg(existing_tree_baseline_validation_mae=("existing_tree_baseline_validation_mae", "mean")),
        on="target", how="left")
    locked = choose_max_two(candidate_table)
    metrics.to_csv(STAGE / "fold_metrics.csv", index=False)
    predictions.to_csv(STAGE / "predictions.csv", index=False)
    selected.to_csv(STAGE / "selected_params_by_fold.csv", index=False)
    details.to_csv(STAGE / "candidate_fold_comparisons.csv", index=False)
    candidate_table.to_csv(STAGE / "all_candidate_gate_results.csv", index=False)
    locked.to_csv(output, index=False)
    report = ["# Stage 1 hybrid development selection", "",
              "- Inputs are only inner GroupKFold OOF metrics from outer folds 0 and 4.",
              "- No outer-test embedding, label, prediction, or metric was read.",
              "- The comparator for gate decisions is the best tree-only pre-registered A1–A4 pipeline within the same outer-train inner CV.",
              "- M3 is user-deferred: its independent run is neither read nor included in this selection, so no partial-M3 comparison is possible.",
              "- `existing_tree_baseline_validation_mae` is retained as a separately reported historical development reference; it is not mixed with the GroupKFold OOF gate statistic.",
              "", f"- Stage-1 passing pipelines: {len(locked)}."]
    report_text = "\n".join(report) + "\n"
    (STAGE / "report.md").write_text(report_text, encoding="utf-8")
    (STAGE / "stage1_report.md").write_text(report_text, encoding="utf-8")
    append_execution(OUTPUT, stage="stage1_development_candidate_selection", target="all", outer_fold="fold_0,fold_4",
                     feature_family="A0-A7,B1-B11", embedding_name="descriptor_branch_raw,fused_embedding,graph_branch_raw",
                     model="M0,M1,M2,M4,M5,M6", status="completed_m3_deferred", output_path=str(STAGE / "selected_candidates.csv"))
    print("STAGE1_SELECTION_COMPLETE", len(locked), output)


if __name__ == "__main__":
    main()
