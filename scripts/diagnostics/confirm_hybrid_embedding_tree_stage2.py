#!/usr/bin/env python3
"""Apply the predeclared fold-1 development gate to Stage-1 hybrid candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_hybrid_embedding_tree_experiment import ROOT, TARGETS, append_execution
from select_hybrid_embedding_tree_candidates import choose_params


OUTPUT = ROOT / "results/hybrid_embedding_tree_exp"
STAGE1 = OUTPUT / "stage1"
STAGE2 = OUTPUT / "stage2"
TREE_ONLY = {"A1", "A2", "A3", "A4"}
COMPLEXITY = {"M0": 0, "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 6}


def key_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["b11_base_key"] = result.b11_base.fillna("__NONE__")
    return result


def stage2_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = STAGE2 / "shards"
    files = sorted(root.glob("*/fold_1_metrics.csv"))
    if not files:
        raise RuntimeError("STAGE2_INCOMPLETE: no fold-1 shard metrics")
    values = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    candidate = values.loc[values.feature_family.isin(TREE_ONLY) | values.feature_family.isin(set(pd.read_csv(STAGE1 / "selected_candidates.csv").feature_family))]
    return values, candidate


def main() -> None:
    STAGE2.mkdir(parents=True, exist_ok=True)
    stage1_selected = pd.read_csv(STAGE1 / "selected_candidates.csv")
    if stage1_selected.empty:
        raise RuntimeError("NO_HYBRID_CANDIDATE_AFTER_STAGE1")
    stage2_metrics, _ = stage2_files()
    stage2_selected = choose_params(stage2_metrics)
    tree = stage2_selected.loc[stage2_selected.feature_family.isin(TREE_ONLY)].copy()
    if tree.empty:
        raise RuntimeError("STAGE2_INCOMPLETE: tree-only inner-CV reference is missing")
    tree = tree.sort_values(["target", "mae", "r2"], ascending=[True, True, False]).groupby("target", as_index=False).first()
    tree = tree.rename(columns={column: f"tree_{column}" for column in tree.columns if column != "target"})
    candidates = key_columns(stage1_selected)
    selected2 = key_columns(stage2_selected)
    target_pairs = candidates[["target", "feature_family", "b11_base_key", "model"]].drop_duplicates()
    fold1 = target_pairs.merge(selected2, on=["target", "feature_family", "b11_base_key", "model"], how="left", validate="one_to_one")
    if fold1.mae.isna().any():
        raise RuntimeError("STAGE2_INCOMPLETE: at least one Stage-1 candidate was not evaluated on fold 1")
    fold1 = fold1.merge(tree, on="target", how="left", validate="many_to_one")
    fold1["mae_change_pct_vs_inner_tree"] = (fold1.mae - fold1.tree_mae) / fold1.tree_mae
    fold1["r2_change_vs_inner_tree"] = fold1.r2 - fold1.tree_r2
    fold1["spearman_change_vs_inner_tree"] = fold1.spearman - fold1.tree_spearman
    fold1["std_distance_change_vs_inner_tree"] = (fold1.std_ratio - 1).abs() - (fold1.tree_std_ratio - 1).abs()

    stage1_detail = key_columns(pd.read_csv(STAGE1 / "candidate_fold_comparisons.csv"))
    stage1_detail = stage1_detail.merge(target_pairs, on=["target", "feature_family", "b11_base_key", "model"], how="inner")
    stage1_detail = stage1_detail[["outer_fold", "target", "feature_family", "b11_base_key", "model", "mae", "r2", "spearman", "std_ratio",
                                  "tree_mae", "tree_r2", "tree_spearman", "tree_std_ratio", "mae_change_pct_vs_inner_tree"]]
    fold1_detail = fold1.rename(columns={"tree_mae": "tree_mae", "tree_r2": "tree_r2", "tree_spearman": "tree_spearman", "tree_std_ratio": "tree_std_ratio"})[
        ["target", "feature_family", "b11_base_key", "model", "mae", "r2", "spearman", "std_ratio", "tree_mae", "tree_r2", "tree_spearman", "tree_std_ratio", "mae_change_pct_vs_inner_tree"]].copy()
    fold1_detail["outer_fold"] = 1
    development = pd.concat([stage1_detail, fold1_detail], ignore_index=True).sort_values(["target", "feature_family", "model", "outer_fold"])
    rows = []
    for keys, part in development.groupby(["target", "feature_family", "b11_base_key", "model"], dropna=False):
        part = part.sort_values("outer_fold")
        change = part.mae_change_pct_vs_inner_tree.to_numpy(float)
        pass_rules = {
            "at_least_two_of_three_mae_improve": bool((change < 0).sum() >= 2),
            "mean_mae_improves": bool(part.mae.mean() < part.tree_mae.mean()),
            "mean_r2_not_down": bool(part.r2.mean() >= part.tree_r2.mean()),
            "mean_spearman_not_down": bool(part.spearman.mean() >= part.tree_spearman.mean()),
            "mean_std_ratio_not_worse": bool((part.std_ratio.sub(1).abs().mean()) <= (part.tree_std_ratio.sub(1).abs().mean()) + .02),
            "fold1_not_catastrophic": bool(float(part.loc[part.outer_fold.eq(1), "mae_change_pct_vs_inner_tree"].iloc[0]) <= .03),
        }
        rows.append({"target": keys[0], "feature_family": keys[1], "b11_base": None if keys[2] == "__NONE__" else keys[2], "model": keys[3],
                     "mean_development_mae": float(part.mae.mean()), "mean_tree_mae": float(part.tree_mae.mean()),
                     "mean_development_r2": float(part.r2.mean()), "mean_tree_r2": float(part.tree_r2.mean()),
                     "mean_development_spearman": float(part.spearman.mean()), "mean_tree_spearman": float(part.tree_spearman.mean()),
                     "mean_development_std_ratio": float(part.std_ratio.mean()), "mean_tree_std_ratio": float(part.tree_std_ratio.mean()),
                     "fold1_mae_change_pct_vs_tree": float(part.loc[part.outer_fold.eq(1), "mae_change_pct_vs_inner_tree"].iloc[0]),
                     "development_improved_fold_count": int((change < 0).sum()), "stage2_gate_pass": bool(all(pass_rules.values())), **pass_rules})
    gate = pd.DataFrame(rows)
    passing = gate.loc[gate.stage2_gate_pass].copy()
    passing["complexity"] = passing.model.map(COMPLEXITY)
    locked = passing.sort_values(["target", "mean_development_mae", "mean_development_r2", "complexity"], ascending=[True, True, False, True]).groupby("target", as_index=False).first()
    status = "READY_FOR_UNTOUCHED_CONFIRMATION" if not locked.empty else "HYBRID_NOT_STABLE_ON_DEVELOPMENT_FOLDS"
    stage2_metrics.to_csv(STAGE2 / "development_fold_metrics.csv", index=False)
    fold1.to_csv(STAGE2 / "fold1_candidate_metrics.csv", index=False)
    development.to_csv(STAGE2 / "three_fold_development_comparison.csv", index=False)
    gate.to_csv(STAGE2 / "stage2_gate_results.csv", index=False)
    locked.to_csv(STAGE2 / "locked_pipeline.csv", index=False)
    (STAGE2 / "locked_pipeline.json").write_text(json.dumps({"status": status, "pipelines": locked.to_dict(orient="records"),
                                                               "selection_rule": "Stage-1 and fold-1 development gates only; outer tests unopened"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_text = "\n".join(["# Stage 2 confirmation", "", f"Status: `{status}`.",
        "All comparisons use development-fold inner GroupKFold OOF metrics; no outer test was read.",
        f"Locked target pipelines: {len(locked)}."]) + "\n"
    (STAGE2 / "report.md").write_text(report_text, encoding="utf-8")
    (STAGE2 / "stage2_report.md").write_text(report_text, encoding="utf-8")
    append_execution(OUTPUT, stage="stage2_development_confirmation", target="all", outer_fold="fold_1", feature_family="locked_stage1_candidates",
                     embedding_name="locked_by_target", model="locked_stage1_models", status=status, output_path=str(STAGE2 / "locked_pipeline.csv"))
    print("STAGE2_CONFIRMATION_COMPLETE", status, len(locked))


if __name__ == "__main__":
    main()
