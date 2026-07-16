#!/usr/bin/env python3
"""Materialize required reports when the Stage-1 gate finds no hybrid candidate."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from prepare_hybrid_embedding_tree_experiment import ROOT, append_execution


OUTPUT = ROOT / "results/hybrid_embedding_tree_exp"


def empty_csv(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def main() -> None:
    candidates = pd.read_csv(OUTPUT / "stage1/selected_candidates.csv")
    if not candidates.empty:
        raise RuntimeError("Early-stop finalizer is only valid when Stage 1 selected zero candidates")
    residual_screen = pd.read_csv(OUTPUT / "redundancy/residual_signal_screening.csv")
    stage2 = OUTPUT / "stage2"
    confirmation = OUTPUT / "confirmation"
    interpretation = OUTPUT / "interpretability"
    residual = OUTPUT / "residual"
    stage2.mkdir(exist_ok=True)
    confirmation.mkdir(exist_ok=True)
    interpretation.mkdir(exist_ok=True)
    for filename, columns in {
        "development_fold_metrics.csv": ["status", "reason"], "fold1_candidate_metrics.csv": ["status", "reason"],
        "three_fold_development_comparison.csv": ["status", "reason"], "stage2_gate_results.csv": ["status", "reason"],
        "locked_pipeline.csv": ["target", "feature_family", "model", "status"],
    }.items():
        empty_csv(stage2 / filename, columns)
    lock_payload = {"status": "NO_HYBRID_CANDIDATE_AFTER_STAGE1", "pipelines": [],
                    "reason": "No embedding-containing candidate passed the two development-fold inner GroupKFold gate; outer test was never opened."}
    (stage2 / "locked_pipeline.json").write_text(json.dumps(lock_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stage2_report = "# Stage 2 not run\n\nStatus: `NO_HYBRID_CANDIDATE_AFTER_STAGE1`. Fold 1 and all outer-test folds were intentionally not opened.\n"
    (stage2 / "stage2_report.md").write_text(stage2_report, encoding="utf-8")
    (stage2 / "report.md").write_text(stage2_report, encoding="utf-8")
    for filename, columns in {
        "untouched_fold_metrics.csv": ["status", "reason"], "pooled_oof_metrics.csv": ["status", "reason"],
        "pooled_oof_predictions.csv": ["status", "reason"], "paired_bootstrap.csv": ["status", "reason"],
    }.items():
        empty_csv(confirmation / filename, columns)
    (confirmation / "confirmation_report.md").write_text("# Confirmation not run\n\nThe Stage-1 gate stopped the protocol before fold 1/2/3 test evaluation.\n", encoding="utf-8")
    for filename, columns in {
        "grouped_permutation_importance.csv": ["status", "reason"], "block_ablation_metrics.csv": ["status", "reason"],
        "sample_prediction_deltas.csv": ["status", "reason"], "subgroup_embedding_gain.csv": ["status", "reason"],
    }.items():
        empty_csv(interpretation / filename, columns)
    residual_screen.to_csv(residual / "residual_probe_metrics.csv", index=False)
    empty_csv(residual / "alpha_selection.csv", ["status", "reason"])
    empty_csv(residual / "residual_predictions.csv", ["status", "reason"])
    (residual / "report.md").write_text("# Residual route\n\nNot run: direct hybrid candidates did not pass Stage 1, and residual screening did not establish the required stable signal gate.\n", encoding="utf-8")
    report = """# Frozen GraphGPS embedding + tree feature experiment

Final status: `TREE_ONLY_REMAINS_BEST` with the development-gate detail `NO_HYBRID_CANDIDATE_AFTER_STAGE1`.

M3 was explicitly deferred while its independent long-running queue continues.  It was not read, merged, or used for any decision in this report; this report covers M0/M1/M2/M4/M5/M6 only.

1. `descriptor_branch_raw` did not become a candidate above the direct F0 tree input.
2. The descriptor encoder did not demonstrate stable incremental information beyond tree features in development GroupKFold.
3. Its redundancy is substantial: it is the audited 55-D raw descriptor branch and F2–F4 reconstruct large portions of it.
4. `fused_embedding` did not establish a stable Recovery-only increment at this gate.
5. No embedding-containing F1–F4 hybrid passed the predeclared MAE/stability rules.
6. No target obtained a confirmed hybrid gain.
7. Recovery did not exceed the current tree baseline in this protocol because no candidate reached confirmation.
8. No confirmed R² or Spearman improvement was established.
9. No confirmed std-ratio improvement was established.
10. A five-fold win count is intentionally unavailable: outer tests stayed sealed.
11. Folds 2/3 were not opened after the Stage-1 stop.
12. High-dimensional embeddings showed no development evidence sufficient to justify outer confirmation.
13. The residual route was not run; its screen did not clear the required stable-signal gate.
14. No final hybrid block can have stable nonzero importance without a locked candidate.
15. Subgroup gains were not estimated because no outer prediction was made.
16. The frozen encoder should not enter the production tree pipeline on this evidence.
17. The historical GraphGPS head remains the principal neural-model research need.
18. Feedback was not read and must not be re-evaluated from this failed gate.

| target | selected_features | model | pooled_mae | tree_mae | mae_delta | pooled_r2 | tree_r2 | pooled_spearman | tree_spearman | std_ratio | folds_won | untouched_confirmed | decision |
| ------ | ----------------- | ----- | ---------: | -------: | --------: | --------: | ------: | --------------: | ------------: | --------: | --------: | ------------------- | -------- |
| all | none | none | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | not run | TREE_ONLY_REMAINS_BEST |
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    append_execution(OUTPUT, stage="early_stop_finalization", target="all", outer_fold="none", feature_family="none", embedding_name="none",
                     model="M0,M1,M2,M4,M5,M6", status="NO_HYBRID_CANDIDATE_AFTER_STAGE1_M3_DEFERRED", output_path=str(OUTPUT / "report.md"))
    print("1. descriptor embedding是否具有增量: 否（Stage 1未通过）")
    print("2. fused embedding是否具有增量: 否（Stage 1未通过）")
    print("3. 最佳target: 无")
    print("4. 最佳特征组合: 无")
    print("5. 最佳模型: 无")
    print("6. pooled MAE/R²/Spearman/std_ratio: 未运行")
    print("7. 与nested tree baseline差异: 未进入outer确认")
    print("8. 改善fold数: 未运行")
    print("9. untouched folds是否确认: 否，未打开")
    print("10. residual路线是否运行: 否")
    print("11. 是否生成feedback_ready.json: 否")
    print("12. 最终状态: TREE_ONLY_REMAINS_BEST / NO_HYBRID_CANDIDATE_AFTER_STAGE1")
    print(f"13. report.md路径: {OUTPUT / 'report.md'}")
    print("14. 未完成项及原因: Stage 1预注册门控失败，按协议提前停止")


if __name__ == "__main__":
    main()
