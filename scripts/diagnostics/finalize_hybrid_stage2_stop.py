#!/usr/bin/env python3
"""Write the mandatory terminal artifacts after a Stage-2 development failure.

This is intentionally separate from the Stage-1 early-stop writer: Stage-1
and Stage-2 development evidence remains intact, while every outer-test and
interpretability artifact is explicitly marked unopened/not run.
"""

from __future__ import annotations

import json

import pandas as pd

from prepare_hybrid_embedding_tree_experiment import ROOT, append_execution


OUTPUT = ROOT / "results/hybrid_embedding_tree_exp"


def empty_csv(path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def main() -> None:
    lock_path = OUTPUT / "stage2/locked_pipeline.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("status") != "HYBRID_NOT_STABLE_ON_DEVELOPMENT_FOLDS":
        raise RuntimeError(f"Stage-2 stop writer requires HYBRID_NOT_STABLE_ON_DEVELOPMENT_FOLDS, got {payload.get('status')}")

    confirmation = OUTPUT / "confirmation"
    interpretation = OUTPUT / "interpretability"
    residual = OUTPUT / "residual"
    for filename, columns in {
        "untouched_fold_metrics.csv": ["status", "reason"],
        "pooled_oof_metrics.csv": ["status", "reason"],
        "pooled_oof_predictions.csv": ["status", "reason"],
        "paired_bootstrap.csv": ["status", "reason"],
    }.items():
        empty_csv(confirmation / filename, columns)
    (confirmation / "confirmation_report.md").write_text(
        "# Confirmation not run\n\nThe candidate failed the fold-1 development gate; folds 2/3 and all outer-test labels remained sealed.\n",
        encoding="utf-8",
    )
    for filename, columns in {
        "grouped_permutation_importance.csv": ["status", "reason"],
        "block_ablation_metrics.csv": ["status", "reason"],
        "sample_prediction_deltas.csv": ["status", "reason"],
        "subgroup_embedding_gain.csv": ["status", "reason"],
    }.items():
        empty_csv(interpretation / filename, columns)
    screen = pd.read_csv(OUTPUT / "redundancy/residual_signal_screening.csv")
    screen.to_csv(residual / "residual_probe_metrics.csv", index=False)
    empty_csv(residual / "alpha_selection.csv", ["status", "reason"])
    empty_csv(residual / "residual_predictions.csv", ["status", "reason"])
    (residual / "report.md").write_text(
        "# Residual route\n\nNot run: the direct-hybrid development gate did not lock a pipeline, and the prerequisite residual screen did not clear its stable-signal gate.\n",
        encoding="utf-8",
    )

    report = """# Frozen GraphGPS embedding + tree feature experiment

Final status: `HYBRID_NOT_STABLE` (`HYBRID_NOT_STABLE_ON_DEVELOPMENT_FOLDS`).

M3 was explicitly deferred while its independent long-running queue continued. It was not read or included in this decision; the development gate used M0/M1/M2/M4/M5/M6 only.

1. Descriptor and fused hybrid candidates were evaluated on Stage-1 development folds, but no candidate was stable enough across all three development folds to enter outer confirmation.
2. No claim of incremental encoder information is confirmed under this decision path.
3. The redundancy audit remains the evidence for the relationship between embeddings and F1–F4.
4. Recovery has no confirmed fused-embedding gain because its candidate did not clear the fold-1 gate.
5. No outer-test MAE was opened for any hybrid after the Stage-2 stop.
6. No target has a confirmed gain.
7. Recovery did not exceed the nested tree baseline in a confirmed evaluation.
8. Confirmed R² and Spearman gains are unavailable because outer tests stayed sealed.
9. Confirmed std-ratio gains are unavailable for the same reason.
10. A five-fold win count is intentionally unavailable.
11. Folds 2 and 3 were not opened.
12. High-dimensional representations did not establish sufficiently stable development evidence.
13. The residual route was not run because its predeclared signal gate did not clear.
14. No final hybrid importance block is available without a locked pipeline.
15. Subgroup gains were not estimated without outer predictions.
16. The frozen encoder should not enter the production tree pipeline on this evidence.
17. The historical GraphGPS head remains the principal neural-model research need.
18. Feedback was not read and must not be used to override this gate.

| target | selected_features | model | pooled_mae | tree_mae | mae_delta | pooled_r2 | tree_r2 | pooled_spearman | tree_spearman | std_ratio | folds_won | untouched_confirmed | decision |
| ------ | ----------------- | ----- | ---------: | -------: | --------: | --------: | ------: | --------------: | ------------: | --------: | --------: | ------------------- | -------- |
| all | no Stage-2 lock | none | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | not run | HYBRID_NOT_STABLE |
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    append_execution(
        OUTPUT,
        stage="stage2_stop_finalization",
        target="all",
        outer_fold="none",
        feature_family="none",
        embedding_name="none",
        model="M0,M1,M2,M4,M5,M6",
        status="HYBRID_NOT_STABLE_ON_DEVELOPMENT_FOLDS_M3_DEFERRED",
        output_path=str(OUTPUT / "report.md"),
    )
    print("1. descriptor embedding是否具有增量: 未确认（Stage 2未通过）")
    print("2. fused embedding是否具有增量: 未确认（Stage 2未通过）")
    print("3. 最佳target: 无")
    print("4. 最佳特征组合: 无")
    print("5. 最佳模型: 无")
    print("6. pooled MAE/R²/Spearman/std_ratio: 未运行")
    print("7. 与nested tree baseline差异: 未进入outer确认")
    print("8. 改善fold数: 未运行")
    print("9. untouched folds是否确认: 否，未打开")
    print("10. residual路线是否运行: 否")
    print("11. 是否生成feedback_ready.json: 否")
    print("12. 最终状态: HYBRID_NOT_STABLE")
    print(f"13. report.md路径: {OUTPUT / 'report.md'}")
    print("14. 未完成项及原因: Stage 2开发门控失败；M3按用户要求独立延后且未读取")


if __name__ == "__main__":
    main()
