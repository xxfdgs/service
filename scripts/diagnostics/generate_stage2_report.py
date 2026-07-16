#!/usr/bin/env python3
"""Render the stage-two decision report from verified diagnostic artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_common import add_stage2_arguments, record_execution, stage2_output  # noqa: E402


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def table(frame: pd.DataFrame, columns: list[str], decimals: int = 3) -> str:
    if frame.empty:
        return "_No completed records._"
    selected = frame[[column for column in columns if column in frame.columns]].copy()
    for column in selected.select_dtypes(include="number"):
        selected[column] = selected[column].map(lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}")
    lines = ["| " + " | ".join(selected.columns) + " |",
             "| " + " | ".join("---" for _ in selected.columns) + " |"]
    for values in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value).replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_stage2_arguments(parser)
    arguments = parser.parse_args()
    output_dir = stage2_output(arguments.output_dir)
    audit_dir = output_dir / "data_audit"
    reproduce_dir = output_dir / "reproducibility"
    label_dir = output_dir / "label_shift"
    group_dir = output_dir / "group_cv"
    stable_dir = output_dir / "stable_features"
    ad_dir = output_dir / "applicability_domain"
    replicate_dir = output_dir / "replicate_experiments"
    classifications = read_csv(audit_dir / "duplicate_group_classification.csv")
    repeat_stats = read_csv(audit_dir / "replicate_statistics.csv")
    shifts = read_csv(label_dir / "train_feedback_label_shift.csv")
    residuals = read_csv(label_dir / "residual_analysis.csv")
    reproducibility = read_csv(reproduce_dir / "reproducibility_metrics.csv")
    comparisons = read_csv(reproduce_dir / "legacy_explicit_prediction_comparison.csv")
    repeat_comparisons = read_csv(reproduce_dir / "legacy_repeat_comparison.csv")
    summary = read_csv(group_dir / "summary_metrics.csv")
    paired = read_csv(group_dir / "paired_model_comparison.csv")
    stable_metrics = read_csv(stable_dir / "metrics_by_feature_set.csv")
    ad_correlations = read_csv(ad_dir / "ad_error_correlations.csv")
    replicate_metrics = read_csv(replicate_dir / "replicate_treatment_metrics.csv")
    optional = []
    for name, path in {
        "reproducibility": reproduce_dir / "reproducibility_metrics.csv",
        "label_shift": label_dir / "residual_analysis.csv",
        "group_cv": group_dir / "fold_metrics.csv",
        "stable_features": stable_dir / "metrics_by_feature_set.csv",
        "applicability_domain": ad_dir / "ad_error_correlations.csv",
        "replicate_experiments": replicate_dir / "replicate_treatment_metrics.csv",
    }.items():
        optional.append({"artifact": name, "completed": path.is_file(), "path": str(path)})
    class_counts = classifications["group_class"].value_counts().to_dict() if not classifications.empty else {}
    class_share = ({key: value / max(1, sum(class_counts.values())) for key, value in class_counts.items()}
                   if class_counts else {})
    repeat_summary = pd.DataFrame()
    if not repeat_stats.empty:
        repeat_summary = (
            repeat_stats.groupby(["group_class", "target"], as_index=False)
            .agg(
                duplicate_groups=("group_size", "size"),
                median_group_size=("group_size", "median"),
                median_std=("std", "median"),
                median_mad=("mad", "median"),
                median_range=("range", "median"),
                max_range_overall_std=("range_overall_std", "max"),
            )
        )
    replicate_summary = pd.DataFrame()
    if not replicate_metrics.empty:
        replicate_summary = (
            replicate_metrics.groupby(["version", "protocol", "target", "model", "used_sample_weight"], as_index=False)
            .agg(
                completed_folds=("mae", "size"),
                mean_mae=("mae", "mean"),
                mean_r2=("r2", "mean"),
            )
        )
    best_stable = pd.DataFrame()
    if not stable_metrics.empty:
        best_stable = stable_metrics.groupby(["protocol", "feature_set", "target", "model"], as_index=False)["mae"].mean()
        best_stable = best_stable.sort_values(["protocol", "target", "mae"]).groupby(["protocol", "target"], as_index=False).first()
    primary_ad = pd.DataFrame()
    if not ad_correlations.empty:
        primary_ad = ad_correlations.groupby("score", as_index=False)["spearman_r"].median().sort_values("spearman_r", ascending=False)
    graph_summary = summary.loc[summary.get("model") == "GraphGPS_coarse_mordred_ensemble"] if not summary.empty else pd.DataFrame()
    graph_complete = not graph_summary.empty and (graph_summary["completed_folds"] >= 5).all()
    graph_beats_tree = (paired["graphgps_win_fraction"] > 0.5).all() if not paired.empty else False
    decision = "A"
    decision_text = "GraphGPS 尚未在两种完整 group CV 中稳定优于 ExtraTrees；暂停扩大 GraphGPS，优先数据清理和低维配方基线。"
    if graph_complete and graph_beats_tree:
        decision = "B"
        decision_text = "GraphGPS 仅应在明确获胜的目标进入 task-specific head / 目标独立模型试验。"
    if not replicate_metrics.empty:
        raw_mae = replicate_metrics.loc[replicate_metrics["version"] == "raw_records", "mae"].mean()
        median_mae = replicate_metrics.loc[replicate_metrics["version"] == "replicate_median", "mae"].mean()
        if np.isfinite(raw_mae) and np.isfinite(median_mae) and median_mae < raw_mae:
            decision_text += " 重复测量处理有改善信号，后续训练应统一采用经审计版本。"
    report = [
        "# 外部泛化诊断第二阶段", "",
        "## 完成状态", table(pd.DataFrame(optional), ["artifact", "completed", "path"]), "",
        "## 复现协议", table(reproducibility, ["protocol", "evaluation_set", "target", "best_epoch", "mae", "rmse", "r2"]), "",
        "### Legacy 与显式 Manifest", table(comparisons, ["evaluation_set", "target", "same_sample_ids", "mae_difference", "max_single_prediction_difference", "threshold_exceeded"]), "",
        "### Legacy 重复运行", table(repeat_comparisons, ["comparison", "evaluation_set", "target", "mae_difference", "max_single_prediction_difference"]), "",
        "- 应使用显式 manifest 的 81/9/10 协议以复现现有加载器；80/10/10 仅为未实际执行的 YAML 标称值。",
        "- legacy/manifest 差异须以 `reproducibility_diagnosis.md` 的阈值审计为准，避免把不同 split 误认为模型退化。", "",
        "## 重复配方审计", f"- 分组分类计数：{class_counts}；占比：{class_share}。", table(repeat_summary, ["group_class", "target", "duplicate_groups", "median_group_size", "median_std", "median_mad", "median_range", "max_range_overall_std"]), "",
        "- 数据版本严格保留原记录；聚合仅应用于 true_replicate，hidden_condition_difference 不聚合。", "",
        "## 标签偏移与预测偏差", table(shifts, ["target", "mean_difference_feedback_minus_train", "standardized_mean_difference", "wasserstein_distance", "ks_statistic", "feedback_outside_train_q01_q99_fraction"]), "",
        table(residuals, ["model", "target", "mean_residual", "true_std_over_prediction_std", "spearman", "regression_slope", "dominant_error_factors"]), "",
        "- feedback 的负 R²应从残差均值、方差压缩、Spearman 与极端样本标记共同判断，不使用 feedback 拟合校准器。", "",
        "## 重复 Group CV", table(summary, ["protocol", "target", "model", "completed_folds", "mean_mae", "std_mae", "mae_ci95_low", "mae_ci95_high", "mean_r2"]), "",
        table(paired, ["protocol", "target", "completed_paired_folds", "mean_graphgps_minus_extratrees_mae", "graphgps_win_fraction"]), "",
        "## 稳定低维特征", table(best_stable, ["protocol", "target", "feature_set", "model", "mae"]), "",
        "## Applicability Domain", table(primary_ad, ["score", "spearman_r"]), "",
        "## 重复数据版本对照", table(replicate_summary, ["version", "protocol", "target", "model", "completed_folds", "mean_mae", "mean_r2", "used_sample_weight"]), "",
        "## 明确结论", 
        f"1. 显式 manifest 是后续唯一可审计的训练协议；当前执行比例应按 81/9/10 记录。",
        f"2. 完全相同配方的差异分类及占比如上；`true_replicate` 才允许中位数聚合，其他记录保留。",
        "3. feedback 负 R²的主导因素由 `residual_analysis.csv` 的偏置、压缩与排序字段给出；不应通过 feedback 校准修复。",
        f"4. GraphGPS 完整五折状态：{graph_complete}；因此其是否稳定优于 ExtraTrees 的结论目前为 {'已验证' if graph_complete else '尚未完成完整五折验证'}。",
        f"5. 最稳定低维方案见上表；最可学习目标应以两种 Group CV 的均值/CI 共同判断。",
        "6. 适用域只采用内部 OOF 误差正相关的指标；domain classifier 概率仅作为辅助列。",
        f"7. **Decision gate {decision}: {decision_text}**", "",
        "## 下一步优先级", 
        "1. 完成缺失的 GraphGPS outer folds 后再作稳定优越性结论。",
        "2. 复核 hidden_condition_difference 与 suspected_record_errors 的实验元数据。",
        "3. 以稳定低维特征和经过审计的数据版本作为下一轮比较基线。",
        "4. 只有支持集覆盖改善后，再考虑网络结构改动或域方法。",
    ]
    (output_dir / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    record_execution(output_dir, Path(__file__).name, details={"seed": arguments.seed, "n_jobs": arguments.n_jobs})
    print(f"Wrote {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
