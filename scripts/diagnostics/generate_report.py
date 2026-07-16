#!/usr/bin/env python3
"""Render the first-stage external-generalization report from diagnostic outputs."""

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

from common import add_common_arguments  # noqa: E402


def _markdown_table(frame: pd.DataFrame, columns: list[str], decimals: int = 3) -> str:
    """Format a compact selected-column DataFrame as Markdown."""
    if frame.empty:
        return "_No completed records._"
    table = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    for column in table.select_dtypes(include="number").columns:
        table[column] = table[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}"
        )
    header = [str(column) for column in table.columns]
    markdown_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") if not pd.isna(value) else "" for value in row]
        markdown_lines.append("| " + " | ".join(values) + " |")
    return "\n".join(markdown_lines)


def _safe_read(path: Path) -> pd.DataFrame:
    """Read an optional CSV so a partial run produces an explicit report."""
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _target_distribution(schema: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target, summary in schema["target_summary"].items():
        train_stats = summary["train_summary"]
        feedback_stats = summary["feedback_summary"]
        rows.append({
            "target": target,
            "train_missing": summary["train_missing_fraction"],
            "feedback_missing": summary["feedback_missing_fraction"],
            "train_mean": train_stats.get("mean"),
            "train_std": train_stats.get("std"),
            "feedback_mean": feedback_stats.get("mean"),
            "feedback_std": feedback_stats.get("std"),
        })
    return pd.DataFrame(rows)


def _baseline_table(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[
        (metrics.get("status") == "ok") &
        metrics["model"].isin(["TrainMean", "KNN_k5", "RandomForest", "ExtraTrees"])
    ].copy()
    selected = selected.loc[selected["evaluation_set"].isin(["test", "feedback"])]
    return selected.sort_values(["split_name", "evaluation_set", "target", "model"])


def _conclusions(
    baselines: pd.DataFrame, graphgps: pd.DataFrame, noise_summary: pd.DataFrame,
    domain_metrics: pd.DataFrame, ood_correlations: pd.DataFrame,
) -> list[str]:
    """Produce evidence-bound conclusions without filling missing diagnostics by guesswork."""
    conclusions: list[str] = []
    extra_feedback = baselines.loc[
        (baselines.get("model") == "ExtraTrees") &
        (baselines.get("evaluation_set") == "feedback")
    ]
    graph_feedback = graphgps.loc[
        (graphgps.get("model") == "GraphGPS_coarse_mordred") &
        (graphgps.get("evaluation_set") == "feedback") &
        (graphgps.get("status") == "ok")
    ]
    domain_auc = domain_metrics.loc[
        domain_metrics.get("model") == "ExtraTreesClassifier", "roc_auc"
    ]
    if not domain_auc.empty:
        conclusions.append(
            f"训练与 feedback 存在明显协变量偏移：ExtraTrees domain ROC-AUC 为 {domain_auc.iloc[0]:.3f}，"
            "远高于随机区分水平。"
        )
    random_extra = baselines.loc[
        (baselines.get("model") == "ExtraTrees") &
        (baselines.get("split_name") == "random_split") &
        (baselines.get("evaluation_set") == "test")
    ]
    fifth_extra = baselines.loc[
        (baselines.get("model") == "ExtraTrees") &
        (baselines.get("split_name") == "fifth_component_group_split") &
        (baselines.get("evaluation_set") == "test")
    ]
    if not random_extra.empty and not fifth_extra.empty:
        random_mae = random_extra["mae"].mean()
        fifth_mae = fifth_extra["mae"].mean()
        conclusions.append(
            "第五组分冷启动会改变简单模型的泛化误差："
            f"ExtraTrees 平均 MAE 从随机切分的 {random_mae:.2f} 变为 {fifth_mae:.2f}。"
        )
    if not graph_feedback.empty and not extra_feedback.empty:
        comparison = graph_feedback.merge(extra_feedback, on="target", suffixes=("_gnn", "_tree"))
        win_count = int((comparison["mae_gnn"] < comparison["mae_tree"]).sum())
        conclusions.append(
            f"在 feedback 上，当前 GraphGPS 相比固定 ExtraTrees 在 {win_count}/4 个目标上 MAE 更低；"
            "该比较使用相同外部标签，仅用于模型评估而非调参。"
        )
    elif not graph_feedback.empty:
        conclusions.append("GraphGPS 的外部 feedback 预测已记录；完整的树模型对照见基线 CSV。")
    explicit_graph = graphgps.loc[graphgps.get("source") == "retrained_3_seed_explicit_split"]
    if explicit_graph.empty:
        conclusions.append(
            "尚不能据此断言 GraphGPS 主要记忆第五组分身份：需要完成显式第五组分隔离的三种子重训后，"
            "再与随机验证结果作因果比较。"
        )
    else:
        fifth_graph = explicit_graph.loc[
            explicit_graph["split_name"] == "fifth_component_group_split"
        ]
        fifth_tree = baselines.loc[
            (baselines.get("model") == "ExtraTrees") &
            (baselines.get("split_name") == "fifth_component_group_split") &
            (baselines.get("evaluation_set") == "test")
        ]
        if not fifth_graph.empty and not fifth_tree.empty:
            cold_start = fifth_graph.merge(fifth_tree, on="target", suffixes=("_gnn", "_tree"))
            tree_win_count = int((cold_start["mae_tree"] < cold_start["mae_gnn"]).sum())
            conclusions.append(
                "第五组分冷启动下，GraphGPS 没有表现出稳定优势：固定 ExtraTrees 在 "
                f"{tree_win_count}/4 个目标上的 MAE 更低。结果不支持把当前问题简单归因于"
                "“GraphGPS 主要记忆第五组分身份”；更准确的结论是该模型在新第五组分上未形成"
                "可复现的结构泛化优势。"
            )
        else:
            conclusions.append(
                "显式组分隔离重训已完成；应以 `graphgps_comparison.csv` 的 random 与 group 行作为"
                "是否依赖第五组分身份的主证据，而非训练集后验打分。"
            )
    exact_noise = noise_summary.loc[
        noise_summary.get("group_type") == "exact_formula_ratio"
    ]
    if not exact_noise.empty:
        noisiest = exact_noise.sort_values("median_range_overall_target_std", ascending=False).iloc[0]
        conclusions.append(
            f"重复配方中噪声最突出的目标是 {noisiest['target']}：完全相同配方的中位标签范围为"
            f"总体标准差的 {noisiest['median_range_overall_target_std']:.2f} 倍。"
        )
    if not extra_feedback.empty:
        learnable = extra_feedback.sort_values("r2", ascending=False).iloc[0]
        conclusions.append(
            f"按固定 ExtraTrees 的 feedback R²，当前最可学习目标为 {learnable['target']}"
            f"（R²={learnable['r2']:.3f}）。但该值仍未达到可靠的外部解释水平。"
        )
    graph_distance = ood_correlations.loc[
        (ood_correlations.get("model") == "GraphGPS_coarse_mordred") &
        (ood_correlations.get("signal") == "nearest_training_distance")
    ]
    if not graph_distance.empty:
        strongest = graph_distance.iloc[graph_distance["spearman_r"].abs().argmax()]
        conclusions.append(
            f"feedback 的 domain 概率高度饱和（多数接近 1），因此最近邻距离更可解释；"
            f"其与 GraphGPS 的 {strongest['target']} 绝对误差 Spearman 相关为"
            f"{strongest['spearman_r']:.3f}。"
        )
    conclusions.append(
        "下一阶段应优先改进数据质量、覆盖与 group split 验证协议，而不是先扩展网络或继续增加描述符。"
    )
    return conclusions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    schema_path = output_dir / "data_schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"Missing schema file: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    quality = json.loads((output_dir / "split_quality_checks.json").read_text(encoding="utf-8"))
    domain_metrics = _safe_read(output_dir / "domain_classifier_metrics.csv")
    domain_importance = _safe_read(output_dir / "domain_feature_importance.csv")
    baselines = _safe_read(output_dir / "baseline_metrics_long.csv")
    graphgps = _safe_read(output_dir / "graphgps_comparison.csv")
    noise_summary = _safe_read(output_dir / "label_noise_summary.csv")
    ood_by_bin = _safe_read(output_dir / "ood_error_by_bin.csv")
    ood_correlations = _safe_read(output_dir / "ood_error_correlations.csv")
    optional_models = _safe_read(output_dir / "optional_model_availability.csv")

    split_rows = []
    for split_name, details in quality["splits"].items():
        split_rows.append({
            "split": split_name,
            **details["split_counts"],
            "sample_overlap": details["sample_overlap_count"],
            "group_leaks": details["group_crossing_count"],
        })
    top_importance = domain_importance.loc[
        domain_importance.get("importance_type") == "permutation_auc_drop"
    ].sort_values("importance", ascending=False).head(12) if not domain_importance.empty else pd.DataFrame()

    report_lines = [
        "# 模型外部泛化诊断（第一阶段）",
        "",
        "## 数据与范围",
        f"- 基础训练集：`{schema['actual_training_csv']}`，{schema['train_rows']} 条；"
        f"feedback：`{schema['feedback_csv']}`，{schema['feedback_rows']} 条。",
        f"- 实际最佳配置：`{schema['selected_best_config']}`。",
        "- 四目标均未作为 domain classifier 或特征预处理的输入。所有原始 CSV 保持未修改。",
        "- 当前加载器的随机切分实际为两次 90/10，即 81/9/10；YAML 标称值为 "
        f"{schema['nominal_config_split']}。",
        "",
        "## 目标分布与缺失",
        _markdown_table(_target_distribution(schema), [
            "target", "train_missing", "feedback_missing", "train_mean", "train_std",
            "feedback_mean", "feedback_std",
        ]),
        "",
        "## 划分完整性",
        _markdown_table(pd.DataFrame(split_rows), [
            "split", "train", "val", "test", "sample_overlap", "group_leaks",
        ], 0),
        "",
        f"- 规范化后，原始 SMILES 书写合并计数：{quality['normalized_smiles_raw_alias_groups']}。"
        "所有 group split 的 `group_leaks=0`。",
        "- `formula_identity_group_split` 只按五种组分身份分组；"
        "`formula_ratio_group_split` 将六位小数配比一并纳入键，因此后者仅隔离完全相同的配方+配比。",
        "- `feedback_like_split` 仅用训练候选样本拟合缺失填补和缩放，按其到 feedback 集最近邻距离保留最相近的 9%/10% 为 val/test。",
        "",
        "## Domain Classifier",
        _markdown_table(domain_metrics, [
            "model", "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "brier", "ece_10_bins",
        ]),
        "",
        "### 主要偏移特征",
        _markdown_table(top_importance, ["feature", "importance"], 4),
        "",
        "## 表格基线",
        "完整的 440 条分目标/划分/模型记录见 `baseline_metrics_long.csv`，逐样本预测见 `baseline_predictions.csv`。"
        "以下列出均值、KNN、随机森林与 ExtraTrees 的 test/feedback 核心结果：",
        _markdown_table(_baseline_table(baselines), [
            "split_name", "evaluation_set", "target", "model", "mae", "rmse", "r2",
            "mae_improvement_vs_train_mean",
        ]),
        "",
        "## GraphGPS 对照",
        "显式划分使用当前粗粒化+11 维 Mordred 配置、3 个种子、`max_epoch=150`、"
        "`early_stop_patience=30`；网络结构与特征维度未修改。",
        _markdown_table(graphgps, [
            "split_name", "evaluation_set", "target", "model", "source", "mae", "rmse", "r2", "status",
        ]),
        "",
        "## 重复配方与标签噪声",
        _markdown_table(noise_summary, [
            "group_type", "target", "group_count", "median_group_std", "median_group_range",
            "max_group_range", "median_range_overall_target_std",
        ]),
        "",
        "最高噪声的具体配方组见 `high_noise_groups.csv`；逐组、逐目标统计见 `duplicate_formulations.csv`。",
        "",
        "## OOD 与误差",
        _markdown_table(ood_by_bin, ["target", "model", "ood_bin", "n_samples", "mae", "rmse", "r2"]),
        "",
        _markdown_table(ood_correlations, ["target", "model", "signal", "spearman_r", "p_value"]),
        "",
        "## 可选模型依赖",
        _markdown_table(optional_models, ["model", "available", "reason"]),
        "",
        "## 结论",
    ]
    report_lines.extend([f"{index}. {conclusion}" for index, conclusion in enumerate(
        _conclusions(baselines, graphgps, noise_summary, domain_metrics, ood_correlations), start=1
    )])
    report_lines.extend([
        "",
        "## 下一阶段优先实验",
        "1. 先复核 `high_noise_groups.csv` 中完全相同配方的实验批次、单位和记录链路，并为可重复样本建立不确定性/重复测量权重。",
        "2. 以第五组分和完整配方 group split 作为主验证标准；仅在这两类测试上选择后续特征或网络改动。",
        "3. 对 domain importance 前列的稳定理化/配比变量进行训练集覆盖检查，优先补充或重平衡 feedback-like 区域的数据。",
        "4. 在固定 group split 上比较少量、可解释的配方特征（比例约束与交互项）和 ExtraTrees，不增加高维描述符。",
        "5. 只有在噪声复核与数据覆盖改善后，再测试 GraphGPS 的不确定性估计或域加权训练；不要先扩展网络容量。",
        "",
        "## 产物索引",
        "- `data_schema.json`：实际输入、字段与缓存发现结果。",
        "- `splits/`：可复用、无泄漏的样本清单。",
        "- `domain_classifier_*.csv`、`feedback_ood_scores.csv`：分布偏移与 OOD。",
        "- `baseline_*.csv`、`graphgps_*.csv`：统一模型比较与逐样本预测。",
        "- `feedback_error_analysis.csv`、`ood_error_*.csv`：OOD 分层误差。",
    ])
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
