#!/usr/bin/env python3
"""Build a TeX/PDF comparison of the requested O12 feedback-71 experiments.

The report deliberately treats ``new_validation.csv`` as the independent
external set.  The 71 rows appended to the augmented training data originate
from ``20260703_validation.csv``; therefore that 97-row table is retained as
diagnostic evidence only, not used to rank external generalisation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/input_graphgps_optimization"


@dataclass(frozen=True)
class Experiment:
    key: str
    label: str
    directory: str
    data_scope: str
    expected_groups: tuple[str, ...]
    layout: str = "standard"

    @property
    def root(self) -> Path:
        return RESULTS / self.directory


EXPERIMENTS = (
    Experiment("input700", "I: input-only 700",
               "o12_input_700_multitasks_lr01",
               "700 input-only rows; 560/70/70 per split", ("core4", "norm2"), "input700_current"),
    Experiment("augmented", "A: augmented",
               "o12_multitask_seed100_109_lr001_input_plus_feedback71",
               "700 input + 71 feedback-only rows; 631/70/70 per split", ("core4", "norm2")),
    Experiment("feedback_only", "B: feedback-only",
               "o12_multitask_seed100_109_lr001_feedback_only",
               "71 feedback-only rows; 57/7/7 per split", ("core4", "norm2")),
)

CORE_TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]
NORM_TARGETS = ["Norm_before", "Norm_after"]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def tex_table(headers: list[str], rows: list[list[object]], column_spec: str) -> str:
    body = [rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule"]
    body.append(" & ".join(tex_escape(header) for header in headers) + r" \\")
    body.append(r"\midrule")
    for row in rows:
        body.append(" & ".join(tex_escape(item) for item in row) + r" \\")
    body.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(body)


def concise_target(name: str) -> str:
    return {
        "Aerosolization_Efficiency": "Aerosolization",
        "mRNA_Recovery_Efficiency": "mRNA recovery",
    }.get(name, name)


def load_experiment(experiment: Experiment) -> dict[str, object]:
    if experiment.layout in {"legacy", "input700_current"}:
        macro = read_csv(experiment.root / "corresponding_split_single_inference/metrics_macro_10seed_mean_variance.csv")
        macro = macro.rename(columns={"mean_mae": "mean_macro_mae", "mean_r2": "mean_macro_r2"})
        targets = read_csv(experiment.root / "corresponding_split_single_inference/metrics_target_10seed_mean_variance.csv")
    else:
        macro = read_csv(experiment.root / "validation_test_metrics_macro_average.csv")
        targets = read_csv(experiment.root / "validation_test_metrics_target_average.csv")
    feedback_rows = []
    if experiment.layout == "input700_current":
        for group in experiment.expected_groups:
            path = experiment.root / "new_validation_ensemble/new_validation" / f"scored_{group}/metrics_ensemble.csv"
            frame = read_csv(path)
            if not frame.empty:
                frame.insert(0, "dataset", "new_validation")
                frame.insert(0, "target_group", group)
                feedback_rows.append(frame)
    else:
        for path in sorted(experiment.root.glob("feedback_*_ensemble/*/metrics_ensemble.csv")):
            frame = read_csv(path)
            if frame.empty:
                continue
            group = path.parents[1].name.replace("feedback_", "").replace("_ensemble", "")
            frame.insert(0, "dataset", path.parent.name)
            frame.insert(0, "target_group", group)
            feedback_rows.append(frame)
    feedback = pd.concat(feedback_rows, ignore_index=True) if feedback_rows else pd.DataFrame()
    settings_paths = sorted(experiment.root.glob("O12_*_split100/run_settings.json"))
    if not settings_paths:
        settings_paths = sorted(experiment.root.glob("*/O12_split100/run_settings.json"))
    settings = json.loads(settings_paths[0].read_text(encoding="utf-8")) if settings_paths else {}
    completed_groups = tuple(sorted(macro.target_group.unique())) if not macro.empty else ()
    return {"experiment": experiment, "macro": macro, "targets": targets,
            "feedback": feedback, "settings": settings, "completed_groups": completed_groups}


def plot_internal_macro(records: list[dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.2), constrained_layout=True)
    for row_index, group in enumerate(("core4", "norm2")):
        rows = []
        for record in records:
            macro = record["macro"]
            subset = macro.loc[macro.target_group.eq(group)] if not macro.empty else macro
            if subset.empty:
                continue
            for split in ("val", "test"):
                value = subset.loc[subset.split.eq(split)]
                if not value.empty:
                    rows.append((record["experiment"].key, record["experiment"].label.split(":")[0], split,
                                 float(value.iloc[0].mean_macro_mae), float(value.iloc[0].mean_macro_r2)))
        for column, metric in enumerate(("MAE", "R2")):
            axis = axes[row_index, column]
            if not rows:
                axis.set_visible(False)
                continue
            labels = list(dict.fromkeys(row[1] for row in rows))
            x = np.arange(len(labels))
            width = .34
            for offset, split, color in ((-.17, "val", "#4c78a8"), (.17, "test", "#f58518")):
                values = [next(row[3 if metric == "MAE" else 4] for row in rows
                               if row[1] == label and row[2] == split) for label in labels]
                axis.bar(x + offset, values, width, label=split, color=color)
            axis.set_xticks(x, labels)
            axis.set_title(f"{group}: internal {metric}")
            axis.set_ylabel(metric)
            axis.grid(axis="y", alpha=.25)
            axis.legend(fontsize=8)
    figure.suptitle("Mean macro metrics across ten fixed splits", fontsize=14)
    figure.savefig(output / "internal_macro_metrics.png", dpi=200)
    plt.close(figure)


def plot_external_metrics(records: list[dict[str, object]], group: str, targets: list[str], output: Path) -> None:
    available = []
    for record in records:
        feedback = record["feedback"]
        part = feedback.loc[(feedback.dataset == "new_validation") & (feedback.target_group == group)] if not feedback.empty else feedback
        if not part.empty:
            available.append((record["experiment"].label.split(":")[0], part.set_index("target")))
    if not available:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    x = np.arange(len(targets))
    width = .76 / len(available)
    colors = plt.get_cmap("tab10").colors
    for metric, axis in zip(("mae", "r2"), axes):
        for index, (label, frame) in enumerate(available):
            values = [float(frame.loc[target, metric]) if target in frame.index else np.nan for target in targets]
            axis.bar(x - .38 + width / 2 + index * width, values, width, label=label,
                     color=colors[index])
        axis.set_xticks(x, [concise_target(target) for target in targets], rotation=17, ha="right")
        axis.set_ylabel(metric.upper())
        axis.set_title(f"Independent new_validation: {group} {metric.upper()}")
        axis.grid(axis="y", alpha=.25)
        axis.legend(fontsize=8)
    figure.savefig(output / f"new_validation_{group}_metrics.png", dpi=200)
    plt.close(figure)


def add_image_grid(items: list[tuple[str, Path]], columns: int, output: Path, title: str) -> bool:
    usable = [(label, path) for label, path in items if path.is_file()]
    if not usable:
        return False
    rows = int(np.ceil(len(usable) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.3 * columns, 4.2 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, (label, path) in zip(axes, usable):
        axis.imshow(mpimg.imread(path))
        axis.set_title(label, fontsize=10)
        axis.axis("off")
    for axis in axes[len(usable):]:
        axis.axis("off")
    figure.suptitle(title, fontsize=14)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return True


def make_scatter_composites(output: Path) -> tuple[bool, bool, bool]:
    input_only = RESULTS / "o12_input_700_multitasks_lr01"
    a = RESULTS / "o12_multitask_seed100_109_lr001_input_plus_feedback71"
    b = RESULTS / "o12_multitask_seed100_109_lr001_feedback_only"
    core_items = []
    for label, root, directory in (
        ("I", input_only, "new_validation_ensemble/new_validation/scored_core4/scatter_by_target"),
        ("A", a, "feedback_core4_ensemble/new_validation/scatter_by_target"),
        ("B", b, "feedback_core4_ensemble/new_validation/scatter_by_target"),
    ):
        for target in CORE_TARGETS:
            core_items.append((
                f"{label}: {concise_target(target)}",
                root / directory / f"{target}_true_vs_pred.png",
            ))
    # A was generated by the newer per-target plotter while B also retains an
    # older two-panel image.  Use the per-target files for both experiments so
    # neither one is silently omitted because of an output-format difference.
    # I uses the newer scored-output layout, whereas A/B use the historical
    # feedback output layout.
    norm_items = []
    for target in NORM_TARGETS:
        for label, root, directory in (
            ("I", input_only, "new_validation_ensemble/new_validation/scored_norm2/scatter_by_target"),
            ("A", a, "feedback_norm2_ensemble/new_validation/scatter_by_target"),
            ("B", b, "feedback_norm2_ensemble/new_validation/scatter_by_target"),
        ):
            norm_items.append((f"{label}: {target}", root / directory / f"{target}_true_vs_pred.png"))
    input_only_items = [
        (f"I: {concise_target(target)}",
         input_only / "new_validation_ensemble/new_validation" / f"scored_{group}/scatter_by_target" /
         f"{target}_true_vs_pred.png")
        for group, targets in (("core4", CORE_TARGETS), ("norm2", NORM_TARGETS))
        for target in targets
    ]
    return (
        add_image_grid(core_items, 4, output / "new_validation_core4_scatter_comparison.png",
                       "Core4 scatter plots on independent new_validation"),
        add_image_grid(norm_items, 3, output / "new_validation_norm2_scatter_comparison.png",
                       "Norm2 scatter plots on independent new_validation"),
        add_image_grid(input_only_items, 3, output / "new_validation_input_only700_scatter.png",
                       "Input-only 700: six-target scatter plots on new_validation"),
    )


def write_report(records: list[dict[str, object]], output: Path, has_core_scatter: bool,
                 has_norm_scatter: bool) -> Path:
    inventory_rows = []
    for record in records:
        experiment = record["experiment"]
        inventory_rows.append([
            experiment.label, experiment.data_scope,
            ", ".join(record["completed_groups"]) or "no completed summary",
        ])
    inventory_rows.extend([
        ["E: later4_input_plus_feedback71", "Dataset/manifests only: 700+71=771 rows", "no model run"],
        ["F: later4 zscore augmented", "Only O12_norm2_split100 intermediate files", "incomplete; no metrics/plots"],
    ])

    internal_rows = []
    for record in records:
        macro = record["macro"]
        for group in ("core4", "norm2"):
            part = macro.loc[macro.target_group.eq(group)] if not macro.empty else macro
            if part.empty:
                continue
            val, test = part.set_index("split").loc["val"], part.set_index("split").loc["test"]
            internal_rows.append([
                record["experiment"].label.split(":")[0], group,
                f"{val.mean_macro_mae:.3f}", f"{val.mean_macro_r2:.3f}",
                f"{test.mean_macro_mae:.3f}", f"{test.mean_macro_r2:.3f}",
            ])

    feedback_rows: dict[str, list[list[object]]] = {"core4": [], "norm2": []}
    for group, targets in (("core4", CORE_TARGETS), ("norm2", NORM_TARGETS)):
        for target in targets:
            row = [concise_target(target)]
            for record in records:
                feedback = record["feedback"]
                value = feedback.loc[(feedback.dataset == "new_validation") &
                                     (feedback.target_group == group) &
                                     (feedback.target == target)] if not feedback.empty else feedback
                row.append("--" if value.empty else f"{value.iloc[0].mae:.3f} / {value.iloc[0].r2:.3f}")
            feedback_rows[group].append(row)

    tex = rf"""\documentclass[UTF8,a4paper,11pt]{{ctexart}}
\usepackage[margin=1.7cm]{{geometry}}
\usepackage{{booktabs,graphicx,float,array}}
\graphicspath{{{{figures/}}}}
\setlength{{\parskip}}{{0.45em}}
\title{{O12 feedback-71 实验差异分析}}
\author{{自动汇总已有运行产物}}
\date{{2026-08-11}}
\begin{{document}}
\maketitle

\section{{范围与可比性}}
本报告仅重算并汇总指定目录中已经保存的指标和图表；没有重新训练或用 feedback 标签调参。每个完成实验均使用十个固定 seed（100--109），内部指标为每个 seed 指标的均值。

\begin{{table}}[H]\centering\small
\caption{{实验清单与完成状态}}
{tex_table(["实验", "训练数据/划分", "完成目标组"], inventory_rows, "p{3.2cm}p{7.0cm}p{3.0cm}")}
\end{{table}}

独立外部集定义为 \texttt{{new\_validation.csv}}（26 条）。增广训练集加入的 71 条来自 \texttt{{20260703\_validation.csv}}，故后者的 97 条评估中含有训练样本；其结果可以用于诊断拟合，但不能作为独立外部泛化排名。feedback-only 训练的内部 val/test 仅各 7 条，方差很大，也不应与 70 条内部划分的结果作强结论式比较。

\section{{内部验证与测试}}
表中为十个 seed 宏平均：同一目标组内各性质的 MAE/R\textsuperscript{{2}} 先平均，再跨 seed 平均。core4 的 MAE 为百分点，norm2 为原始 Norm 单位，二者不应横向比较绝对 MAE。

\begin{{table}}[H]\centering\small
\caption{{内部宏平均指标}}
{tex_table(["实验", "目标组", "Val MAE", "Val R2", "Test MAE", "Test R2"], internal_rows, "llrrrr")}
\end{{table}}

\begin{{figure}}[H]\centering
\includegraphics[width=\linewidth]{{internal_macro_metrics.png}}
\caption{{内部验证/测试宏平均；不同目标组分面显示。}}
\end{{figure}}

\section{{独立 new\_validation 外部结果}}
下表每个单元格依次为 ``MAE / R\textsuperscript{{2}}''，取十个 checkpoint 输出的非加权均值后计算。``--'' 表示目录中没有该目标组的完成模型。

\begin{{table}}[H]\centering\scriptsize
\caption{{Core4 外部结果（MAE / R2）}}
{tex_table(["性质", *[record["experiment"].label.split(":")[0] for record in records]], feedback_rows["core4"], "lrrrr")}
\end{{table}}

\begin{{table}}[H]\centering\small
\caption{{Norm2 外部结果（MAE / R2）}}
{tex_table(["性质", *[record["experiment"].label.split(":")[0] for record in records]], feedback_rows["norm2"], "lrrrr")}
\end{{table}}

\begin{{figure}}[H]\centering
\includegraphics[width=\linewidth]{{new_validation_core4_metrics.png}}
\caption{{Core4 在独立外部集上的指标。}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=\linewidth]{{new_validation_norm2_metrics.png}}
\caption{{Norm2 在独立外部集上的指标。}}
\end{{figure}}
"""
    if has_core_scatter:
        tex += r"""\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{new_validation_core4_scatter_comparison.png}
\caption{Core4 外部散点图对照：A 为增广训练，B 为 feedback-only 训练。}
\end{figure}
"""
    if has_norm_scatter:
        tex += r"""\begin{figure}[H]\centering
\includegraphics[width=0.92\linewidth]{new_validation_norm2_scatter_comparison.png}
\caption{Norm2 外部散点图对照。}
\end{figure}
"""
    tex += r"""
\section{差异解读}
\begin{enumerate}
  \item 在完整的 core4 对照中，A 在 \texttt{EE\_before}（MAE 14.280 对 18.914）和 mRNA recovery（24.691 对 26.928）较 B 低；B 在 \texttt{EE\_after}（15.098 对 17.440）和 Aerosolization（22.611 对 28.404）较低。因此 71 条训练的引入没有在四个性质上形成一致的外部改善。
  \item 对 norm2，B（feedback-only）在 \texttt{Norm\_after} 的外部表现最佳（MAE 0.886，R2 0.727）。\texttt{Norm\_before} 的外部 MAE 与 A 接近；在 26 条外部样本上不宜据此宣称稳定优势。
  \item E 是数据与划分资产而非模型运行；F 不仅未完成十 seed，也实际记录为 \texttt{O12\_norm2\_split100}，而不是完成的 later4 四目标模型。因此本报告不将 F 视作 later4/z-score 的有效对照，必须补齐十个 \texttt{O12\_later4\_split100...109} checkpoint、验证/测试预测、两套 feedback 推理后才能比较 target normalization 的影响。
\end{enumerate}

\section{数据来源}
内部指标来自各完成目录的 \texttt{validation\_test\_metrics\_macro\_average.csv} 与 \texttt{validation\_test\_metrics\_target\_average.csv}；外部指标来自各实验保存的 \texttt{metrics\_ensemble.csv}。本报告同时导出了对应的 CSV 摘要，便于复核。
\end{document}
"""
    report = output / "o12_feedback71_experiment_comparison.tex"
    report.write_text(tex, encoding="utf-8")
    return report


def write_beamer_report(records: list[dict[str, object]], output: Path,
                        has_core_scatter: bool, has_norm_scatter: bool,
                        has_input_only_scatter: bool) -> Path:
    """Render the same evidence as a compact 16:9 Chinese Beamer deck."""
    inventory_rows = []
    for record in records:
        experiment = record["experiment"]
        inventory_rows.append([
            experiment.label, experiment.data_scope,
            ", ".join(record["completed_groups"]) or "no completed summary",
        ])
    internal_rows = []
    for record in records:
        macro = record["macro"]
        for group in ("core4", "norm2"):
            part = macro.loc[macro.target_group.eq(group)] if not macro.empty else macro
            if part.empty:
                continue
            indexed = part.set_index("split")
            internal_rows.append([
                record["experiment"].label.split(":")[0], group,
                f"{indexed.loc['val'].mean_macro_mae:.3f}", f"{indexed.loc['val'].mean_macro_r2:.3f}",
                f"{indexed.loc['test'].mean_macro_mae:.3f}", f"{indexed.loc['test'].mean_macro_r2:.3f}",
            ])

    feedback_rows: dict[str, list[list[object]]] = {"core4": [], "norm2": []}
    for group, targets in (("core4", CORE_TARGETS), ("norm2", NORM_TARGETS)):
        for target in targets:
            row: list[object] = [concise_target(target)]
            for record in records:
                feedback = record["feedback"]
                value = feedback.loc[(feedback.dataset == "new_validation") &
                                     (feedback.target_group == group) &
                                     (feedback.target == target)] if not feedback.empty else feedback
                row.append("--" if value.empty else f"{value.iloc[0].mae:.3f} / {value.iloc[0].r2:.3f}")
            feedback_rows[group].append(row)

    inventory_table = tex_table(
        ["实验", "训练数据/划分", "完成目标组"], inventory_rows, "p{3.1cm}p{8.0cm}p{3.0cm}"
    )
    internal_table = tex_table(
        ["实验", "目标组", "Val MAE", "Val R2", "Test MAE", "Test R2"], internal_rows, "llrrrr"
    )
    core_table = tex_table(
        ["性质", *[record["experiment"].label.split(":")[0] for record in records]],
        feedback_rows["core4"], "lrrrrr"
    )
    norm_table = tex_table(
        ["性质", *[record["experiment"].label.split(":")[0] for record in records]],
        feedback_rows["norm2"], "lrrrrr"
    )
    scatter_frames = ""
    if has_core_scatter:
        scatter_frames += r"""
\begin{frame}{外部散点图：Core4}
\centering\includegraphics[width=\textwidth,height=.79\textheight,keepaspectratio]{new_validation_core4_scatter_comparison.png}
\end{frame}
"""
    if has_norm_scatter:
        scatter_frames += r"""
\begin{frame}{外部散点图：Norm2}
\centering\includegraphics[width=.86\textwidth,height=.79\textheight,keepaspectratio]{new_validation_norm2_scatter_comparison.png}
\end{frame}
"""
    if has_input_only_scatter:
        scatter_frames += r"""
\begin{frame}{外部散点图：I（input-only 700）}
\centering\includegraphics[width=.94\textwidth,height=.79\textheight,keepaspectratio]{new_validation_input_only700_scatter.png}
\end{frame}
"""
    tex = r"""\documentclass[aspectratio=169,10pt,UTF8]{ctexbeamer}
\usetheme{default}
\setbeamertemplate{navigation symbols}{}
\usepackage{booktabs,graphicx,array}
\graphicspath{{figures/}}
\title{O12 feedback-71 实验差异分析}
\subtitle{已保存运行指标、外部推理图表与可比性审计}
\author{自动汇总已有实验产物}
\date{2026-08-11}
\begin{document}

\begin{frame}
\titlepage
\end{frame}

\begin{frame}[t]{范围与可比性}
\begin{itemize}
  \item 本报告只读取已有运行产物；未重新训练，未用 feedback 标签调参。
  \item 每个完成实验使用固定 seed 100--109；内部结果为先按 seed 计算、再平均的指标。
  \item 独立外部集为 \texttt{new\_validation.csv}（26 条）。\texttt{20260703\_validation.csv} 中有 71 条已并入增广训练集，仅作诊断，不用于外部泛化排名。
  \item feedback-only 的内部 val/test 各只有 7 条，因此内部方差大，不能与 70 条划分作强结论式比较。
\end{itemize}
\end{frame}

\begin{frame}[t]{实验清单与完成状态}
\centering\tiny
\resizebox{\textwidth}{!}{__INVENTORY_TABLE__}
\end{frame}

\begin{frame}[t]{内部验证与测试：十 seed 宏平均}
\centering\small
\resizebox{.72\textwidth}{!}{__INTERNAL_TABLE__}
\end{frame}

\begin{frame}{内部验证与测试：图形比较}
\centering
\includegraphics[width=.86\textwidth,height=.76\textheight,keepaspectratio]{internal_macro_metrics.png}
\end{frame}

\begin{frame}[t]{独立外部集：Core4}
\centering\scriptsize
每个单元格为十模型均值的 \textbf{MAE / R2}；core4 的 MAE 单位为百分点。
\vspace{.3em}
\resizebox{.78\textwidth}{!}{__CORE_TABLE__}
\vfill
\includegraphics[width=.88\textwidth,height=.49\textheight,keepaspectratio]{new_validation_core4_metrics.png}
\end{frame}

\begin{frame}[t]{独立外部集：Norm2}
\centering\scriptsize
每个单元格为十模型均值的 \textbf{MAE / R2}；Norm 为原始单位。
\vspace{.3em}
\resizebox{.78\textwidth}{!}{__NORM_TABLE__}
\vfill
\includegraphics[width=.88\textwidth,height=.49\textheight,keepaspectratio]{new_validation_norm2_metrics.png}
\end{frame}

__SCATTER_FRAMES__

\begin{frame}[t]{差异解读}
\begin{enumerate}
  \item Core4：A 在 \texttt{EE\_before} 和 mRNA recovery 外部 MAE 更低；B 在 \texttt{EE\_after} 和 Aerosolization 更低。加入 71 条训练样本未形成四性质一致改善。
  \item Norm2：在当前保留的 A/B 对照中，B（feedback-only）的 \texttt{Norm\_before} 与 \texttt{Norm\_after} 外部 MAE 均低于 A；尤其 \texttt{Norm\_after} 为 0.886，R2 为 0.727。
  \item I 是纯 700 条 input-only 的六性质基线：内部 test 宏平均为 core4 MAE/R2=14.485/-0.054、norm2=1.358/0.086。其在相同 \texttt{new\_validation} 上的 Norm MAE 为 0.687、1.794，两个值均高于 A/B。外部集仅 26 条，结论应结合散点图谨慎解释。
\end{enumerate}
\end{frame}

\begin{frame}[t]{数据来源与复核}
\begin{itemize}
  \item 内部：各完成目录的 \texttt{validation\_test\_metrics\_macro\_average.csv} 与 \texttt{validation\_test\_metrics\_target\_average.csv}。
  \item input-only 700 基线：内部指标来自其保存的 \texttt{corresponding\_split\_single\_inference/}；外部指标与散点图来自相同模型在 \texttt{new\_validation} 上的十模型集成预测。
  \item 外部：每个实验对应的 \texttt{metrics\_ensemble.csv}。
  \item 本报告目录额外导出 \texttt{internal\_macro\_metrics\_comparison.csv} 与 \texttt{feedback\_metrics\_comparison.csv}，可逐项追溯。
\end{itemize}
\vfill
\centering\Large 谢谢
\end{frame}
\end{document}
"""
    tex = (tex.replace("__INVENTORY_TABLE__", inventory_table)
           .replace("__INTERNAL_TABLE__", internal_table)
           .replace("__CORE_TABLE__", core_table)
           .replace("__NORM_TABLE__", norm_table)
           .replace("__SCATTER_FRAMES__", scatter_frames))
    report = output / "o12_feedback71_experiment_comparison.tex"
    report.write_text(tex, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=RESULTS / "o12_feedback71_experiment_comparison_report")
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    records = [load_experiment(experiment) for experiment in EXPERIMENTS]

    internal = pd.concat([
        record["macro"].assign(experiment=record["experiment"].key)
        for record in records if not record["macro"].empty
    ], ignore_index=True)
    external = pd.concat([
        record["feedback"].assign(experiment=record["experiment"].key)
        for record in records if not record["feedback"].empty
    ], ignore_index=True)
    internal.to_csv(output / "internal_macro_metrics_comparison.csv", index=False)
    external.to_csv(output / "feedback_metrics_comparison.csv", index=False)
    plot_internal_macro(records, figures)
    plot_external_metrics(records, "core4", CORE_TARGETS, figures)
    plot_external_metrics(records, "norm2", NORM_TARGETS, figures)
    has_core_scatter, has_norm_scatter, has_input_only_scatter = make_scatter_composites(figures)
    report = write_beamer_report(
        records, output, has_core_scatter, has_norm_scatter, has_input_only_scatter
    )
    if not args.skip_pdf:
        compiler = shutil.which("xelatex")
        if not compiler:
            raise RuntimeError("xelatex is required to compile the Chinese TeX report.")
        command = [compiler, "-interaction=nonstopmode", "-halt-on-error", report.name]
        for _ in range(2):
            completed = subprocess.run(command, cwd=output, text=True, capture_output=True)
            if completed.returncode:
                raise RuntimeError(f"XeLaTeX failed:\n{completed.stdout}\n{completed.stderr}")
    print(json.dumps({
        "tex": str(report), "pdf": str(report.with_suffix(".pdf")),
        "internal_summary": str(output / "internal_macro_metrics_comparison.csv"),
        "external_summary": str(output / "feedback_metrics_comparison.csv"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
