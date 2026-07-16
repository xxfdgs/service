#!/usr/bin/env python3
"""Write the evidence-based final report for the fusion/head redesign study."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'results/fusion_head_redesign_exp'
STAGE = EXP / 'stage1'


def macro(metrics, candidate, fold):
    rows = metrics.loc[(metrics.candidate == candidate) & (metrics.fold == fold)]
    if rows.empty:
        return {key: float('nan') for key in ('mae', 'r2', 'spearman', 'std_ratio')}
    return rows.groupby('target')[['mae', 'r2', 'spearman', 'std_ratio']].mean().mean().to_dict()


def fmt(value):
    return '—' if pd.isna(value) else f'{value:.4f}'


def main():
    metrics = pd.read_csv(STAGE / 'fold_metrics.csv')
    inventory = pd.read_csv(STAGE / 'run_inventory.csv')
    baseline0, baseline4 = macro(metrics, 'A0', 'fold_0'), macro(metrics, 'A0', 'fold_4')
    summaries = {(candidate, fold): macro(metrics, candidate, fold)
                 for candidate in sorted(metrics.candidate.unique())
                 for fold in ('fold_0', 'fold_4')}
    b4_gate = pd.read_csv(EXP / 'dynamics/gate_statistics.csv')
    b4_gate = b4_gate.loc[(b4_gate.candidate == 'B4') & (b4_gate.split == 'val') & (b4_gate.gate_type == 'feature_gate')]
    b4_low_gate = float(b4_gate.below_005_fraction.mean())
    b4_high_gate = float(b4_gate.above_095_fraction.mean())
    selection = json.loads((STAGE / 'head_selection.json').read_text())

    table_rows = []
    for candidate in ('A0', 'A1', 'A2', 'A3', 'A4', 'B0', 'B1', 'B2', 'B3', 'B4'):
        f0, f4 = summaries.get((candidate, 'fold_0'), {}), summaries.get((candidate, 'fold_4'), {})
        selected = 'baseline control' if candidate == 'A0' else 'rejected'
        table_rows.append(
            f"| {candidate} | {fmt(f0.get('mae'))} | — | {fmt(f4.get('mae'))} | "
            f"{fmt(f4.get('std_ratio'))} | {fmt(f4.get('spearman'))} | skipped: no valid candidate | — | — | — | {selected} |")
    table = '\n'.join(table_rows)
    report = f"""# GraphGPS 融合层与预测头受控修复实验

## Final status

`NO_SAFE_FUSION_HEAD_FIX`

该结论来自 20 个规范的 Stage-1 运行（A0–A4 与 B0–B4，各 fold-0/fold-4、seed=0），仅使用 inner validation 选择或拒绝架构。另有一个完整的 A2/fold-0 重复 attempt 被保留在 inventory 中，但不重复计入比较。没有读取 outer test、feedback 标签或 fold-1/2/3。

历史 baseline 等价性已通过：严格 checkpoint 加载、state-dict 哈希及参数量一致，归一化输出最大绝对差为 `7.629e-08`（阈值 `<1e-6`）。

## Stage-1 macro validation comparison

| architecture | fold0_val_mae | fold1_val_mae | fold4_val_mae | fold4_std_ratio | fold4_spearman | untouched_result | pooled_mae | pooled_r2 | pooled_spearman | selected |
| ------------ | ------------: | ------------: | ------------: | --------------: | -------------: | ---------------- | ---------: | --------: | --------------: | -------- |
{table}

Baseline A0 is {fmt(baseline0['mae'])} MAE / {fmt(baseline0['spearman'])} Spearman / {fmt(baseline0['std_ratio'])} std-ratio on fold-0, and {fmt(baseline4['mae'])} / {fmt(baseline4['spearman'])} / {fmt(baseline4['std_ratio'])} on fold-4.

## Required answers

1. **Prediction head 是否是主要塌缩点？** 不是唯一或可安全独立修复的主因。A1–A4 没有任何方案通过双折门槛；A2 虽改善 fold-4，却使 fold-0 MAE 增加约 4.3%。
2. **softmax fusion 是否是主要塌缩点？** 不能据此断定。B0–B4 均未通过跨折标准；替换融合可改善 fold-4，但没有安全泛化。
3. **linear head 是否恢复输出方差？** A1 在 fold-4 将 std-ratio 拉至约 2.405，却使 MAE 恶化为约 52.55、Spearman 为负；这是无效的方差放大。
4. **target-specific head 是否改善多任务学习？** 否。A4 fold-0 MAE 约 62.34、R² 约 -20.70，并且参数量约高 33%。
5. **concat 是否优于 softmax sum？** 否。B1 在两折均严重退化（fold-0/4 MAE 约 67.77/123.93）。
6. **residual fusion 是否更稳定？** 否。B3 fold-0/4 MAE 约 19.61/27.83，均低于 baseline 稳定性。
7. **gated-concat 是否发生新的饱和？** 是。B4 验证 gate 的平均 `gate < 0.05` 比例约 {b4_low_gate:.1%}，`gate > 0.95` 比例约 {b4_high_gate:.1%}；其大量通道趋近关闭，且 fold-0 MAE 恶化。
8. **哪个模块最先发生方差丢失？** A0 fold-4 的 softmax branch 权重超过 0.98 最早于 epoch 41，预测 std-ratio 低于 0.10 于 epoch 42，熵低于 0.05 于 epoch 43；因此先出现的是 fusion gate 饱和，随后是预测输出压缩。
9. **新方案是否消除了 epoch 41–42 collapse？** 没有获得可确认的消除方案。B4 fold-4 的近常数事件仍在 epoch 68 记录，且它在 fold-0 的泛化失败。
10. **fold-4 validation MAE 是否改善？** 局部改善存在：B4 为 {fmt(summaries[('B4','fold_4')]['mae'])}（A0 为 {fmt(baseline4['mae'])}），A2 为 {fmt(summaries[('A2','fold_4')]['mae'])}；但都没有与 fold-0 同时成立。
11. **fold-4 std ratio 和 Spearman 是否改善？** B4 改善至 {fmt(summaries[('B4','fold_4')]['std_ratio'])}/{fmt(summaries[('B4','fold_4')]['spearman'])}（A0 {fmt(baseline4['std_ratio'])}/{fmt(baseline4['spearman'])}），但其 fold-0 MAE 为 {fmt(summaries[('B4','fold_0')]['mae'])}，不满足安全条件。
12. **fold-0 和 fold-1 是否保持稳定？** fold-0 没有任何 fold-4 改善方案保持 ≤2% MAE 退化；fold-1 按 protocol 未运行，因为没有候选通过 Stage-1。
13. **untouched fold-2 和 fold-3 是否确认收益？** 否。它们未触碰，符合“只在 Stage-2 通过后才能确认”的约束。
14. **pooled 五折是否优于原 baseline？** 未评估；没有锁定架构，不能合法读取更多 validation/test folds 来构造 pooled 指标。
15. **方差改善是否对应正确排序？** 在 B4 fold-4，Spearman 从 {fmt(baseline4['spearman'])} 提升至 {fmt(summaries[('B4','fold_4')]['spearman'])}，但这一局部排序信号未跨 fold 保持，因此不能作为正确泛化排序的证明。
16. **是否需要重跑完整 GraphGPS 五折？** 不需要为这些 fusion/head 改动重跑。先前的基线仍有效；只有新的、预先锁定且通过 Stage-1/2 的方案才应进入完整五折。
17. **是否需要进入单任务实验？** 当前不建议。target-specific heads 已作为较轻的多任务冲突测试而失败；本研究没有产生支持完整单任务五折的证据。
18. **是否仍应保留 GraphGPS 研究路线？** 可保留为研究路线，但本次证据不支持把任何 tested fusion/head 变体作为 production candidate。后续需形成不同的、可检验的优化/表示假设。

## Protocol consequences

Head-selection status is `{selection['head_selection_status']}`. 因此没有满足第 5 阶段门槛的候选，按 protocol 跳过：

- 同 seed 重复验证；
- fold-1 开发验证；
- fold-2/3 untouched 确认；
- outer-test、pooled 五折与 production promotion。

所有 Stage-1 predictions 均含 `sample_id`，并只由训练/validation split 生成。完整命令、哈希和输出路径在 `execution_manifest.json`；逐 epoch 指标在 `dynamics/`。
"""
    (EXP / 'report.md').write_text(report)
    skipped = {
        EXP / 'reproducibility/reproducibility_report.md': '# Reproducibility\n\nSkipped: no Stage-1 candidate passed the mandatory fold-0/fold-4 gates.\n',
        EXP / 'stage2/stage2_report.md': '# Stage 2\n\nSkipped: no candidate passed Stage-1; fold-1 was deliberately not run.\n',
        EXP / 'confirmation/confirmation_report.md': '# Confirmation\n\nSkipped: architecture was not locked; folds 2 and 3 remain untouched.\n',
    }
    for path, content in skipped.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(f'Wrote {EXP / "report.md"}')


if __name__ == '__main__':
    main()
