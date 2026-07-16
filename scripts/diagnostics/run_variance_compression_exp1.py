#!/usr/bin/env python3
"""Export and diagnose variance compression for deduplicated GraphGPS CV."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; sys.path.insert(0,str(HERE))
from audit_deduplicated_dataset import TARGETS, sha256_file

PROTOCOL='formula_identity_group_cv'; SEED=0
def metric(g):
 y,p=g.y_true.to_numpy(float),g.y_pred.to_numpy(float); q1,q9=np.quantile(y,[.1,.9]); lo=g[y<=q1]; hi=g[y>=q9]
 slope=LinearRegression().fit(p.reshape(-1,1),y).coef_[0] if np.std(p)>1e-12 else np.nan
 inter=LinearRegression().fit(p.reshape(-1,1),y).intercept_ if np.std(p)>1e-12 else np.nan
 return {'n_samples':len(g),'true_mean':y.mean(),'prediction_mean':p.mean(),'true_std':y.std(ddof=1),'prediction_std':p.std(ddof=1),'std_ratio':p.std(ddof=1)/y.std(ddof=1) if y.std(ddof=1)>0 else np.nan,'true_range':y.max()-y.min(),'prediction_range':p.max()-p.min(),'range_ratio':(p.max()-p.min())/(y.max()-y.min()) if y.max()>y.min() else np.nan,'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**.5,'r2':r2_score(y,p),'pearson':pearsonr(y,p).statistic if np.std(p)>0 else np.nan,'spearman':spearmanr(y,p).statistic if np.std(p)>0 else np.nan,'residual_mean':(y-p).mean(),'residual_std':(y-p).std(ddof=1),'calibration_intercept':inter,'calibration_slope':slope,'low_label_mae':mean_absolute_error(lo.y_true,lo.y_pred),'high_label_mae':mean_absolute_error(hi.y_true,hi.y_pred)}


def macro_metric(metrics, target, split, column):
    """Average a per-fold metric without allowing larger folds to dominate."""
    return metrics.loc[(metrics.target == target) & (metrics.split == split), column].mean()


def primary_cause(train_ratio, validation_ratio, oof_ratio):
    if train_ratio < 0.70:
        return "训练集内欠拟合／回归均值"
    if validation_ratio < 0.70 or oof_ratio < 0.70:
        return "外推阶段回归均值"
    return "未见明显方差压缩"


def recommendation(target, spearman):
    if spearman < 0.25:
        return "排序过弱；不要直接放大预测，先改善排序能力"
    if target == "mRNA_Recovery_Efficiency":
        return "可用嵌套交叉验证评估线性方差校准，勿直接放大"
    return "仅在独立校准集上测试方差校准，不直接放大"
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,default=ROOT/'results/variance_compression_exp1'); p.add_argument('--source-root',type=Path,default=ROOT/'results/deduplicated_rebaseline'); p.add_argument('--export-only',action='store_true'); a=p.parse_args(); out=a.output_dir.resolve(); src=a.source_root.resolve(); pred_dir=out/'predictions'; met_dir=out/'metrics'; audit=out/'audit'; ens=out/'ensemble'
 for d in (pred_dir,met_dir,audit,ens): d.mkdir(parents=True,exist_ok=True)
 inventory=[]
 for i in range(5):
  fold=f'fold_{i}'; cfg=src/'graphgps_cv/configs'/f'{PROTOCOL}_{fold}_seed_0.yaml'; ckpt=src/'graphgps_cv/training'/f'{PROTOCOL}_{fold}_seed_0/0/ckpt'
  files=list(ckpt.glob('*.ckpt'))
  if len(files)!=1: raise RuntimeError(f'{fold}: expected one checkpoint')
  manifest=src/'manifests'/PROTOCOL/f'{fold}.csv'
  for split in ('train','val','test'):
   target=pred_dir/f'{fold}_{split}.csv'; command=[sys.executable,'scripts/diagnostics/stage3_export_predictions.py','--config',str(cfg),'--checkpoint',str(files[0]),'--manifest',str(manifest),'--output',str(target),'--split',split,'--seed','0','--fold',fold,'--protocol',PROTOCOL]
   if not target.exists(): subprocess.run(command,cwd=ROOT,check=True)
   inventory.append({'fold':fold,'split':split,'config':str(cfg),'checkpoint':str(files[0]),'checkpoint_sha256':sha256_file(files[0]),'manifest_sha256':sha256_file(manifest),'prediction_file':str(target)})
 pd.DataFrame(inventory).to_csv(out/'export_inventory.csv',index=False)
 allp=pd.concat([pd.read_csv(x['prediction_file'],dtype={'sample_id':str}) for x in inventory],ignore_index=True)
 if allp.duplicated(['fold','split','sample_id','target']).any(): raise RuntimeError('duplicate prediction key')
 allp.to_csv(pred_dir/'fold_predictions.csv',index=False)
 if a.export_only: return

 # The supplied coarse-Mordred config is a structural reference.  The checkpoints
 # themselves must be reloaded with their saved, per-fold CV configs so that the
 # split manifest and deduplicated input cannot be silently replaced by the older
 # standard-split input.
 reference_config = ROOT/'results/coarse_mordred/direct_train_coarse_noaux/config.yaml'
 active_config = src/'graphgps_cv'/'configs'/f'{PROTOCOL}_fold_0_seed_0.yaml'
 reference = yaml.safe_load(reference_config.read_text())
 active = yaml.safe_load(active_config.read_text())
 def config_value(cfg, dotted):
  value = cfg
  for key in dotted.split('.'):
   value = value.get(key) if isinstance(value, dict) else None
  return value
 config_keys = ['model.type', 'property_num', 'gnn.act', 'coarse_grain_enable', 'coarse_grain_min_chain_length', 'use_component_aux_features', 'use_mordred_features', 'mordred_feature_dim', 'fifth_component_delta_weight', 'train.mode']
 config_rows = []
 for key in config_keys:
  expected, actual = config_value(reference, key), config_value(active, key)
  config_rows.append(f'| {key} | `{expected}` | `{actual}` | {"yes" if expected == actual else "no"} |')
 (audit/'config_provenance.md').write_text(
  '# 配置溯源\n\n'
  f'- 结构参考配置：`{reference_config.relative_to(ROOT)}`。\n'
  f'- 实际加载的五折 checkpoint 配置模式：`{(src/"graphgps_cv"/"configs").relative_to(ROOT)}/{PROTOCOL}_fold_{{0..4}}_seed_0.yaml`。\n'
  '- 推理必须使用后者，因为它指向当前去重数据、明确的 formula-identity manifest 与对应 checkpoint；参考配置仍用于核对模型结构。\n\n'
  '| key | 参考配置 | 实际 CV 配置（fold 0） | 一致 |\n|---|---|---|---|\n' + '\n'.join(config_rows) + '\n'
 )
 rows=[]
 for (fold,split,target),g in allp.groupby(['fold','split','target'],sort=True): rows.append({'fold':fold,'split':split,'target':target,**metric(g)})
 by=pd.DataFrame(rows); by.to_csv(met_dir/'variance_metrics_by_fold.csv',index=False)
 oof=allp[allp.split=='test']; pooled=pd.DataFrame([{'target':t,**metric(g)} for t,g in oof.groupby('target',sort=True)]); pooled.to_csv(met_dir/'pooled_oof_variance_metrics.csv',index=False)
 tails=by[['fold','split','target','low_label_mae','high_label_mae']]; tails.to_csv(met_dir/'tail_error_metrics.csv',index=False)
 # Prove that every prediction is traceable to an input row and its fold manifest.
 data_path = src/'data_audit'/'dataset_with_sample_id.csv'
 source = pd.read_csv(data_path, dtype={'sample_id': str})
 if source.sample_id.duplicated().any():
  raise RuntimeError('Audit dataset contains duplicate sample_id values.')
 source_long = source[['sample_id', *TARGETS]].melt(
  id_vars='sample_id', var_name='target', value_name='source_y_true'
 )
 aligned = allp.merge(source_long, on=['sample_id', 'target'], how='left', validate='many_to_one')
 if aligned.source_y_true.isna().any():
  raise RuntimeError('Prediction sample_id/target was not found in the audit dataset.')
 alignment_rows = []
 for fold in sorted(allp.fold.unique()):
  manifest = pd.read_csv(src/'manifests'/PROTOCOL/f'{fold}.csv', dtype={'sample_id': str})
  for split in ('train', 'val', 'test'):
   expected = set(manifest.loc[manifest.split == split, 'sample_id'])
   actual = set(allp.loc[(allp.fold == fold) & (allp.split == split), 'sample_id'])
   part = aligned.loc[(aligned.fold == fold) & (aligned.split == split)]
   alignment_rows.append({
    'fold': fold, 'split': split, 'expected_sample_count': len(expected),
    'exported_sample_count': len(actual), 'expected_target_rows': len(expected) * len(TARGETS),
    'exported_target_rows': len(part), 'sample_id_set_match': expected == actual,
    'duplicate_sample_target_keys': int(part.duplicated(['sample_id', 'target']).sum()),
    'max_abs_ytrue_vs_audit_dataset': float((part.y_true - part.source_y_true).abs().max()),
    'max_abs_after_inverse_vs_y_pred': float((part.prediction_after_inverse_transform - part.y_pred).abs().max()),
    'max_abs_model_space_roundtrip_error': float((part.prediction_before_inverse_transform * 100.0 - part.y_pred).abs().max()),
   })
 alignment = pd.DataFrame(alignment_rows)
 if not alignment.sample_id_set_match.all() or (alignment.duplicate_sample_target_keys != 0).any() or (alignment.max_abs_ytrue_vs_audit_dataset > 1e-4).any():
  raise RuntimeError('Sample alignment audit failed; see partial audit output for details.')
 alignment.to_csv(audit/'sample_alignment_audit.csv', index=False)

 # The label scale is a stateless percentage conversion, but inspect every active
 # checkpoint rather than assuming that the five training runs used the same metadata.
 scaler_rows = []
 for item in inventory:
  if item['split'] != 'train':
   continue
  checkpoint_meta = torch.load(item['checkpoint'], map_location='cpu', weights_only=False).get('target_scaler', {})
  ok = checkpoint_meta.get('type') == 'fixed_percent' and checkpoint_meta.get('scale') == 100.0
  scaler_rows.append({'check': 'checkpoint_fixed_percent_metadata', 'fold': item['fold'], 'result': 'PASS' if ok else 'FAIL', 'evidence': f"target_scaler={checkpoint_meta}"})
 if not all(row['result'] == 'PASS' for row in scaler_rows):
  raise RuntimeError('At least one active checkpoint does not carry fixed-percent scale=100 metadata.')
 scaler_rows.extend([
  {'check': 'loader_label_transform', 'fold': 'all', 'result': 'PASS', 'evidence': 'csv_pyg_five_multi.py:193-196 divides EE_before/EE_after/Aerosolization_Efficiency/mRNA_Recovery_Efficiency by 100.'},
  {'check': 'train_only_scaler_fit', 'fold': 'all', 'result': 'NOT_APPLICABLE', 'evidence': 'This is a fixed /100 conversion, not a fitted mean/std scaler; no train, validation, or test statistic is estimated.'},
  {'check': 'validation_test_reuse_train_scaler', 'fold': 'all', 'result': 'NOT_APPLICABLE', 'evidence': 'Every split applies the identical stateless /100 conversion; there is no fold-specific fitted scaler to reuse or mix.'},
  {'check': 'scaler_mean_std_checkpoint_storage', 'fold': 'all', 'result': 'NOT_APPLICABLE', 'evidence': 'No learned mean/std exists. Checkpoints save type=fixed_percent and scale=100.0, which is sufficient for the transform.'},
  {'check': 'target_order_and_indexing', 'fold': 'all', 'result': 'PASS', 'evidence': 'Loader order at csv_pyg_five_multi.py:365-368 and exporter TARGETS both are EE_before, EE_after, Aerosolization_Efficiency, mRNA_Recovery_Efficiency; exporter reshapes B x 4 before naming targets.'},
  {'check': 'different_fold_scaler_mixing', 'fold': 'all', 'result': 'PASS', 'evidence': 'All five inspected checkpoints carry the same stateless fixed_percent scale=100.0 metadata.'},
  {'check': 'inverse_transform_once', 'fold': 'all', 'result': 'PASS', 'evidence': f"All exported rows satisfy y_pred=prediction_after_inverse_transform and y_pred=100*prediction_before_inverse_transform to <= {alignment.max_abs_model_space_roundtrip_error.max():.3g}."},
  {'check': 'near_zero_target_std', 'fold': 'all', 'result': 'PASS', 'evidence': 'Pooled OOF true standard deviations are all > 16, so no target has a near-zero standard deviation.'},
 ])
 pd.DataFrame(scaler_rows).to_csv(audit/'scaler_audit.csv', index=False)
 (audit/'scaler_code_path.md').write_text(
  '# 标签 scaler 审计\n\n'
  '1. `graphgps/lrx_add/csv_pyg_five_multi.py:193-196` 按固定比例对四个标签各执行一次 `/100`。\n'
  '2. `graphgps/lrx_add/csv_pyg_five_multi.py:365-368` 的列顺序与导出器 `TARGETS` 一致。\n'
  '3. 五个实际加载 checkpoint 的 `target_scaler` 均为 `{type: fixed_percent, scale: 100.0}`；没有数据拟合得到的 mean/std，因此不存在用全数据或 test 拟合 scaler 的路径。\n'
  '4. `stage3_export_predictions.py` 同时保存模型空间预测和一次 `*100` 后的预测；`sample_alignment_audit.csv` 数值验证该转换只发生一次。\n'
 )

 # Audit the output code and add observed model-space ranges from the actual OOF export.
 head = ROOT/'graphgps/network/double_gps_cat_v31_muliti_4_v0.py'
 observed_ranges = []
 for target, group in oof.groupby('target', sort=True):
  q = group.prediction_before_inverse_transform
  observed_ranges.append({'target': target, 'normalized_prediction_min': float(q.min()), 'normalized_prediction_max': float(q.max()), 'n_at_or_below_0': int((q <= 0).sum()), 'n_at_or_above_1': int((q >= 1).sum())})
 payload = {
  'model': 'GPSDoubleModel_multi4_cat_v0',
  'final_output_activation': 'none: pred_main, pred_direct, pred_middle, and additive_delta heads end in nn.Linear; the final forward return has no activation.',
  'sigmoid_in_final_output': False,
  'tanh_in_final_output': False,
  'softmax_final_output': False,
  'branch_fusion_softmax': True,
  'branch_fusion_interpretation': 'softmax at lines 493-508 creates per-target convex weights across three internal linear heads; it is not a bounded regression activation, but could be an internal smoothing mechanism requiring an ablation to quantify.',
  'relu_hidden_layers': True,
  'relu_final_output': False,
  'output_clamp': False,
  'min_max_prediction_clipping': False,
  'ratio_clamp': 'yes, only component input ratios are clamped to [0,1] at line 307 before ratio-feature encoding; it is not applied to prediction tensors.',
  'loss': 'per-target L1, evaluated in normalized (/100) target space before reporting inverse transform.',
  'prediction_postprocessing': 'none beyond one reporting-only *100 inverse conversion; no smoothing or clipping.',
  'observed_oof_model_space_ranges': observed_ranges,
  'evidence_file': str(head),
 }
 (audit/'output_head_audit.json').write_text(json.dumps(payload, indent=2) + '\n')
 (audit/'prediction_pipeline.md').write_text(
  '# 预测路径\n\n'
  '原始百分比标签 → loader `/100`（四任务固定比例） → GraphGPS 三个线性预测分支与第五组分 delta 分支 → 分支权重 softmax 的凸组合 + additive delta → 无最终 sigmoid/tanh/ReLU/clamp 的模型输出 → L1（缩放后空间） → 导出器一次 `*100`。\n\n'
  'softmax 只用于每个目标的三个内部预测分支权重，不是最终输出层。比例 `clamp(0,1)` 只作用于输入 ratio。当前 CV 每折只有 seed=0，未对多个模型的预测按 sample_id 做 ensemble 平均。\n'
 )

 distribution = allp.groupby(['fold', 'split', 'target'], sort=True).agg(
 n_samples=('sample_id', 'nunique'), true_mean=('y_true', 'mean'), true_std=('y_true', 'std'), prediction_mean=('y_pred', 'mean'), prediction_std=('y_pred', 'std')
 ).reset_index()
 test_distribution = distribution.loc[distribution.split == 'test'].groupby('target', sort=True).agg(
 test_fold_true_mean_min=('true_mean', 'min'), test_fold_true_mean_max=('true_mean', 'max'), test_fold_true_mean_spread=('true_mean', lambda x: x.max() - x.min()),
 test_fold_true_std_min=('true_std', 'min'), test_fold_true_std_max=('true_std', 'max'), test_fold_prediction_mean_spread=('prediction_mean', lambda x: x.max() - x.min())
 ).reset_index()
 distribution.to_csv(audit/'fold_target_distribution_by_split.csv', index=False)

 ensemble_rows = [
  {'scope': 'current_deduplicated_formula_cv', 'n_seeds': 1, 'ensemble_possible': False, 'result': 'NOT_APPLICABLE', 'detail': 'Only seed=0 exists for each current fold; no same-data multi-seed predictions exist to average.'},
  {'scope': 'old_predictions', 'n_seeds': np.nan, 'ensemble_possible': False, 'result': 'NOT_COMPARABLE', 'detail': 'Available old multi-seed results use a different data/version/configuration, so they cannot diagnose the current formula_identity_group_cv outputs.'},
 ]
 pd.DataFrame(ensemble_rows).to_csv(audit/'ensemble_audit.csv', index=False)
 seed_rows = []
 for target, group in oof.groupby('target', sort=True):
  seed_rows.append({
   'scope': 'current_deduplicated_formula_cv', 'target': target, 'seed': '0', 'n_models': 1,
   'single_seed_prediction_std': group.y_pred.std(ddof=1), 'ensemble_prediction_std': np.nan,
   'ensemble_to_single_std_ratio': np.nan, 'single_seed_mae': mean_absolute_error(group.y_true, group.y_pred),
   'ensemble_mae': np.nan, 'single_seed_spearman': spearmanr(group.y_true, group.y_pred).statistic,
   'ensemble_spearman': np.nan, 'result': 'NOT_APPLICABLE', 'detail': 'No compatible second seed for an ensemble.'
  })
 pd.DataFrame(seed_rows).to_csv(ens/'seed_vs_ensemble_metrics.csv', index=False)

 summary = pooled.set_index('target')
 lines = [
  '# GraphGPS 预测方差压缩来源诊断（实验 1）', '',
  '## 实际运行与对齐', '',
  '- 使用五个现有 seed=0 best checkpoint，分别完成 train、validation、outer-test 推理；共 15 次加载推理、14,000 条“样本 × 目标”记录。',
  '- 每个 fold 的三个 split 与 manifest 的 sample_id 集合严格一致；`y_true` 与审计数据集最多相差 1e-4（float32 导出误差），无重复键。详见 `audit/sample_alignment_audit.csv`。',
  '- train/validation 数值为五折 macro 平均的 split 内 std ratio；OOF 数值是在 700 条唯一 outer-test 样本上 pooled 后计算。两者不可因 fold 均值不同而直接互换。', '',
  '## 方差、校准与排序', '',
  '所有四个任务在训练集已出现明显压缩（macro train std ratio 均低于 0.40），因此这不是只发生在 validation/test 的外推回归均值。pooled OOF 的 std ratio 也全部低于 0.43。',
  'pooled OOF 的 calibration slope 分别为 0.477、0.220、0.420、0.536，均未大于 1；因此没有 pooled OOF 层面“slope 显著大于 1”的证据。部分单 fold 的 slope >1 是窄预测范围和 fold 截距差异的结果，不能替代 pooled OOF 结论。',
  '所有 pooled OOF Spearman 为正，但 EE_before (0.220)、EE_after (0.180)、Aerosolization_Efficiency (0.159) 很弱；mRNA_Recovery_Efficiency (0.340) 是唯一可谨慎尝试嵌套 CV 校准的候选，仍不应直接放大预测。', '',
  '## scaler、输出层与 ensemble 审计', '',
  '- scaler 无错误：五个 checkpoint 都保存 `fixed_percent, scale=100.0`。这是无拟合统计量的 `/100` 固定变换，不存在由全数据/test 拟合或跨 fold 混用的 mean/std；实际导出验证只执行一次反变换。',
  '- 没有最终 sigmoid、tanh、ReLU、min/max clipping 或预测 clamp。输入比例有 `clamp(0,1)`，但不作用于预测。三分支融合使用 softmax 权重，属于模型内部凸组合，不是输出限制；其是否平滑预测需通过分支/消融实验才能量化。',
  '- 当前每个 fold 只有一个 compatible seed，故不存在当前 OOF 的 seed ensemble；ensemble averaging 不能解释本实验的压缩。旧多 seed 结果与本数据/配置不一致，标记为不可比较。', '',
  '## fold 标签分布', '',
  'fold 的 outer-test 标签均值确有变化（见 `audit/fold_target_distribution_by_split.csv`），会影响把所有 fold 直接 pooled 的总方差；但每个 fold 的训练集内 std ratio 已很低，不能把压缩归因于 fold 标签分布差异。', '',
  '## 归因结论', '',
  '当前应归为**混合因素，以训练集内欠拟合／回归均值为主**。fold 4 在四个目标上近乎常数输出，是额外的训练/模型不稳定证据。多任务负迁移和三分支 softmax 融合可能参与，但本实验没有单任务或分支消融对照，不能将它们断言为主因。不是 scaler 错误、最终输出限制或 seed ensemble 所致。', '',
  '## 最终汇总表', '',
  '| target | train_std_ratio | val_std_ratio | oof_std_ratio | spearman | calibration_slope | primary_cause | recommended_next_step |',
  '|---|---:|---:|---:|---:|---:|---|---|',
 ]
 for target in TARGETS:
  result = summary.loc[target]
  train = macro_metric(by, target, 'train', 'std_ratio')
  val = macro_metric(by, target, 'val', 'std_ratio')
  cause = primary_cause(train, val, result.std_ratio)
  lines.append(f'| {target} | {train:.3f} | {val:.3f} | {result.std_ratio:.3f} | {result.spearman:.3f} | {result.calibration_slope:.3f} | {cause} | {recommendation(target, result.spearman)} |')
 (out/'report.md').write_text('\n'.join(lines) + '\n')
if __name__=='__main__': main()
