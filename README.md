# O12 模型与数据集使用说明

本文档是本仓库中 **O12（`OneHotEmbedGPS`）** 的文件索引和操作手册。范围包括直接实现、配置、训练/评估/预测脚本、训练数据、描述符、数据划分和已有模型产物；仅在 O12/O22 对比中使用、但不改变 O12 本身的文件，也会明确标注为“对比/集成”。生成的 cache、日志和按种子重复的结果以目录模式完整说明，避免把同构的数千个文件逐一展开。

> 运行目录必须是仓库根目录。数据集、checkpoint 和结果均是本地文件；不要将其中的原始数据、`.pt`、`.ckpt` 或生成结果提交到版本库。

## 1. O12 是什么

O12 的注册模型名为 `OneHotEmbedGPS`。它服务于五组分配方回归：前四组分（IL、HL、Chol、PEG）以“组分位置独立的分子身份 embedding + 配比调制”编码，第五组分以两层 GraphGPS（GINE + Transformer）编码；随后融合五个表示、五个摩尔比例、五组 Mordred 描述符，并输出性质预测。

默认 O12 核心任务 `core4` 有四个输出：`EE_before`、`EE_after`、`Aerosolization_Efficiency`、`mRNA_Recovery_Efficiency`。后续冻结模型还提供 `norm2`（`Norm_before`、`Norm_after`）的十模型集成。`O12_model_structure_summary.md` 记录了 64 维隐藏层、136 维 RDKit/Morgan 辅助特征、11 维 Mordred 特征和 380 维融合输入等结构细节。

## 2. 文件与目录总览

### 2.1 模型实现和框架入口

| 路径 | 含义 |
| --- | --- |
| `O12_model_structure_summary.md` | O12 架构、数据流及其与原始 GraphGPS 的差异说明。 |
| `graphgps/network/onehot_embed_gps.py` | **O12 主实现**；注册 `OneHotEmbedGPS`，包含前四组分 embedding/配比调制、第五组分 `Comp5GraphEncoder`、可选 input-derived `Fifth_class` embedding 和预测 MLP。 |
| `graphgps/component_vocab.py` | 前四组分的 embedding 词表和兼容旧数据的 atom-count 映射。新实验应使用输入 CSV 推导的词表。 |
| `graphgps/lrx_add/csv_pyg_five_multi.py` | 五组分 CSV 到 PyG 图数据的加载器；生成 canonical SMILES 词表、136 维 Morgan/RDKit 特征、Mordred 特征、比例和 `sample_uid`。 |
| `graphgps/lrx_add/mordred_lookup.py` | 按 canonical SMILES 查找 Mordred 特征；无匹配时返回零向量。 |
| `graphgps/lrx_add/compute_loss_multi4.py` | 四性质多任务损失实现。 |
| `graphgps/layer/gps_layer.py` | O12 第五组分使用的 GraphGPS layer（GINE + Transformer）。 |
| `loader_5.py` | 构建五个组分的 train/validation/test PyG loader。 |
| `graph_feature.py` | 基于 RDKit 的 SMILES 图特征构建。 |
| `graphgps/__init__.py`、`graphgps/config/config_gps.py`、`graphgps/loader/master_loader.py`、`graphgps/train/five_predict.py` | GraphGPS 插件注册、配置默认值、loader 注册和 O12 预测模式所依赖的框架胶水代码。 |
| `main.py` / `main_predict.py` | GraphGym 的通用训练入口 / checkpoint 推理入口。 |

### 2.2 配置与训练、测试、预测脚本

| 路径 | 用途 |
| --- | --- |
| `configs/GPS/O12_predict.yaml` | 旧版单 checkpoint O12 推理配置：`OneHotEmbedGPS`、四输出、辅助特征和 11 维 Mordred 已启用；默认读取 `input/20260703_sum_utf8.csv`，checkpoint 为 `results/O12_predict_checkpoint`。 |
| `scripts/O12_train_10_seeds.sh` | 在 split seed 110--135 上训练早期 O12 `core4` 多任务实验。 |
| `scripts/train_o12_strict_vocab_multitask_seed100_109.sh` | 在固定 split seed 100--109 上训练 `core4` 和 `norm2` O12 多任务模型；启用严格词表 `[2, 3, 2, 3]`。推荐作为可重复的多任务训练脚本。 |
| `scripts/train_o12_log1p_norm2_seed100_109.sh` | 在随机行切分 seed 100--109 上训练连续 `log1p` Norm O12；反变换到原尺度后按 validation MAE 选 checkpoint。 |
| `scripts/train_o12_log1p_norm2_fifth_group_seed200_209.sh` | 在第五组分身份互斥的 seed 200--209 上训练 OOD 版连续 `log1p` Norm O12。 |
| `scripts/train_o12_log1p_norm2_fifth_group_class_seed200_209.sh` | 上述 OOD 版的可选 `Fifth_class` embedding 候选；类别词表只由 input 建立。 |
| `scripts/train_o12_single_task_six_targets_seed100_109.sh` | 对六个性质逐一训练单输出 O12；共 6 × 10 个运行。 |
| `scripts/train_o12_o22_norm2_all_splits.sh` | 用全部已保存切分训练 O12/O22 的 `norm2` 对比；其中 O12 分支为 `concat_mlp`。 |
| `scripts/predict_o12_10seed_ensemble_on_predict_sets.sh` | 以两组各十个冻结 O12 checkpoint 批量预测三个 `raw/predict` 表，并合并 `core4`/`norm2` 结果。 |
| `scripts/diagnostics/run_fusion_head_experiment.py` | 上述训练脚本调用的核心可恢复训练器；接收切分清单、目标集和 O12 的 `concat_mlp` 设置，保存 selected-best checkpoint。 |
| `scripts/diagnostics/create_fifth_group_split_manifests.py` | 仅根据 input 的 canonical `Fifth_SMILE` 建立组间零交集的 train/validation/test 清单，不读取目标列。 |
| `scripts/diagnostics/summarize_o12_log1p_norm2.py` | 审计十个 log1p O12 checkpoint 的输入来源、切分、外部集隔离，并汇总原尺度连续 MAE/RMSE/R²。 |
| `scripts/diagnostics/fit_input_only_o12_continuous_blend.py` | 只用 input validation 连续 MAE 冻结旧 O12 与 log1p O12 的凸组合权重，不计算性质阈值。 |
| `scripts/diagnostics/apply_frozen_o12_continuous_blend.py` | 对同 seed 的两个冻结 GraphGPS 预测应用已冻结权重，再做十 seed 平均。 |
| `scripts/diagnostics/evaluate_o12_log1p_norm_feedback.py` | 模型和权重冻结后的最终外部评估；此时才额外计算 1.0 两侧一致率并绘制两个 Norm 性质散点图。 |
| `scripts/diagnostics/summarize_o12_strict_vocab_multitask.py` | 汇总严格词表 O12 的验证集和测试集指标。 |
| `scripts/diagnostics/summarize_selected_checkpoints_test.py` | 汇总任意完成的 `O12*_split*` 运行的 selected-best 测试指标。 |
| `scripts/diagnostics/predict_o12_10seed_ensemble.py` | 十模型 O12 集成的实现；校验列、屏蔽标签、产生均值与模型间标准差。 |
| `scripts/diagnostics/merge_o12_10seed_prediction_groups.py` | 合并相同输入表的 `core4` 和 `norm2` 集成预测。 |
| `scripts/diagnostics/plot_o12_ensemble_feedback.py` | 对有标签的六性质 O12 十模型集成结果计算 MAE/RMSE/R²，并输出带模型间标准差误差棒的 true-vs-predicted 散点图。 |
| `scripts/diagnostics/train_input_only_norm_log_rf.py` | 只读取 input 数据，以整 input 系列留一验证的连续 MAE 选择 `Norm_before`/`Norm_after` 的 log-RF 参数，再训练十 seed 集成；不读取 feedback，也不使用性质阈值。 |
| `scripts/diagnostics/predict_input_only_norm_log_rf.py` | 使用冻结的 input-only Norm 十 seed 集成预测任意同结构配方表。 |
| `scripts/diagnostics/evaluate_input_only_norm_feedback.py` | 模型冻结后的外部评估；在此阶段才计算 1.0 两侧一致率，并生成基线对比图和六性质混合预测表。 |
| `scripts/diagnostics/evaluate_saved_checkpoints_on_feedback.py` | 用有标签 feedback 表评测保存的 O12 checkpoint。 |
| `scripts/diagnostics/plot_o12_repeat10_test.py` | 绘制 repeat-10 O12 测试结果。 |
| `scripts/diagnostics/build_input_only_o12_residual_tree_head.py` | 在 O12 预测之上训练输入特征残差树头的诊断实验，不是基础 O12 结构。 |
| `scripts/predict_with_split_plots.py` | 对 O12 或 O12/O22 集成生成按 split 的预测和图；O12/O22 模式属于对比/集成。 |
| `scripts/diagnostics/predict_o12_o22_feedback_ensemble.py`、`scripts/diagnostics/combine_o12_o22_six_property_metrics.py`、`scripts/diagnostics/run_o12_o22_repeat_benchmark.py` | O12/O22 的 feedback 集成、指标合并和重复基准；不用于单独训练 O12。 |
| `python/val_average.py` | 从 `O12_split100`--`O12_split109` 的 `predictions.csv` 汇总指定 split（默认 validation）的 `Norm_before` / `Norm_after` MAE；适用于命名与其默认根目录一致的旧 `norm2` 运行。 |

`scripts/diagnostics/build_feedback_mean_prediction_comparison.py` 与 `scripts/diagnostics/summarize_frozen_validation_predictions.py` 也会读取 O12 的已保存预测，用于对比分析。`scripts/O22_train_10_seeds.sh` 虽引用 O12 的源配置，但训练的是 O22，故不属于 O12 训练入口。

### 2.3 原始数据、派生数据和特征文件

| 路径 | 含义 |
| --- | --- |
| `datasets_lrx/raw/input/20260703_sum_utf8.csv` | O12 主训练输入（700 行，UTF-8 CSV）。`20260703_sum.csv` 是同源版本；`.xlsx` 和 `*_deleted.csv` 为原始/清洗变体。训练脚本最终使用的已消毒版本由其源配置指定。 |
| `datasets_lrx/raw/feedback/20260703_validation.csv` | 97 行、有标签的外部反馈/验证集；用于独立 checkpoint 评测，不应混入训练。`20260703_validation.xlsx` 和 `20260724-validation(1).csv` 为同类反馈文件。 |
| `datasets_lrx/raw/feedback/new_validation.csv` | 26 行最终外部评估集；不得用于训练、切分、checkpoint/超参数/融合权重选择，只有冻结预测形成后才读取标签计算 MAE 和同侧率。 |
| `datasets_lrx/raw/predict/20260723-DOPE-peptide-predict2.csv` | 5,001 行待预测配方表。 |
| `datasets_lrx/raw/predict/20260723-library-single-predict.csv` | 第二个待预测配方表。 |
| `datasets_lrx/raw/predict/20260723-validation.xlsx` | 第三个待预测表（Excel）；批量预测脚本会读取它。`.~lock.*` 是办公软件临时锁文件，不能作为输入。 |
| `datasets_lrx/subset/processed/{train,val,test}{,_2,_3,_4,_5}.pt` | 五组分 PyG 的已处理缓存；由 loader 生成/使用，不手工编辑。`pre_filter.pt`、`pre_transform.pt` 是 PyG 元数据。 |
| `datasets_lrx/.cache/*O12*` | O12 推理期间生成的 loader cache；可删除并在下次运行时重建。 |
| `results/input_graphgps_optimization/features/mordred11_train_standardized.csv` | O12 所用的 74 个 canonical SMILES 的 11 维标准化 Mordred 查找表。 |
| `results/input_graphgps_optimization/features/mordred11_train_standardized.json` | 上述查找表的特征均值/标准差、训练来源和 SHA-256 元数据；集成预测必需。 |
| `results/input_graphgps_optimization/five_split_manifests/split_manifest_seed100.csv` ... `split_manifest_seed135.csv` | 固定的 36 份数据划分。列为 `sample_id`、`split`、`original_row_index`、`split_order`；100--109 是十 seed 冻结/严格词表训练用的主集合。 |
| `results/input_graphgps_optimization/fifth_group_split_manifests/` | seed 200--209 的 input-only 第五组分身份分组清单、SHA-256 inventory 和协议；每份清单的三个 split 之间 canonical `Fifth_SMILE` 交集为 0。 |
| `results/new_dataset_benchmark_20260713/input_sanitized_utf8.csv` | O12 源实验和词表来源数据；其路径保存在已保存配置及 Mordred metadata 中。 |
| `results/new_dataset_benchmark_20260713/split_manifest.csv` | O12 源实验的初始 train/val/test 切分清单。 |

训练 CSV 必须包含以下字段：

```text
ID
IL_SMILE, HL_SMILE, Chol_SMILE, PEG_SMILE, Fifth_SMILE
mol%_IL, mol%_HL, mol%_Chol, mol%_PEG, mol%_Fifth
EE_before, EE_after, Aerosolization_Efficiency, mRNA_Recovery_Efficiency,
Norm_before, Norm_after
```

纯预测表至少需要 `ID`、五个 `*_SMILE` 和五个 `mol%_*` 字段；`ID` 必须非空且唯一。预测器会将可能存在的六个标签置零后才交给 loader，避免标签泄漏。

### 2.4 已保存的 O12 模型、运行结果与产物

| 路径 | 含义 |
| --- | --- |
| `results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/` | 初始 O12 实验的完整运行目录。`source_config.yaml` 是原始基配置，`effective_config.yaml` 是实际生效配置，`summary.json`/`run_settings.json` 是汇总与可复现元数据。 |
| `.../checkpoints/selected_best.pt` | 初始 O12 所选择的最佳 checkpoint（结构总结中为 epoch 74）。同目录 `best_candidate_epoch_*.pt` 是训练过程中候选最佳 checkpoint。 |
| `.../{epoch_metrics,branch_statistics,fusion_statistics,gate_statistics,gradient_statistics,head_statistics,collapse_events}.csv` | 每轮性能、分支/融合/门控/梯度/头部统计和异常事件。`predictions.csv` 是保存 checkpoint 的预测；`resume_state.pt` 可恢复训练；`cache/`、`cache_build.log` 可重建。 |
| `results/O12_predict_checkpoint/43/ckpt/0.ckpt` | `configs/GPS/O12_predict.yaml` 指向的旧版 GraphGym checkpoint。`43/val/stats.json` 是其验证统计。 |
| `results/input_graphgps_optimization/O12-10-seeds-prediction-models/{core4,norm2}/O12_split100` ... `O12_split109` | 部署用冻结十模型集合：每个目标组各 10 个模型。每个 run 均包含 `checkpoints/selected_best.pt`、`effective_config.yaml`、`source_config.yaml`、`run_settings.json`、`summary.json`、`predictions.csv`、各诊断 CSV、`resume_state.pt` 和 cache/log。十模型集成脚本要求这些文件完整存在。 |
| `results/input_graphgps_optimization/o12_log1p_norm2_graphgps_10seed/` | 随机行切分的十个 log1p Norm GraphGPS 运行、input-only 审计、连续融合权重和冻结外部评估产物。 |
| `results/input_graphgps_optimization/o12_log1p_norm2_fifth_group_10seed/` | 第五组分身份互斥的十个 grouped-OOD GraphGPS 运行及连续指标。 |
| `results/input_graphgps_optimization/o12_log1p_norm2_fifth_group_class_10seed/` | 可选 grouped-OOD + input `Fifth_class` embedding 候选运行。 |
| `results/input_graphgps_optimization/O12-10-seeds-prediction-models/predict_ensemble_10seed/` | 三个预测表的集成产物。每个输入表子目录包含 `ensemble_mean_predictions_{core4,norm2}.csv`、`predictions_by_model_long_*.csv`、`ensemble_prediction_summary_*.csv`、`provenance_*.json`、staging CSV、Mordred lookup 和 cache；根目录包含 `run_summary_*.csv` 与合并结果。 |
| `results/input_graphgps_optimization/o12_strict_vocab_multitask_seed100_109/` | 严格词表训练的运行根目录：`O12_{core4,norm2}_split100` ... `split109`、`logs/` 和指标汇总 CSV。 |
| `results/input_graphgps_optimization/single_task_o12_six_targets/` | 六个单任务 O12 的运行根目录；子目录模式为 `O12_{target}_split100` ... `split109`，并含 `logs/` 与 `checkpoint_test_metrics/`。 |
| `results/input_graphgps_optimization/five_split_runs/O12_split110` ... `O12_split135` | `O12_train_10_seeds.sh` 产生的早期 core4 多任务运行。 |
| `results/input_graphgps_optimization/norm2_five_split_runs/O12_split100` ... `O12_split135` | `norm2` 的 O12 运行；每个 run 的内容与上表一致。该根目录也含 O22 对比运行。 |
| `results/input_graphgps_optimization/repeat10_o12_o22/`、`results/input_graphgps_optimization/calibration/O12_*`、`results/input_graphgps_optimization/feedback_inference/O12_*` | O12/O22 重复比较、校准和反馈集成产物；这些不是单模型 O12 checkpoint。 |
| `results/O12_split_predictions/`、`runs/*O12_predict*/` | 旧版单 checkpoint 的输出和运行日志；可依据需要清理。 |

## 3. 环境准备

使用包含 CUDA PyTorch、PyTorch Geometric、RDKit、GraphGym、pandas、scikit-learn、SciPy、Mordred、yacs 和（读取 Excel 所需）openpyxl 的 Python 环境。仓库的 `requirements.txt` 是依赖起点：

```bash
python -m pip install -r requirements.txt
python -c "import torch, torch_geometric, rdkit, pandas; print(torch.cuda.is_available())"
```

所有 shell 训练脚本默认使用 `/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python`。环境路径不同可通过 `PYTHON` 或 `PYTHON_BIN` 覆盖，例如：

```bash
PYTHON="$(command -v python)" bash scripts/train_o12_strict_vocab_multitask_seed100_109.sh
```

运行前检查 YAML 或脚本中的 `gpu_serial`、数据路径、`BASE_CONFIG`、`MORDRED`、`MANIFESTS` 和输出目录。GPU 内存不足时，不要同时启动两个 target group；可修改严格词表脚本末尾的并行逻辑，或直接调用训练器训练一个 group。

## 4. 训练与测试

### 4.1 推荐：十个固定切分上的多任务训练

这会在每个 seed 100--109 上训练 `core4` 与 `norm2`，并启用严格输入词表。脚本支持中断恢复：已有 `summary.json` 和 `predictions.csv` 的运行会跳过，有 `resume_state.pt` 的运行会恢复。

```bash
PYTHON="$(command -v python)" \
  bash scripts/train_o12_strict_vocab_multitask_seed100_109.sh
```

默认输出到 `results/input_graphgps_optimization/o12_strict_vocab_multitask_seed100_109/`。完成后查看：

```bash
python scripts/diagnostics/summarize_o12_strict_vocab_multitask.py \
  --runs-root results/input_graphgps_optimization/o12_strict_vocab_multitask_seed100_109
```

结果文件 `validation_test_metrics_by_seed_target.csv` 给出每个 seed、split 和目标的 MAE/RMSE/R²/Pearson/Spearman；`validation_test_metrics_target_average.csv` 和 `validation_test_metrics_macro_average.csv` 给出平均结果。训练器带 `--include-test`，因此每个 selected-best checkpoint 都会输出对应固定 test split 的预测；模型选择仍以 validation 为准。

### 4.2 单性质训练

若要分别拟合六个输出而非共享多任务头：

```bash
PYTHON="$(command -v python)" \
  bash scripts/train_o12_single_task_six_targets_seed100_109.sh
```

完成后查看测试汇总：

```bash
python scripts/diagnostics/summarize_selected_checkpoints_test.py \
  --runs-root results/input_graphgps_optimization/single_task_o12_six_targets \
  --output-dir results/input_graphgps_optimization/single_task_o12_six_targets/checkpoint_test_metrics
```

若使用旧的 `norm2_five_split_runs/O12_split100`--`O12_split109` 目录，还可仅汇总验证集 MAE：

```bash
python python/val_average.py \
  --root-folder results/input_graphgps_optimization/norm2_five_split_runs \
  --split val
```

### 4.3 指定一个切分的最小 smoke run

下面命令只训练一个 O12 `core4` 运行，适合检查 CUDA、CSV、描述符和切分是否可用。真实实验仍应使用第 4.1 节的固定十 seed 脚本。

```bash
python scripts/diagnostics/run_fusion_head_experiment.py \
  --config results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/source_config.yaml \
  --run-dir results/smoke_o12_core4_split100 \
  --target-set core4 \
  --split-manifest results/input_graphgps_optimization/five_split_manifests/split_manifest_seed100.csv \
  --fold split100 --group B --candidate O12Smoke \
  --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
  --seed 43 --base-lr 0.001 --weight-decay 1e-5 \
  --gt-dropout 0.1 --gt-attn-dropout 0.2 \
  --use-mordred-features --mordred-feature-dim 11 \
  --mordred-feature-path results/input_graphgps_optimization/features/mordred11_train_standardized.csv \
  --use-component-aux-features --strict-component-vocab \
  --execution-max-epochs 1 --include-test
```

检查 `results/smoke_o12_core4_split100/summary.json`、`epoch_metrics.csv`、`predictions.csv` 和 `checkpoints/selected_best.pt`。这是一次新实验，不会覆盖冻结模型。

### 4.4 旧版 GraphGym checkpoint 的单模型推理

`configs/GPS/O12_predict.yaml` 是旧推理流程，不会重新训练。它把整个输入表置于 validation loader 中输出一次预测：

```bash
python main_predict.py --cfg_file configs/GPS/O12_predict.yaml --repeat 1
```

运行前按需求修改 YAML 中的 `read_csv`、`pretrained.dir`、`property_num`、`component_vocab_source` 和 GPU 设置。该配置默认使用四输出 checkpoint `results/O12_predict_checkpoint`，与后述十模型 `core4`/`norm2` 集合不是同一套部署接口。

## 5. 冻结十模型集成预测

推荐通过现成包装脚本预测三个归档表：

```bash
PYTHON_BIN="$(command -v python)" \
  bash scripts/predict_o12_10seed_ensemble_on_predict_sets.sh
```

要预测自己的一个 CSV，只运行目标组对应的命令：

```bash
python scripts/diagnostics/predict_o12_10seed_ensemble.py \
  --model-root results/input_graphgps_optimization/O12-10-seeds-prediction-models \
  --input-files /absolute/path/to/formulations.csv \
  --output-root results/o12_custom_prediction \
  --target-group core4

python scripts/diagnostics/predict_o12_10seed_ensemble.py \
  --model-root results/input_graphgps_optimization/O12-10-seeds-prediction-models \
  --input-files /absolute/path/to/formulations.csv \
  --output-root results/o12_custom_prediction \
  --target-group norm2

python scripts/diagnostics/merge_o12_10seed_prediction_groups.py \
  --output-root results/o12_custom_prediction
```

每个目标组会将十个 `selected_best.pt` 的预测做**等权算术平均**；`pred_<target>_std_10models` 是十模型间的总体标准差，表示模型分歧，并非校准后的置信区间。输出 CSV 位于 `results/o12_custom_prediction/<输入文件名>/`。

## 6. 独立有标签测试与常见检查

在 feedback 数据上重新评测保存模型（不训练）可运行：

```bash
python scripts/diagnostics/evaluate_saved_checkpoints_on_feedback.py \
  --feedback-csv datasets_lrx/raw/feedback/20260703_validation.csv
```

执行前可用以下检查快速定位问题：

```bash
# 检查冻结模型集是否完整（每个 group 需 split100--109）
find results/input_graphgps_optimization/O12-10-seeds-prediction-models \
  -path '*/O12_split*/checkpoints/selected_best.pt' | wc -l

# 查看一个运行选中的 checkpoint、配置和指标
ls results/input_graphgps_optimization/O12-10-seeds-prediction-models/core4/O12_split100
```

常见失败原因：

- 自定义表缺少五个 SMILES 或五个比例列，或 `ID` 重复；预测器会在载入前报出缺失字段。
- 训练/推理 CSV 的 SMILES 不可由 RDKit 解析；第五组分空值使用占位符，但严格词表训练会拒绝前四组分缺失或不合法值。
- 使用了不匹配的 config 和 checkpoint；十模型预测必须读取同一 run 目录下的 `effective_config.yaml` 与 `selected_best.pt`。
- `mordred11_train_standardized.json` 或 `.csv` 缺失；十模型预测无法重建标准化 lookup。
- 没有 CUDA、GPU 编号不对或显存不足；调整 `accelerator`/`gpu_serial`，并先执行第 4.3 节 smoke run。

## 7. Input-only Norm 改进模型

该补充模型只拟合 `Norm_before` 和 `Norm_after`。训练及超参数选择均只使用 input；目标始终是连续 `log1p` 回归，按 input 系列留一验证的 MAE 选参，不将 1.0 阈值写入训练或选择过程。

```bash
python scripts/diagnostics/train_input_only_norm_log_rf.py \
  --input-csv datasets_lrx/raw/input/20260703_sum_utf8.csv \
  --manifests results/input_graphgps_optimization/five_split_manifests \
  --output-dir results/input_only_norm_log_rf_10seed

python scripts/diagnostics/predict_input_only_norm_log_rf.py \
  --model results/input_only_norm_log_rf_10seed/input_only_norm_log_rf_10seed.joblib \
  --input-csv /absolute/path/to/formulations.csv \
  --output-csv results/input_only_norm_log_rf_10seed/predictions.csv
```

`protocol.json` 保存 input SHA-256、候选网格、选中参数和“不读取外部验证集/不使用阈值”的协议声明。只有在模型冻结后，才可用 `evaluate_input_only_norm_feedback.py` 对有标签外部表报告 MAE 与额外的阈值两侧一致率。

## 8. 连续 log1p Norm GraphGPS 与最终外部评估

这一流程仍是 O12/`OneHotEmbedGPS`，不是第 7 节的 RF。训练损失位于连续 `log1p` 空间，但每轮 validation、checkpoint 选择和保存预测都会先用 `expm1` 返回原始 Norm 单位；训练器不包含 1.0 阈值指标。

随机行切分版本：

```bash
MAX_PARALLEL=3 \
  bash scripts/train_o12_log1p_norm2_seed100_109.sh

python scripts/diagnostics/summarize_o12_log1p_norm2.py \
  --runs-root results/input_graphgps_optimization/o12_log1p_norm2_graphgps_10seed
```

更严格的第五组分 OOD 版本先建立只依赖 input 身份的分组清单，再训练十 seed：

```bash
python scripts/diagnostics/create_fifth_group_split_manifests.py \
  --input-csv datasets_lrx/raw/input/20260703_sum_utf8.csv \
  --output-dir results/input_graphgps_optimization/fifth_group_split_manifests \
  --first-seed 200 --seeds 10

MAX_PARALLEL=3 \
  bash scripts/train_o12_log1p_norm2_fifth_group_seed200_209.sh

python scripts/diagnostics/summarize_o12_log1p_norm2.py \
  --runs-root results/input_graphgps_optimization/o12_log1p_norm2_fifth_group_10seed \
  --run-prefix O12Group_split --first-seed 200 --seed-count 10
```

十个 checkpoint 和所有 input-only 选择结果冻结之后，才对外部表做前向预测。预测器会先复制输入表，再将六个标签列全部置零；模型输出形成后，最终评估脚本才读取原表中的标签：

```bash
python scripts/diagnostics/predict_o12_10seed_ensemble.py \
  --model-root results/input_graphgps_optimization/o12_log1p_norm2_fifth_group_10seed \
  --direct-run-root --run-prefix O12Group_split \
  --first-seed 200 --seed-count 10 \
  --input-files datasets_lrx/raw/feedback/new_validation.csv \
  --output-root results/input_graphgps_optimization/o12_log1p_norm2_fifth_group_10seed/frozen_new_validation \
  --target-group norm2

python scripts/diagnostics/evaluate_o12_log1p_norm_feedback.py \
  --predictions results/input_graphgps_optimization/o12_log1p_norm2_fifth_group_10seed/frozen_new_validation/new_validation/ensemble_mean_predictions_norm2.csv \
  --baseline results/o12_10seed_feedback_new_validation/new_validation/ensemble_mean_predictions_norm2.csv \
  --output-dir results/input_graphgps_optimization/o12_log1p_norm2_fifth_group_10seed/frozen_new_validation/new_validation/evaluation
```

最终目录中的 `new_validation_norm_metrics.csv` 报告 MAE、RMSE、R² 和仅用于最终诊断的同侧率；`new_validation_o12_optimized_norm_scatter.{png,pdf}` 分别展示 `Norm_before` 与 `Norm_after`，`evaluation_protocol.json` 保存模型冻结和预测 provenance。
