"""
@Name:  predict_average.py
@Auth:  rongxing
@Date:  2023/4/25-下午4:04
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME}
average predicited datas from different pretrain_finetuning_models
and analysis function of average
output pictures and csv for 6 properties
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap


def read_data(serial, file_name, save_path, true_num):
    """读取单个模型的预测数据"""
    read_path = save_path + file_name + '/' + str(serial) + 'test_true_pred_sum.csv'
    read_data = pd.read_csv(read_path, index_col=0)[:true_num]
    # 确保列名无重复
    read_data = read_data.loc[:, ~read_data.columns.duplicated()]
    return read_data

def se(true_list, predict_list):
    """计算标准误差"""
    se_ = np.std(np.array((predict_list)-np.array(true_list))/np.sqrt(len(predict_list)))
    se = se_ / 1
    return se

def save_picture(path, csv, prop_name, true_col, pred_col):
    """为单个性质绘制预测vs真实值图"""
    fig, ax1 = plt.subplots(figsize=(8, 8))
    fontsize = 20

    # 提取当前性质的真实值和平均预测值
    pred_value = csv[pred_col]
    true_value = csv[true_col]

    # 根据性质名称适配坐标轴参考线（保留原有逻辑，可根据实际需求调整）
    criterion_list = []
    if 'EE' in prop_name:
        criterion_list = list(range(0, 101))  # 假设EE是百分比，0-100
    elif 'Efficiency' in prop_name:
        criterion_list = list(range(0, 101))  # 效率类指标0-100
    elif 'Norm' in prop_name:
        criterion_list = list(np.arange(0, 3.1, 0.2))  # 归一化指标0-1
    else:
        criterion_list = list(range(0, 101))  # 默认兜底

    # 绘制参考线（y=x）和预测点
    plt.plot(criterion_list, criterion_list, markersize=3, alpha=0.8, label='true',
             linewidth=1.5)
    plt.plot(true_value, pred_value, 'r+', markersize=8, alpha=0.8, label='pre')

    # 计算评估指标
    r_2 = r2_score(true_value, pred_value)
    MAE = mean_absolute_error(true_value, pred_value)

    # 标题（可根据需求调整单位，比如Efficiency无单位则去掉℃）
    title_label = f'{prop_name} MAE = {MAE:.2f}  R²={r_2:.3f}'
    print(f'{prop_name} MAE = {MAE}, R² = {r_2}')

    fig.suptitle(title_label, fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)

    # 设置坐标轴粗细
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Experiment', fontsize=14)
    plt.ylabel('Model-prediction', fontsize=14)

    # 保存图片（按性质命名）
    plt_save_name = f'predict_average_{prop_name}.png'
    plt.savefig(path + plt_save_name, dpi=100)
    plt.close()

def average_accuracy(true_list, average_list):
    """计算单个性质的平均准确率"""
    result_list = []
    for true_val, avg_val in zip(true_list, average_list):
        if true_val != 0:  # 避免除零错误
            cha_ = abs(true_val - avg_val) / true_val
            result_list.append(cha_)
    avg_acc = np.mean(result_list) if result_list else 0
    print(f'average accuracy = {avg_acc:.4f}')
    return avg_acc

def main_run_multi(file_name, save_path, loop_num, csv_file, property_num,
                   model_weights=None):
    """
    主函数：读取10个模型数据，计算6个性质的平均预测值，输出图片和CSV
    :param file_name: 文件夹名
    :param save_path: 保存根路径
    :param loop_num: 模型数量（设置为10）
    :param csv_file: 原始CSV路径
    """
    if property_num ==4:
        PROPERTY_MAP = {
            'true_EE_before': 'pred_EE_before',
            'true_EE_after': 'pred_EE_after',
            'true_Aero_Efficiency': 'pred_Aero_Efficiency',
            'true_Recovery_Efficiency': 'pred_Recovery_Efficiency'
        }
    elif property_num ==2:
        PROPERTY_MAP = {
            'true_Norm_before': 'pred_Norm_before',
            'true_Norm_after': 'pred_Norm_after'
        }
    # 读取原始文件，确定数据行数
    read_csv = pd.read_csv(csv_file)
    read_num = len(read_csv)  # 按行号匹配，无需依赖canonical smiles列

    # 初始化存储所有模型数据的字典（按性质分类）
    all_model_data = {}
    for true_col, pred_prefix in PROPERTY_MAP.items():
        all_model_data[true_col] = {'true': None, 'pred_models': []}

    if model_weights is None:
        model_weights = np.ones(loop_num, dtype=float) / loop_num
    else:
        model_weights = np.asarray(model_weights, dtype=float)
        if len(model_weights) != loop_num:
            raise ValueError('Number of model weights must match loop_num.')
        if np.any(model_weights < 0) or model_weights.sum() == 0:
            raise ValueError('Model weights must be non-negative with a positive sum.')
        model_weights = model_weights / model_weights.sum()

    # 读取模型预测数据
    for serial in range(loop_num):
        model_data = read_data(serial, file_name, save_path, read_num)
        # 遍历每个性质，提取当前模型的预测值
        for true_col, pred_prefix in PROPERTY_MAP.items():
            pred_col = pred_prefix  # 单个模型的预测列名
            if pred_col not in model_data.columns:
                raise ValueError(f'模型{serial}缺少列：{pred_col}')
            if all_model_data[true_col]['true'] is None:
                # 仅第一次读取真实值（所有模型真实值一致）
                all_model_data[true_col]['true'] = model_data[true_col].values
            # 存储当前模型的预测值
            all_model_data[true_col]['pred_models'].append(model_data[pred_col].values)

    # 初始化最终输出的DataFrame
    final_df = pd.DataFrame()
    # 先存储真实值
    for true_col in PROPERTY_MAP.keys():
        final_df[true_col] = all_model_data[true_col]['true']

    # 计算每个性质的平均预测值，并存储
    for true_col, pred_prefix in PROPERTY_MAP.items():
        pred_models = all_model_data[true_col]['pred_models']
        # 按验证集性能权重计算模型预测值
        pred_array = np.array(pred_models).T  # 转置：(样本数, 模型数)
        avg_pred = np.round(np.average(pred_array, axis=1,
                                       weights=model_weights), 2)
        # 定义平均预测列名
        avg_pred_col = f'{pred_prefix}_average'
        final_df[avg_pred_col] = avg_pred

        # 计算当前性质的评估指标
        true_list = all_model_data[true_col]['true']
        r2 = r2_score(true_list, avg_pred)
        mae = mean_absolute_error(true_list, avg_pred)
        se_val = se(true_list, avg_pred)
        print(f'\n===== 性质：{true_col} =====')
        print(f'R2 = {r2:.4f}, MAE = {mae:.4f}, SE = {se_val:.4f}')
        average_accuracy(true_list, avg_pred)

        # 绘制并保存当前性质的可视化图
        save_picture_path = save_path + file_name + '/'
        prop_name = true_col.replace('true_', '')  # 简化性质名称用于绘图
        save_picture(save_picture_path, final_df, prop_name, true_col, avg_pred_col)

    # 保存最终的平均预测结果CSV
    final_df = final_df.round(2)
    final_df.to_csv(save_path + file_name + '/predicted_average_6props.csv', index=False)
    pd.DataFrame({'model_index': range(loop_num), 'weight': model_weights}).to_csv(
        os.path.join(save_path, file_name, 'ensemble_weights.csv'), index=False)
    print('\n----- 所有性质处理完成 -----')

if __name__ == '__main__':
    # 配置参数（修改为10个模型）
    loop_num = 10  # 模型数量设置为10
    read_csv = '/home/lrx/dataset/cooperation/gps/datasets_lrx/raw/feedback/20260116_Prediction_7_96_top_center_bottom.csv'
    file_name = 'gps_predict_num4_2000'
    save_path = '/home/lrx/dataset/cooperation/gps/results/predict_202604/'

    # 执行主函数
    main_run_multi(file_name, save_path, loop_num, read_csv,4)
    print('----- all over -----')
