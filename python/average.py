"""
@Name:  average.py
@Auth:  rongxing
@Date:  2026/4/21-下午5:14
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""
"""
@Name:  predict_average.py
@Auth:  rongxing
@Date:  2023/4/25-下午4:04
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME}
average predicited datas from different pretrain_finetuning_models
and analysis function of average
output csv for properties (移除图片输出，直接读取10个模型CSV文件)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def read_data(file_path):
    """读取单个模型的预测数据（修改：直接读取指定路径的CSV文件）"""
    read_data = pd.read_csv(file_path, index_col=0)
    # 确保列名无重复
    read_data = read_data.loc[:, ~read_data.columns.duplicated()]
    return read_data


def se(true_list, predict_list):
    """计算标准误差"""
    se_ = np.std(np.array((predict_list) - np.array(true_list)) / np.sqrt(len(predict_list)))
    se = se_ / 1
    return se


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


def main_run_multi(data_path, save_path, property_num):
    """
    主函数：读取10个模型数据（0-9test_true_pred_sum.csv），计算性质的平均预测值，输出CSV
    :param data_path: 模型CSV文件所在路径（存放0-9test_true_pred_sum.csv的文件夹）
    :param save_path: 结果保存路径
    :param property_num: 性质数量（4或2）
    """
    # 定义性质映射关系
    if property_num == 4:
        PROPERTY_MAP = {
            'true_EE_before': 'pred_EE_before',
            'true_EE_after': 'pred_EE_after',
            'true_Aero_Efficiency': 'pred_Aero_Efficiency',
            'true_Recovery_Efficiency': 'pred_Recovery_Efficiency'
        }
    elif property_num == 2:
        PROPERTY_MAP = {
            'true_Norm_before': 'pred_Norm_before',
            'true_Norm_after': 'pred_Norm_after'
        }
    else:
        raise ValueError("property_num仅支持4或2")

    # 初始化存储所有模型数据的字典（按性质分类）
    all_model_data = {}
    for true_col, pred_col in PROPERTY_MAP.items():
        all_model_data[true_col] = {'true': None, 'pred_models': []}

    # 读取10个模型的预测数据（0到9）
    for serial in range(10):
        # 构造单个模型CSV文件路径
        csv_file_path = f"{data_path}/{serial}test_true_pred_sum.csv"
        try:
            model_data = read_data(csv_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"未找到模型文件：{csv_file_path}")

        # 遍历每个性质，提取当前模型的预测值
        for true_col, pred_col in PROPERTY_MAP.items():
            if pred_col not in model_data.columns:
                raise ValueError(f'模型{serial}缺少列：{pred_col}')
            if true_col not in model_data.columns:
                raise ValueError(f'模型{serial}缺少列：{true_col}')

            if all_model_data[true_col]['true'] is None:
                # 仅第一次读取真实值（所有模型真实值一致）
                all_model_data[true_col]['true'] = model_data[true_col].values

            # 存储当前模型的预测值（确保行数一致）
            pred_vals = model_data[pred_col].values[:len(all_model_data[true_col]['true'])]
            all_model_data[true_col]['pred_models'].append(pred_vals)

    # 初始化最终输出的DataFrame
    final_df = pd.DataFrame()
    # 先存储真实值
    for true_col in PROPERTY_MAP.keys():
        final_df[true_col] = all_model_data[true_col]['true']

    # 计算每个性质的平均预测值，并存储
    for true_col, pred_col in PROPERTY_MAP.items():
        pred_models = all_model_data[true_col]['pred_models']
        # 计算10个模型预测值的均值（按行平均）
        pred_array = np.array(pred_models).T  # 转置：(样本数, 模型数)
        avg_pred = np.round(np.mean(pred_array, axis=1), 2)  # 按样本平均，保留2位小数

        # 定义平均预测列名
        avg_pred_col = f'{pred_col}_average'
        final_df[avg_pred_col] = avg_pred

        # 计算当前性质的评估指标
        true_list = all_model_data[true_col]['true']
        r2 = r2_score(true_list, avg_pred)
        mae = mean_absolute_error(true_list, avg_pred)
        se_val = se(true_list, avg_pred)
        print(f'\n===== 性质：{true_col} =====')
        print(f'R2 = {r2:.4f}, MAE = {mae:.4f}, SE = {se_val:.4f}')
        average_accuracy(true_list, avg_pred)

    # 保存最终的平均预测结果CSV
    final_df = final_df.round(2)
    final_df.to_csv(f"{save_path}/predicted_average_results.csv", index=False)
    print('\n----- 所有性质处理完成，结果已保存 -----')


if __name__ == '__main__':
    # 配置参数
    data_path = '/home/lrx/dataset/cooperation/gps/results/predict_202604/gps_predict_num2_end'  # 存放0-9test_true_pred_sum.csv的文件夹路径
    save_path = '/home/lrx/dataset/cooperation/gps/results/predict_202604/gps_predict_num2_end'  # 结果保存路径
    property_num = 2  # 性质数量（4或2）

    # 执行主函数
    main_run_multi(data_path, save_path, property_num)
    print('----- all over -----')