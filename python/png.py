"""
@Name:  png.py
@Auth:  rongxing
@Date:  2026/1/23-下午1:04
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

def plot_true_vs_pred(csv_path, true_col='true value', pred_col='pred value',
                      title='True Value vs Predicted Value',
                      xlabel='True Value', ylabel='Predicted Value',
                      figsize=(8, 6), dpi=100):
    """
    读取CSV文件中的真实值和预测值列，绘制散点图并添加对角线参考线

    参数:
    --------
    csv_path : str
        CSV文件的路径（绝对路径或相对路径）
    true_col : str, 可选
        真实值列的列名，默认是'true value'
    pred_col : str, 可选
        预测值列的列名，默认是'pred value'
    title : str, 可选
        图表标题
    xlabel : str, 可选
        x轴标签
    ylabel : str, 可选
        y轴标签
    figsize : tuple, 可选
        图表尺寸，默认(8,6)
    dpi : int, 可选
        图表分辨率，默认100

    返回:
    --------
    None
        直接显示绘制好的图表
    """
    # 1. 读取CSV文件，处理文件读取异常
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{csv_path}'，请检查文件路径是否正确")
        return
    except Exception as e:
        print(f"读取文件时出错：{str(e)}")
        return

    # 2. 检查指定列是否存在
    if true_col not in df.columns:
        print(f"错误：文件中没有名为 '{true_col}' 的列")
        print(f"文件中可用的列名：{list(df.columns)}")
        return
    if pred_col not in df.columns:
        print(f"错误：文件中没有名为 '{pred_col}' 的列")
        print(f"文件中可用的列名：{list(df.columns)}")
        return

    # 3. 提取数据并去除空值
    true_vals = df[true_col].dropna()
    pred_vals = df[pred_col].dropna()

    # 确保两个数组长度一致（去除任意一个为空的行）
    valid_indices = true_vals.index.intersection(pred_vals.index)
    true_vals = true_vals.loc[valid_indices]
    pred_vals = pred_vals.loc[valid_indices]

    if len(true_vals) == 0:
        print("错误：没有有效的真实值和预测值数据（可能全为空）")
        return

    # 4. 设置绘图样式
    plt.figure(figsize=figsize, dpi=dpi)
    plt.style.use('seaborn-v0_8-whitegrid')  # 美观的样式

    # 5. 绘制散点图
    plt.scatter(true_vals, pred_vals, alpha=0.6, s=30, c='steelblue', edgecolors='white')

    # 6. 绘制对角线参考线（覆盖数据的最大/最小值范围）
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')

    # 7. 设置图表标签和标题
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.legend(fontsize=10)

    # 8. 保证x/y轴等比例，避免视觉误导
    plt.axis('equal')
    plt.tight_layout()  # 自动调整布局

    # 9. 显示图表
    plt.show()

# ------------------- 使用示例 -------------------
# 假设你的CSV文件路径是 './predictions.csv'
plot_true_vs_pred('/home/lrx/dataset/cooperation/gps/results/gps_predict/0test_true_pred_sum.csv')

# # 如果列名不是默认值（比如列名是'true'和'pred'），可以指定列名：
# plot_true_vs_pred('./predictions.csv', true_col='true', pred_col='pred')