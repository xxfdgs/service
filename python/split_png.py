"""
@Name:  split_png.py
@Auth:  rongxing
@Date:  2026/4/17-下午2:48
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('TkAgg')

# =========================
# 需要修改的参数
# =========================
input_dir = r"/home/lrx/dataset/cooperation/gps/results/gps_predict"          # 存放10个csv文件的目录
output_dir = r"/home/lrx/dataset/cooperation/gps/results/gps_predict"     # 图片输出目录
property_num = 2                           # 只能是 2 或 4


# =========================
# 性质映射
# =========================
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
    raise ValueError("property_num 只能设置为 2 或 4")


def se(true_list, predict_list):
    """计算标准误差"""
    true_arr = np.array(true_list, dtype=float)
    pred_arr = np.array(predict_list, dtype=float)
    se_ = np.std((pred_arr - true_arr) / np.sqrt(len(pred_arr)))
    return se_


def average_accuracy(true_list, average_list):
    """计算平均相对误差"""
    result_list = []
    for true_val, avg_val in zip(true_list, average_list):
        if true_val != 0:
            cha_ = abs(true_val - avg_val) / abs(true_val)
            result_list.append(cha_)

    avg_acc = np.mean(result_list) if result_list else 0
    return avg_acc


def make_output_dir(path):
    os.makedirs(path, exist_ok=True)


def get_axis_reference(prop_name, true_value, pred_value):
    """根据性质和数据自动设置参考线范围"""
    true_value = np.array(true_value, dtype=float)
    pred_value = np.array(pred_value, dtype=float)

    data_min = min(np.min(true_value), np.min(pred_value))
    data_max = max(np.max(true_value), np.max(pred_value))

    if 'EE' in prop_name:
        ref_min, ref_max = 0, 100
    elif 'Efficiency' in prop_name:
        ref_min, ref_max = 0, 100
    elif 'Norm' in prop_name:
        ref_min, ref_max = 0, 3
    else:
        ref_min, ref_max = data_min, data_max

    ref_min = min(ref_min, data_min)
    ref_max = max(ref_max, data_max)

    padding = 0.05 * (ref_max - ref_min) if ref_max > ref_min else 1.0
    plot_min = ref_min - padding
    plot_max = ref_max + padding

    criterion_list = np.linspace(plot_min, plot_max, 200)
    return criterion_list, plot_min, plot_max


def save_picture(save_path, csv_df, prop_name, true_col, pred_col, file_stem):
    """绘制单个性质的预测值-真实值散点图"""
    fontsize = 18

    if true_col not in csv_df.columns:
        print(f"[警告] {file_stem} 中缺少列: {true_col}，跳过。")
        return None
    if pred_col not in csv_df.columns:
        print(f"[警告] {file_stem} 中缺少列: {pred_col}，跳过。")
        return None

    tmp_df = csv_df[[true_col, pred_col]].copy().dropna()

    if len(tmp_df) == 0:
        print(f"[警告] {file_stem} - {prop_name} 无有效数据，跳过。")
        return None

    true_value = tmp_df[true_col].astype(float).values
    pred_value = tmp_df[pred_col].astype(float).values

    # 指标
    r_2 = r2_score(true_value, pred_value)
    mae = mean_absolute_error(true_value, pred_value)
    se_value = se(true_value, pred_value)
    avg_acc = average_accuracy(true_value, pred_value)

    print(f'{file_stem} | {prop_name}: MAE={mae:.6f}, R²={r_2:.6f}, SE={se_value:.6f}, AvgRelErr={avg_acc:.6f}')

    # 画图
    fig, ax = plt.subplots(figsize=(8, 8))
    criterion_list, plot_min, plot_max = get_axis_reference(prop_name, true_value, pred_value)

    # y=x参考线
    ax.plot(criterion_list, criterion_list, linewidth=1.8, alpha=0.9, label='y = x')

    # 散点
    ax.scatter(true_value, pred_value, s=50, alpha=0.85, marker='o', label='Prediction')

    # 坐标范围
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect('equal', adjustable='box')

    # 标题
    title_label = f'{file_stem} - {prop_name}\nMAE={mae:.3f}  R²={r_2:.3f}  SE={se_value:.3f}'
    fig.suptitle(title_label, fontsize=fontsize)

    # 坐标轴
    ax.set_xlabel('Experiment', fontsize=16)
    ax.set_ylabel('Model-prediction', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=13)

    # 边框
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    ax.legend(loc='best', fontsize=13)
    ax.grid(alpha=0.25)

    # 保存
    save_name = f'{file_stem}_{prop_name}.png'
    save_file = os.path.join(save_path, save_name)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        "file_name": file_stem,
        "property": prop_name,
        "true_col": true_col,
        "pred_col": pred_col,
        "num_samples": len(true_value),
        "MAE": mae,
        "R2": r_2,
        "SE": se_value,
        "AvgRelErr": avg_acc,
        "save_file": save_file
    }


def process_one_csv(csv_path, output_dir):
    """处理单个csv文件"""
    file_name = os.path.basename(csv_path)
    file_stem = os.path.splitext(file_name)[0]

    print(f"\n开始处理: {file_name}")

    if not os.path.exists(csv_path):
        print(f"[警告] 文件不存在: {csv_path}")
        return []

    csv_df = pd.read_csv(csv_path)

    # 为每个csv单独创建输出子目录
    file_output_dir =output_dir  #os.path.join(output_dir, file_stem)
    # make_output_dir(file_output_dir)

    results = []
    for true_col, pred_col in PROPERTY_MAP.items():
        prop_name = true_col.replace("true_", "")
        result = save_picture(file_output_dir, csv_df, prop_name, true_col, pred_col, file_stem)
        if result is not None:
            results.append(result)

    # 保存该csv的指标汇总
    if results:
        result_df = pd.DataFrame(results)
        result_csv_path = os.path.join(file_output_dir, f"{file_stem}_metrics_summary.csv")
        result_df.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
        print(f"已保存指标汇总: {result_csv_path}")

    return results


def main():
    make_output_dir(output_dir)

    all_results = []

    # 依次处理 0~9 共10个文件
    for i in range(10):
        file_name = f"{i}test_true_pred_sum.csv"
        csv_path = os.path.join(input_dir, file_name)

        results = process_one_csv(csv_path, output_dir)
        all_results.extend(results)

    # 保存总汇总
    if all_results:
        all_df = pd.DataFrame(all_results)
        all_summary_path = os.path.join(output_dir, "all_metrics_summary.csv")
        all_df.to_csv(all_summary_path, index=False, encoding='utf-8-sig')
        print(f"\n全部文件处理完成，总汇总已保存到: {all_summary_path}")
    else:
        print("\n没有成功生成任何图片，请检查CSV文件路径和列名。")


if __name__ == "__main__":
    main()