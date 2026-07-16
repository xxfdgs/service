import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# 基础配置
plt.rcParams['axes.unicode_minus'] = False

# ===================== 核心配置项 =====================
# 1. 6个CSV文件路径
csv_paths = [
    "/home/lrx/dataset/cooperation/gps/results/predict/single_property/cat/list0_EE_before/direct_layer1_batch4_single_cat_list0_EE_before/predicted_average.csv",
    "/home/lrx/dataset/cooperation/gps/results/predict/single_property/cat/list1_EE_after/direct_layer3_batch4_single_cat_list1_EE_after/predicted_average.csv",
    "/home/lrx/dataset/cooperation/gps/results/predict/single_property/cat/list2_Aero_Efficiency/direct_layer2_batch4_single_cat_list2_Aero_Efficiency/predicted_average.csv",
    "/home/lrx/dataset/cooperation/gps/results/predict/single_property/cat/list3_Recovery/direct_layer1_batch4_single_cat_list3_Recovery/predicted_average.csv",
    "/home/lrx/dataset/cooperation/gps/results/predict/single_property/cat/list4_Norm_before/v1/direct_layer1_batch4_single_cat_v1_list4_Norm_before/predicted_average.csv",
    "/home/lrx/dataset/cooperation/gps/results/predict/single_property/cat/list5_Norm_after/v1/direct_layer1_batch4_single_cat_v1_list5_Norm_after/predicted_average.csv"
]

# 2. 性质名称
property_names = [
    "EE_before",
    "EE_after",
    "Aero_Efficiency",
    "Recovery",
    "Norm_before",
    "Norm_after"
]

# 3. 目标保存路径（统一存储图片和CSV）
target_dir = "/home/lrx/dataset/cooperation/gps/python/predict_error_analyses"
# 4. 输出文件名
csv_filename = "property_true_value_and_abs_error.csv"
heatmap_filename = "prediction_error_heatmap_all_abs.png"


# ===================== 路径校验与创建 =====================
def check_and_create_dir(dir_path):
    """检查目录是否存在，不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"📁 目录不存在，已创建：{dir_path}")
    else:
        print(f"📁 目录已存在：{dir_path}")


# ===================== 读取数据+计算误差（全部绝对误差） =====================
def load_data_and_calculate_errors():
    """
    读取所有文件，计算所有性质的绝对误差，返回：
    - result_df: 包含数据索引、各性质真实值、各性质绝对误差的DataFrame
    - error_matrix: 仅绝对误差的矩阵（用于绘图）
    """
    result_dict = {
        "Data_Index": range(1, 96)  # 数据索引1-95
    }
    error_list = []

    # 遍历每个性质（全部计算绝对误差）
    for i, (path, prop) in enumerate(zip(csv_paths, property_names)):
        # 读取原始CSV
        df = pd.read_csv(path)
        true_vals = df['true value'].values  # 真实值
        pred_vals = df['average predict'].values  # 预测值
        abs_err = np.abs(true_vals - pred_vals)  # 所有性质都用绝对误差

        # 存储真实值和绝对误差
        result_dict[f"{prop}_TrueValue"] = true_vals
        result_dict[f"{prop}_AbsError"] = abs_err

        # 存储误差值到列表（用于绘图）
        error_list.append(abs_err)

    # 转换为DataFrame
    result_df = pd.DataFrame(result_dict)
    # 误差矩阵（95行×6列）
    error_matrix = np.array(error_list).T

    return result_df, error_matrix


# ===================== 导出CSV文件到指定路径 =====================
def export_to_csv(df, dir_path, filename):
    """将DataFrame导出为CSV到指定目录"""
    full_csv_path = os.path.join(dir_path, filename)
    # 保留6位小数，避免索引列，UTF-8编码
    df.round(6).to_csv(full_csv_path, index=False, encoding='utf-8')
    print(f"✅ CSV文件已保存至：{full_csv_path}")
    print(f"📊 数据量：{len(df)} 行 × {len(df.columns)} 列")


# ===================== 绘制热力图并保存到指定路径 =====================
def plot_and_save_heatmap(error_matrix, dir_path, filename):
    """绘制分尺度热力图（全部绝对误差）并保存到指定目录"""
    # 拆分数据：前4列（0~100）、第5列、第6列（各自真实值范围）
    data_100 = error_matrix[:, :4]  # 前4个性质
    data_5 = error_matrix[:, 4].reshape(-1, 1)  # 第5个（Norm_before）
    data_6 = error_matrix[:, 5].reshape(-1, 1)  # 第6个（Norm_after）

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 14))

    # 绘制三部分独立热图（全部绝对误差，分尺度映射）
    im1 = ax.imshow(data_100, cmap='Reds', aspect='auto', extent=[0, 4, 95, 0], vmin=0, vmax=100)
    im2 = ax.imshow(data_5, cmap='Blues', aspect='auto', extent=[4, 5, 95, 0])
    im3 = ax.imshow(data_6, cmap='Greens', aspect='auto', extent=[5, 6, 95, 0])

    # 图表标注
    ax.set_title('Prediction Error Heatmap (All Absolute Error)', fontsize=14)
    ax.set_xlabel('Properties', fontsize=12)
    ax.set_ylabel('Data Index (1–95)', fontsize=12)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    ax.set_xticklabels(property_names, rotation=15)
    ax.set_yticks(range(0, 95, 5), range(1, 96, 5))

    # 分隔线
    ax.axvline(4, color='k', lw=2)
    ax.axvline(5, color='k', lw=2)
    ax.text(2, -2.8, 'Abs Error (0~100)', ha='center', fontweight='bold', fontsize=11)
    ax.text(4.5, -2.8, 'Abs Error', ha='center', fontweight='bold', fontsize=11)
    ax.text(5.5, -2.8, 'Abs Error', ha='center', fontweight='bold', fontsize=11)

    # 独立颜色条
    cbar1 = plt.colorbar(im1, ax=ax, shrink=0.3, pad=0.02)
    cbar1.set_label('Absolute Error (0~100)', fontsize=10)
    cbar2 = plt.colorbar(im2, ax=ax, shrink=0.3, pad=0.03)
    cbar2.set_label('Absolute Error (Norm_before)', fontsize=10)
    cbar3 = plt.colorbar(im3, ax=ax, shrink=0.3, pad=0.02)
    cbar3.set_label('Absolute Error (Norm_after)', fontsize=10)

    # 保存图片到指定路径
    full_heatmap_path = os.path.join(dir_path, filename)
    plt.tight_layout()
    plt.savefig(full_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭画布释放内存
    print(f"✅ 热力图已保存至：{full_heatmap_path}")


# ===================== 主程序执行 =====================
if __name__ == "__main__":
    # 1. 校验并创建目标目录
    check_and_create_dir(target_dir)

    # 2. 读取数据+计算所有性质的绝对误差
    result_df, error_matrix = load_data_and_calculate_errors()

    # 3. 导出CSV到指定路径
    export_to_csv(result_df, target_dir, csv_filename)

    # 4. 绘制并保存热力图到指定路径
    plot_and_save_heatmap(error_matrix, target_dir, heatmap_filename)

    print("\n🎉 所有文件已成功保存到指定目录！")