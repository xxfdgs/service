import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# 基础配置
plt.rcParams['axes.unicode_minus'] = False

# ===================== 核心配置项 =====================
# 1. 设置要输出的性质数量（2/4/6）
property_num = 2  # 可修改为 4 或 2

# 2. 6个CSV文件路径
csv_paths = [
    "/home/lrx/dataset/cooperation/gps/results/predict/single_property/cat/list0_EE_before/direct_layer1_batch4_single_cat_list0_EE_before/predicted_average.csv",
    "/home/lrx/dataset/cooperation/gps/results/predict/single_property/cat/list1_EE_after/direct_layer3_batch4_single_cat_list1_EE_after/predicted_average.csv",
    "/home/lrx/dataset/cooperation/gps/results/predict/single_property/cat/list2_Aero_Efficiency/direct_layer2_batch4_single_cat_list2_Aero_Efficiency/predicted_average.csv",
    "/home/lrx/dataset/cooperation/gps/results/predict/single_property/cat/list3_Recovery/direct_layer1_batch4_single_cat_list3_Recovery/predicted_average.csv",
    "/home/lrx/dataset/cooperation/gps/results/predict/single_property/cat/list4_Norm_before/v1/direct_layer1_batch4_single_cat_v1_list4_Norm_before/predicted_average.csv",
    "/home/lrx/dataset/cooperation/gps/results/predict/single_property/cat/list5_Norm_after/v1/direct_layer1_batch4_single_cat_v1_list5_Norm_after/predicted_average.csv"
]

# 3. 性质名称
property_names = [
    "EE_before",
    "EE_after",
    "Aero_Efficiency",
    "Recovery",
    "Norm_before",
    "Norm_after"
]

# 4. 目标保存路径（统一存储图片和CSV）
target_dir = "/home/lrx/dataset/cooperation/gps/python/predict_error_analyses"
# 5. 输出文件名（会自动追加property_num）
csv_filename = 'property_true_value_and_abs_error'+ str(property_num)+'.csv'
heatmap_filename = 'prediction_error_heatmap_all_abs'+str(property_num)  # 去掉.png后缀，后续拼接


# ===================== 路径校验与创建 =====================
def check_and_create_dir(dir_path):
    """检查目录是否存在，不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"📁 目录不存在，已创建：{dir_path}")
    else:
        print(f"📁 目录已存在：{dir_path}")


# ===================== 读取数据+计算误差（按指定数量筛选） =====================
def load_data_and_calculate_errors(selected_num):
    """
    读取指定数量的文件，计算对应性质的绝对误差，返回：
    - result_df: 包含数据索引、指定性质真实值、指定性质绝对误差的DataFrame
    - error_matrix: 仅指定性质绝对误差的矩阵（用于绘图）
    - selected_props: 选中的性质名称列表
    """
    result_dict = {
        "Data_Index": range(1, 96)  # 数据索引1-95
    }
    error_list = []

    # 根据selected_num选择对应的性质
    if selected_num == 6:
        selected_indices = range(6)  # 所有6个性质
    elif selected_num == 4:
        selected_indices = range(4)  # 前4个性质
    elif selected_num == 2:
        selected_indices = range(4, 6)  # 最后2个性质
    else:
        raise ValueError("property_num仅支持6、4、2三个值")

    selected_props = [property_names[i] for i in selected_indices]
    selected_paths = [csv_paths[i] for i in selected_indices]

    # 遍历选中的性质（计算绝对误差）
    for i, (path, prop) in enumerate(zip(selected_paths, selected_props)):
        # 读取原始CSV
        df = pd.read_csv(path)
        true_vals = df['true value'].values  # 真实值
        pred_vals = df['average predict'].values  # 预测值
        abs_err = np.abs(true_vals - pred_vals)  # 绝对误差

        # 存储真实值和绝对误差
        result_dict[f"{prop}_TrueValue"] = true_vals
        result_dict[f"{prop}_AbsError"] = abs_err

        # 存储误差值到列表（用于绘图）
        error_list.append(abs_err)

    # 转换为DataFrame
    result_df = pd.DataFrame(result_dict)
    # 误差矩阵（95行×N列，N为选中的性质数量）
    error_matrix = np.array(error_list).T

    return result_df, error_matrix, selected_props


# ===================== 导出CSV文件到指定路径 =====================
def export_to_csv(df, dir_path, filename, selected_num):
    """将DataFrame导出为CSV到指定目录，文件名追加selected_num"""
    # 给CSV文件名追加selected_num
    csv_name_with_num = filename.replace(".csv", f"_{selected_num}.csv")
    full_csv_path = os.path.join(dir_path, csv_name_with_num)
    # 保留6位小数，避免索引列，UTF-8编码
    df.round(6).to_csv(full_csv_path, index=False, encoding='utf-8')
    print(f"✅ CSV文件已保存至：{full_csv_path}")
    print(f"📊 数据量：{len(df)} 行 × {len(df.columns)} 列")


# ===================== 绘制热力图并保存到指定路径 =====================
def plot_and_save_heatmap(error_matrix, selected_props, dir_path, filename, selected_num):
    """绘制分尺度热力图（全部绝对误差）并保存到指定目录，文件名追加selected_num"""
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 14))

    # 根据选中的性质数量和类型绘制对应热力图
    if selected_num == 6:
        # 原始逻辑：前4列(0~100)、第5列、第6列
        data_100 = error_matrix[:, :4]  # 前4个性质
        data_5 = error_matrix[:, 4].reshape(-1, 1)  # 第5个（Norm_before）
        data_6 = error_matrix[:, 5].reshape(-1, 1)  # 第6个（Norm_after）

        im1 = ax.imshow(data_100, cmap='Reds', aspect='auto', extent=[0, 4, 95, 0], vmin=0, vmax=100)
        im2 = ax.imshow(data_5, cmap='Blues', aspect='auto', extent=[4, 5, 95, 0])
        im3 = ax.imshow(data_6, cmap='Greens', aspect='auto', extent=[5, 6, 95, 0])

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

        ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

    elif selected_num == 4:
        # 仅前4个性质（0~100尺度）
        data_100 = error_matrix[:, :4]
        im1 = ax.imshow(data_100, cmap='Reds', aspect='auto', extent=[0, 4, 95, 0], vmin=0, vmax=100)

        ax.text(2, -2.8, 'Abs Error (0~100)', ha='center', fontweight='bold', fontsize=11)

        # 仅前4个的颜色条
        cbar1 = plt.colorbar(im1, ax=ax, shrink=0.3, pad=0.02)
        cbar1.set_label('Absolute Error (0~100)', fontsize=10)

        ax.set_xticks([0.5, 1.5, 2.5, 3.5])

    elif selected_num == 2:
        # 仅最后2个性质（分别用蓝、绿色条）
        data_5 = error_matrix[:, 0].reshape(-1, 1)  # Norm_before
        data_6 = error_matrix[:, 1].reshape(-1, 1)  # Norm_after

        im2 = ax.imshow(data_5, cmap='Blues', aspect='auto', extent=[0, 1, 95, 0])
        im3 = ax.imshow(data_6, cmap='Greens', aspect='auto', extent=[1, 2, 95, 0])

        # 分隔线
        ax.axvline(1, color='k', lw=2)
        ax.text(0.5, -2.8, 'Abs Error', ha='center', fontweight='bold', fontsize=11)
        ax.text(1.5, -2.8, 'Abs Error', ha='center', fontweight='bold', fontsize=11)

        # 最后2个的颜色条
        cbar2 = plt.colorbar(im2, ax=ax, shrink=0.3, pad=0.03)
        cbar2.set_label('Absolute Error (Norm_before)', fontsize=10)
        cbar3 = plt.colorbar(im3, ax=ax, shrink=0.3, pad=0.02)
        cbar3.set_label('Absolute Error (Norm_after)', fontsize=10)

        ax.set_xticks([0.5, 1.5])

    # 通用图表标注
    ax.set_title(f'Prediction Error Heatmap (All Absolute Error) - {selected_num} Properties', fontsize=14)
    ax.set_xlabel('Properties', fontsize=12)
    ax.set_ylabel('Data Index (1–95)', fontsize=12)
    ax.set_xticklabels(selected_props, rotation=15)
    ax.set_yticks(range(0, 95, 5), range(1, 96, 5))

    # 给图片文件名追加selected_num并补全后缀
    full_heatmap_path = os.path.join(dir_path, f"{filename}_{selected_num}.png")
    plt.tight_layout()
    plt.savefig(full_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭画布释放内存
    print(f"✅ 热力图已保存至：{full_heatmap_path}")


# ===================== 主程序执行 =====================
if __name__ == "__main__":
    # 1. 校验并创建目标目录
    check_and_create_dir(target_dir)

    # 2. 读取数据+计算指定数量性质的绝对误差
    result_df, error_matrix, selected_props = load_data_and_calculate_errors(property_num)

    # 3. 导出CSV到指定路径（文件名追加property_num）
    export_to_csv(result_df, target_dir, csv_filename, property_num)

    # 4. 绘制并保存热力图到指定路径（文件名追加property_num）
    plot_and_save_heatmap(error_matrix, selected_props, target_dir, heatmap_filename, property_num)

    print(f"\n🎉 所有{property_num}个性质的文件已成功保存到指定目录！")