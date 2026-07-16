import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# 基础配置
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

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

# 2. 6个性质名称
property_names = [
    "EE_before",
    "EE_after",
    "Aero_Efficiency",
    "Recovery",
    "Norm_before",
    "Norm_after"
]

# 3. 图片保存路径
save_dir = "/home/lrx/dataset/cooperation/gps/python/predict_error_analyses"
save_filename = "true_value_heatmap.png"
full_save_path = os.path.join(save_dir, save_filename)


# ===================== 数据读取与处理 =====================
def load_true_value_data(property_num=6):
    """
    读取指定数量的CSV的true value列，固定取95行数据
    :param property_num: 要显示的性质数量，可选6/4/2
    :return: 对应数量的true value矩阵、后2个性质的最大值（仅当property_num=6/2时有效）
    """
    # 校验输入参数合法性
    if property_num not in [2, 4, 6]:
        raise ValueError("❌ property_num只能是2、4、6中的一个！")

    true_value_list = []
    norm_max_vals = []  # 后2个性质的最大值

    # 根据property_num筛选要读取的文件索引
    if property_num == 6:
        read_indices = [0, 1, 2, 3, 4, 5]
    elif property_num == 4:
        read_indices = [0, 1, 2, 3]
    else:  # 2
        read_indices = [4, 5]

    for idx in read_indices:
        # 读取CSV并提取true value列，固定取前95行
        df = pd.read_csv(csv_paths[idx])
        true_vals = df['true value'].values[:95]

        # 数据清洗：替换空值为0
        true_vals = np.nan_to_num(true_vals, nan=0.0)

        if idx < 4:
            # 前4个性质：截断到0~100
            true_vals = np.clip(true_vals, 0, 100)
            true_value_list.append(true_vals)
        else:
            # 后2个性质：保留原始值，记录最大值
            true_value_list.append(true_vals)
            norm_max_vals.append(np.max(true_vals))

    # 转换为95行×property_num列的矩阵
    true_matrix = np.array(true_value_list).T
    print(f"✅ 数据读取完成！矩阵维度：{true_matrix.shape}")
    if norm_max_vals:
        print(
            f"📌 后2个性质最大值：{norm_max_vals if len(norm_max_vals) == 2 else [norm_max_vals[0]] if len(norm_max_vals) == 1 else '无'}")
    return true_matrix, norm_max_vals


# ===================== 绘制热力图（支持颜色反转控制） =====================
def plot_true_value_heatmap(true_matrix, norm_max_vals, property_num=6, reverse_=True):
    """
    绘制true value热力图
    :param true_matrix: true value矩阵
    :param norm_max_vals: 后2个性质的最大值
    :param property_num: 显示的性质数量（2/4/6）
    :param reverse_: 是否反转颜色（True=数值越大颜色越浅，False=数值越大颜色越深）
    """
    # 确定颜色后缀（_r表示反转）
    color_suffix = '_r' if reverse_ else ''

    # 创建画布（根据性质数量调整宽度）
    fig_width = {2: 5, 4: 8, 6: 10}[property_num]
    fig, ax = plt.subplots(figsize=(10, 14))

    # 根据property_num拆分数据并绘制
    if property_num == 6:
        # 6个性质：前4+后2
        data_100 = true_matrix[:, :4]
        data_5 = true_matrix[:, 4].reshape(-1, 1)
        data_6 = true_matrix[:, 5].reshape(-1, 1)

        # 绘制前4列（Reds系列）
        im1 = ax.imshow(data_100, cmap=f'Reds{color_suffix}', aspect='auto',
                        extent=[0, 4, 95, 0], vmin=0, vmax=100)
        # 绘制第5列（Blues系列）
        im2 = ax.imshow(data_5, cmap=f'Blues{color_suffix}', aspect='auto',
                        extent=[4, 5, 95, 0])
        # 绘制第6列（Greens系列）
        im3 = ax.imshow(data_6, cmap=f'Greens{color_suffix}', aspect='auto',
                        extent=[5, 6, 95, 0])

        # 设置x轴刻度和标签
        ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        ax.set_xticklabels(property_names, rotation=15)

        # 分隔线
        ax.axvline(4, color='k', lw=2)
        ax.axvline(5, color='k', lw=2)

        # 文本标注
        ax.text(2, -2.8, 'True Value (0~100)', ha='center', fontweight='bold', fontsize=11)
        ax.text(4.5, -2.8, f'True Value (0~{norm_max_vals[0]:.2f})', ha='center', fontweight='bold', fontsize=11)
        ax.text(5.5, -2.8, f'True Value (0~{norm_max_vals[1]:.2f})', ha='center', fontweight='bold', fontsize=11)

        # 颜色条
        cbar1 = plt.colorbar(im1, ax=ax, shrink=0.3, pad=0.02)
        cbar1.set_label('True Value (0~100)', fontsize=10)
        cbar2 = plt.colorbar(im2, ax=ax, shrink=0.3, pad=0.03)
        cbar2.set_label(f'True Value (Norm_before, 0~{norm_max_vals[0]:.2f})', fontsize=10)
        cbar3 = plt.colorbar(im3, ax=ax, shrink=0.3, pad=0.02)
        cbar3.set_label(f'True Value (Norm_after, 0~{norm_max_vals[1]:.2f})', fontsize=10)

    elif property_num == 4:
        # 4个性质：仅前4个
        data_100 = true_matrix
        im1 = ax.imshow(data_100, cmap=f'Reds{color_suffix}', aspect='auto',
                        extent=[0, 4, 95, 0], vmin=0, vmax=100)

        # 设置x轴刻度和标签
        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xticklabels(property_names[:4], rotation=15)

        # 文本标注
        ax.text(2, -2.8, 'True Value (0~100)', ha='center', fontweight='bold', fontsize=11)

        # 颜色条
        cbar1 = plt.colorbar(im1, ax=ax, shrink=0.3, pad=0.02)
        cbar1.set_label('True Value (0~100)', fontsize=10)

    else:  # 2个性质：仅后2个
        data_5 = true_matrix[:, 0].reshape(-1, 1)
        data_6 = true_matrix[:, 1].reshape(-1, 1)

        # 绘制第5列（Blues系列）
        im2 = ax.imshow(data_5, cmap=f'Blues{color_suffix}', aspect='auto',
                        extent=[0, 1, 95, 0])
        # 绘制第6列（Greens系列）
        im3 = ax.imshow(data_6, cmap=f'Greens{color_suffix}', aspect='auto',
                        extent=[1, 2, 95, 0])

        # 设置x轴刻度和标签
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(property_names[4:], rotation=15)

        # 分隔线
        ax.axvline(1, color='k', lw=2)

        # 文本标注
        ax.text(0.5, -2.8, f'True Value (0~{norm_max_vals[0]:.2f})', ha='center', fontweight='bold', fontsize=11)
        ax.text(1.5, -2.8, f'True Value (0~{norm_max_vals[1]:.2f})', ha='center', fontweight='bold', fontsize=11)

        # 颜色条
        cbar2 = plt.colorbar(im2, ax=ax, shrink=0.3, pad=0.03)
        cbar2.set_label(f'True Value (Norm_before, 0~{norm_max_vals[0]:.2f})', fontsize=10)
        cbar3 = plt.colorbar(im3, ax=ax, shrink=0.3, pad=0.02)
        cbar3.set_label(f'True Value (Norm_after, 0~{norm_max_vals[1]:.2f})', fontsize=10)

    # 通用图表配置
    color_desc = "Lighter = Larger Value" if reverse_ else "Deeper = Larger Value"
    ax.set_title(f'True Value Heatmap ({color_desc})', fontsize=14)
    ax.set_xlabel('Properties', fontsize=12)
    ax.set_ylabel('Data Index (1–95)', fontsize=12)
    ax.set_yticks(range(0, 95, 5), range(1, 96, 5))

    # 保存图片
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.tight_layout()
    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 热力图已保存至：{full_save_path}")


# ===================== 主程序执行 =====================
if __name__ == "__main__":
    try:
        # 可配置参数（核心修改点）
        property_num = 2  # 可选：2/4/6
        reverse_ = False  # 可选：True/False

        # 1. 读取true value数据
        true_mat, max_vals = load_true_value_data(property_num=property_num)
        # 2. 绘制并保存热力图
        plot_true_value_heatmap(true_mat, max_vals, property_num=property_num, reverse_=reverse_)

        # 输出执行结果提示
        prop_desc = {
            2: "最后两个性质（Norm_before、Norm_after）",
            4: "前四个性质（EE_before、EE_after、Aero_Efficiency、Recovery）",
            6: "全部六个性质"
        }[property_num]
        color_desc = "数值越大颜色越浅" if reverse_ else "数值越大颜色越深"
        print(f"\n🎉 热力图绘制完成！\n📋 显示范围：{prop_desc}\n🎨 颜色规则：{color_desc}")

    except Exception as e:
        print(f"❌ 程序执行出错：{str(e)}")