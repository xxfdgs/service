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
save_filename = "true_value_heatmap_reversed.png"
full_save_path = os.path.join(save_dir, save_filename)


# ===================== 数据读取与处理 =====================
def load_true_value_data():
    """读取6个CSV的true value列，固定取95行数据"""
    true_value_list = []
    norm_max_vals = []  # 后2个性质的最大值

    for idx, path in enumerate(csv_paths):
        # 读取CSV并提取true value列，固定取前95行
        df = pd.read_csv(path)
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

    # 转换为95行×6列的矩阵
    true_matrix = np.array(true_value_list).T
    print(f"✅ 数据读取完成！矩阵维度：{true_matrix.shape}")
    print(f"📌 后2个性质最大值：Norm_before={norm_max_vals[0]:.2f}, Norm_after={norm_max_vals[1]:.2f}")
    return true_matrix, norm_max_vals


# ===================== 绘制热力图（颜色反转：数值越大颜色越浅） =====================
def plot_true_value_heatmap(true_matrix, norm_max_vals):
    """绘制颜色反转的true value热力图（数值越大颜色越浅）"""
    # 拆分数据（与参考代码一致）
    data_100 = true_matrix[:, :4]  # 前4个性质（0~100）
    data_5 = true_matrix[:, 4].reshape(-1, 1)  # 第5个（Norm_before）
    data_6 = true_matrix[:, 5].reshape(-1, 1)  # 第6个（Norm_after）

    # 创建画布（尺寸与参考代码一致）
    fig, ax = plt.subplots(figsize=(10, 14))

    # 核心修改：色系后加_r实现反转（数值越大颜色越浅）
    # 前4列：Reds_r（原Reds反转）
    im1 = ax.imshow(data_100, cmap='Reds_r', aspect='auto',
                    extent=[0, 4, 95, 0], vmin=0, vmax=100)
    # 第5列：Blues_r（原Blues反转）
    im2 = ax.imshow(data_5, cmap='Blues_r', aspect='auto',
                    extent=[4, 5, 95, 0])
    # 第6列：Greens_r（原Greens反转）
    im3 = ax.imshow(data_6, cmap='Greens_r', aspect='auto',
                    extent=[5, 6, 95, 0])

    # 图表标注（与参考代码一致）
    ax.set_title('True Value Heatmap (Lighter = Larger Value)', fontsize=14)
    ax.set_xlabel('Properties', fontsize=12)
    ax.set_ylabel('Data Index (1–95)', fontsize=12)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    ax.set_xticklabels(property_names, rotation=15)
    ax.set_yticks(range(0, 95, 5), range(1, 96, 5))

    # 分隔线（与参考代码一致）
    ax.axvline(4, color='k', lw=2)
    ax.axvline(5, color='k', lw=2)
    # 文本标注（适配颜色反转）
    ax.text(2, -2.8, 'True Value (0~100)', ha='center', fontweight='bold', fontsize=11)
    ax.text(4.5, -2.8, f'True Value (0~{norm_max_vals[0]:.2f})', ha='center', fontweight='bold', fontsize=11)
    ax.text(5.5, -2.8, f'True Value (0~{norm_max_vals[1]:.2f})', ha='center', fontweight='bold', fontsize=11)

    # 独立颜色条（与参考代码一致，色系自动反转）
    cbar1 = plt.colorbar(im1, ax=ax, shrink=0.3, pad=0.02)
    cbar1.set_label('True Value (0~100)', fontsize=10)
    cbar2 = plt.colorbar(im2, ax=ax, shrink=0.3, pad=0.03)
    cbar2.set_label(f'True Value (Norm_before, 0~{norm_max_vals[0]:.2f})', fontsize=10)
    cbar3 = plt.colorbar(im3, ax=ax, shrink=0.3, pad=0.02)
    cbar3.set_label(f'True Value (Norm_after, 0~{norm_max_vals[1]:.2f})', fontsize=10)

    # 保存图片
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.tight_layout()
    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 颜色反转的热力图已保存至：{full_save_path}")


# ===================== 主程序执行 =====================
if __name__ == "__main__":
    try:
        # 1. 读取true value数据
        true_mat, max_vals = load_true_value_data()
        # 2. 绘制并保存颜色反转的热力图
        plot_true_value_heatmap(true_mat, max_vals)
        print("\n🎉 颜色反转热力图绘制完成！数值越大，颜色越浅")
    except Exception as e:
        print(f"❌ 程序执行出错：{str(e)}")