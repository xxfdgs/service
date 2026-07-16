"""
@Name:  distribution.py
@Auth:  rongxing
@Date:  2026/3/3-下午4:13
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 基础配置（确保中文/负号正常显示，若仅需英文可保留）
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 英文无衬线字体，适配Linux/Mac/Windows

# ===================== 核心配置项 =====================
# 1. 数据文件路径（替换为你的数据文件路径，支持CSV/Excel）
data_file_path = "/home/lrx/dataset/cooperation/gps/datasets_lrx/raw/20251215-528-norm.csv"

# 2. 6个目标列名（需与数据文件中的列名完全一致）
target_columns = [
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
    "Norm_before",
    "Norm_after"
]

# 3. 各性质的数值范围（用于限定图表坐标轴）
value_ranges = {
    "EE_before": (0, 100),
    "EE_after": (0, 100),
    "Aerosolization_Efficiency": (0, 100),
    "mRNA_Recovery_Efficiency": (0, 100),
    "Norm_before": (0, 14),
    "Norm_after": (0, 18)
}

# 4. 输出图片路径（可自定义）
output_fig_path = "/home/lrx/dataset/cooperation/gps/python/distribution/property_distribution.png"


# ===================== 读取数据 =====================
def load_data(file_path, columns):
    """
    读取数据文件，提取指定列，返回清理后的DataFrame
    """
    # 读取数据（自动识别CSV/Excel）
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("仅支持CSV/Excel格式的文件！")

    # 检查目标列是否存在
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据文件中缺少以下列：{missing_cols}")

    # 提取目标列并去除空值/异常值
    df_target = df[columns].dropna()
    # 按数值范围过滤异常值（仅保留范围内的数据）
    for col in columns:
        min_val, max_val = value_ranges[col]
        df_target = df_target[(df_target[col] >= min_val) & (df_target[col] <= max_val)]

    print(f"✅ 数据读取完成！有效数据量：{len(df_target)} 行")
    return df_target


# ===================== 绘制数据分布图 =====================
def plot_property_distribution(df, ranges, output_path):
    """
    绘制6个性质的数据分布图（2行3列布局）
    每个子图包含：直方图（频数）+ 核密度曲线（概率密度）
    """
    # 创建2行3列的画布（尺寸适配6个子图）
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()  # 将2x3的轴数组展平为一维，方便遍历

    # 定义颜色列表（区分不同性质）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # 遍历每个性质绘制子图
    for idx, (col, color) in enumerate(zip(target_columns, colors)):
        ax = axes[idx]
        min_val, max_val = ranges[col]

        # 绘制直方图（bins根据范围自适应，密度=True用于和核密度曲线匹配）
        df[col].plot(
            kind='hist',
            bins=30,  # 分箱数，可调整
            density=True,
            alpha=0.7,
            color=color,
            ax=ax,
            edgecolor='black',
            linewidth=0.5
        )

        # 绘制核密度曲线（平滑的分布趋势）
        df[col].plot(
            kind='kde',
            color='black',
            linewidth=2,
            ax=ax
        )

        # 设置子图标题和标签
        ax.set_title(f'Distribution of {col}', fontsize=14, pad=10)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)

        # 限定X轴范围（匹配性质的数值范围）
        ax.set_xlim(min_val, max_val)
        # 优化刻度（避免拥挤）
        ax.tick_params(axis='both', labelsize=10)
        # 添加网格线（提升可读性）
        ax.grid(alpha=0.3, linestyle='--')

    # 调整子图间距，避免重叠
    plt.tight_layout()

    # 保存高清图片（300DPI）
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭画布释放内存
    print(f"✅ 数据分布图已保存至：{output_path}")


# ===================== 主程序执行 =====================
if __name__ == "__main__":
    try:
        # 1. 读取数据
        data_df = load_data(data_file_path, target_columns)

        # 2. 绘制并保存分布图
        plot_property_distribution(data_df, value_ranges, output_fig_path)

        print("\n🎉 所有操作完成！")
    except Exception as e:
        print(f"❌ 程序执行出错：{str(e)}")