import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,r2_score
import os
import matplotlib
matplotlib.use('TkAgg')
# 设置图表字体（纯英文展示）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_pred_vs_true(csv_path, true_col, pred_col, save_path, title="Predicted vs True Values"):
    """
    绘制单组预测值vs真实值散点图，计算MAE并保存PNG
    调整样式：白色背景、取消虚线、对角线浅蓝色、数据点红色+；
    坐标范围：前四个性质0-100，最后两个0-8
    参数:
    csv_path (str): CSV文件路径
    true_col (str): 真实值列名
    pred_col (str): 预测值列名
    save_path (str): 图片保存路径（含.png）
    title (str): 图表标题
    """
    try:
        # 读取数据并校验列名
        df = pd.read_csv(csv_path)
        if true_col not in df.columns:
            raise ValueError(f"True column '{true_col}' not found")
        if pred_col not in df.columns:
            raise ValueError(f"Pred column '{pred_col}' not found")

        # 提取数据并去除缺失值
        true_vals = df[true_col].dropna()
        pred_vals = df[pred_col].loc[true_vals.index].dropna()

        # 判断有效数据量，为空则跳过
        if len(true_vals) == 0 or len(pred_vals) == 0:
            print(f"⚠️ 跳过 {true_col} vs {pred_col}：无有效非空数据")
            return None

        # 计算MAE
        mae = mean_absolute_error(true_vals, pred_vals)
        r_2 = r2_score(true_vals, pred_vals)
        print(r_2, mae)
        # 绘图 - 核心样式调整
        plt.figure(figsize=(8, 8))
        # 设置白色背景
        plt.gca().set_facecolor('white')
        plt.gcf().patch.set_facecolor('white')

        # 数据点：红色+、大小8、透明度0.8
        plt.plot(true_vals, pred_vals, 'r+', markersize=8, alpha=0.8, label='pre')

        # 按列名判断坐标范围
        top4_cols = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]
        # end2_cols =
        if true_col in top4_cols:
            # 前四个性质：坐标范围0-100
            x_min, x_max = -5, 105
            y_min, y_max = -5, 105
        # else:
        #     # 最后两个性质：坐标范围0-8
        #     x_min, x_max = 0, 10
        #     y_min, y_max = 0, 10
        else:
            # 最后两个性质：坐标范围0-8
            x_min, x_max = 0, 10
            y_min, y_max = 0, 10
        fontsize = 20
        # 对角线（y=x）：浅蓝色、无虚线、线宽1.5、透明度0.8、标记大小3
        plt.plot([x_min, x_max], [y_min, y_max], color='lightblue', markersize=3,
                 alpha=0.8, label='true', linewidth=1.5)

        # 坐标范围限制
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        # 样式调整：取消网格虚线（直接移除grid或设置linestyle为''）
        plt.xlabel('True Values', fontsize=14)
        plt.ylabel('Predicted Values', fontsize=14)
        plt.title(f'{title}\nMAE = {mae:.4f}', fontsize=fontsize)
        plt.legend(fontsize=20)
        # 取消网格虚线（如需保留网格但取消虚线，可改linestyle='-'，此处按需求取消）
        plt.grid(False)

        # 保存图片
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 保存成功: {save_path} | MAE: {mae:.4f}")
        return mae

    except FileNotFoundError:
        print(f"❌ 错误：文件 '{csv_path}' 不存在")
        return None
    except Exception as e:
        print(f"❌ 错误 ({true_col} vs {pred_col}): {str(e)}")
        return None

def batch_plot_pred_vs_true(csv_path, save_dir="pred_vs_true_plots/"):
    """
    批量生成所有预测值vs真实值的对比图（空值组自动跳过）
    参数:
    csv_path (str): CSV文件路径
    save_dir (str): 图片保存目录（自动创建）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 预测值-真实值列名对
    pred_true_pairs = [
        ("EE_before", "EE_before_pred"),
        ("EE_after", "EE_after_pred"),
        ("Aerosolization_Efficiency", "Aerosolization_Efficiency_pred"),
        ("mRNA_Recovery_Efficiency", "mRNA_Recovery_Efficiency_pred"),
        ("Norm_before", "Norm_before_pred"),
        ("Norm_after", "Norm_after_pred")
    ]

    # 存储所有MAE结果
    mae_results = {}

    # 批量绘图
    for true_col, pred_col in pred_true_pairs:
        # 生成保存文件名
        save_filename = f"{true_col.lower()}_vs_pred.png"
        save_path = os.path.join(save_dir, save_filename)
        # 生成图表标题
        plot_title = f"{true_col}: Predicted vs True Values"
        # 绘图并记录MAE
        mae = plot_pred_vs_true(csv_path, true_col, pred_col, save_path, plot_title)
        mae_results[true_col] = mae

    # 打印汇总结果
    print("\n===== MAE 汇总结果 =====")
    for col, mae in mae_results.items():
        if mae is None:
            print(f"{col}: 跳过（无有效数据/计算失败）")
        else:
            print(f"{col}: {mae:.4f}")

    return mae_results

# -------------------------- 示例使用 --------------------------
if __name__ == "__main__":
    input_csv = '/home/lrx/dataset/cooperation/gps/datasets_lrx/raw/feedback/20260116_Prediction_7_96_top_center_bottom.csv'
    save_path = '/home/lrx/dataset/cooperation/gps/python/png/'
    batch_plot_pred_vs_true(csv_path=input_csv, save_dir=save_path)