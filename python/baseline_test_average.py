import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    input_path = Path(
        "results/input_graphgps_optimization/"
        "multitask_baselines_seed100_109/"
        "test_metrics/baseline_test_metrics_by_target.csv"
    )

    output_path = Path(
        "results/input_graphgps_optimization/"
        "multitask_baselines_seed100_109/"
        "test_metrics/baseline_test_metrics_average.csv"
    )

    models = [
        "GCN",
        "GIN",
        "MLP",
        "MPNN",
        "Transformer",
    ]

    targets = [
        "EE_before",
        "EE_after",
        "Aerosolization_Efficiency",
        "mRNA_Recovery_Efficiency",
        "Norm_before",
        "Norm_after",
    ]

    # 读取数据
    df = pd.read_csv(input_path)

    # 输入文件应包含：
    # model, target, mae, r2
    required_columns = {
        "model",
        "target",
        "mae",
        "r2",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"输入文件缺少列：{sorted(missing_columns)}\n"
            f"当前列名：{df.columns.tolist()}"
        )

    # 只保留指定模型和性质
    filtered_df = df[
        df["model"].isin(models)
        & df["target"].isin(targets)
    ].copy()

    # 检查每个模型-target组合是否有10个种子结果
    expected_groups = pd.MultiIndex.from_product(
        [models, targets],
        names=["model", "target"],
    )

    counts = (
        filtered_df
        .groupby(["model", "target"])
        .size()
        .reindex(expected_groups, fill_value=0)
    )

    invalid_counts = counts[counts != 10]

    if not invalid_counts.empty:
        raise ValueError(
            "以下模型-target组合的记录数不等于10：\n"
            f"{invalid_counts.to_string()}"
        )

    # 统计10个随机种子的均值与样本标准差
    summary_df = (
        filtered_df
        .groupby(["model", "target"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
    )

    # 将MAE与R²合并到同一个单元格
    # 单元格格式：
    # MAE均值 ± MAE标准差 / R²均值 ± R²标准差
    summary_df["metrics"] = summary_df.apply(
        lambda row: (
            f"{row['mae_mean']:.3f} ± {row['mae_std']:.3f}"
            f" / "
            f"{row['r2_mean']:.3f}"
        ),
        axis=1,
    )

    # 转换为宽表：
    # 行 = target
    # 列 = model
    # 值 = MAE与R²的组合字符串
    output_df = summary_df.pivot(
        index="target",
        columns="model",
        values="metrics",
    )

    # 按指定顺序排列行和列
    output_df = output_df.reindex(
        index=targets,
        columns=models,
    )

    # 将索引名称改为target
    output_df.index.name = "target"

    # 保存CSV
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(output_path)

    print(f"结果已保存至：{output_path}")
    print()
    print("每个单元格格式：MAE ± std / R² ± std")
    print()
    print(output_df.to_string())