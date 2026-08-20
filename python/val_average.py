#!/usr/bin/env python3
"""计算多任务模型在验证集上的 MAE，并汇总 seed100–109 的均值与标准差。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = [
    #"EE_before",
    #"EE_after",
    #"Aerosolization_Efficiency",
    #"mRNA_Recovery_Efficiency",
    "Norm_before",
    "Norm_after",
]

SEEDS = list(range(100, 110))

REQUIRED_COLUMNS = {
    "split",
    "target",
    "y_true",
    "y_pred",
}


def calculate_mae(frame: pd.DataFrame) -> float:
    """计算 MAE，并拒绝空数据或非有限数值。"""
    if frame.empty:
        raise ValueError("Cannot calculate MAE from an empty DataFrame.")

    y_true = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(frame["y_pred"], errors="coerce").to_numpy(dtype=float)

    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if not valid_mask.all():
        invalid_count = int((~valid_mask).sum())
        raise ValueError(
            f"Found {invalid_count} rows with NaN or infinite y_true/y_pred."
        )

    return float(np.mean(np.abs(y_pred - y_true)))


def load_seed_predictions(
    input_path: Path,
    seed: int,
    split_name: str,
) -> pd.DataFrame:
    """读取并校验单个 seed 的预测文件。"""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Prediction file does not exist for seed {seed}: {input_path}"
        )

    frame = pd.read_csv(input_path)

    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{input_path} misses required columns: "
            f"{sorted(missing_columns)}"
        )

    frame = frame.loc[frame["split"].astype(str) == split_name].copy()

    if frame.empty:
        raise ValueError(
            f"No rows with split={split_name!r} in {input_path}"
        )

    frame["target"] = frame["target"].astype(str)

    unexpected_targets = sorted(set(frame["target"]) - set(TARGETS))
    if unexpected_targets:
        print(
            f"Warning: seed {seed} contains targets that will be ignored: "
            f"{unexpected_targets}"
        )

    return frame.loc[frame["target"].isin(TARGETS)].copy()


def calculate_seed_metrics(
    frame: pd.DataFrame,
    seed: int,
) -> list[dict[str, int | str | float]]:
    """计算单个 seed 的各 target 验证集 MAE。"""
    rows: list[dict[str, int | str | float]] = []

    observed_targets = set(frame["target"])
    missing_targets = set(TARGETS) - observed_targets

    if missing_targets:
        raise ValueError(
            f"Seed {seed} misses validation predictions for targets: "
            f"{sorted(missing_targets)}"
        )

    for target in TARGETS:
        target_frame = frame.loc[frame["target"] == target].copy()

        mae = calculate_mae(target_frame)

        rows.append(
            {
                "seed": seed,
                "target": target,
                "n_val": len(target_frame),
                "mae": mae,
            }
        )

    return rows


def build_target_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """汇总每个 target 在不同 seed 上的 MAE。"""
    summary = (
        seed_metrics.groupby("target", as_index=False, sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_val_min=("n_val", "min"),
            n_val_max=("n_val", "max"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_min=("mae", "min"),
            mae_max=("mae", "max"),
        )
    )

    target_order = {
        target: index
        for index, target in enumerate(TARGETS)
    }

    summary["target_order"] = summary["target"].map(target_order)
    summary = (
        summary.sort_values("target_order")
        .drop(columns="target_order")
        .reset_index(drop=True)
    )

    summary["mae_mean_std"] = summary.apply(
        lambda row: f"{row['mae_mean']:.6f} ± {row['mae_std']:.6f}",
        axis=1,
    )

    return summary


def build_seed_macro_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个 seed 的六任务宏平均 MAE。

    每个 target 权重相同，而不是按照各 target 的样本数加权。
    """
    summary = (
        seed_metrics.groupby("seed", as_index=False)
        .agg(
            n_targets=("target", "nunique"),
            macro_mae=("mae", "mean"),
        )
        .sort_values("seed")
        .reset_index(drop=True)
    )

    return summary


def build_pooled_metrics(
    all_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    计算合并所有 seed 预测行后的 pooled MAE。

    注意：
    pooled MAE 与“各 seed MAE 的算术平均”含义不同。
    当每个 seed 的样本数量完全相同时，两者通常相同或非常接近。
    """
    rows = []

    for target in TARGETS:
        target_frame = all_predictions.loc[
            all_predictions["target"] == target
        ]

        rows.append(
            {
                "target": target,
                "n_prediction_rows": len(target_frame),
                "pooled_mae": calculate_mae(target_frame),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--root-folder",
        type=Path,
        default=Path(
            "results/input_graphgps_optimization/norm2_five_split_runs"
        ),
        help="包含 O12+split100 等目录的根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果输出目录。默认保存到 root-folder/val_mae_summary。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="要评估的数据划分，默认为 val。",
    )
    parser.add_argument(
        "--allow-different-val-counts",
        action="store_true",
        help="允许不同 seed 的同一 target 验证样本数不同。",
    )
    args = parser.parse_args()

    root_folder = args.root_folder.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root_folder / "val_mae_summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, int | str | float]] = []
    prediction_frames: list[pd.DataFrame] = []

    for seed in SEEDS:
        input_path = (
            root_folder
            / f"O12_split{seed}"
            / "predictions.csv"
        )

        frame = load_seed_predictions(
            input_path=input_path,
            seed=seed,
            split_name=args.split,
        )

        metric_rows.extend(
            calculate_seed_metrics(
                frame=frame,
                seed=seed,
            )
        )

        frame = frame.copy()
        frame["seed"] = seed
        prediction_frames.append(frame)

    seed_metrics = pd.DataFrame(metric_rows)

    expected_row_count = len(SEEDS) * len(TARGETS)
    if len(seed_metrics) != expected_row_count:
        raise RuntimeError(
            f"Expected {expected_row_count} seed-target metric rows, "
            f"but found {len(seed_metrics)}."
        )

    if not seed_metrics.groupby("target")["seed"].nunique().eq(
        len(SEEDS)
    ).all():
        raise RuntimeError(
            "Not every target has metrics for all expected seeds."
        )

    # 检查同一 target 在不同 seed 中的验证样本数量是否一致。
    count_summary = (
        seed_metrics.groupby("target")["n_val"]
        .agg(["min", "max"])
    )
    inconsistent_counts = count_summary.loc[
        count_summary["min"] != count_summary["max"]
    ]

    if not inconsistent_counts.empty:
        message = (
            "Validation sample counts differ across seeds:\n"
            f"{inconsistent_counts.to_string()}"
        )

        if args.allow_different_val_counts:
            print(f"Warning: {message}")
        else:
            raise RuntimeError(
                message
                + "\nUse --allow-different-val-counts only if this is expected."
            )

    target_summary = build_target_summary(seed_metrics)
    seed_macro_summary = build_seed_macro_summary(seed_metrics)

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    pooled_metrics = build_pooled_metrics(all_predictions)

    # 每行一个 target，每列一个 seed 的 MAE。
    seed_mae_wide = seed_metrics.pivot(
        index="target",
        columns="seed",
        values="mae",
    ).reindex(
        index=TARGETS,
        columns=SEEDS,
    )

    seed_mae_wide.columns = [
        f"seed{seed}_mae"
        for seed in seed_mae_wide.columns
    ]

    seed_mae_wide = seed_mae_wide.reset_index()

    # 将 target 汇总信息合并到宽表末尾。
    combined_summary = seed_mae_wide.merge(
        target_summary[
            [
                "target",
                "mae_mean",
                "mae_std",
                "mae_mean_std",
                "n_val_min",
                "n_val_max",
            ]
        ],
        on="target",
        how="left",
        validate="one_to_one",
    )

    seed_metrics.to_csv(
        output_dir / "val_mae_by_seed_and_target.csv",
        index=False,
    )

    target_summary.to_csv(
        output_dir / "val_mae_target_summary.csv",
        index=False,
    )

    seed_macro_summary.to_csv(
        output_dir / "val_macro_mae_by_seed.csv",
        index=False,
    )

    pooled_metrics.to_csv(
        output_dir / "val_mae_pooled_predictions.csv",
        index=False,
    )

    combined_summary.to_csv(
        output_dir / "val_mae_seed_wide_summary.csv",
        index=False,
    )

    overall_macro_mean = float(
        seed_macro_summary["macro_mae"].mean()
    )
    overall_macro_std = float(
        seed_macro_summary["macro_mae"].std()
    )

    print("\nPer-target validation MAE across seed100–109:")
    print(
        target_summary[
            [
                "target",
                "mae_mean",
                "mae_std",
                "mae_min",
                "mae_max",
                "n_val_min",
                "n_val_max",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nMacro-average MAE by seed:")
    print(
        seed_macro_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print(
        "\nOverall macro MAE across seeds: "
        f"{overall_macro_mean:.6f} ± {overall_macro_std:.6f}"
    )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()