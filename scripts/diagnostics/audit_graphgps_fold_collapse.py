#!/usr/bin/env python3
"""Static, sample-id-aligned audit for a GraphGPS fold-collapse investigation.

The script intentionally reads only the deduplicated training data, manifests,
feature artifacts, and already-built per-fold caches.  It never changes a CSV,
cache, checkpoint, or model parameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
]
SPLITS = ("train", "val", "test")
QUANTILES = (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99)
RATIO_COLUMNS = ("mol%_IL", "mol%_HL", "mol%_Chol", "mol%_PEG", "mol%_Fifth")


def number(value: float | np.floating) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def numeric_summary(values: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(values, errors="coerce")
    non_missing = values.dropna()
    result: dict[str, float | int] = {
        "n_non_missing": int(non_missing.size),
        "missing_rate": float(values.isna().mean()),
        "mean": number(non_missing.mean()),
        "std": number(non_missing.std(ddof=1)),
        "min": number(non_missing.min()),
        "max": number(non_missing.max()),
    }
    for q in QUANTILES:
        result[f"q{int(q * 100):02d}"] = number(non_missing.quantile(q))
    return result


def cache_file(cache_root: Path, fold: int, split: str, component: int) -> Path:
    pattern = (
        f".cache/double_deduplicated_graphgps_cv_formula_identity_group_cv_"
        f"fold_{fold}_seed_0_seed_0/subset/processed/{split}"
    )
    suffix = "" if component == 1 else f"_{component}"
    path = cache_root / f"{pattern}{suffix}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_collated(path: Path) -> tuple[object, dict[str, torch.Tensor]]:
    data, slices = torch.load(path, map_location="cpu", weights_only=False)
    return data, slices


def per_graph(data: object, slices: dict[str, torch.Tensor], key: str) -> list[torch.Tensor]:
    values = getattr(data, key)
    bounds = slices[key].tolist()
    return [values[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def label_matrix(data: object, slices: dict[str, torch.Tensor]) -> np.ndarray:
    columns = []
    for key in ("y", "y1", "y2", "y3"):
        columns.append(np.asarray([float(item.reshape(-1)[0]) for item in per_graph(data, slices, key)]))
    return np.column_stack(columns)


def cache_stats(data: object, slices: dict[str, torch.Tensor]) -> dict[str, object]:
    samples = np.asarray([int(item.reshape(-1)[0]) for item in per_graph(data, slices, "sample_uid")])
    labels = label_matrix(data, slices)
    nodes = np.asarray([int(item.numel() // data.x.shape[1]) for item in per_graph(data, slices, "x")])
    edges = np.asarray([int(item.shape[1]) for item in per_graph(data, slices, "edge_index")])
    ratios = np.asarray([float(item.reshape(-1)[0]) for item in per_graph(data, slices, "ratio")])
    masks = np.asarray([float(item.reshape(-1)[0]) for item in per_graph(data, slices, "mask")])
    mordred = np.vstack([item.reshape(-1).numpy() for item in per_graph(data, slices, "mordred_feat")])
    return {
        "n": int(samples.size), "sample_uid": samples, "labels": labels,
        "nodes": nodes, "edges": edges, "ratios": ratios, "masks": masks,
        "mordred": mordred, "x_dtype": str(data.x.dtype), "x_shape": tuple(data.x.shape),
        "mordred_dtype": str(data.mordred_feat.dtype), "mordred_shape": tuple(data.mordred_feat.shape),
        "has_nonfinite": bool(
            not torch.isfinite(data.x.float()).all()
            or not torch.isfinite(data.mordred_feat.float()).all()
            or not torch.isfinite(torch.as_tensor(labels)).all()
        ),
    }


def component_count(frame: pd.DataFrame) -> pd.Series:
    columns = [f"canonical_component_{index}_smiles" for index in range(1, 6)]
    return frame[columns].notna().sum(axis=1)


def write_report(path: Path, status: str, reasons: list[str], group_leakage: pd.DataFrame,
                 cache_rows: pd.DataFrame, feature_rows: pd.DataFrame) -> None:
    bad_cache = cache_rows.loc[cache_rows.status != "PASS"]
    bad_features = feature_rows.loc[feature_rows.status != "PASS"]
    lines = [
        "# Fold 4 静态数据与缓存审计", "",
        f"**结论：{status}**。", "",
        "## 检查范围", "",
        "fold_0、fold_1 与 fold_4 的 train/validation/outer-test manifest、当前去重 CSV、图缓存和 11 维 Mordred 缓存。所有关联以 `sample_id` 或 manifest 的 `original_row_index` 完成。", "",
        "## DATA_ERROR 判定", "",
    ]
    lines.extend([f"- {reason}" for reason in reasons] or ["- 未触发 DATA_ERROR 条件。"])
    lines += ["", "## 结果摘要", ""]
    lines += [
        f"- manifest group 泄漏行数：{int((group_leakage.leaked_group_count > 0).sum())}。",
        f"- 缓存/样本对齐失败行数：{len(bad_cache)}。",
        f"- 描述符/图数值异常行数：{len(bad_features)}。",
        "- 四目标标签在缓存中按 `y, y1, y2, y3` 顺序保存，已与 CSV 目标顺序及数值（/100）逐项核对。",
        "- 详细数值见同目录 CSV；此审计不使用 outer-test 标签作任何训练或超参数选择。",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/fold4_collapse_audit/static_audit")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 4])
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = ROOT / "results/deduplicated_rebaseline"
    data = pd.read_csv(source_root / "data_audit/dataset_with_sample_id.csv", dtype={"sample_id": str})
    if data.sample_id.duplicated().any():
        raise RuntimeError("The audit dataset has duplicate sample_id values.")
    data_by_index = data.set_index("original_row_index", verify_integrity=True)
    mordred_lookup = pd.read_csv(source_root / "artifacts/mordred_11_lookup.csv")
    lookup_smiles = set(mordred_lookup.smiles.astype(str))
    graph_cache = pd.read_csv(source_root / "artifacts/graph_cache.csv", dtype={"sample_id": str})

    fold_split_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    graph_rows: list[dict[str, object]] = []
    cache_rows: list[dict[str, object]] = []
    unknown_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    data_errors: list[str] = []

    for fold in args.folds:
        manifest = pd.read_csv(source_root / "manifests/formula_identity_group_cv" / f"fold_{fold}.csv", dtype={"sample_id": str})
        if manifest.sample_id.duplicated().any():
            data_errors.append(f"fold_{fold}: manifest has duplicate sample_id values")
        if manifest.original_row_index.duplicated().any():
            data_errors.append(f"fold_{fold}: manifest has duplicate original_row_index values")
        if set(manifest.sample_id) != set(data.sample_id):
            data_errors.append(f"fold_{fold}: manifest sample_id universe differs from audit CSV")
        if len(manifest) != len(data):
            data_errors.append(f"fold_{fold}: manifest row count differs from audit CSV")
        group_sets = {split: set(manifest.loc[manifest.split == split, "group_id"]) for split in SPLITS}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            leaked = group_sets[left] & group_sets[right]
            leakage_rows.append({"fold": f"fold_{fold}", "left_split": left, "right_split": right, "leaked_group_count": len(leaked), "status": "PASS" if not leaked else "FAIL"})
            if leaked:
                data_errors.append(f"fold_{fold}: group leakage between {left} and {right}")

        cache_root = source_root / "graphgps_cv/isolated_cache_roots" / f"fold_{fold}_seed_0"
        for split in SPLITS:
            section = manifest.loc[manifest.split == split].copy()
            frame = section.merge(data, on=["sample_id", "original_row_index"], how="left", validate="one_to_one")
            if len(frame) != len(section) or frame[TARGETS].isna().all(axis=None):
                data_errors.append(f"fold_{fold}/{split}: manifest-to-dataset alignment failed")
            component_counts = component_count(frame)
            ratio_sum = frame.loc[:, RATIO_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            fifth_missing = frame["canonical_component_5_smiles"].isna() | frame["canonical_component_5_smiles"].astype(str).eq("nan")
            fold_split_rows.append({
                "fold": f"fold_{fold}", "split": split, "n_samples": len(frame), "n_groups": frame.group_id.nunique(),
                "sample_id_unique": not frame.sample_id.duplicated().any(), "fifth_component_missing_rate": float(fifth_missing.mean()),
                "ratio_sum_not_100_rate": float((~np.isclose(ratio_sum, 100.0, atol=1e-6)).mean()),
                "component_count_mean": float(component_counts.mean()), "component_count_min": int(component_counts.min()), "component_count_max": int(component_counts.max()),
                "formula_identity_count": int(frame.formula_identity_key.nunique()), "formula_ratio_count": int(frame.formula_ratio_key.nunique()),
            })
            for target in TARGETS:
                target_rows.append({"fold": f"fold_{fold}", "split": split, "target": target, **numeric_summary(frame[target])})

            train = manifest.loc[manifest.split == "train"].merge(data, on=["sample_id", "original_row_index"], how="left")
            for position in range(1, 6):
                column = f"canonical_component_{position}_smiles"
                known = set(train[column].dropna().astype(str))
                values = frame[column].dropna().astype(str)
                # The data audit uses <missing> as a stable canonical key;
                # the loader deliberately converts that sentinel to [Fr] when
                # making the masked fifth-component graph and descriptor.
                lookup_values = values.replace("<missing>", "[Fr]")
                unknown_rows.append({
                    "fold": f"fold_{fold}", "split": split, "component_position": position,
                    "n_non_missing": len(values), "unknown_to_train_count": int((~values.isin(known)).sum()),
                    "unknown_to_train_rate": float((~values.isin(known)).mean()) if len(values) else 0.0,
                    "missing_mordred_lookup_count": int((~lookup_values.isin(lookup_smiles)).sum()),
                    "status": "PASS" if lookup_values.isin(lookup_smiles).all() else "FAIL",
                })

            expected_source_indices = set(section.original_row_index.astype(int))
            for component in range(1, 6):
                path = cache_file(cache_root, fold, split, component)
                collated, slices = load_collated(path)
                stats = cache_stats(collated, slices)
                actual_source_indices = set(stats["sample_uid"].tolist())
                labels = stats["labels"] * 100.0
                expected_labels = data_by_index.loc[stats["sample_uid"], TARGETS].to_numpy(dtype=float)
                max_label_error = float(np.max(np.abs(labels - expected_labels))) if len(labels) else 0.0
                alignment_ok = (
                    stats["n"] == len(section)
                    and actual_source_indices == expected_source_indices
                    and max_label_error < 1e-4
                    and not stats["has_nonfinite"]
                )
                cache_rows.append({
                    "fold": f"fold_{fold}", "split": split, "component_position": component,
                    "cache_file": str(path), "n_graphs": stats["n"], "expected_n_graphs": len(section),
                    "sample_uid_set_match": actual_source_indices == expected_source_indices,
                    "max_abs_label_error_after_x100": max_label_error, "has_nonfinite": stats["has_nonfinite"],
                    "x_dtype": stats["x_dtype"], "x_shape": json.dumps(stats["x_shape"]),
                    "mordred_dtype": stats["mordred_dtype"], "mordred_shape": json.dumps(stats["mordred_shape"]),
                    "status": "PASS" if alignment_ok else "FAIL",
                })
                if not alignment_ok:
                    data_errors.append(f"fold_{fold}/{split}/component_{component}: cache alignment, label, or finite-value failure")

                graph_rows.append({
                    "fold": f"fold_{fold}", "split": split, "component_position": component,
                    **{f"nodes_{key}": value for key, value in numeric_summary(pd.Series(stats["nodes"])).items()},
                    **{f"edges_{key}": value for key, value in numeric_summary(pd.Series(stats["edges"])).items()},
                    "ratio_mean": float(np.mean(stats["ratios"])), "ratio_std": float(np.std(stats["ratios"], ddof=1)),
                    "masked_rate": float(np.mean(stats["masks"] > 0)),
                })
                for index in range(stats["mordred"].shape[1]):
                    values = stats["mordred"][:, index]
                    finite = np.isfinite(values)
                    extreme = np.abs(values[finite]) > 1e6
                    feature_rows.append({
                        "fold": f"fold_{fold}", "split": split, "component_position": component, "feature_index": index,
                        "mean": float(np.nanmean(values)), "std": float(np.nanstd(values, ddof=1)),
                        "zero_variance": bool(np.nanstd(values) < 1e-12), "nan_count": int(np.isnan(values).sum()),
                        "inf_count": int(np.isinf(values).sum()), "extreme_abs_gt_1e6_count": int(extreme.sum()),
                        "status": "PASS" if finite.all() else "FAIL",
                    })
                if not np.isfinite(stats["mordred"]).all():
                    data_errors.append(f"fold_{fold}/{split}/component_{component}: non-finite Mordred feature")

            # Batch validity is derived from the exact cached first-component labels.
            first, first_slices = load_collated(cache_file(cache_root, fold, split, 1))
            labels = label_matrix(first, first_slices)
            for start in range(0, len(labels), 8):
                stop = min(start + 8, len(labels))
                for index, target in enumerate(TARGETS):
                    valid_count = int(np.isfinite(labels[start:stop, index]).sum())
                    batch_rows.append({"fold": f"fold_{fold}", "split": split, "batch_index": start // 8, "target": target, "batch_size": stop - start, "valid_label_count": valid_count, "too_few": valid_count < 1})

        # Validate artifacts independently of the PyG caches.
        graph_part = graph_cache.loc[graph_cache.sample_id.isin(manifest.sample_id)]
        if len(graph_part) != len(manifest) * 5 or graph_part.duplicated(["sample_id", "component_position"]).any():
            data_errors.append(f"fold_{fold}: graph_cache does not map one-to-one to manifest sample_id/component_position")

    fold_split = pd.DataFrame(fold_split_rows)
    targets = pd.DataFrame(target_rows)
    batches = pd.DataFrame(batch_rows)
    features = pd.DataFrame(feature_rows)
    graphs = pd.DataFrame(graph_rows)
    caches = pd.DataFrame(cache_rows)
    unknowns = pd.DataFrame(unknown_rows)
    leakage = pd.DataFrame(leakage_rows)
    if (unknowns.status != "PASS").any():
        data_errors.append("At least one manifest component is absent from the Mordred lookup.")
    if batches.too_few.any():
        data_errors.append("At least one loader batch has no valid target label.")

    fold_split.to_csv(output / "fold_split_statistics.csv", index=False)
    targets.to_csv(output / "target_distribution_comparison.csv", index=False)
    batches.to_csv(output / "batch_valid_label_counts.csv", index=False)
    features.to_csv(output / "feature_distribution_comparison.csv", index=False)
    graphs.to_csv(output / "graph_distribution_comparison.csv", index=False)
    caches.to_csv(output / "cache_alignment_audit.csv", index=False)
    unknowns.to_csv(output / "unknown_category_audit.csv", index=False)
    status = "DATA_ERROR" if data_errors else "PASS"
    write_report(output / "static_audit_report.md", status, data_errors, leakage, caches, features)
    print(f"Wrote static fold-collapse audit to {output}; status={status}")


if __name__ == "__main__":
    main()
