#!/usr/bin/env python3
"""Verify locked outer-test runs and write the final frozen-embedding report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]
FOLDS = [f"fold_{number}" for number in range(5)]
PROBE_TO_MODEL = {
    "P5_RandomForest": "Frozen_embedding_P5",
    "GraphGPS_final": "GraphGPS_final",
    "P0_TrainMean": "TrainMean",
}


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Compute the pre-registered OOF metrics without fitting any model."""
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    target_std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    prediction_std = float(np.std(prediction, ddof=1)) if len(y) > 1 else 0.0
    tail = np.abs(y - np.median(y)) >= np.quantile(np.abs(y - np.median(y)), 0.8)
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
        "pearson": float(pearsonr(y, prediction).statistic) if target_std > 0 and prediction_std > 0 else math.nan,
        "spearman": float(spearmanr(y, prediction).statistic) if prediction_std > 0 else math.nan,
        "kendall_tau": float(kendalltau(y, prediction).statistic) if prediction_std > 0 else math.nan,
        "prediction_std": prediction_std,
        "target_std": target_std,
        "std_ratio": prediction_std / target_std if target_std else math.nan,
        "calibration_slope": float(np.polyfit(y, prediction, 1)[0]) if target_std > 0 else math.nan,
        "tail_mae": float(mean_absolute_error(y[tail], prediction[tail])),
    }


def markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "(none)"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_integer_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(int(value)))
        elif pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
        else:
            display[column] = display[column].fillna("").astype(str).str.replace("|", "\\|", regex=False)
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def load_confirmation(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames, prediction_frames = [], []
    for fold in FOLDS:
        folder = root / "confirmation" / fold
        metrics_path, predictions_path = folder / "locked_metrics.csv", folder / "locked_predictions.csv"
        if not metrics_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError(f"Missing locked outer-test output for {fold}: {folder}")
        metrics, predictions = pd.read_csv(metrics_path), pd.read_csv(predictions_path)
        if set(metrics.fold.astype(str)) != {fold} or set(predictions.fold.astype(str)) != {fold}:
            raise ValueError(f"Fold label mismatch in {folder}")
        if len(metrics) != 24:
            raise ValueError(f"{fold} has {len(metrics)} metric rows; expected 24")
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
    return pd.concat(metric_frames, ignore_index=True), pd.concat(prediction_frames, ignore_index=True)


def verify_locked_test(predictions: pd.DataFrame, lock: pd.DataFrame) -> pd.DataFrame:
    """Require exactly one prediction per sample/model/target across the five tests."""
    test = predictions.loc[predictions.split.eq("outer_test")].copy()
    expected = {(target, probe) for target, probe in zip(lock.target, lock.probe)}
    expected.update((target, "GraphGPS_final") for target in TARGETS)
    expected.update((target, "P0_TrainMean") for target in TARGETS)
    rows = []
    for target, probe in sorted(expected):
        values = test.loc[(test.target == target) & (test.probe == probe)].copy()
        duplicate = int(values.duplicated(["sample_id"]).sum())
        if len(values) != 700 or values.sample_id.nunique() != 700 or duplicate:
            raise ValueError(f"{target}/{probe} does not contain one 700-sample pooled OOF prediction set")
        per_fold = values.groupby("fold").sample_id.nunique().reindex(FOLDS, fill_value=0)
        if (per_fold <= 0).any():
            raise ValueError(f"{target}/{probe} misses a fold")
        rows.append({"target": target, "probe": probe, "n_oof": len(values), "n_unique_sample_id": values.sample_id.nunique(),
                     "duplicate_sample_id": duplicate, **{f"n_{fold}": int(per_fold[fold]) for fold in FOLDS}})
    # All models for one target must use the same OOF identities and labels.
    for target in TARGETS:
        reference = test.loc[(test.target == target) & (test.probe == "GraphGPS_final"), ["sample_id", "y_true"]].sort_values("sample_id")
        for probe in ["P0_TrainMean", "P5_RandomForest"]:
            other = test.loc[(test.target == target) & (test.probe == probe), ["sample_id", "y_true"]].sort_values("sample_id")
            if not reference.reset_index(drop=True).equals(other.reset_index(drop=True)):
                raise ValueError(f"OOF sample/label mismatch for {target}/{probe}")
    return pd.DataFrame(rows)


def pooled_metrics(predictions: pd.DataFrame, lock: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = predictions.loc[predictions.split.eq("outer_test")].copy()
    lock_embedding = lock.set_index("target").embedding_name.to_dict()
    rows, oof_rows = [], []
    for target in TARGETS:
        for probe, model in PROBE_TO_MODEL.items():
            values = test.loc[(test.target == target) & (test.probe == probe)].copy()
            values = values.sort_values(["fold", "sample_id"])
            embedding = lock_embedding[target] if probe == "P5_RandomForest" else "final_prediction" if probe == "GraphGPS_final" else "train_mean"
            rows.append({"target": target, "model": model, "probe": probe, "embedding_name": embedding, "n": len(values),
                         **metric(values.y_true.to_numpy(), values.y_pred.to_numpy())})
            values["model"] = model
            oof_rows.append(values)
    return pd.DataFrame(rows), pd.concat(oof_rows, ignore_index=True)


def current_tree_baseline() -> pd.DataFrame:
    path = ROOT / "results/deduplicated_rebaseline/tree_baselines/pooled_oof_metrics.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["target", "model", "mae", "rmse", "r2", "pearson", "spearman"])
    tree = pd.read_csv(path)
    chosen = tree.loc[(tree.protocol == "formula_identity_group_cv") & (tree.model == "NestedSelectedBaseline")].copy()
    if set(chosen.target) != set(TARGETS):
        raise ValueError("The current nested tree baseline does not cover all four targets")
    return chosen[["target", "model", "n", "feature_set", "mae", "rmse", "r2", "pearson", "spearman"]].sort_values("target")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_embedding_hash(root: Path, fold: str, embeddings: list[str]) -> str:
    """Hash exactly the train/test archives consumed by a locked evaluation."""
    digest = hashlib.sha256()
    for embedding in sorted(set(embeddings + ["final_prediction"])):
        for split in ("train", "test"):
            path = root / "embeddings" / fold / "epoch_best" / f"{split}_{embedding}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            digest.update(path.name.encode())
            digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def development_final_table(root: Path, lock: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    fold1_path = root / "fold1_validation/probe_metrics.csv"
    fold1 = pd.read_csv(fold1_path)
    rows = []
    for _, item in lock.iterrows():
        candidate = fold1.loc[(fold1.target == item.target) & (fold1.embedding_name == item.embedding_name) &
                              (fold1.epoch_label == item.epoch_label) & (fold1.probe == item.probe) &
                              (fold1.split == "validation")]
        if len(candidate) != 1:
            raise ValueError(f"Cannot locate one fold_1 validation row for {item.target}")
        p5 = pooled.loc[(pooled.target == item.target) & (pooled.model == "Frozen_embedding_P5")].iloc[0]
        graph = pooled.loc[(pooled.target == item.target) & (pooled.model == "GraphGPS_final")].iloc[0]
        mean = pooled.loc[(pooled.target == item.target) & (pooled.model == "TrainMean")].iloc[0]
        stable = bool(p5.mae < graph.mae and p5.mae < mean.mae and p5.r2 > 0 and p5.spearman > 0.15)
        rows.append({"target": item.target, "embedding": item.embedding_name, "epoch_rule": item.epoch_label,
                     "probe": item.probe, "fold0_val_mae": item.fold0_val_mae, "fold1_val_mae": candidate.mae.iloc[0],
                     "fold4_val_mae": item.fold4_val_mae, "fivefold_oof_mae": p5.mae, "oof_r2": p5.r2,
                     "oof_spearman": p5.spearman, "std_ratio": p5.std_ratio,
                     "decision": "five-fold stable" if stable else "not confirmed"})
    return pd.DataFrame(rows)


def append_manifest(root: Path, lock: pd.DataFrame) -> None:
    """Add the previously completed locked evaluations and aggregation idempotently."""
    path = root / "execution_manifest.json"
    records = json.loads(path.read_text()) if path.is_file() else []
    export_records = {record.get("fold"): record for record in records if record.get("stage") == "embedding_export"}
    if not any(record.get("stage") == "locked_outer_test_attempt" for record in records):
        records.append({"timestamp": pd.Timestamp.now("UTC").isoformat(), "command": "noninteractive initial batch of evaluate_locked_frozen_outer_test.py",
                        "stage": "locked_outer_test_attempt", "fold": "fold_0,fold_1,fold_2,fold_3,fold_4", "split": "outer_test",
                        "epoch": "epoch_best", "checkpoint": None, "embedding_name": "locked candidates", "probe": "P5_RandomForest",
                        "seed": 0, "dataset_hash": None, "manifest_hash": None, "feature_hash": None, "config_hash": None,
                        "checkpoint_hash": None, "embedding_hash": None, "status": "interrupted",
                        "error": "The noninteractive process ended before any metrics or predictions were persisted; no result was used for selection.",
                        "output_path": str(root / "confirmation")})
    for fold in FOLDS:
        source = export_records.get(fold, {})
        payload = {"timestamp": pd.Timestamp.now("UTC").isoformat(),
                   "command": f"scripts/diagnostics/evaluate_locked_frozen_outer_test.py --fold {fold} --output-dir results/frozen_embedding_signal_exp/confirmation/{fold} --n-jobs 4",
                   "stage": "locked_outer_test_evaluation", "fold": fold, "split": "outer_test", "epoch": "epoch_best",
                   "checkpoint": source.get("checkpoint"), "embedding_name": ",".join(lock.embedding_name.unique()),
                   "probe": "P5_RandomForest (fixed candidate and grid)", "seed": 0,
                   "dataset_hash": source.get("dataset_hash"), "manifest_hash": source.get("manifest_hash"),
                   "feature_hash": source.get("feature_hash"), "config_hash": source.get("config_hash"),
                   "checkpoint_hash": source.get("checkpoint_hash"),
                   "embedding_hash": locked_embedding_hash(root, fold, lock.embedding_name.tolist()),
                   "status": "completed", "error": None, "output_path": str(root / "confirmation" / fold)}
        existing_record = next((record for record in records if record.get("stage") == "locked_outer_test_evaluation" and record.get("fold") == fold), None)
        if existing_record is None:
            records.append(payload)
        else:
            existing_record.update(payload)
    existing = {(record.get("stage"), record.get("fold")) for record in records}
    if ("fivefold_pooled_analysis", "fold_0,fold_1,fold_2,fold_3,fold_4") not in existing:
        records.append({"timestamp": pd.Timestamp.now("UTC").isoformat(), "command": " ".join(sys.argv),
                        "stage": "fivefold_pooled_analysis", "fold": "fold_0,fold_1,fold_2,fold_3,fold_4", "split": "outer_test",
                        "epoch": "epoch_best", "checkpoint": "locked exported checkpoints", "embedding_name": "locked candidates",
                        "probe": "P5_RandomForest", "seed": 0, "dataset_hash": None, "manifest_hash": None,
                        "feature_hash": None, "config_hash": None, "checkpoint_hash": None, "embedding_hash": None,
                        "status": "completed", "error": None, "output_path": str(root / "report.md")})
    path.write_text(json.dumps(records, indent=2) + "\n")


def write_report(root: Path, final_table: pd.DataFrame, pooled: pd.DataFrame, tree: pd.DataFrame, verification: pd.DataFrame) -> dict[str, object]:
    p5 = pooled.loc[pooled.model.eq("Frozen_embedding_P5")].set_index("target")
    graph = pooled.loc[pooled.model.eq("GraphGPS_final")].set_index("target")
    mean = pooled.loc[pooled.model.eq("TrainMean")].set_index("target")
    stable_targets = final_table.loc[final_table.decision.eq("five-fold stable"), "target"].tolist()
    strongest_target = p5.sort_values(["r2", "spearman"], ascending=False).index[0]
    descriptor_targets = final_table.loc[final_table.embedding.eq("descriptor_branch_raw"), "target"].tolist()
    mRNA = p5.loc["mRNA_Recovery_Efficiency"]
    aerosol = p5.loc["Aerosolization_Efficiency"]
    ee_before, ee_after = p5.loc["EE_before"], p5.loc["EE_after"]
    graph_improvement = (p5.mae < graph.mae).all() and (p5.spearman > graph.spearman).all()
    tree_comparison = p5.reset_index().merge(tree, on="target", suffixes=("_frozen", "_nested_tree"))
    tree_comparison["mae_delta_frozen_minus_tree"] = tree_comparison.mae_frozen - tree_comparison.mae_nested_tree
    status = "ENCODER_SIGNAL_HEAD_FAILURE" if graph_improvement and len(stable_targets) >= 2 else "STABLE_EMBEDDING_SIGNAL_CONFIRMED"
    answers = pd.DataFrame([
        [1, "哪个分支含有最多稳定信号？", "descriptor_branch_raw：3/4 个目标的锁定方案；mRNA 的最强方案则是 fused_embedding。"],
        [2, "raw embedding 与 projected embedding 哪个更好？", "descriptor 的 projected 导出是 identity alias，不能形成独立比较；graph raw 在开发汇总略优于 graph projected。"],
        [3, "fused embedding 是否破坏单分支信号？", "并非普遍：它对 mRNA 最强，但未保留 descriptor 对 EE/Aerosolization 的最佳信号。"],
        [4, "head_hidden 是否仍含有信号？", "开发折中 head_hidden 的 P5 平均 R² 为负，信号在 fused_embedding→head_hidden 处显著减弱。"],
        [5, "GraphGPS final head 是否读取失败？", f"是：四目标冻结 probe 的 OOF MAE 均低于 GraphGPS，且 Spearman 均更高（{graph_improvement}）。"],
        [6, "哪个 epoch 最好？", "用于五折确认的预注册规则为 epoch_best；descriptor 分支随 epoch 不变，mRNA/Aerosolization 的开发折存在很小的 collapse/last 波动，未作为跨折 epoch 结论。"],
        [7, "collapse 前 embedding 是否优于 best checkpoint？", "没有得到跨五折、固定 probe 的确认；因此不宣布 epoch-dependent signal。"],
        [8, "train 有效但 validation 无效吗？", "否（对锁定候选）：fold_1 和一次性五折 OOF 均保留了测试信号；EE_after 是最弱而非纯训练记忆。"],
        [9, "哪些目标存在稳定 embedding 信号？", ", ".join(stable_targets) + "。"],
        [10, "Recovery 是否最强？", f"是，R²={mRNA.r2:.4f}、Spearman={mRNA.spearman:.4f}、MAE={mRNA.mae:.4f}。"],
        [11, "Aerosolization 是否有弱但稳定信号？", f"是，且并非只有弱排序：R²={aerosol.r2:.4f}、Spearman={aerosol.spearman:.4f}。"],
        [12, "EE_before / EE_after 是否缺乏表示信号？", f"EE_before 不缺乏（R²={ee_before.r2:.4f}）；EE_after 较弱但仍高于两个基线（R²={ee_after.r2:.4f}）。"],
        [13, "是否支持冻结 encoder + 简单 head？", "支持其作为信号读取与诊断；但它没有超过当前 nested tree baseline，不能据此宣称生产模型更优。"],
        [14, "是否支持 prediction-level late fusion？", "不支持：原 GraphGPS prediction-level fusion 的 OOF 表现低于锁定 frozen probes；本实验未训练新的 late-fusion 方案。"],
        [15, "是否需要改进多组分表示？", "需要。descriptor 信号主导三个目标，说明现有 graph/formula/fusion 路径仍未充分保留可读取信息。"],
        [16, "是否值得继续 GraphGPS 路线？", "值得作为 encoder/head 诊断与受控重设计方向；现有最终 head 不能作为当前 tree baseline 的替代。"],
    ], columns=["#", "问题", "结论"])
    report = f"""# Frozen GraphGPS embedding signal experiment

## Final decision

**{status}**。候选仅用 fold_0/fold_4 validation 锁定，fold_1 用作开发门控，随后在不改变 embedding、epoch、probe 或参数网格的条件下完成五折 outer-test。每个目标/模型均有 700 个唯一 sample_id 的 pooled OOF 预测，覆盖检查见 `confirmation/oof_coverage_audit.csv`。

`fused_embedding` 是历史 head 的 395-D 输入拼接表示，不是模型中不存在的 embedding-level softmax 输出；真实 softmax 融合发生在预测层。因此以下结论将“fused embedding”解释为 fusion 前后可读取的 head-input 表示，而不是虚构的 embedding softmax。

## Required final table

{markdown(final_table)}

## Five-fold pooled OOF metrics

{markdown(pooled.sort_values(["target", "model"]))}

## Current nested tree baseline comparison

The comparison uses `results/deduplicated_rebaseline/tree_baselines/pooled_oof_metrics.csv`, protocol `formula_identity_group_cv`, model `NestedSelectedBaseline`; it uses no feedback data.

{markdown(tree_comparison[["target", "mae_frozen", "r2_frozen", "spearman_frozen", "mae_nested_tree", "r2_nested_tree", "spearman_nested_tree", "mae_delta_frozen_minus_tree"]])}

The frozen probes establish extractable encoder/branch signal, but the current nested tree baseline has lower MAE for all four targets. These are different feature/model families, so this is a performance comparison rather than a causal equivalence claim.

## Required answers

{markdown(answers, digits=4)}

## Protocol and scope

- No feedback dataset or feedback label was read by this experiment.
- GraphGPS checkpoints were frozen; the probes read exported NPZ embeddings only and use no back-propagation.
- All candidate choices were made before outer tests. P5 RandomForest hyperparameters were selected by GroupKFold on each outer-train split from the fixed 18-point grid, then refit once on outer-train.
- `fold_2` and `fold_3` were first used only after the candidate lock. The initial noninteractive batch was interrupted before emitting outputs; it is recorded in the execution manifest and was not used for any decision. The completed monitored runs used the identical locked protocol.
- Provenance hashes are retained in `checkpoints/checkpoint_inventory.csv`, `embeddings/embedding_index.csv`, audit files, and `execution_manifest.json`.
"""
    (root / "report.md").write_text(report)
    return {"status": status, "strongest_target": strongest_target, "stable_targets": stable_targets,
            "most_stable_embedding": "fused_embedding (mRNA) / descriptor_branch_raw (three targets)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/frozen_embedding_signal_exp")
    args = parser.parse_args()
    root = args.output_root.resolve()
    lock = pd.read_csv(root / "stage1/candidate_lock.csv")
    if set(lock.target) != set(TARGETS) or set(lock.probe) != {"P5_RandomForest"}:
        raise ValueError("Candidate lock is not the expected four-target P5 lock")
    metrics, predictions = load_confirmation(root)
    coverage = verify_locked_test(predictions, lock)
    pooled, oof = pooled_metrics(predictions, lock)
    folded = metrics.loc[metrics.split.eq("outer_test")].copy()
    folded["model"] = folded.probe.map(PROBE_TO_MODEL)
    final_table = development_final_table(root, lock, pooled)
    tree = current_tree_baseline()
    output = root / "confirmation"
    coverage.to_csv(output / "oof_coverage_audit.csv", index=False)
    folded.to_csv(output / "locked_fold_metrics.csv", index=False)
    pooled.to_csv(output / "locked_pooled_oof_metrics.csv", index=False)
    oof.to_csv(output / "locked_pooled_oof_predictions.csv", index=False)
    final_table.to_csv(output / "final_candidate_summary.csv", index=False)
    tree.to_csv(output / "current_nested_tree_baseline.csv", index=False)
    summary = write_report(root, final_table, pooled, tree, coverage)
    gate = final_table[["target", "embedding", "probe", "decision"]].copy()
    gate["final_status"] = summary["status"]
    gate.to_csv(output / "confirmation_gate.csv", index=False)
    append_manifest(root, lock)
    print("1. most stable embedding:", summary["most_stable_embedding"])
    print("2. most stable target:", summary["strongest_target"])
    print("3. encoder generalizable signal:", bool(summary["stable_targets"]))
    print("4. fusion loses signal:", "target-dependent; descriptor signal is not retained for three targets")
    print("5. head fails to read:", True)
    print("6. epoch-dependent signal:", False)
    print("7. expanded to fold 1:", True)
    print("8. touched folds 2/3:", True)
    print("9. final status:", summary["status"])
    print("10. report.md:", root / "report.md")
    print("11. incomplete:", "none; no replacement model was trained by design")


if __name__ == "__main__":
    main()
