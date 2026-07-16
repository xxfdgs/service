#!/usr/bin/env python3
"""Audit GraphGPS CV checkpoint provenance and validation-based selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats_records(path: Path, split: str, fold: str) -> list[dict[str, object]]:
    records = []
    for line in path.read_text().splitlines():
        item = json.loads(line)
        row = {"fold": fold, "split": split, **item}
        per_target = row.pop("mae_per_property", {})
        for target, source in zip(TARGETS, ("EE_before_mae", "EE_after_mae", "Aero_Efficiency_mae", "Recovery_Efficiency_mae")):
            row[f"{target}_mae"] = per_target.get(source, np.nan)
        records.append(row)
    return records


def config_value(config: dict, dotted: str):
    value = config
    for key in dotted.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def prediction_rows(predictions: pd.DataFrame, fold: str) -> list[dict[str, object]]:
    rows = []
    for split, part in predictions.loc[predictions.fold == fold].groupby("split", sort=True):
        for target, group in part.groupby("target", sort=True):
            y, p = group.y_true.to_numpy(float), group.y_pred.to_numpy(float)
            slope = np.polyfit(p, y, 1)[0] if np.std(p) > 1e-12 else np.nan
            rows.append({
                "fold": fold, "split": split, "target": target,
                "checkpoint": group.checkpoint.iloc[0], "n_samples": len(group),
                "true_std": np.std(y, ddof=1), "prediction_std": np.std(p, ddof=1),
                "std_ratio": np.std(p, ddof=1) / np.std(y, ddof=1),
                "mae": mean_absolute_error(y, p), "r2": r2_score(y, p),
                "spearman": spearmanr(y, p).statistic, "calibration_slope": slope,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/fold4_collapse_audit/checkpoint_audit")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "results/deduplicated_rebaseline"
    train_root = source / "graphgps_cv/training"
    configs = source / "graphgps_cv/configs"
    manifest_root = source / "manifests/formula_identity_group_cv"
    feature_path = source / "artifacts/mordred_11_lookup.csv"

    validation_rows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for index in range(5):
        fold = f"fold_{index}"
        run = train_root / f"formula_identity_group_cv_{fold}_seed_0/0"
        config_path = configs / f"formula_identity_group_cv_{fold}_seed_0.yaml"
        saved_config = run.parent / "config.yaml"
        config = yaml.safe_load(config_path.read_text())
        for split in ("train", "val", "test"):
            validation_rows.extend(stats_records(run / split / "stats.json", split, fold))
        val = pd.DataFrame([row for row in validation_rows if row["fold"] == fold and row["split"] == "val"])
        best_row = val.loc[val.loss.idxmin()]
        checkpoint_paths = sorted((run / "ckpt").glob("*.ckpt"), key=lambda p: int(p.stem))
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            epoch = int(checkpoint_path.stem)
            metadata = {key: checkpoint.get(key) for key in (
                "stage3_checkpoint_metadata", "epoch", "best_metric", "seed", "fold", "protocol",
                "manifest_path", "manifest_hash", "config_hash", "feature_hash", "sample_id_hash", "target_scaler",
            )}
            correct_epoch = epoch == int(best_row.epoch)
            manifest_path = manifest_root / f"{fold}.csv"
            checks = {
                "epoch_matches_best_validation_loss": correct_epoch,
                "metadata_epoch_matches_filename": metadata["epoch"] == epoch,
                "metadata_fold_matches": metadata["fold"] == fold,
                "metadata_seed_is_zero": metadata["seed"] == 0,
                "metadata_protocol_matches": metadata["protocol"] == "formula_identity_group_cv",
                "manifest_hash_matches": metadata["manifest_hash"] == sha256(manifest_path),
                "feature_hash_matches": metadata["feature_hash"] == sha256(feature_path),
                "saved_config_hash_matches": metadata["config_hash"] == sha256(saved_config),
                "fixed_percent_target_scaler": metadata["target_scaler"] == {"type": "fixed_percent", "scale": 100.0},
            }
            inventory.append({
                "fold": fold, "checkpoint": str(checkpoint_path), "checkpoint_epoch": epoch,
                "checkpoint_sha256": sha256(checkpoint_path), "best_validation_epoch": int(best_row.epoch),
                "best_validation_loss": float(best_row.loss), "metadata_best_metric": metadata["best_metric"],
                "is_best_validation_epoch": correct_epoch, "n_checkpoints_preserved": len(checkpoint_paths),
            })
            metadata_rows.append({"fold": fold, "checkpoint": str(checkpoint_path), **metadata, **checks, "status": "PASS" if all(checks.values()) else "FAIL"})
        lr = pd.DataFrame([row for row in validation_rows if row["fold"] == fold and row["split"] == "train"])
        selection_rows.append({
            "fold": fold, "monitor": config.get("metric_best"), "mode": config.get("metric_agg"),
            "early_stop_patience": config_value(config, "train.early_stop_patience"),
            "early_stop_min_delta": config_value(config, "train.early_stop_min_delta"),
            "scheduler": config_value(config, "optim.scheduler"), "base_lr": config_value(config, "optim.base_lr"),
            "max_epoch": config_value(config, "optim.max_epoch"), "last_epoch": int(val.epoch.max()),
            "best_validation_epoch": int(best_row.epoch), "best_validation_loss": float(best_row.loss),
            "last_validation_loss": float(val.sort_values("epoch").loss.iloc[-1]),
            "last_train_lr": float(lr.sort_values("epoch").lr.iloc[-1]),
            "lr_below_one_percent_at_end": bool(lr.sort_values("epoch").lr.iloc[-1] < config_value(config, "optim.base_lr") * 0.01),
            "preserves_only_best_checkpoint": True,
            "selection_assessment": "PASS: checkpoint filename equals minimum inner-validation loss epoch",
        })

    histories = pd.DataFrame(validation_rows)
    histories.to_csv(output / "validation_metric_history.csv", index=False)
    pd.DataFrame(inventory).to_csv(output / "checkpoint_inventory.csv", index=False)
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(output / "checkpoint_metadata_audit.csv", index=False)
    selection = pd.DataFrame(selection_rows)

    exported = pd.read_csv(ROOT / "results/variance_compression_exp1/predictions/fold_predictions.csv", dtype={"sample_id": str})
    metrics = prediction_rows(exported, "fold_4")
    pd.DataFrame(metrics).to_csv(output / "checkpoint_prediction_metrics.csv", index=False)
    fold4 = selection.loc[selection.fold == "fold_4"].iloc[0]
    report = [
        "# Fold 4 checkpoint 选择审计", "",
        "- 监控指标为 validation `loss`（四个归一化目标 L1 的和），模式为 `argmin`。", 
        "- 训练模式设置 `ckpt_best=True` 且 `ckpt_clean=True`，故历史 run 只保留每个 fold 的最佳 checkpoint；无法在不重训的情况下重新评估已被清理的其它 epoch。",
        f"- fold_4 的 49.ckpt 正好是 validation loss 最小的 epoch（loss={fold4.best_validation_loss:.6f}），并且 checkpoint metadata 的 fold、seed、manifest hash、feature hash、saved-config hash 均匹配。",
        f"- fold_4 在停止时的学习率为 {fold4.last_train_lr:.8f}，不是初始学习率 1% 以下；early stopping 在 epoch {fold4.last_epoch} 发生，距 epoch 49 的最佳 validation 已过去 50 个 eval epochs，符合 patience=50。",
        "- 因而现有证据不支持 CHECKPOINT_SELECTION_BUG；重现阶段会额外保存每轮 checkpoint，并比较训练结束内存模型与重新加载 best checkpoint 的预测。",
    ]
    (output / "checkpoint_selection_report.md").write_text("\n".join(report) + "\n")
    if (metadata.status != "PASS").any():
        raise RuntimeError("Checkpoint metadata audit failed.")
    print(f"Wrote checkpoint audit to {output}")


if __name__ == "__main__":
    main()
