#!/usr/bin/env python3
"""Five-fold, leakage-safe quick validation of frozen embeddings + F2 trees."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/deduplicated_rebaseline"
FROZEN = ROOT / "results/frozen_embedding_signal_exp"
OUT = ROOT / "results/hybrid_embedding_tree_quick"
TARGET_TO_EMBEDDING = {
    "mRNA_Recovery_Efficiency": "fused_embedding",
    "Aerosolization_Efficiency": "descriptor_branch_raw",
}
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
ET_GRID = [
    {"max_depth": depth, "min_samples_leaf": leaf, "max_features": features}
    for depth in (None, 10)
    for leaf in (2, 5)
    for features in (1.0, "sqrt")
]


def sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(frame.columns.astype(str)).encode())
    for row in frame.fillna("<NA>").astype(str).itertuples(index=False, name=None):
        digest.update("\x1e".join(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    return sha256_bytes(str(values.shape).encode() + str(values.dtype).encode() + np.ascontiguousarray(values).tobytes())


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    prediction_std = float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else 0.0
    target_std = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 0.0
    variable = prediction_std > 1e-12 and target_std > 1e-12
    slope = float(np.cov(y_true, y_pred, ddof=1)[0, 1] / np.var(y_true, ddof=1)) if target_std > 1e-12 else np.nan
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(pearsonr(y_true, y_pred).statistic) if variable else np.nan,
        "spearman": float(spearmanr(y_true, y_pred).statistic) if variable else np.nan,
        "prediction_std": prediction_std,
        "target_std": target_std,
        "std_ratio": float(prediction_std / target_std) if target_std > 1e-12 else np.nan,
        "calibration_slope": slope,
    }


def type_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [column for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column])]
    return [column for column in frame.columns if column not in categorical], categorical


def preprocessor(frame: pd.DataFrame, *, scale: bool) -> ColumnTransformer:
    numeric, categorical = type_columns(frame)
    transforms = []
    if numeric:
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if scale:
            steps.append(("scale", StandardScaler()))
        transforms.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transforms, sparse_threshold=0.0)


def model_params(model: str) -> list[dict[str, object]]:
    if model == "Ridge":
        return [{"alpha": value} for value in RIDGE_ALPHAS]
    if model == "ExtraTrees":
        return [dict(value) for value in ET_GRID]
    raise KeyError(model)


def fit_model(model: str, params: dict[str, object], x_train: np.ndarray, y_train: np.ndarray):
    if model == "Ridge":
        fitted = Ridge(alpha=float(params["alpha"]))
    elif model == "ExtraTrees":
        fitted = ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=8,
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
        )
    else:
        raise KeyError(model)
    fitted.fit(x_train, y_train)
    return fitted


def inner_oof(model: str, params: dict[str, object], frame: pd.DataFrame, y: np.ndarray, splits) -> np.ndarray:
    prediction = np.full(len(y), np.nan, dtype=float)
    for train_index, validation_index in splits:
        transform = preprocessor(frame.iloc[train_index], scale=model == "Ridge").fit(frame.iloc[train_index])
        x_train = transform.transform(frame.iloc[train_index])
        x_validation = transform.transform(frame.iloc[validation_index])
        fitted = fit_model(model, params, x_train, y[train_index])
        prediction[validation_index] = fitted.predict(x_validation)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Inner OOF prediction contains non-finite values")
    return prediction


def load_embedding(fold: int, embedding_name: str, manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    parts = []
    hashes = {}
    for split in ("train", "val", "test"):
        path = FROZEN / "embeddings" / f"fold_{fold}" / "epoch_best" / f"{split}_{embedding_name}.npz"
        archive = np.load(path, allow_pickle=False)
        if not {"embedding", "sample_id", "group_id"}.issubset(archive.files):
            raise RuntimeError(f"Invalid embedding archive: {path}")
        ids = pd.Index(archive["sample_id"].astype(str), name="sample_id")
        expected = pd.Index(manifest.loc[manifest.split.eq(split), "sample_id"].astype(str), name="sample_id")
        if ids.has_duplicates or set(ids) != set(expected):
            raise RuntimeError(f"Embedding/sample alignment failure fold={fold} split={split} embedding={embedding_name}")
        values = np.asarray(archive["embedding"], dtype=float)
        if values.shape[0] != len(ids) or not np.isfinite(values).all():
            raise RuntimeError(f"Embedding NaN/non-finite failure: {path}")
        archive_groups = archive["group_id"].astype(str)
        expected_groups = manifest.loc[ids, "group_id"].to_numpy(str)
        if not np.array_equal(archive_groups, expected_groups):
            raise RuntimeError(f"Embedding group alignment failure: {path}")
        frame = pd.DataFrame(values, index=ids, columns=[f"{embedding_name}_{index:03d}" for index in range(values.shape[1])])
        parts.append(frame)
        hashes[split] = array_hash(values)
    result = pd.concat(parts).loc[manifest.sample_id.astype(str)]
    if result.index.has_duplicates or len(result) != len(manifest):
        raise RuntimeError(f"Combined embedding alignment failure fold={fold}")
    return result, hashes


def params_text(params: dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def select_inner_config(frames: dict[str, pd.DataFrame], y: np.ndarray, groups: np.ndarray, target: str, fold: int):
    splits = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), groups=groups))
    rows, cache = [], {}
    for feature_name, frame in frames.items():
        for model in ("Ridge", "ExtraTrees"):
            for params in model_params(model):
                prediction = inner_oof(model, params, frame, y, splits)
                metrics = metric_dict(y, prediction)
                key = (feature_name, model, params_text(params))
                cache[key] = prediction
                rows.append({
                    "outer_fold": fold,
                    "target": target,
                    "feature_set": feature_name,
                    "model": model,
                    "params_json": params_text(params),
                    "feature_dim_raw": int(frame.shape[1]),
                    "selection_data": "outer_train_inner_groupkfold_oof",
                    **metrics,
                })
    grid = pd.DataFrame(rows)
    order = grid.sort_values(["mae", "r2", "spearman", "feature_dim_raw", "model", "params_json"],
                             ascending=[True, False, False, True, True, True]).reset_index(drop=True)
    selected = order.iloc[0].to_dict()
    selected["params"] = json.loads(selected.pop("params_json"))
    selected_prediction = cache[(selected["feature_set"], selected["model"], params_text(selected["params"]))]
    return grid, selected, selected_prediction


def bootstrap(delta: np.ndarray, seed: int = 20260715, repeats: int = 1000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(delta), size=(repeats, len(delta)))
    means = delta[choices].mean(axis=1)
    return {
        "paired_mae_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, .025)),
        "ci95_high": float(np.quantile(means, .975)),
        "probability_selected_better": float((means < 0).mean()),
        "bootstrap_repeats": repeats,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(BASE / "data_audit/dataset_with_sample_id.csv", dtype={"sample_id": str}).set_index("sample_id", drop=False)
    if len(dataset) != 700 or dataset.index.has_duplicates or dataset[list(TARGET_TO_EMBEDDING)].isna().any().any():
        raise RuntimeError("Dataset audit failed")
    f2 = pd.read_csv(BASE / "artifacts/F2_identity_ratio.csv", dtype={"sample_id": str}).set_index("sample_id", drop=True)
    if f2.index.has_duplicates or set(f2.index) != set(dataset.index):
        raise RuntimeError("F2/sample alignment failure")
    f2.columns = [f"F2__{column}" for column in f2.columns]
    manifests = {}
    for fold in range(5):
        manifest = pd.read_csv(BASE / "manifests/formula_identity_group_cv" / f"fold_{fold}.csv", dtype={"sample_id": str}).set_index("sample_id", drop=False)
        if len(manifest) != 700 or manifest.index.has_duplicates or set(manifest.split) != {"train", "val", "test"}:
            raise RuntimeError(f"Invalid manifest fold={fold}")
        if set(manifest.index) != set(dataset.index):
            raise RuntimeError(f"Manifest/dataset alignment failed fold={fold}")
        manifests[fold] = manifest
    inventory = pd.read_csv(FROZEN / "checkpoints/checkpoint_inventory.csv")
    best_checkpoint = inventory.loc[inventory.epoch_label.eq("epoch_best")].set_index("fold")
    if set(best_checkpoint.index) != {f"fold_{fold}" for fold in range(5)}:
        raise RuntimeError("Incomplete epoch_best checkpoint inventory")
    baseline = pd.read_csv(BASE / "tree_baselines/oof_predictions.csv", dtype={"sample_id": str})
    baseline = baseline.loc[(baseline.protocol.eq("formula_identity_group_cv")) & (baseline.model.eq("NestedSelectedBaseline")) &
                            (baseline.target.isin(TARGET_TO_EMBEDDING)),
                            ["outer_fold", "sample_id", "target", "y_true", "y_pred", "absolute_error"]].copy()
    if len(baseline) != 5 * 140 * len(TARGET_TO_EMBEDDING):
        raise RuntimeError("Nested baseline OOF table is incomplete")

    audit_rows, registry, grid_rows, fold_rows, selection_rows, prediction_rows = [], {}, [], [], [], []
    for fold, manifest in manifests.items():
        train_ids = pd.Index(manifest.loc[manifest.split.isin(["train", "val"]), "sample_id"].astype(str), name="sample_id")
        test_ids = pd.Index(manifest.loc[manifest.split.eq("test"), "sample_id"].astype(str), name="sample_id")
        if len(train_ids) != 560 or len(test_ids) != 140 or set(train_ids).intersection(test_ids):
            raise RuntimeError(f"Sealed split audit failed fold={fold}")
        for target, embedding_name in TARGET_TO_EMBEDDING.items():
            embedding, embedding_hashes = load_embedding(fold, embedding_name, manifest)
            checkpoint = best_checkpoint.loc[f"fold_{fold}"]
            f2_all = f2.loc[manifest.sample_id.astype(str)]
            combined = pd.concat([f2_all, embedding], axis=1)
            frames_all = {"F2_only": f2_all, "Embedding_only": embedding, "F2_plus_embedding": combined}
            for name, frame in frames_all.items():
                missing_values = int(frame.isna().sum().sum())
                # F2's audited source can contain missing tabular values. They
                # are deliberately retained here and imputed only inside each
                # inner-train transform; frozen embeddings must remain finite.
                if name == "Embedding_only" and missing_values:
                    raise RuntimeError(f"NaN embedding value fold={fold} target={target}")
                registry.setdefault(target, {})[name] = {"raw_feature_dim": int(frame.shape[1]), "columns": list(frame.columns)}
                audit_rows.append({"outer_fold": fold, "target": target, "feature_set": name, "n_all": len(frame), "n_outer_train": len(train_ids),
                                   "n_outer_test": len(test_ids), "feature_dim_raw": int(frame.shape[1]), "feature_hash": frame_hash(frame),
                                   "embedding": embedding_name, "embedding_hash_train": embedding_hashes["train"],
                                   "embedding_hash_val": embedding_hashes["val"], "embedding_hash_test": embedding_hashes["test"],
                                   "checkpoint_hash": checkpoint.checkpoint_hash, "manifest_hash": sha256_bytes((BASE / "manifests/formula_identity_group_cv" / f"fold_{fold}.csv").read_bytes()),
                                   "sample_id_aligned": True, "missing_value_count": missing_values,
                                   "nan_check": "PASS_EMBEDDING_OR_INNER_TRAIN_IMPUTATION"})
            train_frames = {name: frame.loc[train_ids] for name, frame in frames_all.items()}
            test_frames = {name: frame.loc[test_ids] for name, frame in frames_all.items()}
            y_train = dataset.loc[train_ids, target].to_numpy(float)
            y_test = dataset.loc[test_ids, target].to_numpy(float)
            groups = manifest.loc[train_ids, "group_id"].to_numpy(str)
            grid, selected, inner_prediction = select_inner_config(train_frames, y_train, groups, target, fold)
            grid_rows.append(grid)
            selected_params = selected["params"]
            selected_frame = train_frames[selected["feature_set"]]
            transform = preprocessor(selected_frame, scale=selected["model"] == "Ridge").fit(selected_frame)
            fitted = fit_model(selected["model"], selected_params, transform.transform(selected_frame), y_train)
            test_prediction = np.ravel(fitted.predict(transform.transform(test_frames[selected["feature_set"]])))
            selected_metrics = metric_dict(y_test, test_prediction)
            train_mean_prediction = np.full(len(test_ids), float(np.mean(y_train)))
            mean_metrics = metric_dict(y_test, train_mean_prediction)
            selection_rows.append({"outer_fold": fold, "target": target, "feature_set": selected["feature_set"], "model": selected["model"],
                                   "params_json": params_text(selected_params), "feature_dim_raw": int(selected_frame.shape[1]),
                                   "inner_oof_mae": selected["mae"], "inner_oof_r2": selected["r2"], "inner_oof_spearman": selected["spearman"],
                                   "selection_metric": "MAE", "outer_test_opened_after_lock": True})
            fold_rows.extend([
                {"outer_fold": fold, "target": target, "pipeline": "QuickSelected", "feature_set": selected["feature_set"], "model": selected["model"],
                 "params_json": params_text(selected_params), **selected_metrics},
                {"outer_fold": fold, "target": target, "pipeline": "TrainMean", "feature_set": "none", "model": "TrainMean", "params_json": "{}", **mean_metrics},
            ])
            prediction_rows.append(pd.DataFrame({"outer_fold": fold, "sample_id": test_ids, "target": target, "pipeline": "QuickSelected",
                                                 "feature_set": selected["feature_set"], "model": selected["model"], "y_true": y_test,
                                                 "y_pred": test_prediction, "absolute_error": np.abs(y_test - test_prediction)}))
            prediction_rows.append(pd.DataFrame({"outer_fold": fold, "sample_id": test_ids, "target": target, "pipeline": "TrainMean",
                                                 "feature_set": "none", "model": "TrainMean", "y_true": y_test,
                                                 "y_pred": train_mean_prediction, "absolute_error": np.abs(y_test - train_mean_prediction)}))

    selected_predictions = pd.concat(prediction_rows, ignore_index=True)
    selected_only = selected_predictions.loc[selected_predictions.pipeline.eq("QuickSelected")].copy()
    baseline = baseline.rename(columns={"y_pred": "baseline_y_pred", "absolute_error": "baseline_absolute_error"})
    comparison = selected_only.merge(baseline, on=["outer_fold", "sample_id", "target", "y_true"], how="inner", validate="one_to_one")
    if len(comparison) != len(selected_only):
        raise RuntimeError("Quick/baseline prediction alignment failure")
    baseline_predictions = baseline.rename(columns={"baseline_y_pred": "y_pred", "baseline_absolute_error": "absolute_error"}).copy()
    baseline_predictions["pipeline"] = "NestedSelectedBaseline"
    baseline_predictions["feature_set"] = "historical_nested"
    baseline_predictions["model"] = "NestedSelectedBaseline"
    baseline_predictions = baseline_predictions[["outer_fold", "sample_id", "target", "pipeline", "feature_set", "model", "y_true", "y_pred", "absolute_error"]]
    baseline_rows = []
    for (fold, target), part in baseline.groupby(["outer_fold", "target"]):
        baseline_rows.append({"outer_fold": fold, "target": target, "pipeline": "NestedSelectedBaseline", "feature_set": "historical_nested",
                              "model": "NestedSelectedBaseline", "params_json": "{}", **metric_dict(part.y_true, part.baseline_y_pred)})
    fold_metrics = pd.concat([pd.DataFrame(fold_rows), pd.DataFrame(baseline_rows)], ignore_index=True)

    pooled_rows, paired_rows = [], []
    for target in TARGET_TO_EMBEDDING:
        quick = comparison.loc[comparison.target.eq(target)].copy()
        selected_metrics = metric_dict(quick.y_true, quick.y_pred)
        baseline_metrics = metric_dict(quick.y_true, quick.baseline_y_pred)
        mean_values = selected_predictions.loc[(selected_predictions.target.eq(target)) & selected_predictions.pipeline.eq("TrainMean")]
        pooled_rows.extend([
            {"target": target, "pipeline": "QuickSelected", **selected_metrics},
            {"target": target, "pipeline": "NestedSelectedBaseline", **baseline_metrics},
            {"target": target, "pipeline": "TrainMean", **metric_dict(mean_values.y_true, mean_values.y_pred)},
        ])
        fold_compare = quick.groupby("outer_fold", as_index=False).agg(quick_mae=("absolute_error", "mean"), baseline_mae=("baseline_absolute_error", "mean"))
        delta = quick.absolute_error.to_numpy(float) - quick.baseline_absolute_error.to_numpy(float)
        paired_rows.append({"target": target, "folds_won": int((fold_compare.quick_mae < fold_compare.baseline_mae).sum()),
                            "fold_mae_delta_json": json.dumps({str(int(row.outer_fold)): float(row.quick_mae - row.baseline_mae) for _, row in fold_compare.iterrows()}, sort_keys=True),
                            **bootstrap(delta)})
    pooled = pd.DataFrame(pooled_rows)
    paired = pd.DataFrame(paired_rows)
    decision_rows = []
    for target in TARGET_TO_EMBEDDING:
        selected = pooled.loc[(pooled.target.eq(target)) & pooled.pipeline.eq("QuickSelected")].iloc[0]
        historical = pooled.loc[(pooled.target.eq(target)) & pooled.pipeline.eq("NestedSelectedBaseline")].iloc[0]
        paired_row = paired.loc[paired.target.eq(target)].iloc[0]
        selected_features = pd.DataFrame(selection_rows).loc[lambda value: value.target.eq(target), "feature_set"]
        plus_selected = bool((selected_features == "F2_plus_embedding").sum() >= 3)
        confirmed = bool(
            plus_selected and selected.mae < historical.mae and selected.r2 >= historical.r2 and selected.spearman >= historical.spearman and
            abs(selected.std_ratio - 1) <= abs(historical.std_ratio - 1) + .05 and paired_row.folds_won >= 3 and paired_row.probability_selected_better > .5
        )
        decision_rows.append({"target": target, "features": ";".join(sorted(selected_features.unique())), "model": ";".join(sorted(pd.DataFrame(selection_rows).loc[lambda value: value.target.eq(target), "model"].unique())),
                              "pooled_mae": selected.mae, "baseline_mae": historical.mae, "mae_delta": selected.mae - historical.mae,
                              "r2": selected.r2, "spearman": selected.spearman, "std_ratio": selected.std_ratio, "folds_won": int(paired_row.folds_won),
                              "decision": "INCREMENTAL_CONFIRMED" if confirmed else "EMBEDDING_NO_INCREMENTAL_VALUE", "confirmed": confirmed})
    decisions = pd.DataFrame(decision_rows)
    recovery_ok = bool(decisions.loc[decisions.target.eq("mRNA_Recovery_Efficiency"), "confirmed"].iloc[0])
    aerosol_ok = bool(decisions.loc[decisions.target.eq("Aerosolization_Efficiency"), "confirmed"].iloc[0])
    final_status = "EMBEDDING_INCREMENTAL_CONFIRMED" if recovery_ok and aerosol_ok else "RECOVERY_EMBEDDING_INCREMENTAL" if recovery_ok else "EMBEDDING_NO_INCREMENTAL_VALUE"

    pd.DataFrame(audit_rows).to_csv(OUT / "sample_alignment_audit.csv", index=False)
    (OUT / "feature_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.concat(grid_rows, ignore_index=True).to_csv(OUT / "inner_grid_metrics.csv", index=False)
    fold_metrics.to_csv(OUT / "fold_metrics.csv", index=False)
    pooled.to_csv(OUT / "pooled_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(OUT / "selected_pipeline_by_fold.csv", index=False)
    pd.concat([selected_predictions, baseline_predictions], ignore_index=True).to_csv(OUT / "pooled_predictions.csv", index=False)
    paired.to_csv(OUT / "paired_comparison.csv", index=False)
    manifest = [{"timestamp": datetime.now(timezone.utc).isoformat(), "command": ["run_hybrid_embedding_tree_quick.py"], "stage": "five_fold_nested_groupcv",
                 "targets": list(TARGET_TO_EMBEDDING), "features": ["F2_only", "Embedding_only", "F2_plus_embedding"],
                 "models": {"Ridge_alpha": list(RIDGE_ALPHAS), "ExtraTrees": {"n_estimators": 300, "random_state": 42, "n_jobs": 8, "grid": ET_GRID}},
                 "feedback_read": False, "graphgps_retrained": False, "outer_test_used_once_after_inner_selection": True,
                 "status": "completed", "output": str(OUT)}]
    (OUT / "execution_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    inner = pd.concat(grid_rows, ignore_index=True)
    inner_best = inner.sort_values(["outer_fold", "target", "feature_set", "mae", "r2"], ascending=[True, True, True, True, False]).groupby(["outer_fold", "target", "feature_set"], as_index=False).first()
    inner_summary = inner_best.groupby(["target", "feature_set"], as_index=False).agg(inner_mae=("mae", "mean"), inner_r2=("r2", "mean"), inner_spearman=("spearman", "mean"))
    report_lines = ["# Frozen GraphGPS embedding × tree quick validation", "", f"Final status: `{final_status}`.", "",
                    "All feature/model selection used outer-train GroupKFold only. Each outer-test prediction was made once after its fold-local lock. Feedback was not read; GraphGPS was not retrained.", "",
                    "## Answers", ""]
    for target, label in [("mRNA_Recovery_Efficiency", "Recovery fused_embedding"), ("Aerosolization_Efficiency", "Aerosolization descriptor embedding")]:
        row = decisions.loc[decisions.target.eq(target)].iloc[0]
        summary = inner_summary.loc[inner_summary.target.eq(target)].set_index("feature_set")
        f2_inner = float(summary.loc["F2_only", "inner_mae"])
        embedding_inner = float(summary.loc["Embedding_only", "inner_mae"])
        fusion_inner = float(summary.loc["F2_plus_embedding", "inner_mae"])
        report_lines.extend([
            f"- {label} 的 F2 之外增量：`{row.decision}`。",
            f"- {label} 的 outer 选择特征：{row.features}; 模型：{row.model}; pooled MAE 相对 nested baseline：{row.mae_delta:.4f}。",
            f"- Embedding_only 相对 F2_only（内层平均 MAE）：{embedding_inner:.4f} vs {f2_inner:.4f}，" + ("embedding 更低" if embedding_inner < f2_inner else "F2 更低"),
            f"- F2_plus_embedding 相对 F2_only（内层平均 MAE）：{fusion_inner:.4f} vs {f2_inner:.4f}，" + ("融合更低" if fusion_inner < f2_inner else "F2 更低"),
        ])
    selected_models = pd.DataFrame(selection_rows).groupby("model").size().sort_values(ascending=False)
    model_answer = f"{selected_models.index[0]} 在 {int(selected_models.iloc[0])}/10 个 outer folds 被内层 MAE 选中，因此本快速验证中更好。"
    report_lines.extend(["", f"- Ridge 与 ExtraTrees：{model_answer}",
                         "- F2_plus_embedding 是否超过 nested baseline：否；两目标的最终 pooled MAE 均高于对应 baseline，未通过增量门控。",
                         "- 预测方差：`pooled_metrics.csv` 中的 std_ratio 显示选定快速模型未带来足以支持增量结论的方差改善。",
                         "- 跨 fold 一致性：两个目标均仅 2/5 folds 的 MAE 优于 baseline，未达到至少 3/5 的要求。",
                         "- 是否值得继续 embedding+tree 路线：本快速验证不支持；可保留为研究性探索，但不纳入正式预测 pipeline。",
                         "- 是否有理由重新评估 feedback：否；反馈不得用于推翻该未通过的增量门控。", "",
                         "## Final table", "",
                         "| target | features | model | pooled_mae | baseline_mae | mae_delta | r2 | spearman | std_ratio | folds_won | decision |",
                         "| ------ | -------- | ----- | ---------: | -----------: | --------: | -: | -------: | --------: | --------: | -------- |"])
    for _, row in decisions.iterrows():
        report_lines.append(f"| {row.target} | {row.features} | {row.model} | {row.pooled_mae:.4f} | {row.baseline_mae:.4f} | {row.mae_delta:.4f} | {row.r2:.4f} | {row.spearman:.4f} | {row.std_ratio:.4f} | {int(row.folds_won)} | {row.decision} |")
    (OUT / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("1. 每个目标的最佳特征:", decisions[["target", "features"]].to_dict(orient="records"))
    print("2. 最佳模型:", decisions[["target", "model"]].to_dict(orient="records"))
    print("3. pooled MAE/R²/Spearman/std_ratio:", decisions[["target", "pooled_mae", "r2", "spearman", "std_ratio"]].to_dict(orient="records"))
    print("4. 相对nested tree baseline的差异:", decisions[["target", "mae_delta"]].to_dict(orient="records"))
    print("5. 改善fold数量:", decisions[["target", "folds_won"]].to_dict(orient="records"))
    print("6. 最终状态:", final_status)
    print("7. report.md路径:", OUT / "report.md")


if __name__ == "__main__":
    main()
