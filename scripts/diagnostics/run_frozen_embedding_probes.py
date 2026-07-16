#!/usr/bin/env python3
"""Nested group-aware probes on frozen GraphGPS embeddings.

Only outer-train and the explicit validation split are read.  The archived
outer-test arrays are intentionally never opened by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]
EMBEDDINGS = [
    "graph_branch_raw", "descriptor_branch_raw", "formula_branch_raw",
    "graph_branch_projected", "descriptor_branch_projected", "formula_branch_projected",
    "fused_embedding", "head_hidden", "final_prediction",
]
EPOCH_ORDER = ["epoch_initial", "epoch_precollapse", "epoch_collapse", "epoch_best", "epoch_last"]


@dataclass
class ProbeResult:
    probe: str
    params: dict[str, Any]
    inner_mae: float
    inner_r2: float
    inner_spearman: float
    train_prediction: np.ndarray
    validation_prediction: np.ndarray


def stable_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    finite = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[finite], prediction[finite]
    if len(y) == 0:
        return {name: math.nan for name in ("mae", "rmse", "r2", "pearson", "spearman", "kendall_tau", "prediction_std", "target_std", "std_ratio", "calibration_slope", "tail_mae")}
    target_std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    pred_std = float(np.std(prediction, ddof=1)) if len(y) > 1 else 0.0
    tail = np.abs(y - np.median(y)) >= np.quantile(np.abs(y - np.median(y)), .8)
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)) if len(y) > 1 else math.nan,
        "pearson": float(pearsonr(y, prediction).statistic) if len(y) > 2 and target_std > 0 and pred_std > 0 else math.nan,
        "spearman": float(spearmanr(y, prediction).statistic) if len(y) > 2 and pred_std > 0 else math.nan,
        "kendall_tau": float(kendalltau(y, prediction).statistic) if len(y) > 2 else math.nan,
        "prediction_std": pred_std, "target_std": target_std,
        "std_ratio": float(pred_std / target_std) if target_std > 0 else math.nan,
        "calibration_slope": float(np.polyfit(y, prediction, 1)[0]) if len(y) > 2 and target_std > 0 else math.nan,
        "tail_mae": float(mean_absolute_error(y[tail], prediction[tail])) if tail.any() else math.nan,
    }


def candidates(probe: str, dimension: int, n_train: int) -> list[dict[str, Any]]:
    if probe == "P1_Ridge":
        return [{"alpha": value} for value in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)]
    if probe == "P2_ElasticNet":
        return [{"alpha": alpha, "l1_ratio": l1_ratio}
                for alpha, l1_ratio in itertools.product((1e-4, 1e-3, 1e-2, 1e-1, 1.0), (0.0, .1, .5, .9, 1.0))]
    if probe == "P3_PLS":
        values = {2, 4, 8, 16, min(32, dimension)}
        return [{"n_components": value} for value in sorted(values) if value <= min(dimension, n_train - 1)]
    if probe in {"P4_ExtraTrees", "P5_RandomForest"}:
        return [{"n_estimators": 300, "max_depth": depth, "min_samples_leaf": leaf, "max_features": feature}
                for depth, leaf, feature in itertools.product((None, 5, 10), (1, 3, 5), (1.0, "sqrt"))]
    raise ValueError(probe)


def estimator(probe: str, params: dict[str, Any]):
    if probe == "P1_Ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    if probe == "P2_ElasticNet":
        return make_pipeline(StandardScaler(), ElasticNet(**params, max_iter=20000, tol=1e-4, random_state=0))
    if probe == "P3_PLS":
        return PLSRegression(**params, scale=True, max_iter=1000, tol=1e-6)
    if probe == "P4_ExtraTrees":
        return ExtraTreesRegressor(**params, random_state=0, n_jobs=8)
    if probe == "P5_RandomForest":
        return RandomForestRegressor(**params, random_state=0, n_jobs=8)
    raise ValueError(probe)


def fit_prediction(model: Any, x_train: np.ndarray, y_train: np.ndarray, x_predict: np.ndarray) -> tuple[np.ndarray, bool]:
    # PLS cannot form a latent variable if every input feature is constant in
    # the current inner-train split.  A train-mean prediction is the unique
    # well-defined no-information limit.  This is fitted on inner-train only,
    # and prevents a numerical failure from silently deleting the required
    # P3 evaluation row for a collapsed representation.
    if isinstance(model, PLSRegression) and not np.any(np.std(x_train, axis=0) > 1e-12):
        return np.full(len(x_predict), float(np.mean(y_train))), False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)
        prediction = np.asarray(model.predict(x_predict)).reshape(-1)
    warning_seen = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return prediction, warning_seen


def choose_probe(x: np.ndarray, y: np.ndarray, group: np.ndarray, probe: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cv = GroupKFold(n_splits=min(5, len(np.unique(group))))
    search_rows: list[dict[str, Any]] = []
    options = candidates(probe, x.shape[1], len(x))
    if not options:
        return {}, [{"probe": probe, "status": "no_valid_parameter", "params": "{}"}]
    for params in options:
        predictions = np.full(len(y), np.nan)
        warning_seen = False
        status = "ok"
        try:
            for train_index, valid_index in cv.split(x, y, group):
                prediction, warned = fit_prediction(estimator(probe, params), x[train_index], y[train_index], x[valid_index])
                predictions[valid_index] = prediction
                warning_seen |= warned
            value = metrics(y, predictions)
        except Exception as error:  # retain failed grid points instead of silently omitting them
            value = {"mae": math.inf, "r2": math.nan, "spearman": math.nan}
            status = f"failed:{type(error).__name__}"
        search_rows.append({"probe": probe, "params": json.dumps(params, sort_keys=True), "status": status,
                            "convergence_warning": warning_seen, "inner_mae": value["mae"],
                            "inner_r2": value["r2"], "inner_spearman": value["spearman"]})
    valid = [row for row in search_rows if row["status"] == "ok" and np.isfinite(row["inner_mae"])]
    if not valid:
        return {}, search_rows
    best = min(valid, key=lambda row: (row["inner_mae"], -np.nan_to_num(row["inner_spearman"], nan=-1.0)))
    return json.loads(best["params"]), search_rows


def execute_representation(x_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray,
                           x_val: np.ndarray, y_val: np.ndarray,
                           probes: list[str]) -> tuple[list[ProbeResult], list[dict[str, Any]]]:
    results: list[ProbeResult] = []
    selection_rows: list[dict[str, Any]] = []
    if "P0_TrainMean" in probes:
        mean = np.full(len(y_train), y_train.mean())
        val_mean = np.full(len(y_val), y_train.mean())
        results.append(ProbeResult("P0_TrainMean", {}, math.nan, math.nan, math.nan, mean, val_mean))
        selection_rows.append({"probe": "P0_TrainMean", "params": "{}", "status": "fixed", "convergence_warning": False,
                               "inner_mae": math.nan, "inner_r2": math.nan, "inner_spearman": math.nan})
    for probe in ("P1_Ridge", "P2_ElasticNet", "P3_PLS", "P4_ExtraTrees", "P5_RandomForest"):
        if probe not in probes:
            continue
        best, rows = choose_probe(x_train, y_train, groups, probe)
        selection_rows.extend(rows)
        if not best:
            continue
        chosen = next(row for row in rows if row["status"] == "ok" and row["params"] == json.dumps(best, sort_keys=True))
        model = estimator(probe, best)
        train_prediction, _ = fit_prediction(model, x_train, y_train, x_train)
        validation_prediction, _ = fit_prediction(model, x_train, y_train, x_val)
        results.append(ProbeResult(probe, best, float(chosen["inner_mae"]), float(chosen["inner_r2"]),
                                   float(chosen["inner_spearman"]), train_prediction, validation_prediction))
    return results, selection_rows


def load_embedding(root: Path, fold: str, epoch_label: str, split: str, embedding: str) -> dict[str, np.ndarray]:
    archive = np.load(root / "embeddings" / fold / epoch_label / f"{split}_{embedding}.npz", allow_pickle=False)
    return {name: archive[name] for name in archive.files}


def append_metrics(rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]], *, fold: str,
                   epoch_label: str, embedding: str, target: str, probe: str, params: dict[str, Any],
                   inner_mae: float, inner_r2: float, inner_spearman: float, split: str,
                   sample_ids: np.ndarray, y: np.ndarray, prediction: np.ndarray) -> None:
    row = {"fold": fold, "epoch_label": epoch_label, "embedding_name": embedding, "target": target,
           "probe": probe, "selected_params": json.dumps(params, sort_keys=True), "inner_cv_mae": inner_mae,
           "inner_cv_r2": inner_r2, "inner_cv_spearman": inner_spearman, "split": split, "n_samples": len(y), **metrics(y, prediction)}
    rows.append(row)
    for sample_id, truth, estimate in zip(sample_ids.astype(str), y, prediction):
        prediction_rows.append({"fold": fold, "epoch_label": epoch_label, "embedding_name": embedding,
                                "target": target, "probe": probe, "split": split, "sample_id": sample_id,
                                "y_true": float(truth), "y_pred": float(estimate)})


def direct_final_rows(rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]], root: Path,
                      fold: str) -> None:
    train = load_embedding(root, fold, "epoch_best", "train", "final_prediction")
    val = load_embedding(root, fold, "epoch_best", "val", "final_prediction")
    for target_index, target in enumerate(TARGETS):
        for split, data in [("train", train), ("validation", val)]:
            append_metrics(rows, prediction_rows, fold=fold, epoch_label="epoch_best", embedding="final_prediction",
                           target=target, probe="GraphGPS_final", params={}, inner_mae=math.nan, inner_r2=math.nan,
                           inner_spearman=math.nan, split=split, sample_ids=data["sample_id"],
                           y=data["labels"][:, target_index], prediction=data["embedding"][:, target_index] * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/frozen_embedding_signal_exp")
    parser.add_argument("--probes-output-dir", type=Path, default=None,
                        help="Optional isolated CSV destination for a parallel embedding shard.")
    parser.add_argument("--skip-manifest", action="store_true",
                        help="Do not append to the shared manifest (used by parallel shards).")
    parser.add_argument("--folds", nargs="*", default=["fold_0", "fold_4"])
    parser.add_argument("--epoch-labels", nargs="*", default=EPOCH_ORDER)
    parser.add_argument("--embeddings", nargs="*", default=EMBEDDINGS)
    parser.add_argument("--probes", nargs="*",
                        default=["GraphGPS_final", "P0_TrainMean", "P1_Ridge", "P2_ElasticNet", "P3_PLS", "P4_ExtraTrees", "P5_RandomForest"])
    args = parser.parse_args()
    root = args.output_root.resolve()
    output = args.probes_output_dir.resolve() if args.probes_output_dir else root / "probes"
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    all_selection: list[dict[str, Any]] = []
    cache: dict[str, tuple[list[ProbeResult], list[dict[str, Any]]]] = {}
    for fold in args.folds:
        if "GraphGPS_final" in args.probes:
            direct_final_rows(all_rows, all_predictions, root, fold)
        for epoch_label in args.epoch_labels:
            for embedding in args.embeddings:
                print(json.dumps({"progress": "representation", "fold": fold, "epoch_label": epoch_label,
                                  "embedding": embedding}), flush=True)
                train = load_embedding(root, fold, epoch_label, "train", embedding)
                validation = load_embedding(root, fold, epoch_label, "val", embedding)
                if not np.array_equal(np.sort(train["sample_id"]), np.sort(train["sample_id"])):
                    raise ValueError("Unexpected sample identity comparison failure")
                if set(train["sample_id"].astype(str)).intersection(set(validation["sample_id"].astype(str))):
                    raise ValueError(f"Train/validation sample leakage in {fold}/{epoch_label}/{embedding}")
                for target_index, target in enumerate(TARGETS):
                    y_train = train["labels"][:, target_index].astype(float)
                    y_val = validation["labels"][:, target_index].astype(float)
                    valid_train = np.isfinite(y_train)
                    valid_val = np.isfinite(y_val)
                    x_train = train["embedding"].astype(float)[valid_train]
                    x_val = validation["embedding"].astype(float)[valid_val]
                    groups = train["group_id"].astype(str)[valid_train]
                    cache_key = stable_hash(x_train, x_val, y_train[valid_train], y_val[valid_val], groups.astype("U")) + target
                    if cache_key not in cache:
                        cache[cache_key] = execute_representation(
                            x_train, y_train[valid_train], groups, x_val, y_val[valid_val], args.probes)
                    results, search_rows = cache[cache_key]
                    for search in search_rows:
                        all_selection.append({"fold": fold, "epoch_label": epoch_label, "embedding_name": embedding,
                                              "target": target, **search})
                    for result in results:
                        append_metrics(all_rows, all_predictions, fold=fold, epoch_label=epoch_label,
                                       embedding=embedding, target=target, probe=result.probe, params=result.params,
                                       inner_mae=result.inner_mae, inner_r2=result.inner_r2,
                                       inner_spearman=result.inner_spearman, split="train",
                                       sample_ids=train["sample_id"][valid_train], y=y_train[valid_train],
                                       prediction=result.train_prediction)
                        append_metrics(all_rows, all_predictions, fold=fold, epoch_label=epoch_label,
                                       embedding=embedding, target=target, probe=result.probe, params=result.params,
                                       inner_mae=result.inner_mae, inner_r2=result.inner_r2,
                                       inner_spearman=result.inner_spearman, split="validation",
                                       sample_ids=validation["sample_id"][valid_val], y=y_val[valid_val],
                                       prediction=result.validation_prediction)
    pd.DataFrame(all_rows).to_csv(output / "probe_metrics.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(output / "probe_predictions.csv", index=False)
    pd.DataFrame(all_selection).to_csv(output / "inner_cv_selection.csv", index=False)
    (output / "protocol.json").write_text(json.dumps({
        "outer_test_opened": False,
        "selection": "GroupKFold(n_splits=5 where possible) on outer-train only; explicit validation is reported after refit.",
        "probes": {"P0": "TrainMean", "P1": "Ridge", "P2": "ElasticNet", "P3": "PLS", "P4": "ExtraTrees", "P5": "RandomForest"},
        "tree_grid": {"n_estimators": 300, "max_depth": [None, 5, 10], "min_samples_leaf": [1, 3, 5], "max_features": [1.0, "sqrt"]},
        "cached_duplicate_embeddings": "Only byte-identical train/validation embeddings reuse a fitted nested-probe result; rows remain emitted for every requested checkpoint/embedding.",
    }, indent=2) + "\n")
    if not args.skip_manifest:
        manifest = root / "execution_manifest.json"
        records = json.loads(manifest.read_text()) if manifest.exists() else []
        records.append({"timestamp": pd.Timestamp.now("UTC").isoformat(), "command": " ".join(sys.argv), "stage": "nested_frozen_probes",
                        "fold": ",".join(args.folds), "split": "outer-train,validation", "epoch": ",".join(args.epoch_labels),
                        "checkpoint": None, "embedding_name": ",".join(args.embeddings), "probe": "P0-P5", "seed": 0,
                        "dataset_hash": None, "manifest_hash": None, "feature_hash": None, "config_hash": None,
                        "checkpoint_hash": None, "embedding_hash": None, "status": "completed", "error": None,
                        "output_path": str(output)})
        manifest.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
