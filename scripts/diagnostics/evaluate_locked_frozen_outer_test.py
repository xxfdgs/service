#!/usr/bin/env python3
"""One-time locked outer-test evaluation for frozen-embedding candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    target_std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    prediction_std = float(np.std(prediction, ddof=1)) if len(y) > 1 else 0.0
    tail = np.abs(y - np.median(y)) >= np.quantile(np.abs(y - np.median(y)), .8)
    return {"mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))), "r2": float(r2_score(y, prediction)),
            "pearson": float(pearsonr(y, prediction).statistic) if target_std > 0 and prediction_std > 0 else math.nan,
            "spearman": float(spearmanr(y, prediction).statistic) if prediction_std > 0 else math.nan,
            "kendall_tau": float(kendalltau(y, prediction).statistic) if prediction_std > 0 else math.nan,
            "prediction_std": prediction_std, "target_std": target_std,
            "std_ratio": prediction_std / target_std if target_std else math.nan,
            "calibration_slope": float(np.polyfit(y, prediction, 1)[0]) if target_std > 0 else math.nan,
            "tail_mae": float(mean_absolute_error(y[tail], prediction[tail]))}


def grid() -> list[dict[str, object]]:
    return [{"n_estimators": 300, "max_depth": depth, "min_samples_leaf": leaf, "max_features": features}
            for depth in (None, 5, 10) for leaf in (1, 3, 5) for features in (1.0, "sqrt")]


def fit(params: dict[str, object], x: np.ndarray, y: np.ndarray, n_jobs: int) -> RandomForestRegressor:
    model = RandomForestRegressor(**params, random_state=0, n_jobs=n_jobs)
    model.fit(x, y)
    return model


def choose(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_jobs: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    rows = []
    for params in grid():
        prediction = np.full(len(y), np.nan)
        for train_index, validation_index in cv.split(x, y, groups):
            prediction[validation_index] = fit(params, x[train_index], y[train_index], n_jobs).predict(x[validation_index])
        value = metric(y, prediction)
        rows.append({"params": json.dumps(params, sort_keys=True), "inner_mae": value["mae"],
                     "inner_r2": value["r2"], "inner_spearman": value["spearman"]})
    selected = min(rows, key=lambda row: (row["inner_mae"], -np.nan_to_num(row["inner_spearman"], nan=-1.0)))
    return json.loads(selected["params"]), rows


def archive(root: Path, fold: str, split: str, embedding: str) -> dict[str, np.ndarray]:
    path = root / "embeddings" / fold / "epoch_best" / f"{split}_{embedding}.npz"
    loaded = np.load(path, allow_pickle=False)
    return {name: loaded[name] for name in loaded.files}


def add(rows: list[dict[str, object]], predictions: list[dict[str, object]], fold: str, target: str,
        embedding: str, probe: str, split: str, ids: np.ndarray, y: np.ndarray, pred: np.ndarray,
        params: dict[str, object] | None = None, inner: dict[str, float] | None = None) -> None:
    rows.append({"fold": fold, "target": target, "embedding_name": embedding, "epoch_rule": "epoch_best",
                 "probe": probe, "split": split, "selected_params": json.dumps(params or {}, sort_keys=True),
                 **(inner or {}), **metric(y, pred)})
    predictions.extend({"fold": fold, "target": target, "embedding_name": embedding, "probe": probe,
                        "split": split, "sample_id": str(sample), "y_true": float(truth), "y_pred": float(estimate)}
                       for sample, truth, estimate in zip(ids, y, pred))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    root = ROOT / "results/frozen_embedding_signal_exp"
    lock = pd.read_csv(root / "stage1/candidate_lock.csv")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    cached: dict[str, tuple[dict[str, object], list[dict[str, object]]]] = {}
    for _, candidate in lock.iterrows():
        target = str(candidate.target)
        index = TARGETS.index(target)
        embedding = str(candidate.embedding_name)
        train, test = archive(root, args.fold, "train", embedding), archive(root, args.fold, "test", embedding)
        if set(train["sample_id"].astype(str)).intersection(set(test["sample_id"].astype(str))):
            raise ValueError(f"train/test leakage in {args.fold}/{embedding}")
        x_train, x_test = train["embedding"].astype(float), test["embedding"].astype(float)
        y_train, y_test = train["labels"][:, index].astype(float), test["labels"][:, index].astype(float)
        fingerprint = f"{embedding}/{index}/{x_train.shape}/{float(x_train.sum()):.10g}"
        if fingerprint not in cached:
            cached[fingerprint] = choose(x_train, y_train, train["group_id"].astype(str), args.n_jobs)
        params, inner_rows = cached[fingerprint]
        selected = next(row for row in inner_rows if row["params"] == json.dumps(params, sort_keys=True))
        for row in inner_rows:
            selection_rows.append({"fold": args.fold, "target": target, "embedding_name": embedding, **row})
        model = fit(params, x_train, y_train, args.n_jobs)
        inner = {"inner_cv_mae": selected["inner_mae"], "inner_cv_r2": selected["inner_r2"],
                 "inner_cv_spearman": selected["inner_spearman"]}
        add(metrics_rows, prediction_rows, args.fold, target, embedding, "P5_RandomForest", "train",
            train["sample_id"], y_train, model.predict(x_train), params, inner)
        add(metrics_rows, prediction_rows, args.fold, target, embedding, "P5_RandomForest", "outer_test",
            test["sample_id"], y_test, model.predict(x_test), params, inner)
        train_mean, test_mean = np.full(len(y_train), y_train.mean()), np.full(len(y_test), y_train.mean())
        add(metrics_rows, prediction_rows, args.fold, target, embedding, "P0_TrainMean", "train",
            train["sample_id"], y_train, train_mean)
        add(metrics_rows, prediction_rows, args.fold, target, embedding, "P0_TrainMean", "outer_test",
            test["sample_id"], y_test, test_mean)
        final_train, final_test = archive(root, args.fold, "train", "final_prediction"), archive(root, args.fold, "test", "final_prediction")
        add(metrics_rows, prediction_rows, args.fold, target, "final_prediction", "GraphGPS_final", "train",
            final_train["sample_id"], final_train["labels"][:, index], final_train["embedding"][:, index] * 100.0)
        add(metrics_rows, prediction_rows, args.fold, target, "final_prediction", "GraphGPS_final", "outer_test",
            final_test["sample_id"], final_test["labels"][:, index], final_test["embedding"][:, index] * 100.0)
    pd.DataFrame(metrics_rows).to_csv(output / "locked_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "locked_predictions.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "locked_inner_selection.csv", index=False)
    (output / "protocol.json").write_text(json.dumps({"fold": args.fold, "outer_test_used_once": True,
        "locked_candidates": lock.to_dict(orient="records"), "probe": "P5 RandomForest fixed grid with outer-train GroupKFold selection"}, indent=2) + "\n")
    print(json.dumps({"fold": args.fold, "metrics": len(metrics_rows), "predictions": len(prediction_rows)}))


if __name__ == "__main__":
    main()
