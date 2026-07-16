#!/usr/bin/env python3
"""One-time sealed outer-test evaluation for development-locked hybrids.

The program refuses to run until Stage 2 has written a successful lock.  For
each outer fold it selects model hyperparameters using that fold's outer-train
GroupKFold only, refits on the full outer train, then opens the test labels
once solely to write metrics.  It never changes the locked feature/model pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_hybrid_embedding_tree_stage as stage
from prepare_hybrid_embedding_tree_experiment import BASE, FROZEN, ROOT, TARGETS, archive, append_execution


OUTPUT = ROOT / "results/hybrid_embedding_tree_exp"
CONFIRM = OUTPUT / "confirmation"
STAGE2 = OUTPUT / "stage2"


def embedding_for_ids(fold: int, alias: str, split_names: tuple[str, ...], sample_ids: pd.Index) -> pd.DataFrame:
    source = stage.EMBEDDING_NAMES[alias]
    chunks = []
    for split in split_names:
        values = archive(FROZEN / "embeddings" / f"fold_{fold}" / "epoch_best" / f"{split}_{source}.npz")
        chunks.append(pd.DataFrame(values["embedding"], index=pd.Index(values["sample_id"].astype(str), name="sample_id")))
    frame = pd.concat(chunks)
    if frame.index.has_duplicates or set(frame.index) != set(sample_ids):
        raise RuntimeError(f"Embedding/test alignment failure fold={fold} alias={alias}")
    frame = frame.loc[sample_ids]
    frame.columns = [f"{alias}_{index:03d}" for index in range(frame.shape[1])]
    return frame


def compose(family: str, target: str, b11_base: str | None, bases: dict[str, pd.DataFrame], embeddings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if family == "B11":
        frame, _ = stage.compose_b11(target, str(b11_base), bases, embeddings)
    else:
        frame, _ = stage.compose_features(family, target, bases, embeddings)
    return frame


def fit_predict(model_name: str, params: dict[str, object], x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame,
                seed: int, tree_n_jobs: int) -> np.ndarray:
    if model_name == "M0":
        return np.full(len(x_test), float(np.mean(y_train)))
    scale = model_name in {"M1", "M2", "M6"}
    transform = stage.preprocessor(x_train, scale=scale).fit(x_train)
    train_values, test_values = transform.transform(x_train), transform.transform(x_test)
    if model_name == "M1":
        model = stage.Ridge(alpha=float(params["alpha"]))
    elif model_name == "M2":
        # Match the bounded, recorded solver policy used for the Stage-1/2
        # inner-CV grid; only alpha and l1_ratio are model-selection knobs.
        model = stage.ElasticNet(alpha=float(params["alpha"]), l1_ratio=float(params["l1_ratio"]), max_iter=1000,
                                 tol=1e-3, random_state=seed)
    elif model_name == "M3":
        model = stage.RandomForestRegressor(**params, random_state=seed, n_jobs=tree_n_jobs)
    elif model_name == "M4":
        model = stage.ExtraTreesRegressor(**params, random_state=seed, n_jobs=tree_n_jobs)
    elif model_name == "M5":
        model = stage.HistGradientBoostingRegressor(**params, random_state=seed)
    elif model_name == "M6":
        model = stage.PLSRegression(n_components=int(params["n_components"]), scale=False, max_iter=1000)
    else:
        raise KeyError(model_name)
    model.fit(train_values, y_train)
    return np.ravel(model.predict(test_values))


def bootstrap_paired(errors_hybrid: np.ndarray, errors_tree: np.ndarray, seed: int, repeats: int) -> dict[str, float]:
    delta = np.asarray(errors_hybrid, float) - np.asarray(errors_tree, float)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(delta), size=(repeats, len(delta)))
    means = delta[choices].mean(axis=1)
    return {"paired_mae_delta": float(delta.mean()), "ci95_low": float(np.quantile(means, .025)),
            "ci95_high": float(np.quantile(means, .975)), "probability_hybrid_better": float((means < 0).mean()),
            "bootstrap_repeats": int(repeats)}


def load_lock() -> pd.DataFrame:
    path = STAGE2 / "locked_pipeline.json"
    if not path.is_file():
        raise RuntimeError("BLOCKED_NO_STAGE2_LOCK")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "READY_FOR_UNTOUCHED_CONFIRMATION":
        raise RuntimeError(f"BLOCKED_STAGE2_STATUS_{payload.get('status')}")
    locked = pd.DataFrame(payload.get("pipelines", []))
    if locked.empty:
        raise RuntimeError("BLOCKED_EMPTY_STAGE2_LOCK")
    return locked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-n-jobs", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    arguments = parser.parse_args()
    locked = load_lock().set_index("target", drop=False)
    CONFIRM.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(BASE / "data_audit/dataset_with_sample_id.csv", dtype={"sample_id": str}).set_index("sample_id", drop=False)
    all_bases = stage.feature_frames()
    selection_rows, metric_rows, prediction_frames = [], [], []
    for fold in range(5):
        manifest = stage.outer_manifest(fold)
        train_ids = pd.Index(manifest.loc[manifest.split.isin(["train", "val"]), "sample_id"].astype(str), name="sample_id")
        test_ids = pd.Index(manifest.loc[manifest.split.eq("test"), "sample_id"].astype(str), name="sample_id")
        groups = manifest.loc[train_ids, "group_id"].to_numpy(str)
        splits = stage.inner_splits(groups)
        train_bases = {name: frame.loc[train_ids] for name, frame in all_bases.items()}
        test_bases = {name: frame.loc[test_ids] for name, frame in all_bases.items()}
        train_embeddings = {alias: embedding_for_ids(fold, alias, ("train", "val"), train_ids) for alias in stage.EMBEDDING_NAMES}
        test_embeddings = {alias: embedding_for_ids(fold, alias, ("test",), test_ids) for alias in stage.EMBEDDING_NAMES}
        for target, lock in locked.iterrows():
            family, model_name = str(lock.feature_family), str(lock.model)
            b11_base = None if pd.isna(lock.get("b11_base")) else str(lock.b11_base)
            x_train = compose(family, target, b11_base, train_bases, train_embeddings)
            x_test = compose(family, target, b11_base, test_bases, test_embeddings)
            y_train = dataset.loc[train_ids, target].to_numpy(float)
            # All hyperparameter selection completes here, before indexing y_test.
            params, inner_predictions, skipped = stage.evaluate_model(model_name, x_train, y_train, splits, 20260715 + fold, arguments.tree_n_jobs)
            chosen = stage.selected_param(params, inner_predictions, y_train)
            chosen_inner = inner_predictions[stage.json_text(chosen)]
            selection_rows.append({"outer_fold": fold, "target": target, "feature_family": family, "b11_base": b11_base,
                                   "model": model_name, "selected_params_json": stage.json_text(chosen),
                                   "inner_selection_mae": stage.metric_dict(y_train, chosen_inner)["mae"],
                                   "selection_data": "outer_train_groupkfold_only", "not_applicable_grid_rows": len(skipped)})
            prediction = fit_predict(model_name, chosen, x_train, y_train, x_test, 20260715 + fold, arguments.tree_n_jobs)
            # Test label access is intentionally after refit/prediction and only for reporting.
            y_test = dataset.loc[test_ids, target].to_numpy(float)
            metric_rows.append({"outer_fold": fold, "target": target, "feature_family": family, "b11_base": b11_base,
                                "model": model_name, "selected_params_json": stage.json_text(chosen), "n": len(test_ids),
                                "outer_test_used_once_after_lock": True, **stage.metric_dict(y_test, prediction)})
            prediction_frames.append(pd.DataFrame({"outer_fold": fold, "sample_id": test_ids, "group_id": manifest.loc[test_ids, "group_id"].to_numpy(str),
                                                   "target": target, "model": "HybridLocked", "feature_family": family, "b11_base": b11_base,
                                                   "selected_params_json": stage.json_text(chosen), "y_true": y_test, "y_pred": prediction,
                                                   "absolute_error": np.abs(y_test - prediction)}))
    selections = pd.DataFrame(selection_rows)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    # This historical OOF table is opened only after the candidate is locked and
    # all hybrid outer predictions have been made.
    tree = pd.read_csv(BASE / "tree_baselines/oof_predictions.csv", dtype={"sample_id": str})
    tree = tree.loc[(tree.protocol == "formula_identity_group_cv") & (tree.model == "NestedSelectedBaseline"),
                    ["outer_fold", "sample_id", "target", "y_true", "y_pred", "absolute_error"]].copy()
    tree = tree.rename(columns={"y_pred": "tree_y_pred", "absolute_error": "tree_absolute_error"})
    comparison = predictions.merge(tree, on=["outer_fold", "sample_id", "target", "y_true"], how="inner", validate="one_to_one")
    if len(comparison) != len(predictions):
        raise RuntimeError("Baseline OOF alignment failed after lock")
    pooled_rows, bootstrap_rows = [], []
    for target, values in comparison.groupby("target"):
        hybrid_metrics = stage.metric_dict(values.y_true, values.y_pred)
        tree_metrics = stage.metric_dict(values.y_true, values.tree_y_pred)
        fold_comparison = values.groupby("outer_fold", as_index=False).apply(
            lambda part: pd.Series({"hybrid_mae": stage.metric_dict(part.y_true, part.y_pred)["mae"],
                                    "tree_mae": stage.metric_dict(part.y_true, part.tree_y_pred)["mae"]}), include_groups=False)
        folds_won = int((fold_comparison.hybrid_mae < fold_comparison.tree_mae).sum())
        bootstrap = bootstrap_paired(values.absolute_error, values.tree_absolute_error, 20260715, arguments.bootstrap_repeats)
        bootstrap_rows.append({"target": target, **bootstrap})
        pooled_rows.append({"target": target, "model": "HybridLocked", "n": len(values), **hybrid_metrics,
                            "tree_mae": tree_metrics["mae"], "tree_r2": tree_metrics["r2"], "tree_spearman": tree_metrics["spearman"],
                            "tree_std_ratio": tree_metrics["std_ratio"], "mae_delta_vs_tree": hybrid_metrics["mae"] - tree_metrics["mae"],
                            "folds_won": folds_won})
    pooled = pd.DataFrame(pooled_rows)
    bootstraps = pd.DataFrame(bootstrap_rows)
    comparison["prediction_delta_vs_tree"] = comparison.y_pred - comparison.tree_y_pred
    comparison["mae_delta_vs_tree"] = comparison.absolute_error - comparison.tree_absolute_error
    untouched = metrics.loc[metrics.outer_fold.isin([2, 3])].merge(
        comparison.loc[comparison.outer_fold.isin([2, 3])].groupby(["outer_fold", "target"], as_index=False).agg(
            tree_mae=("tree_absolute_error", "mean"), hybrid_mae=("absolute_error", "mean")), on=["outer_fold", "target"], how="left")
    untouched["mae_change_pct_vs_tree"] = (untouched.hybrid_mae - untouched.tree_mae) / untouched.tree_mae
    selections.to_csv(CONFIRM / "locked_inner_selection.csv", index=False)
    metrics.to_csv(CONFIRM / "outer_fold_metrics.csv", index=False)
    untouched.to_csv(CONFIRM / "untouched_fold_metrics.csv", index=False)
    pooled.to_csv(CONFIRM / "pooled_oof_metrics.csv", index=False)
    comparison.to_csv(CONFIRM / "pooled_oof_predictions.csv", index=False)
    bootstraps.to_csv(CONFIRM / "paired_bootstrap.csv", index=False)
    (CONFIRM / "confirmation_report.md").write_text("\n".join(["# Locked outer-test confirmation", "",
        "Feature/model pairs were locked from folds 0/4/1 before any test labels were accessed.",
        "Folds 2 and 3 are the untouched confirmation folds; no result-driven model changes were performed."]) + "\n", encoding="utf-8")
    append_execution(OUTPUT, stage="locked_outer_confirmation", target=",".join(locked.index), outer_fold="fold_0..fold_4",
                     feature_family="locked", embedding_name="locked", model="locked", status="completed", output_path=str(CONFIRM))
    print("LOCKED_OUTER_CONFIRMATION_COMPLETE", len(metrics), len(predictions))


if __name__ == "__main__":
    main()
