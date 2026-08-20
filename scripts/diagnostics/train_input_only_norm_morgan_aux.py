#!/usr/bin/env python3
"""Train a grouped-input continuous molecular auxiliary head for O12.

Hyperparameters are chosen only by raw-scale MAE on input validation rows
whose fifth-component identities are absent from the corresponding training
split.  Targets are continuous log1p values; no external table or threshold
criterion is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGETS = ("Norm_before", "Norm_after")
SMILES_COLUMNS = (
    "IL_SMILE", "HL_SMILE", "Chol_SMILE", "PEG_SMILE", "Fifth_SMILE")
RATIO_COLUMNS = (
    "mol%_IL", "mol%_HL", "mol%_Chol", "mol%_PEG", "mol%_Fifth")
FP_SIZE = 128
FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=FP_SIZE)
CANDIDATES = tuple(
    (min_samples_leaf, max_features)
    for min_samples_leaf in (2, 5, 10, 20)
    for max_features in (0.35, 0.70)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def molecule_features(value: object) -> np.ndarray:
    result = np.zeros(FP_SIZE + 8, dtype=np.float32)
    if pd.isna(value) or str(value).strip().lower() in {
        "", "nan", "none", "[fr]",
    }:
        return result
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return result
    DataStructs.ConvertToNumpyArray(
        FP_GENERATOR.GetFingerprint(molecule), result[:FP_SIZE])
    result[FP_SIZE:] = (
        Descriptors.MolWt(molecule) / 1000.0,
        Descriptors.MolLogP(molecule) / 10.0,
        Descriptors.TPSA(molecule) / 200.0,
        Lipinski.NumHDonors(molecule) / 10.0,
        Lipinski.NumHAcceptors(molecule) / 20.0,
        Lipinski.NumRotatableBonds(molecule) / 20.0,
        Lipinski.RingCount(molecule) / 10.0,
        Lipinski.FractionCSP3(molecule),
    )
    return result


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = set((*SMILES_COLUMNS, *RATIO_COLUMNS)).difference(frame.columns)
    if missing:
        raise ValueError(f"Feature table misses columns: {sorted(missing)}")
    component_features = np.concatenate([
        np.stack([molecule_features(value) for value in frame[column]])
        for column in SMILES_COLUMNS
    ], axis=1)
    ratios = frame[list(RATIO_COLUMNS)].to_numpy(float) / 100.0
    ratio_parts = [
        ratios, np.square(ratios), np.sqrt(np.maximum(ratios, 0.0))]
    ratio_parts.extend(
        ratios[:, left:left + 1] * ratios[:, right:right + 1]
        for left in range(5)
        for right in range(left + 1, 5)
    )
    classes = (
        frame["Fifth_class"].fillna("__unknown__").astype(str).str.strip().str.lower()
        if "Fifth_class" in frame
        else pd.Series("__unknown__", index=frame.index)
    )
    class_features = np.column_stack([
        classes.eq("__unknown__").to_numpy(float),
        classes.eq("single").to_numpy(float),
        classes.eq("double").to_numpy(float),
    ])
    return np.concatenate(
        [component_features, *ratio_parts, class_features], axis=1
    ).astype(np.float32)


def estimator(seed: int, min_samples_leaf: int, max_features: float,
              trees: int, jobs: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=seed,
        n_jobs=jobs,
    )


def predict_raw(model: ExtraTreesRegressor, features: np.ndarray) -> np.ndarray:
    return np.maximum(np.expm1(model.predict(features)), 0.0)


def manifest_indices(path: Path) -> dict[str, np.ndarray]:
    manifest = pd.read_csv(path)
    if len(manifest) == 0 or set(manifest["split"]) != {
        "train", "val", "test",
    }:
        raise RuntimeError(f"Invalid grouped split manifest: {path}")
    return {
        split: manifest.loc[
            manifest["split"].eq(split), "original_row_index"].to_numpy(int)
        for split in ("train", "val", "test")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=200)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--selection-trees", type=int, default=100)
    parser.add_argument("--final-trees", type=int, default=400)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()

    source = args.input_csv.resolve()
    manifests_root = args.manifests.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, dtype={"ID": str})
    if frame["ID"].duplicated().any():
        raise ValueError("Input IDs must be unique.")
    if frame[list(TARGETS)].isna().any().any():
        raise ValueError("Norm targets must be complete.")
    features = feature_matrix(frame)
    seeds = list(range(args.first_seed, args.first_seed + args.seed_count))
    split_indices = {
        seed: manifest_indices(
            manifests_root / f"fifth_group_manifest_seed{seed}.csv")
        for seed in seeds
    }

    candidate_rows = []
    for target in TARGETS:
        truth = frame[target].to_numpy(float)
        for min_samples_leaf, max_features in CANDIDATES:
            absolute_error_sum = 0.0
            validation_rows = 0
            fold_maes = []
            for seed in seeds:
                indices = split_indices[seed]
                model = estimator(
                    seed, min_samples_leaf, max_features,
                    args.selection_trees, args.jobs)
                model.fit(
                    features[indices["train"]],
                    np.log1p(truth[indices["train"]]),
                )
                prediction = predict_raw(model, features[indices["val"]])
                absolute_error = np.abs(
                    truth[indices["val"]] - prediction)
                absolute_error_sum += float(absolute_error.sum())
                validation_rows += len(absolute_error)
                fold_maes.append(float(absolute_error.mean()))
            candidate_rows.append({
                "target": target,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "folds": len(fold_maes),
                "validation_rows": validation_rows,
                "pooled_validation_mae": (
                    absolute_error_sum / validation_rows),
                "mean_fold_validation_mae": float(np.mean(fold_maes)),
                "std_fold_validation_mae": float(
                    np.std(fold_maes, ddof=1)),
            })
    candidates = pd.DataFrame(candidate_rows)
    candidates["selected"] = False
    selected: dict[str, dict[str, float | int]] = {}
    for target in TARGETS:
        part = candidates.loc[candidates["target"].eq(target)].sort_values(
            ["pooled_validation_mae", "mean_fold_validation_mae",
             "min_samples_leaf", "max_features"])
        index = part.index[0]
        candidates.loc[index, "selected"] = True
        row = candidates.loc[index]
        selected[target] = {
            "min_samples_leaf": int(row["min_samples_leaf"]),
            "max_features": float(row["max_features"]),
            "pooled_validation_mae": float(
                row["pooled_validation_mae"]),
        }
    candidates.to_csv(output / "grouped_candidate_metrics.csv", index=False)

    models: dict[str, dict[int, ExtraTreesRegressor]] = {
        target: {} for target in TARGETS}
    metric_rows = []
    prediction_rows = []
    for target in TARGETS:
        truth = frame[target].to_numpy(float)
        settings = selected[target]
        for seed in seeds:
            indices = split_indices[seed]
            model = estimator(
                seed,
                int(settings["min_samples_leaf"]),
                float(settings["max_features"]),
                args.final_trees,
                args.jobs,
            )
            model.fit(
                features[indices["train"]],
                np.log1p(truth[indices["train"]]),
            )
            models[target][seed] = model
            for split in ("val", "test"):
                subset = indices[split]
                prediction = predict_raw(model, features[subset])
                metric_rows.append({
                    "split_seed": seed,
                    "split": split,
                    "target": target,
                    "n": len(subset),
                    "mae": float(mean_absolute_error(
                        truth[subset], prediction)),
                    "rmse": float(mean_squared_error(
                        truth[subset], prediction) ** 0.5),
                    "r2": float(r2_score(truth[subset], prediction)),
                })
                prediction_rows.extend({
                    "split_seed": seed,
                    "source_index": int(index),
                    "sample_id": str(frame.iloc[index]["ID"]),
                    "split": split,
                    "target": target,
                    "y_true": float(truth[index]),
                    "y_pred": float(value),
                } for index, value in zip(subset, prediction))
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(output / "grouped_fixed_split_metrics.csv", index=False)
    predictions.to_csv(
        output / "grouped_fixed_split_predictions.csv", index=False)
    metrics.groupby(["split", "target"], as_index=False).agg(
        completed_seeds=("split_seed", "nunique"),
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"),
        mean_r2=("r2", "mean"),
    ).to_csv(output / "grouped_fixed_split_metrics_summary.csv", index=False)
    artifact = {
        "models": models,
        "targets": TARGETS,
        "split_seeds": seeds,
        "selected": selected,
        "target_transform": "log1p",
        "feature_schema": (
            "five_component_morgan128_rdkit8_ratios_class"),
        "input_columns": frame.columns.tolist(),
    }
    model_path = output / "input_only_norm_morgan_aux_10seed.joblib"
    joblib.dump(artifact, model_path, compress=3)
    protocol = {
        "input_only": True,
        "external_feedback_read": False,
        "threshold_or_side_criterion_used": False,
        "input_csv": str(source),
        "input_sha256": sha256(source),
        "manifests": str(manifests_root),
        "split_seeds": seeds,
        "selection_metric": (
            "pooled continuous raw-scale validation MAE on "
            "fifth-identity-disjoint input splits"),
        "target_transform": "log1p",
        "candidate_grid": [
            {"min_samples_leaf": leaf, "max_features": feature_fraction}
            for leaf, feature_fraction in CANDIDATES
        ],
        "selected": selected,
        "model": str(model_path),
        "model_sha256": sha256(model_path),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(candidates.loc[candidates["selected"]].to_string(index=False))
    print()
    print(pd.read_csv(
        output / "grouped_fixed_split_metrics_summary.csv").to_string(
            index=False))


if __name__ == "__main__":
    main()
