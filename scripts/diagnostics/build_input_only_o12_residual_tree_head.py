#!/usr/bin/env python3
"""Validation-selected residual tree head on frozen input-only GraphGPS predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]
SMILES = ["IL_SMILE", "HL_SMILE", "Chol_SMILE", "PEG_SMILE", "Fifth_SMILE"]
RATIOS = ["mol%_IL", "mol%_HL", "mol%_Chol", "mol%_PEG", "mol%_Fifth"]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value: object) -> str:
    if pd.isna(value) or str(value).lower() in {"nan", "none", "[fr]"}:
        return "[Fr]"
    molecule = Chem.MolFromSmiles(str(value))
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else "[Fr]"


def mol_features(value: object) -> list[float]:
    if pd.isna(value) or str(value).lower() in {"nan", "none", "[fr]"}:
        return [0.0] * 7
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return [0.0] * 7
    return [Descriptors.MolWt(molecule), Crippen.MolLogP(molecule),
            rdMolDescriptors.CalcTPSA(molecule), float(Lipinski.NumHDonors(molecule)),
            float(Lipinski.NumHAcceptors(molecule)), float(Lipinski.NumRotatableBonds(molecule)),
            float(Lipinski.RingCount(molecule))]


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in SMILES:
        result[f"{column}_key"] = frame[column].map(canonical)
    ratios = frame[RATIOS].astype(float).to_numpy() / 100.0
    for index, column in enumerate(RATIOS):
        result[column] = ratios[:, index]
        result[f"{column}_sq"] = ratios[:, index] ** 2
    for left in range(5):
        for right in range(left + 1, 5):
            result[f"ratio_{left + 1}_{right + 1}"] = ratios[:, left] * ratios[:, right]
    values = np.stack([np.asarray([mol_features(value) for value in frame[column]])
                       for column in SMILES], axis=1)
    weighted = (values * ratios[:, :, None]).sum(axis=1)
    for index, name in enumerate(("mw", "logp", "tpsa", "hbd", "hba", "rotors", "rings")):
        result[f"weighted_{name}"] = weighted[:, index]
        result[f"fifth_{name}"] = values[:, 4, index]
    return result


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {"mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "r2": float(r2_score(y, prediction)),
            "pearson": float(pearsonr(y, prediction).statistic) if np.std(prediction) else float("nan"),
            "spearman": float(spearmanr(y, prediction).statistic) if np.std(prediction) else float("nan")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    frame = pd.read_csv(arguments.input_csv)
    manifest = pd.read_csv(arguments.manifest, dtype={"sample_id": str})
    base = pd.read_csv(arguments.base_predictions)
    if len(frame) != len(manifest) or set(manifest.split) != {"train", "val", "test"}:
        raise RuntimeError("Input/manifest integrity check failed")
    features = feature_frame(frame)
    features["sample_id"] = manifest.sample_id.to_numpy(str)
    split_map = manifest.set_index("sample_id").split.to_dict()
    base["sample_id"] = base.sample_id.astype(str)
    base["split"] = base.sample_id.map(split_map)
    if base.split.isna().any() or set(base.target) != set(TARGETS):
        raise RuntimeError("Prediction/manifest alignment failed")
    numeric = [column for column in features if column != "sample_id" and not column.endswith("_key")]
    categorical = [column for column in features if column.endswith("_key")]
    preprocessor = ColumnTransformer([
        ("identity", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ])
    output = arguments.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    rows, selected, all_predictions = [], [], []
    for target in TARGETS:
        part = base.loc[base.target.eq(target)].merge(features, on="sample_id", validate="one_to_one")
        train = part.loc[part.split.eq("train")]; validation = part.loc[part.split.eq("val")]
        pipeline = Pipeline([("features", preprocessor), ("model", ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=3, max_features=0.8, random_state=43, n_jobs=-1))])
        pipeline.fit(train[categorical + numeric], train.y_true - train.y_pred)
        val_residual = pipeline.predict(validation[categorical + numeric])
        # Select the residual strength on validation only.  The candidate grid
        # is intentionally compact to avoid turning the 70-row validation set
        # into a high-variance hyperparameter search.
        strengths = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
        val_mae = [mean_absolute_error(validation.y_true, validation.y_pred + value * val_residual)
                   for value in strengths]
        strength = float(strengths[int(np.argmin(val_mae))])
        selected.append({"target": target, "residual_strength": strength,
                         "validation_mae": float(min(val_mae)), "fit_split": "train"})
        # Test rows are used only after the strength and tree have been fixed.
        for split in ("train", "val", "test"):
            subset = part.loc[part.split.eq(split)].copy()
            prediction = subset.y_pred.to_numpy(float) + strength * pipeline.predict(subset[categorical + numeric])
            subset["y_pred_base"] = subset.y_pred
            subset["y_pred"] = prediction
            all_predictions.append(subset[["sample_id", "split", "target", "y_true", "y_pred", "y_pred_base"]])
            rows.append({"split": split, "target": target, "n": len(subset), **metric(subset.y_true.to_numpy(float), prediction)})
    metrics = pd.DataFrame(rows); metrics.to_csv(output / "metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(selected).to_csv(output / "selection.csv", index=False)
    metrics.groupby("split", as_index=False)[["mae", "rmse", "r2", "pearson", "spearman"]].mean().rename(
        columns={"mae": "mean_mae", "rmse": "mean_rmse", "r2": "mean_r2", "pearson": "mean_pearson", "spearman": "mean_spearman"}
    ).to_csv(output / "metrics_summary.csv", index=False)
    (output / "protocol.json").write_text(json.dumps({"input_only": True, "feedback_read": False,
        "base_predictions": str(arguments.base_predictions.resolve()), "base_predictions_sha256": digest(arguments.base_predictions),
        "tree": "ExtraTreesRegressor(n_estimators=500,min_samples_leaf=3,max_features=0.8)",
        "residual_fit_split": "train", "strength_selection_split": "val", "test_read_after_selection_only": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
