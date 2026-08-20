#!/usr/bin/env python3
"""Build a train-only selected RDKit-2D molecular descriptor lookup.

The lookup is intended for the descriptor input of ``OneHotEmbedGPS``.  Its
standardisation and descriptor ranking use *only* rows assigned to the fixed
training split; validation and test labels are never loaded.  Ranking is based
on the average absolute correlation between each ratio-weighted mixture
descriptor and the four training targets, with a simple redundancy filter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


SMILES_COLUMNS = ["IL_SMILE", "HL_SMILE", "Chol_SMILE", "PEG_SMILE", "Fifth_SMILE"]
RATIO_COLUMNS = ["mol%_IL", "mol%_HL", "mol%_Chol", "mol%_PEG", "mol%_Fifth"]
TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]


def canonical(value: object) -> str:
    text = str(value)
    if text in {"", "nan", "None", "[Fr]"}:
        return ""
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise ValueError(f"Cannot canonicalize component SMILES: {text}")
    return Chem.MolToSmiles(molecule, canonical=True)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def descriptor_matrix(keys: list[str]) -> tuple[np.ndarray, list[str]]:
    definitions = list(Descriptors._descList)
    names = [name for name, _ in definitions]
    values = np.full((len(keys), len(definitions)), np.nan, dtype=float)
    for row, key in enumerate(keys):
        molecule = Chem.MolFromSmiles(key) if key else None
        if molecule is None:
            continue
        for column, (_, function) in enumerate(definitions):
            try:
                value = float(function(molecule))
            except Exception:
                continue
            values[row, column] = value if np.isfinite(value) else np.nan
    return values, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dimension", default=64, type=int)
    arguments = parser.parse_args()
    if arguments.dimension < 1:
        raise ValueError("--dimension must be positive")

    input_csv = arguments.input_csv.resolve()
    manifest_path = arguments.manifest.resolve()
    output = arguments.output.resolve()
    frame = pd.read_csv(input_csv)
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    if len(frame) != len(manifest) or set(manifest.split) != {"train", "val", "test"}:
        raise RuntimeError("Input/manifest integrity check failed")
    missing = set(SMILES_COLUMNS + RATIO_COLUMNS + TARGETS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"Input lacks required columns: {sorted(missing)}")
    train_indices = manifest.loc[manifest.split.eq("train"), "original_row_index"].astype(int).to_numpy()
    train = frame.iloc[train_indices].reset_index(drop=True)

    component_keys = frame[SMILES_COLUMNS].map(canonical)
    unique_keys = sorted({key for key in component_keys.to_numpy().ravel() if key})
    key_to_index = {key: index for index, key in enumerate(unique_keys)}
    raw, names = descriptor_matrix(unique_keys)

    # Remove unavailable/non-finite and invariant descriptors using training
    # component occurrences only.  This makes both preprocessing stages input
    # split safe and deterministic.
    train_keys = train[SMILES_COLUMNS].map(canonical).to_numpy().ravel()
    train_rows = np.array([key_to_index[key] for key in train_keys if key], dtype=int)
    train_values = raw[train_rows]
    finite = np.isfinite(train_values).all(axis=0)
    center = np.nanmean(train_values, axis=0)
    scale = np.nanstd(train_values, axis=0)
    valid = finite & np.isfinite(center) & np.isfinite(scale) & (scale > 1e-12)
    if not valid.any():
        raise RuntimeError("No finite, non-constant RDKit descriptors on the training split")
    # A molecule outside the fitting split can occasionally make one RDKit
    # descriptor unavailable.  Impute it with the training mean (zero after
    # scaling), never with information from validation/test molecules.
    standardized = np.nan_to_num(
        (raw[:, valid] - center[valid]) / scale[valid], nan=0.0, posinf=0.0, neginf=0.0)
    available_names = np.asarray(names, dtype=object)[valid]

    # Rank only on the training rows.  A descriptor is first collapsed to its
    # ratio-weighted mixture value, then scored evenly across the four targets.
    train_component_indices = np.array([
        [key_to_index.get(key, -1) for key in row]
        for row in train[SMILES_COLUMNS].map(canonical).to_numpy()
    ], dtype=int)
    mixture = np.zeros((len(train), standardized.shape[1]), dtype=float)
    ratios = train[RATIO_COLUMNS].to_numpy(float) / 100.0
    for position in range(len(SMILES_COLUMNS)):
        indices = train_component_indices[:, position]
        present = indices >= 0
        mixture[present] += ratios[present, position, None] * standardized[indices[present]]
    scores = np.zeros(mixture.shape[1], dtype=float)
    for target in TARGETS:
        y = train[target].to_numpy(float)
        for column in range(mixture.shape[1]):
            x = mixture[:, column]
            correlation = np.corrcoef(x, y)[0, 1] if np.std(x) > 1e-12 else np.nan
            scores[column] += abs(correlation) if np.isfinite(correlation) else 0.0
    scores /= len(TARGETS)
    order = np.argsort(-scores, kind="stable")
    selected: list[int] = []
    for candidate in order:
        if any(abs(np.corrcoef(mixture[:, candidate], mixture[:, old])[0, 1]) >= 0.98
               for old in selected if np.std(mixture[:, old]) > 1e-12):
            continue
        selected.append(int(candidate))
        if len(selected) == arguments.dimension:
            break
    if len(selected) < arguments.dimension:
        raise RuntimeError(f"Only {len(selected)} non-redundant descriptors available")

    result = pd.DataFrame({"smiles": unique_keys})
    for output_column, source_column in enumerate(selected):
        result[f"feature_{output_column}"] = standardized[:, source_column]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    metadata = {
        "input_csv": str(input_csv), "input_sha256": digest(input_csv),
        "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
        "fit_rows": int(len(train)), "fit_component_occurrences": int(len(train_rows)),
        "feature_count": int(len(selected)), "source": "rdkit_2d",
        "selection": "training-only ratio-weighted average absolute target correlation",
        "feedback_read": False,
        "selected_descriptors": [str(available_names[index]) for index in selected],
        "selection_scores": [float(scores[index]) for index in selected],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "unique_molecules": len(result),
                      "feature_count": len(selected), "feedback_read": False}))


if __name__ == "__main__":
    main()
