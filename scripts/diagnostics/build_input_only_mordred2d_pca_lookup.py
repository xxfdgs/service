#!/usr/bin/env python3
"""Build PCA-compressed Mordred 2-D descriptors from the fixed input split.

All imputation, standardisation and PCA fitting use molecular occurrences in
the declared training rows only.  No labels, feedback data, validation rows or
test rows are used to fit the representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.decomposition import PCA


SMILES_COLUMNS = ["IL_SMILE", "HL_SMILE", "Chol_SMILE", "PEG_SMILE", "Fifth_SMILE"]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dimension", default=16, type=int)
    arguments = parser.parse_args()
    if arguments.dimension < 1:
        raise ValueError("--dimension must be positive")
    input_csv, manifest_path, output = (arguments.input_csv.resolve(),
                                        arguments.manifest.resolve(),
                                        arguments.output.resolve())
    frame = pd.read_csv(input_csv)
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    if len(frame) != len(manifest) or set(manifest.split) != {"train", "val", "test"}:
        raise RuntimeError("Input/manifest integrity check failed")
    if set(SMILES_COLUMNS) - set(frame.columns):
        raise RuntimeError("Input lacks a five-component SMILES column")

    keys_by_row = frame[SMILES_COLUMNS].map(canonical)
    keys = sorted({key for key in keys_by_row.to_numpy().ravel() if key})
    index = {key: position for position, key in enumerate(keys)}
    # Mordred 1.2 imports the removed numpy.product alias under NumPy 2.x.
    # Restoring the alias is compatibility-only and does not alter descriptors.
    np.product = np.prod
    from mordred import Calculator, descriptors  # pylint: disable=import-outside-toplevel

    calculator = Calculator(descriptors, ignore_3D=True)
    raw_frame = calculator.pandas([Chem.MolFromSmiles(key) for key in keys], quiet=True)
    raw = raw_frame.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    train_indices = manifest.loc[manifest.split.eq("train"), "original_row_index"].astype(int).to_numpy()
    train_keys = keys_by_row.iloc[train_indices].to_numpy().ravel()
    occurrence_indices = np.array([index[key] for key in train_keys if key], dtype=int)
    training = raw[occurrence_indices]
    medians = np.nanmedian(training, axis=0)
    finite_median = np.isfinite(medians)
    raw = raw[:, finite_median]
    training = training[:, finite_median]
    medians = medians[finite_median]
    raw = np.where(np.isfinite(raw), raw, medians)
    training = raw[occurrence_indices]
    center = training.mean(axis=0)
    scale = training.std(axis=0)
    valid = np.isfinite(center) & np.isfinite(scale) & (scale > 1e-12)
    raw, training, center, scale = raw[:, valid], training[:, valid], center[valid], scale[valid]
    standardized = (raw - center) / scale
    train_standardized = standardized[occurrence_indices]
    if arguments.dimension > min(train_standardized.shape):
        raise RuntimeError("Requested PCA dimension exceeds available training matrix rank")
    pca = PCA(n_components=arguments.dimension, svd_solver="full", random_state=0)
    pca.fit(train_standardized)
    transformed = pca.transform(standardized)
    result = pd.DataFrame({"smiles": keys})
    for column in range(arguments.dimension):
        result[f"feature_{column}"] = transformed[:, column]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    metadata = {
        "input_csv": str(input_csv), "input_sha256": digest(input_csv),
        "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
        "fit_rows": int(len(train_indices)), "fit_component_occurrences": int(len(occurrence_indices)),
        "source": "mordred_2d", "raw_descriptor_count": int(raw_frame.shape[1]),
        "finite_nonconstant_descriptor_count": int(standardized.shape[1]),
        "feature_count": int(arguments.dimension),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "labels_read": False, "feedback_read": False,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "unique_molecules": len(result),
                      "feature_count": arguments.dimension, **{key: metadata[key] for key in
                      ("raw_descriptor_count", "finite_nonconstant_descriptor_count", "feedback_read")}}))


if __name__ == "__main__":
    main()
