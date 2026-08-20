"""Lookup for seed-standardized O13-F Fifth semantic features."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem


@lru_cache(maxsize=16)
def _load_lookup(path: str) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    table = pd.read_csv(path)
    if "smiles" not in table:
        raise ValueError(f"Fifth semantic lookup has no smiles column: {path}")
    columns = tuple(column for column in table.columns if column.startswith("feature_"))
    if not columns or table.smiles.duplicated().any():
        raise ValueError(f"Fifth semantic lookup schema is invalid: {path}")
    return columns, {row.smiles: row[list(columns)].to_numpy(dtype=np.float32)
                     for _, row in table.iterrows()}


def fifth_semantic_vector(smiles: object, enabled: bool, path: str, dimension: int) -> np.ndarray:
    if dimension < 0:
        raise ValueError("Fifth semantic feature dimension must be non-negative.")
    if not enabled:
        return np.zeros(dimension, dtype=np.float32)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Fifth semantic feature file not found: {path}")
    molecule = Chem.MolFromSmiles(str(smiles))
    key = Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else ""
    columns, lookup = _load_lookup(path)
    if len(columns) != dimension:
        raise ValueError(f"Expected {dimension} semantic features, found {len(columns)} in {path}")
    return lookup.get(key, np.zeros(dimension, dtype=np.float32))
