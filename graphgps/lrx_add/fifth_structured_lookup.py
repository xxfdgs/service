"""Train-fitted lookup for O13G low-dimensional Fifth features."""
from __future__ import annotations
import os
from functools import lru_cache
import numpy as np
import pandas as pd
from rdkit import Chem

@lru_cache(maxsize=16)
def _lookup(path: str):
    table = pd.read_csv(path)
    required = {"smiles", "aa_id", "terminal_id", "tail_length_normalized", "tail_length_present_mask"}
    if missing := required - set(table): raise ValueError(f"O13G lookup missing {sorted(missing)}")
    if table.smiles.duplicated().any(): raise ValueError(f"O13G lookup duplicate SMILES: {path}")
    return {row.smiles: row for _, row in table.iterrows()}

def fifth_structured_values(smiles, enabled, path):
    """Return (AA id, terminal id, tail normalized, tail-present mask)."""
    if not enabled: return 0, 0, 0.0, 0.0
    if not path or not os.path.isfile(path): raise FileNotFoundError(f"O13G lookup missing: {path}")
    mol = Chem.MolFromSmiles(str(smiles)); key = Chem.MolToSmiles(mol, canonical=True) if mol else ""
    row = _lookup(path).get(key)
    if row is None: return 0, 0, 0.0, 0.0
    return int(row.aa_id), int(row.terminal_id), float(row.tail_length_normalized), float(row.tail_length_present_mask)
