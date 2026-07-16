import os
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem


@lru_cache(maxsize=8)
def _load_lookup(path):
    table = pd.read_csv(path)
    feature_columns = [column for column in table.columns if column.startswith('feature_')]
    return {row.smiles: row[feature_columns].to_numpy(dtype=np.float32)
            for _, row in table.iterrows()}


def mordred_feature_vector(smiles, enabled, path, dimension):
    if not enabled:
        return np.zeros(dimension, dtype=np.float32)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f'Mordred feature file not found: {path}')
    molecule = Chem.MolFromSmiles(str(smiles))
    key = Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else ''
    vector = _load_lookup(path).get(key)
    if vector is None:
        return np.zeros(dimension, dtype=np.float32)
    if len(vector) != dimension:
        raise ValueError(f'Expected {dimension} Mordred features, got {len(vector)}.')
    return vector
