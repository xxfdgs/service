#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small-data molecular baselines for Norm_before / Norm_after.

Design goals
------------
1. No new_validation label leakage: external labels are used only after inference.
2. Use the project's frozen Fifth-identity OOD manifests for internal evaluation.
3. Compare compact descriptor / fingerprint models against GraphGPS fairly.
4. Report the user's threshold-centric metrics (Recall/F2/FN for y > threshold),
   especially on Fifth_class == 'double'.

Default variants
----------------
- fifth_desc_ridge
- fifth_desc_svr
- fifth_desc_extratrees
- fifth_desc_xgboost
- full_desc_ridge
- full_desc_svr
- full_desc_extratrees
- full_desc_xgboost
- fifth_morgan_ridge
- fifth_morgan_rf
- fifth_morgan_tanimoto_knn
- fifth_desc_morgan_mlp

The script intentionally uses fixed, conservative hyperparameters for the first
screen. Hyperparameter tuning should be performed only after identifying the
strongest representation/model families; this avoids turning ten small OOD
holdouts into a large multiple-comparisons exercise.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVR

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors,
)

try:
    from xgboost import XGBRegressor
except Exception:  # optional at import time
    XGBRegressor = None


COMPONENT_SMILES = [
    "IL_SMILE",
    "HL_SMILE",
    "Chol_SMILE",
    "PEG_SMILE",
    "Fifth_SMILE",
]

RATIO_COLUMNS = [
    "mol%_IL",
    "mol%_HL",
    "mol%_Chol",
    "mol%_PEG",
    "mol%_Fifth",
]

DEFAULT_TARGETS = [
    "Norm_before",
    "Norm_after",
]

DEFAULT_VARIANTS = [
    "fifth_desc_ridge",
    "fifth_desc_svr",
    "fifth_desc_extratrees",
    "fifth_desc_xgboost",
    "fifth_morgan_ridge",
    "fifth_morgan_rf",
    "fifth_morgan_tanimoto_knn",
    "fifth_desc_morgan_mlp",
]

MISSING_SMILES = {
    "",
    "nan",
    "none",
    "null",
    "0",
    "0.0",
    "[fr]",
}

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS,
)


# =============================================================================
# IO / validation
# =============================================================================

def read_csv_robust(path: Path) -> pd.DataFrame:
    errors = []

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
    ):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                dtype={"ID": str},
            )
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")

    raise UnicodeError(
        f"Unable to decode {path}:\n"
        + "\n".join(errors)
    )


def normalize_class(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def validate_frame(
    frame: pd.DataFrame,
    *,
    external: bool,
    targets: list[str],
) -> None:

    required = {
        "ID",
        "Fifth_SMILE",
        "Fifth_class",
        *targets,
    }

    # full descriptor variants need all formulation
    # components + ratios.
    required.update(COMPONENT_SMILES)
    required.update(RATIO_COLUMNS)

    missing = sorted(
        required.difference(frame.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if (
        frame["ID"].isna().any()
        or frame["ID"].astype(str).duplicated().any()
    ):
        kind = (
            "new_validation"
            if external
            else "training"
        )
        raise ValueError(
            f"{kind} ID values must be non-null and unique"
        )


def resolve_manifest(
    manifest_dir: Path,
    seed: int,
) -> Path:

    candidates = [
        manifest_dir
        / f"fifth_identity_manifest_seed{seed}.csv",

        manifest_dir
        / f"split_manifest_seed{seed}.csv",

        manifest_dir
        / f"fifth_group_manifest_seed{seed}.csv",
    ]

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"No Fifth-OOD manifest for seed {seed} "
        f"under {manifest_dir}. Tried: "
        + ", ".join(
            str(x.name)
            for x in candidates
        )
    )


def split_indices_from_manifest(
    data: pd.DataFrame,
    manifest_path: Path,
) -> dict[str, np.ndarray]:

    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "sample_id": str,
            "ID": str,
        },
    )

    if "split" not in manifest.columns:
        raise ValueError(
            f"Manifest lacks split column: "
            f"{manifest_path}"
        )

    id_col = (
        "sample_id"
        if "sample_id" in manifest.columns
        else "ID"
    )

    if id_col not in manifest.columns:
        raise ValueError(
            f"Manifest lacks sample_id/ID: "
            f"{manifest_path}"
        )

    if (
        manifest[id_col].isna().any()
        or manifest[id_col].duplicated().any()
    ):
        raise ValueError(
            f"Manifest IDs are null/duplicated: "
            f"{manifest_path}"
        )

    data_ids = data["ID"].astype(str)

    index_by_id = pd.Series(
        np.arange(len(data), dtype=int),
        index=data_ids,
    )

    missing = sorted(
        set(manifest[id_col].astype(str))
        - set(index_by_id.index)
    )

    if missing:
        raise ValueError(
            "Manifest contains IDs absent from "
            f"training data: {missing[:5]}"
        )

    if len(manifest) != len(data):
        raise ValueError(
            "Manifest must cover all rows "
            f"exactly once: "
            f"{len(manifest)} != {len(data)}"
        )

    out: dict[str, np.ndarray] = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        ids = (
            manifest
            .loc[
                manifest["split"].eq(split),
                id_col,
            ]
            .astype(str)
        )

        out[split] = (
            index_by_id
            .loc[ids]
            .to_numpy(dtype=int)
        )

        if len(out[split]) == 0:
            raise ValueError(
                f"Empty {split} partition "
                f"in {manifest_path}"
            )

    return out


# =============================================================================
# Molecular feature engineering
# =============================================================================

def clean_smiles(value: object) -> str:

    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()

    return (
        ""
        if text.lower() in MISSING_SMILES
        else text
    )


def mol_from_smiles(
    value: object,
) -> Chem.Mol | None:

    text = clean_smiles(value)

    if not text:
        return None

    mol = Chem.MolFromSmiles(text)

    if mol is None:
        raise ValueError(
            "RDKit failed to parse SMILES: "
            f"{text!r}"
        )

    return mol


def safe_float(
    fun: Callable[[], float],
) -> float:

    try:
        value = float(fun())

        return (
            value
            if np.isfinite(value)
            else np.nan
        )

    except Exception:
        return np.nan


def count_atoms(
    mol: Chem.Mol,
    atomic_num: int,
) -> int:

    return sum(
        atom.GetAtomicNum() == atomic_num
        for atom in mol.GetAtoms()
    )


def count_aromatic_atoms(
    mol: Chem.Mol,
) -> int:

    return sum(
        atom.GetIsAromatic()
        for atom in mol.GetAtoms()
    )


def count_ss_bonds(
    mol: Chem.Mol,
) -> int:

    count = 0

    for bond in mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()

        if (
            a.GetAtomicNum() == 16
            and b.GetAtomicNum() == 16
        ):
            count += 1

    return count


def count_smarts(
    mol: Chem.Mol,
    smarts: str,
) -> int:

    pattern = Chem.MolFromSmarts(smarts)

    return (
        len(
            mol.GetSubstructMatches(pattern)
        )
        if pattern is not None
        else 0
    )


DESC_NAMES = [
    "present",
    "MolWt",
    "ExactMolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "RingCount",
    "AromaticRings",
    "AliphaticRings",
    "SaturatedRings",
    "FractionCSP3",
    "HeavyAtomCount",
    "HeteroAtomCount",
    "FormalCharge",
    "ValenceElectrons",
    "NHOHCount",
    "NOCount",
    "LabuteASA",
    "BertzCT",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "AtomC",
    "AtomN",
    "AtomO",
    "AtomP",
    "AtomS",
    "AtomF",
    "AtomCl",
    "AtomBr",
    "AtomI",
    "AromaticAtomCount",
    "AmideCount",
    "EsterCount",
    "CarboxylCount",
    "SSBondCount",
]


def descriptor_vector(
    value: object,
) -> np.ndarray:

    mol = mol_from_smiles(value)

    if mol is None:
        return np.zeros(
            len(DESC_NAMES),
            dtype=np.float32,
        )

    values = [
        1.0,

        safe_float(
            lambda: Descriptors.MolWt(mol)
        ),

        safe_float(
            lambda: Descriptors.ExactMolWt(mol)
        ),

        safe_float(
            lambda: Crippen.MolLogP(mol)
        ),

        safe_float(
            lambda: Crippen.MolMR(mol)
        ),

        safe_float(
            lambda: rdMolDescriptors.CalcTPSA(mol)
        ),

        safe_float(
            lambda: Lipinski.NumHDonors(mol)
        ),

        safe_float(
            lambda: Lipinski.NumHAcceptors(mol)
        ),

        safe_float(
            lambda: Lipinski.NumRotatableBonds(mol)
        ),

        safe_float(
            lambda: Lipinski.RingCount(mol)
        ),

        safe_float(
            lambda: Lipinski.NumAromaticRings(mol)
        ),

        safe_float(
            lambda: Lipinski.NumAliphaticRings(mol)
        ),

        safe_float(
            lambda: Lipinski.NumSaturatedRings(mol)
        ),

        safe_float(
            lambda: rdMolDescriptors.CalcFractionCSP3(mol)
        ),

        safe_float(
            lambda: Lipinski.HeavyAtomCount(mol)
        ),

        safe_float(
            lambda: Lipinski.NumHeteroatoms(mol)
        ),

        safe_float(
            lambda: Chem.GetFormalCharge(mol)
        ),

        safe_float(
            lambda: Descriptors.NumValenceElectrons(mol)
        ),

        safe_float(
            lambda: Lipinski.NHOHCount(mol)
        ),

        safe_float(
            lambda: Lipinski.NOCount(mol)
        ),

        safe_float(
            lambda: rdMolDescriptors.CalcLabuteASA(mol)
        ),

        safe_float(
            lambda: Descriptors.BertzCT(mol)
        ),

        safe_float(
            lambda: Descriptors.Kappa1(mol)
        ),

        safe_float(
            lambda: Descriptors.Kappa2(mol)
        ),

        safe_float(
            lambda: Descriptors.Kappa3(mol)
        ),

        float(
            count_atoms(mol, 6)
        ),

        float(
            count_atoms(mol, 7)
        ),

        float(
            count_atoms(mol, 8)
        ),

        float(
            count_atoms(mol, 15)
        ),

        float(
            count_atoms(mol, 16)
        ),

        float(
            count_atoms(mol, 9)
        ),

        float(
            count_atoms(mol, 17)
        ),

        float(
            count_atoms(mol, 35)
        ),

        float(
            count_atoms(mol, 53)
        ),

        float(
            count_aromatic_atoms(mol)
        ),

        float(
            count_smarts(
                mol,
                "[CX3](=[OX1])[NX3]",
            )
        ),

        float(
            count_smarts(
                mol,
                "[CX3](=[OX1])[OX2][#6]",
            )
        ),

        float(
            count_smarts(
                mol,
                "[CX3](=[OX1])[OX2H1]",
            )
        ),

        float(
            count_ss_bonds(mol)
        ),
    ]

    if len(values) != len(DESC_NAMES):
        raise RuntimeError(
            "Descriptor schema length mismatch"
        )

    return np.asarray(
        values,
        dtype=np.float32,
    )


def morgan_bitvect(
    value: object,
) -> DataStructs.ExplicitBitVect:

    mol = mol_from_smiles(value)

    if mol is None:
        return DataStructs.ExplicitBitVect(
            MORGAN_BITS
        )

    return _MORGAN_GENERATOR.GetFingerprint(
        mol
    )


def bitvect_to_array(
    fp: DataStructs.ExplicitBitVect,
) -> np.ndarray:

    arr = np.zeros(
        (MORGAN_BITS,),
        dtype=np.uint8,
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        arr,
    )

    return arr


@dataclass
class FeatureBundle:
    fifth_desc: np.ndarray
    full_desc: np.ndarray
    fifth_morgan: np.ndarray
    fifth_desc_morgan: np.ndarray
    fifth_fps: list[
        DataStructs.ExplicitBitVect
    ]
    audit: dict


def build_features(
    frame: pd.DataFrame,
) -> FeatureBundle:

    n = len(frame)

    # Cache descriptor vectors per unique
    # SMILES so repeated formulation
    # components are cheap and exactly
    # consistent.
    desc_cache: dict[
        str,
        np.ndarray,
    ] = {}

    fp_cache: dict[
        str,
        DataStructs.ExplicitBitVect,
    ] = {}

    def get_desc(
        value: object,
    ) -> np.ndarray:

        key = clean_smiles(value)

        if key not in desc_cache:
            desc_cache[key] = (
                descriptor_vector(value)
            )

        return desc_cache[key]

    def get_fp(
        value: object,
    ) -> DataStructs.ExplicitBitVect:

        key = clean_smiles(value)

        if key not in fp_cache:
            fp_cache[key] = (
                morgan_bitvect(value)
            )

        return fp_cache[key]

    fifth_desc = np.vstack([
        get_desc(v)
        for v in frame["Fifth_SMILE"]
    ]).astype(np.float32)

    fifth_fps = [
        get_fp(v)
        for v in frame["Fifth_SMILE"]
    ]

    fifth_morgan = np.vstack([
        bitvect_to_array(fp)
        for fp in fifth_fps
    ]).astype(np.float32)

    ratios = (
        frame[RATIO_COLUMNS]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=np.float32)
    )

    # Data are conventionally in mol%;
    # normalize to fractions for numerical scale.
    ratios_frac = ratios / 100.0

    component_mats = []

    for col in COMPONENT_SMILES:
        component_mats.append(
            np.vstack([
                get_desc(v)
                for v in frame[col]
            ]).astype(np.float32)
        )

    stacked_components = np.stack(
        component_mats,
        axis=1,
    )
    # shape:
    # [N, 5, descriptor_dim]

    finite_ratio = np.where(
        np.isfinite(ratios_frac),
        ratios_frac,
        0.0,
    )

    denom = finite_ratio.sum(
        axis=1,
        keepdims=True,
    )

    safe_denom = np.where(
        np.abs(denom) > 1e-12,
        denom,
        1.0,
    )

    norm_ratio = (
        finite_ratio
        / safe_denom
    )

    weighted_desc = (
        stacked_components
        * norm_ratio[:, :, None]
    ).sum(axis=1)

    cls = normalize_class(
        frame["Fifth_class"]
    )

    class_features = np.column_stack([
        cls.eq("single").to_numpy(
            dtype=np.float32
        ),
        cls.eq("double").to_numpy(
            dtype=np.float32
        ),
    ])

    # Full formulation representation:
    #
    # - descriptor of each component
    # - weighted formulation descriptor
    # - mol ratios
    # - single/double indicator
    full_desc = np.concatenate(
        [
            stacked_components.reshape(
                n,
                -1,
            ),
            weighted_desc,
            ratios_frac,
            class_features,
        ],
        axis=1,
    ).astype(np.float32)

    # Combined small-MLP representation:
    #
    # - Fifth descriptors
    # - Fifth Morgan fingerprint
    # - formulation ratios
    # - class indicator
    fifth_desc_morgan = np.concatenate(
        [
            fifth_desc,
            fifth_morgan,
            ratios_frac,
            class_features,
        ],
        axis=1,
    ).astype(np.float32)

    audit = {
        "rows": int(n),

        "descriptor_dim":
            int(len(DESC_NAMES)),

        "fifth_desc_dim":
            int(fifth_desc.shape[1]),

        "full_desc_dim":
            int(full_desc.shape[1]),

        "morgan_bits":
            MORGAN_BITS,

        "fifth_desc_morgan_dim":
            int(
                fifth_desc_morgan.shape[1]
            ),

        "unique_descriptor_smiles":
            int(len(desc_cache)),

        "unique_fingerprint_smiles":
            int(len(fp_cache)),

        "nonfinite_fifth_desc":
            int(
                (
                    ~np.isfinite(
                        fifth_desc
                    )
                ).sum()
            ),

        "nonfinite_full_desc":
            int(
                (
                    ~np.isfinite(
                        full_desc
                    )
                ).sum()
            ),
    }

    return FeatureBundle(
        fifth_desc=fifth_desc,
        full_desc=full_desc,
        fifth_morgan=fifth_morgan,
        fifth_desc_morgan=fifth_desc_morgan,
        fifth_fps=fifth_fps,
        audit=audit,
    )


# =============================================================================
# Models
# =============================================================================

class TanimotoKNNRegressor:

    def __init__(
        self,
        k: int = 5,
        alpha: float = 4.0,
        min_similarity: float = 1e-12,
    ):
        self.k = int(k)
        self.alpha = float(alpha)
        self.min_similarity = float(
            min_similarity
        )

        self.fps_: list[
            DataStructs.ExplicitBitVect
        ] | None = None

        self.y_: np.ndarray | None = None

        self.mean_: float | None = None

    def fit(
        self,
        fps: list[
            DataStructs.ExplicitBitVect
        ],
        y: np.ndarray,
    ):

        if len(fps) != len(y):
            raise ValueError(
                "fps/y length mismatch"
            )

        self.fps_ = list(fps)

        self.y_ = np.asarray(
            y,
            dtype=float,
        )

        self.mean_ = float(
            np.mean(self.y_)
        )

        return self

    def predict(
        self,
        fps: list[
            DataStructs.ExplicitBitVect
        ],
    ) -> np.ndarray:

        if (
            self.fps_ is None
            or self.y_ is None
            or self.mean_ is None
        ):
            raise RuntimeError(
                "TanimotoKNNRegressor "
                "is not fitted"
            )

        out = []

        k = min(
            self.k,
            len(self.fps_),
        )

        for fp in fps:

            sims = np.asarray(
                DataStructs
                .BulkTanimotoSimilarity(
                    fp,
                    self.fps_,
                ),
                dtype=float,
            )

            if k < len(sims):

                idx = np.argpartition(
                    -sims,
                    k - 1,
                )[:k]

            else:
                idx = np.arange(
                    len(sims)
                )

            idx = idx[
                np.argsort(
                    -sims[idx]
                )
            ]

            weights = np.power(
                np.maximum(
                    sims[idx],
                    0.0,
                ),
                self.alpha,
            )

            if (
                float(weights.sum())
                <= self.min_similarity
            ):
                out.append(
                    self.mean_
                )

            else:
                out.append(
                    float(
                        np.dot(
                            weights,
                            self.y_[idx],
                        )
                        / weights.sum()
                    )
                )

        return np.asarray(
            out,
            dtype=float,
        )

    def neighbor_rows(
        self,
        query_fps: list[
            DataStructs.ExplicitBitVect
        ],
        query_ids: Iterable[str],
        train_ids: Iterable[str],
        train_y: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:

        if self.fps_ is None:
            raise RuntimeError(
                "TanimotoKNNRegressor "
                "is not fitted"
            )

        train_ids = list(
            map(
                str,
                train_ids,
            )
        )

        rows = []

        for qid, fp in zip(
            map(str, query_ids),
            query_fps,
        ):

            sims = np.asarray(
                DataStructs
                .BulkTanimotoSimilarity(
                    fp,
                    self.fps_,
                ),
                dtype=float,
            )

            order = np.argsort(
                -sims
            )[
                :min(
                    top_k,
                    len(sims),
                )
            ]

            for rank, idx in enumerate(
                order,
                start=1,
            ):

                rows.append({
                    "query_id":
                        qid,

                    "rank":
                        rank,

                    "train_id":
                        train_ids[idx],

                    "tanimoto":
                        float(
                            sims[idx]
                        ),

                    "train_y":
                        float(
                            train_y[idx]
                        ),
                })

        return rows


def make_model(
    name: str,
    seed: int,
):

    if name.endswith("_ridge"):

        return Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "variance",
                VarianceThreshold(
                    threshold=1e-10
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(
                    alpha=20.0
                ),
            ),
        ])

    if name.endswith("_svr"):

        return Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "variance",
                VarianceThreshold(
                    threshold=1e-10
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                SVR(
                    C=10.0,
                    gamma="scale",
                    epsilon=0.10,
                    kernel="rbf",
                ),
            ),
        ])

    if name.endswith(
        "_extratrees"
    ):

        return Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=600,
                    max_features=0.70,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ])

    if name.endswith(
        "_xgboost"
    ):

        if XGBRegressor is None:
            raise RuntimeError(
                "xgboost is not installed "
                "but an XGBoost variant "
                "was requested"
            )

        return Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                XGBRegressor(
                    objective=
                        "reg:squarederror",

                    n_estimators=500,

                    max_depth=3,

                    learning_rate=0.03,

                    subsample=0.85,

                    colsample_bytree=0.80,

                    reg_lambda=5.0,

                    reg_alpha=0.05,

                    random_state=seed,

                    n_jobs=-1,

                    verbosity=0,
                ),
            ),
        ])

    if name == "fifth_morgan_rf":

        return RandomForestRegressor(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )

    if (
        name
        == "fifth_desc_morgan_mlp"
    ):

        return Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "variance",
                VarianceThreshold(
                    threshold=1e-10
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(
                        64,
                        32,
                    ),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=40,
                    random_state=seed,
                ),
            ),
        ])

    raise KeyError(
        "No model factory for "
        f"variant {name}"
    )


VARIANTS = {
    "fifth_desc_ridge":
        "fifth_desc",

    "fifth_desc_svr":
        "fifth_desc",

    "fifth_desc_extratrees":
        "fifth_desc",

    "fifth_desc_xgboost":
        "fifth_desc",

    "full_desc_ridge":
        "full_desc",

    "full_desc_svr":
        "full_desc",

    "full_desc_extratrees":
        "full_desc",

    "full_desc_xgboost":
        "full_desc",

    "fifth_morgan_ridge":
        "fifth_morgan",

    "fifth_morgan_rf":
        "fifth_morgan",

    "fifth_morgan_tanimoto_knn":
        "fifth_fps",

    "fifth_desc_morgan_mlp":
        "fifth_desc_morgan",
}


def subset_features(
    bundle: FeatureBundle,
    feature_kind: str,
    indices: np.ndarray,
):

    if feature_kind == "fifth_fps":

        return [
            bundle.fifth_fps[
                int(i)
            ]
            for i in indices
        ]

    return getattr(
        bundle,
        feature_kind,
    )[indices]


def fit_predict_variant(
    variant: str,
    bundle: FeatureBundle,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y: np.ndarray,
    seed: int,
):

    feature_kind = VARIANTS[
        variant
    ]

    finite_train = train_idx[
        np.isfinite(
            y[train_idx]
        )
    ]

    finite_test = test_idx[
        np.isfinite(
            y[test_idx]
        )
    ]

    if len(finite_train) < 5:
        raise ValueError(
            "Too few finite training "
            f"labels for {variant}: "
            f"{len(finite_train)}"
        )

    y_train = y[
        finite_train
    ]

    if (
        variant
        == "fifth_morgan_tanimoto_knn"
    ):

        model = (
            TanimotoKNNRegressor(
                k=5,
                alpha=4.0,
            )
        )

        model.fit(
            subset_features(
                bundle,
                feature_kind,
                finite_train,
            ),
            y_train,
        )

        pred = model.predict(
            subset_features(
                bundle,
                feature_kind,
                finite_test,
            )
        )

    else:

        model = make_model(
            variant,
            seed,
        )

        model.fit(
            subset_features(
                bundle,
                feature_kind,
                finite_train,
            ),
            y_train,
        )

        pred = np.asarray(
            model.predict(
                subset_features(
                    bundle,
                    feature_kind,
                    finite_test,
                )
            ),
            dtype=float,
        ).reshape(-1)

    return (
        finite_test,
        pred,
        model,
        finite_train,
    )


# =============================================================================
# Metrics
# =============================================================================

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> dict:

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    n = len(y_true)

    if n == 0:

        return {
            k: np.nan
            for k in [
                "mae",
                "r2",
                "spearman",
                "recall_gt1",
                "precision_gt1",
                "f2_gt1",
                "fn",
                "fp",
                "true_gt1",
                "pred_gt1",
                "mean_pred",
                "std_pred",
            ]
        } | {
            "n": 0
        }

    true_pos = (
        y_true > threshold
    )

    pred_pos = (
        y_pred > threshold
    )

    tp = int(
        np.sum(
            true_pos
            & pred_pos
        )
    )

    fn = int(
        np.sum(
            true_pos
            & ~pred_pos
        )
    )

    fp = int(
        np.sum(
            ~true_pos
            & pred_pos
        )
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else np.nan
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else np.nan
    )

    beta2 = 4.0

    if (
        np.isfinite(recall)
        and np.isfinite(precision)
        and (
            beta2 * precision
            + recall
        ) > 0
    ):

        f2 = (
            (1.0 + beta2)
            * precision
            * recall
            / (
                beta2 * precision
                + recall
            )
        )

    else:
        f2 = np.nan

    if (
        n >= 2
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            spear = float(
                spearmanr(
                    y_true,
                    y_pred,
                ).statistic
            )

    else:
        spear = np.nan

    return {
        "n":
            int(n),

        "mae":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),

        "r2":
            (
                float(
                    r2_score(
                        y_true,
                        y_pred,
                    )
                )
                if n >= 2
                else np.nan
            ),

        "spearman":
            spear,

        "recall_gt1":
            (
                float(recall)
                if np.isfinite(recall)
                else np.nan
            ),

        "precision_gt1":
            (
                float(precision)
                if np.isfinite(precision)
                else np.nan
            ),

        "f2_gt1":
            (
                float(f2)
                if np.isfinite(f2)
                else np.nan
            ),

        "fn":
            fn,

        "fp":
            fp,

        "true_gt1":
            int(
                np.sum(true_pos)
            ),

        "pred_gt1":
            int(
                np.sum(pred_pos)
            ),

        "mean_pred":
            float(
                np.mean(y_pred)
            ),

        "std_pred":
            float(
                np.std(
                    y_pred,
                    ddof=0,
                )
            ),
    }


def add_subset_metrics(
    rows: list[dict],
    *,
    scope: str,
    variant: str,
    target: str,
    seed: int | str,
    frame: pd.DataFrame,
    indices: np.ndarray,
    pred: np.ndarray,
    threshold: float,
) -> None:

    classes = normalize_class(
        frame.iloc[
            indices
        ]["Fifth_class"]
    ).to_numpy()

    y_true = pd.to_numeric(
        frame.iloc[
            indices
        ][target],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    for subset in (
        "all",
        "single",
        "double",
    ):

        if subset == "all":

            mask = np.ones(
                len(indices),
                dtype=bool,
            )

        else:

            mask = (
                classes == subset
            )

        metrics = regression_metrics(
            y_true[mask],
            pred[mask],
            threshold,
        )

        rows.append({
            "scope":
                scope,

            "variant":
                variant,

            "target":
                target,

            "seed":
                seed,

            "subset":
                subset,

            **metrics,
        })


def summarize_ood(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:

    metric_cols = [
        "mae",
        "r2",
        "spearman",
        "recall_gt1",
        "precision_gt1",
        "f2_gt1",
        "fn",
        "fp",
        "mean_pred",
        "std_pred",
    ]

    rows = []

    for keys, group in (
        per_seed.groupby(
            [
                "variant",
                "target",
                "subset",
            ],
            sort=False,
        )
    ):

        (
            variant,
            target,
            subset,
        ) = keys

        row = {
            "variant":
                variant,

            "target":
                target,

            "subset":
                subset,

            "seeds":
                int(
                    group[
                        "seed"
                    ].nunique()
                ),
        }

        for col in metric_cols:

            values = pd.to_numeric(
                group[col],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            finite = values[
                np.isfinite(values)
            ]

            row[
                f"{col}_mean"
            ] = (
                float(
                    np.mean(finite)
                )
                if len(finite)
                else np.nan
            )

            row[
                f"{col}_std"
            ] = (
                float(
                    np.std(
                        finite,
                        ddof=1,
                    )
                )
                if len(finite) > 1
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Experiment driver
# =============================================================================

def run_ood(
    train: pd.DataFrame,
    bundle: FeatureBundle,
    manifest_dir: Path,
    seeds: list[int],
    targets: list[str],
    variants: list[str],
    threshold: float,
    output_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    rows: list[dict] = []

    prediction_rows: list[
        pd.DataFrame
    ] = []

    for seed in seeds:

        manifest_path = (
            resolve_manifest(
                manifest_dir,
                seed,
            )
        )

        split = (
            split_indices_from_manifest(
                train,
                manifest_path,
            )
        )

        print(
            f"[OOD seed={seed}] "
            f"train={len(split['train'])} "
            f"val={len(split['val'])} "
            f"test={len(split['test'])} "
            f"manifest="
            f"{manifest_path.name}",
            flush=True,
        )

        for target in targets:

            y = pd.to_numeric(
                train[target],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            for variant in variants:

                (
                    test_idx,
                    pred,
                    _,
                    _,
                ) = fit_predict_variant(
                    variant,
                    bundle,
                    split["train"],
                    split["test"],
                    y,
                    seed,
                )

                add_subset_metrics(
                    rows,
                    scope=
                        "internal_ood_test",

                    variant=
                        variant,

                    target=
                        target,

                    seed=
                        seed,

                    frame=
                        train,

                    indices=
                        test_idx,

                    pred=
                        pred,

                    threshold=
                        threshold,
                )

                prediction_rows.append(
                    pd.DataFrame({
                        "seed":
                            seed,

                        "target":
                            target,

                        "variant":
                            variant,

                        "ID":
                            train
                            .iloc[test_idx][
                                "ID"
                            ]
                            .astype(str)
                            .to_numpy(),

                        "Fifth_class":
                            normalize_class(
                                train
                                .iloc[test_idx][
                                    "Fifth_class"
                                ]
                            )
                            .to_numpy(),

                        "y_true":
                            y[test_idx],

                        "y_pred":
                            pred,
                    })
                )

    per_seed = pd.DataFrame(
        rows
    )

    summary = summarize_ood(
        per_seed
    )

    per_seed.to_csv(
        output_dir
        / "ood_metrics_per_seed.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "ood_metrics_summary.csv",
        index=False,
    )

    if prediction_rows:

        pd.concat(
            prediction_rows,
            ignore_index=True,
        ).to_csv(
            output_dir
            / "ood_predictions_long.csv",
            index=False,
        )

    return (
        per_seed,
        summary,
    )


def run_external(
    train: pd.DataFrame,
    train_bundle: FeatureBundle,
    external: pd.DataFrame,
    external_bundle: FeatureBundle,
    targets: list[str],
    variants: list[str],
    threshold: float,
    output_dir: Path,
    final_seed: int,
) -> pd.DataFrame:

    rows: list[dict] = []

    long_predictions: list[
        pd.DataFrame
    ] = []

    neighbor_rows: list[
        dict
    ] = []

    all_train_idx = np.arange(
        len(train),
        dtype=int,
    )

    all_external_idx = np.arange(
        len(external),
        dtype=int,
    )

    for target in targets:

        y_train = pd.to_numeric(
            train[target],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        finite_train = all_train_idx[
            np.isfinite(
                y_train
            )
        ]

        y_external = pd.to_numeric(
            external[target],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        for variant in variants:

            feature_kind = VARIANTS[
                variant
            ]

            if (
                variant
                == "fifth_morgan_tanimoto_knn"
            ):

                model = (
                    TanimotoKNNRegressor(
                        k=5,
                        alpha=4.0,
                    )
                )

                model.fit(
                    subset_features(
                        train_bundle,
                        feature_kind,
                        finite_train,
                    ),
                    y_train[
                        finite_train
                    ],
                )

                pred = model.predict(
                    subset_features(
                        external_bundle,
                        feature_kind,
                        all_external_idx,
                    )
                )

                if target == targets[0]:

                    nn = model.neighbor_rows(
                        subset_features(
                            external_bundle,
                            feature_kind,
                            all_external_idx,
                        ),

                        external[
                            "ID"
                        ]
                        .astype(str)
                        .tolist(),

                        train
                        .iloc[
                            finite_train
                        ][
                            "ID"
                        ]
                        .astype(str)
                        .tolist(),

                        y_train[
                            finite_train
                        ],

                        top_k=5,
                    )

                    for row in nn:
                        row[
                            "target"
                        ] = target

                    neighbor_rows.extend(
                        nn
                    )

            else:

                model = make_model(
                    variant,
                    final_seed,
                )

                model.fit(
                    subset_features(
                        train_bundle,
                        feature_kind,
                        finite_train,
                    ),
                    y_train[
                        finite_train
                    ],
                )

                pred = np.asarray(
                    model.predict(
                        subset_features(
                            external_bundle,
                            feature_kind,
                            all_external_idx,
                        )
                    ),
                    dtype=float,
                ).reshape(-1)

            add_subset_metrics(
                rows,
                scope=
                    "new_validation",

                variant=
                    variant,

                target=
                    target,

                seed=
                    "all700",

                frame=
                    external,

                indices=
                    all_external_idx,

                pred=
                    pred,

                threshold=
                    threshold,
            )

            long_predictions.append(
                pd.DataFrame({
                    "target":
                        target,

                    "variant":
                        variant,

                    "ID":
                        external[
                            "ID"
                        ]
                        .astype(str)
                        .to_numpy(),

                    "Fifth":
                        (
                            external[
                                "Fifth"
                            ]
                            .astype(str)
                            .to_numpy()
                            if (
                                "Fifth"
                                in external.columns
                            )
                            else ""
                        ),

                    "Fifth_class":
                        normalize_class(
                            external[
                                "Fifth_class"
                            ]
                        )
                        .to_numpy(),

                    "y_true":
                        y_external,

                    "y_pred":
                        pred,
                })
            )

    metrics = pd.DataFrame(
        rows
    )

    predictions = pd.concat(
        long_predictions,
        ignore_index=True,
    )

    metrics.to_csv(
        output_dir
        / "new_validation_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        output_dir
        / "new_validation_predictions_long.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Plotting:
    #
    # one figure per model
    #
    # left  = Norm_before
    # right = Norm_after
    #
    # Within each subplot:
    # single = circle
    # double = triangle
    #
    # Auxiliary lines:
    # y = x
    # x = threshold
    # y = threshold
    # -------------------------------------------------------------------------

    plot_new_validation_scatter(
        predictions=predictions,
        metrics=metrics,
        output_dir=output_dir,
        threshold=threshold,
    )

    if neighbor_rows:

        pd.DataFrame(
            neighbor_rows
        ).to_csv(
            output_dir
            / (
                "new_validation_"
                "tanimoto_neighbors_top5.csv"
            ),
            index=False,
        )

    # One wide table per target is
    # convenient for direct sample-level
    # inspection.
    for target in targets:

        block = predictions.loc[
            predictions[
                "target"
            ].eq(target)
        ].copy()

        base_cols = [
            "ID",
            "Fifth",
            "Fifth_class",
            "y_true",
        ]

        wide = (
            block[
                base_cols
            ]
            .drop_duplicates(
                "ID"
            )
            .set_index(
                "ID"
            )
        )

        for variant in variants:

            part = (
                block
                .loc[
                    block[
                        "variant"
                    ].eq(variant),
                    [
                        "ID",
                        "y_pred",
                    ],
                ]
                .set_index(
                    "ID"
                )
            )

            wide[
                f"pred_{variant}"
            ] = part[
                "y_pred"
            ]

        wide.reset_index().to_csv(
            output_dir
            / (
                "new_validation_predictions_"
                f"{target}_wide.csv"
            ),
            index=False,
        )

    return metrics


def print_rankings(
    metrics: pd.DataFrame,
    title: str,
) -> None:

    if metrics.empty:
        return

    print(
        "\n"
        + "=" * 100
    )

    print(title)

    print(
        "=" * 100
    )

    # double is the user's primary risk
    # subset; show MAE + Recall/F2 together.
    x = metrics.loc[
        metrics[
            "subset"
        ].eq("double")
    ].copy()

    if "mae_mean" in x.columns:

        cols = [
            "target",
            "variant",
            "mae_mean",
            "r2_mean",
            "spearman_mean",
            "recall_gt1_mean",
            "f2_gt1_mean",
            "fn_mean",
            "fp_mean",
        ]

        x = x.sort_values(
            [
                "target",
                "recall_gt1_mean",
                "mae_mean",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )

    else:

        cols = [
            "target",
            "variant",
            "mae",
            "r2",
            "spearman",
            "recall_gt1",
            "f2_gt1",
            "fn",
            "fp",
        ]

        x = x.sort_values(
            [
                "target",
                "recall_gt1",
                "mae",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )

    cols = [
        c
        for c in cols
        if c in x.columns
    ]

    with pd.option_context(
        "display.max_rows",
        200,
        "display.width",
        180,
    ):
        print(
            x[cols].to_string(
                index=False
            )
        )


# =============================================================================
# Plotting
# =============================================================================

def safe_slug(
    text: str,
) -> str:

    chars = []

    for ch in str(text):

        chars.append(
            ch
            if (
                ch.isalnum()
                or ch in {
                    "-",
                    "_",
                }
            )
            else "_"
        )

    return (
        "".join(chars).strip("_")
        or "item"
    )


def plot_new_validation_scatter(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    threshold: float,
) -> None:
    """Plot new_validation true-vs-predicted scatter by model.

    For every model variant, create one figure with two subplots:

        left  : Norm_before
        right : Norm_after

    In each subplot:

        single -> circle marker
        double -> triangle marker

    Auxiliary lines:

        y = x
        x = threshold
        y = threshold

    With the default threshold=1.0 this means:

        y = x
        x = 1
        y = 1
    """

    if predictions.empty:
        return

    import matplotlib

    matplotlib.use(
        "Agg"
    )

    import matplotlib.pyplot as plt

    plot_dir = (
        output_dir
        / "new_validation_scatter_plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_lookup = (
        metrics.set_index(
            [
                "variant",
                "target",
                "subset",
            ],
            drop=False,
        )
    )

    target_order = [
        "Norm_before",
        "Norm_after",
    ]

    # =====================================================================
    # One figure per MODEL
    # =====================================================================

    for variant, variant_block in (
        predictions.groupby(
            "variant",
            sort=False,
        )
    ):

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(
                13.2,
                5.8,
            ),
            dpi=160,
            constrained_layout=True,
        )

        # =================================================================
        # Left  : Norm_before
        # Right : Norm_after
        # =================================================================

        for ax, target in zip(
            axes,
            target_order,
        ):

            block = (
                variant_block
                .loc[
                    variant_block[
                        "target"
                    ].eq(target)
                ]
                .copy()
            )

            block[
                "Fifth_class"
            ] = normalize_class(
                block[
                    "Fifth_class"
                ]
            )

            block[
                "y_true"
            ] = pd.to_numeric(
                block[
                    "y_true"
                ],
                errors="coerce",
            )

            block[
                "y_pred"
            ] = pd.to_numeric(
                block[
                    "y_pred"
                ],
                errors="coerce",
            )

            block = block.loc[
                np.isfinite(
                    block[
                        "y_true"
                    ]
                )
                & np.isfinite(
                    block[
                        "y_pred"
                    ]
                )
            ].copy()

            if block.empty:

                ax.set_title(
                    f"{target} "
                    "(no valid points)"
                )

                ax.axis(
                    "off"
                )

                continue

            # =============================================================
            # Use the same x/y limits inside each subplot.
            #
            # Include threshold=1 explicitly so x=1/y=1 always remain
            # visible even when all predictions happen to lie on one side.
            # =============================================================

            vals = np.concatenate([
                block[
                    "y_true"
                ].to_numpy(
                    dtype=float
                ),

                block[
                    "y_pred"
                ].to_numpy(
                    dtype=float
                ),

                np.asarray(
                    [threshold],
                    dtype=float,
                ),
            ])

            vals = vals[
                np.isfinite(vals)
            ]

            vmin = float(
                vals.min()
            )

            vmax = float(
                vals.max()
            )

            if math.isclose(
                vmin,
                vmax,
            ):

                pad = max(
                    0.5,
                    abs(vmin) * 0.1,
                )

            else:

                pad = (
                    0.08
                    * (
                        vmax
                        - vmin
                    )
                )

            lo = (
                vmin
                - pad
            )

            hi = (
                vmax
                + pad
            )

            # =============================================================
            # single and double in the SAME subplot
            # =============================================================

            for subset, marker in (
                (
                    "single",
                    "o",
                ),
                (
                    "double",
                    "^",
                ),
            ):

                sub = block.loc[
                    block[
                        "Fifth_class"
                    ].eq(subset)
                ].copy()

                ax.scatter(
                    sub[
                        "y_true"
                    ],

                    sub[
                        "y_pred"
                    ],

                    s=44,

                    alpha=0.82,

                    marker=marker,

                    label=(
                        f"{subset} "
                        f"(n={len(sub)})"
                    ),
                )

            # =============================================================
            # Auxiliary line 1: y = x
            # =============================================================

            ax.plot(
                [
                    lo,
                    hi,
                ],
                [
                    lo,
                    hi,
                ],
                linewidth=1.25,
                linestyle="--",
                label="y = x",
            )

            # =============================================================
            # Auxiliary line 2: x = 1
            # =============================================================

            ax.axvline(
                threshold,
                linewidth=1.10,
                linestyle=":",
                label=(
                    f"x = "
                    f"{threshold:g}"
                ),
            )

            # =============================================================
            # Auxiliary line 3: y = 1
            # =============================================================

            ax.axhline(
                threshold,
                linewidth=1.10,
                linestyle="-.",
                label=(
                    f"y = "
                    f"{threshold:g}"
                ),
            )

            ax.set_xlim(
                lo,
                hi,
            )

            ax.set_ylim(
                lo,
                hi,
            )

            # equal axis scale is important for a true-vs-pred scatter
            ax.set_aspect(
                "equal",
                adjustable="box",
            )

            ax.set_xlabel(
                f"True {target}"
            )

            ax.set_ylabel(
                f"Predicted {target}"
            )

            ax.set_title(
                target
            )

            ax.grid(
                alpha=0.22
            )

            ax.legend(
                loc="best",
                fontsize=8,
            )

            # =============================================================
            # Metric summary
            #
            # all
            # single
            # double
            # =============================================================

            info_lines = []

            for subset_label in (
                "all",
                "single",
                "double",
            ):

                key = (
                    variant,
                    target,
                    subset_label,
                )

                if (
                    key
                    not in metrics_lookup.index
                ):
                    continue

                row = (
                    metrics_lookup
                    .loc[key]
                )

                if isinstance(
                    row,
                    pd.DataFrame,
                ):
                    row = row.iloc[0]

                info = (
                    f"{subset_label}: "
                    f"MAE="
                    f"{row['mae']:.3f}, "
                    f"R²="
                    f"{row['r2']:.3f}, "
                    f"Rec>1="
                    f"{row['recall_gt1']:.3f}, "
                    f"F2>1="
                    f"{row['f2_gt1']:.3f}"
                )

                info_lines.append(
                    info
                )

            if info_lines:

                ax.text(
                    0.03,
                    0.97,

                    "\n".join(
                        info_lines
                    ),

                    transform=
                        ax.transAxes,

                    va="top",

                    ha="left",

                    fontsize=7.6,

                    bbox={
                        "boxstyle":
                            "round,pad=0.30",

                        "facecolor":
                            "white",

                        "alpha":
                            0.78,
                    },
                )

        # =================================================================
        # One title / one output file per model
        # =================================================================

        fig.suptitle(
            f"new_validation — {variant}"
        )

        stem = (
            plot_dir
            / (
                "scatter_"
                f"{safe_slug(variant)}"
            )
        )

        fig.savefig(
            stem.with_suffix(
                ".png"
            ),
            dpi=220,
            bbox_inches="tight",
        )

        fig.savefig(
            stem.with_suffix(
                ".pdf"
            ),
            bbox_inches="tight",
        )

        plt.close(
            fig
        )


# =============================================================================
# Arguments
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--training-csv",
        type=Path,
        default=Path(
            "results/"
            "input_graphgps_optimization/"
            "o12_input_700_multitasks_"
            "lr0001_sigmoid_core4_"
            "ratiofix_20260812_"
            "freshcache_baseline/"
            "staging/"
            "20260812-sum-700_utf8.csv"
        ),
    )

    parser.add_argument(
        "--new-validation",
        type=Path,
        default=Path(
            "datasets_lrx/"
            "raw/"
            "feedback/"
            "new_validation.csv"
        ),
    )

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path(
            "results/"
            "input_graphgps_optimization/"
            "o12_fifth_identity_ood_"
            "seed100_109/"
            "fifth_identity_manifests"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/"
            "simple_molecular_baselines/"
            "fifth_ood_v1"
        ),
    )

    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(
            range(
                100,
                110,
            )
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--final-seed",
        type=int,
        default=20260820,
    )

    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(
            VARIANTS
        ),
        default=DEFAULT_VARIANTS,
    )

    parser.add_argument(
        "--skip-ood",
        action="store_true",
        help=(
            "Skip internal Fifth-OOD "
            "evaluation and only fit "
            "all training rows -> "
            "new_validation."
        ),
    )

    parser.add_argument(
        "--skip-external",
        action="store_true",
        help=(
            "Skip new_validation scoring; "
            "useful for sealed external sets."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    args = parse_args()

    output_dir = (
        args.output_dir
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        args.training_csv
        .resolve()
    )

    external_path = (
        args.new_validation
        .resolve()
    )

    manifest_dir = (
        args.manifest_dir
        .resolve()
    )

    if not train_path.is_file():
        raise FileNotFoundError(
            train_path
        )

    if (
        not args.skip_external
        and not external_path.is_file()
    ):
        raise FileNotFoundError(
            external_path
        )

    if (
        not args.skip_ood
        and not manifest_dir.is_dir()
    ):
        raise FileNotFoundError(
            manifest_dir
        )

    train = read_csv_robust(
        train_path
    )

    validate_frame(
        train,
        external=False,
        targets=args.targets,
    )

    if len(train) != 700:

        warnings.warn(
            "Expected 700 training rows "
            "for the locked experiment, "
            f"got {len(train)}"
        )

    external = None

    if not args.skip_external:

        external = read_csv_robust(
            external_path
        )

        validate_frame(
            external,
            external=True,
            targets=args.targets,
        )

    print(
        "[features] building "
        f"training features for "
        f"{len(train)} rows",
        flush=True,
    )

    train_bundle = build_features(
        train
    )

    audit = {
        "training":
            train_bundle.audit
    }

    external_bundle = None

    if external is not None:

        print(
            "[features] building "
            "new_validation features "
            f"for {len(external)} rows",
            flush=True,
        )

        external_bundle = (
            build_features(
                external
            )
        )

        audit[
            "new_validation"
        ] = external_bundle.audit

    provenance = {
        "training_csv":
            str(train_path),

        "new_validation":
            (
                str(external_path)
                if external is not None
                else None
            ),

        "manifest_dir":
            (
                str(manifest_dir)
                if not args.skip_ood
                else None
            ),

        "targets":
            args.targets,

        "variants":
            args.variants,

        "seeds":
            args.seeds,

        "threshold":
            args.threshold,

        "morgan_radius":
            MORGAN_RADIUS,

        "morgan_bits":
            MORGAN_BITS,

        "descriptor_names":
            DESC_NAMES,

        "no_external_label_tuning":
            True,

        "feature_audit":
            audit,
    }

    (
        output_dir
        / "experiment_provenance.json"
    ).write_text(
        json.dumps(
            provenance,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =====================================================================
    # Internal Fifth-identity OOD
    # =====================================================================

    if not args.skip_ood:

        (
            _,
            ood_summary,
        ) = run_ood(
            train,
            train_bundle,
            manifest_dir,
            args.seeds,
            args.targets,
            args.variants,
            args.threshold,
            output_dir,
        )

        print_rankings(
            ood_summary,
            (
                "Internal Fifth-identity "
                "OOD test summary — "
                "double subset"
            ),
        )

    # =====================================================================
    # Fit all 700 rows -> new_validation
    # =====================================================================

    if (
        external is not None
        and external_bundle is not None
    ):

        external_metrics = (
            run_external(
                train,
                train_bundle,
                external,
                external_bundle,
                args.targets,
                args.variants,
                args.threshold,
                output_dir,
                args.final_seed,
            )
        )

        print_rankings(
            external_metrics,
            (
                "new_validation — "
                "double subset"
            ),
        )

    print(
        "\nDone. Outputs: "
        f"{output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()