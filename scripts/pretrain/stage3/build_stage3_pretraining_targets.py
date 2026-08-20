#!/usr/bin/env python3
"""
Stage 3 — build pretraining targets for the frozen Stage-2C molecular library.

This stage does NOT use Norm labels.

It prepares:
    1. deterministic train/val/test split;
    2. physicochemical descriptor regression targets;
    3. train-only descriptor normalization;
    4. Morgan fingerprint multi-label targets;
    5. fingerprint prevalence statistics;
    6. frozen schemas/manifests for the GraphGPS pretraining runner.

Default design
--------------
Split:
    ~80/10/10 train/val/test.

Observed training Fifth structures:
    kept in PRETRAIN TRAIN by default.

Why?
    The purpose of Stage 3 is representation pretraining, not estimating the
    final downstream generalization error. We want the encoder to see all
    chemistry already present in the downstream training set. Validation/test
    are used only to monitor the pretraining tasks.

Synthetic structures:
    stratified by scaffold family and sequence length before deterministic
    random splitting.

Descriptors:
    MolWt
    MolLogP
    MolMR
    TPSA
    HBD
    HBA
    NumRotatableBonds
    RingCount
    NumAromaticRings
    FractionCSP3
    HeavyAtomCount
    HeteroAtomCount
    LabuteASA
    BertzCT
    NHOHCount
    NOCount
    AmideBondCount
    FreeCarboxylCount
    AmineCount
    AromaticAtomCount
    DisulfideBondCount
    FormalCharge

Morgan fingerprint:
    radius = 2
    nBits = 1024
    useChirality = False

This matches the current downstream setting in which stereochemistry is not
encoded by GraphGPS.

Normalization
-------------
Descriptor mean/std are fit on PRETRAIN TRAIN ONLY.

Descriptors whose train-set standard deviation is effectively zero are NOT
silently removed. They are retained in the raw targets and schema, but their
normalization scale is set to 1.0 and they are flagged as constant_train=True.
The future pretraining runner may mask them from the regression loss.

Outputs
-------
pretraining_split.csv

descriptor_targets_raw.csv
descriptor_targets_scaled.npz
descriptor_scaler.json
descriptor_train_statistics.csv

morgan_fp_1024.npz
morgan_fp_train_statistics.npz
morgan_fp_bit_statistics.csv

target_schema.json
stage3_manifest.json

Hard gates
----------
- Stage-2C IDs are unique.
- Stage-2C model-visible graph identities are unique.
- All molecules are RDKit-valid.
- No [Fr] placeholder is present.
- Descriptor matrix contains no NaN/Inf.
- Fingerprints are binary and have exactly 1024 bits.
- Train/val/test are disjoint and cover all molecules.
- Every observed downstream-training Fifth graph is in pretraining train.
- Descriptor scaler is fit only from train rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Crippen,
    Descriptors,
    GraphDescriptors,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors,
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_FP_RADIUS = 2
DEFAULT_FP_BITS = 1024
DEFAULT_SEED = 20260818

DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "HBD",
    "HBA",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
    "HeteroAtomCount",
    "LabuteASA",
    "BertzCT",
    "NHOHCount",
    "NOCount",
    "AmideBondCount",
    "FreeCarboxylCount",
    "AmineCount",
    "AromaticAtomCount",
    "DisulfideBondCount",
    "FormalCharge",
]

# SMARTS patterns are intentionally simple, interpretable structural counts.
FREE_CARBOXYL_SMARTS = Chem.MolFromSmarts(
    "[CX3](=O)[OX2H1,O-]"
)
AMINE_SMARTS = Chem.MolFromSmarts(
    "[NX3;H2,H1,H0;!$(N-C=O);!$(N-S=O);!$(N-P=O)]"
)
DISULFIDE_SMARTS = Chem.MolFromSmarts(
    "[#16X2]-[#16X2]"
)

if any(
    patt is None
    for patt in (
        FREE_CARBOXYL_SMARTS,
        AMINE_SMARTS,
        DISULFIDE_SMARTS,
    )
):
    raise RuntimeError("Failed to compile Stage-3 SMARTS patterns.")


# =============================================================================
# Utilities
# =============================================================================

def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def bool_from_any(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    text = clean(value).lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n", ""}:
        return False

    raise ValueError(f"Cannot parse boolean value {value!r}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def stable_seed(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{text}".encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def mol_or_fail(smiles: str, label: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"RDKit failed to parse {label}: {smiles}"
        )

    Chem.SanitizeMol(mol)
    return mol


def nonisomeric_key(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )


def first_pipe_token(value: Any) -> str:
    text = clean(value)

    if not text:
        return ""

    return text.split("|")[0].strip()


def first_integer_pipe_token(value: Any) -> str:
    text = clean(value)

    if not text:
        return "none"

    token = text.split("|")[0].strip()

    if not token:
        return "none"

    try:
        return str(int(float(token)))
    except ValueError:
        return token


# =============================================================================
# Descriptor calculation
# =============================================================================

def count_matches(
    mol: Chem.Mol,
    pattern: Chem.Mol,
) -> int:
    return int(
        len(
            mol.GetSubstructMatches(
                pattern,
                uniquify=True,
            )
        )
    )


def aromatic_atom_count(mol: Chem.Mol) -> int:
    return int(
        sum(
            atom.GetIsAromatic()
            for atom in mol.GetAtoms()
        )
    )


def calc_amide_bonds(mol: Chem.Mol) -> int:
    # Current RDKit versions expose CalcNumAmideBonds.
    # Keep a fallback for older builds.
    if hasattr(
        rdMolDescriptors,
        "CalcNumAmideBonds",
    ):
        return int(
            rdMolDescriptors.CalcNumAmideBonds(mol)
        )

    patt = Chem.MolFromSmarts(
        "[NX3][CX3](=[OX1])"
    )

    if patt is None:
        raise RuntimeError(
            "Failed to construct fallback amide SMARTS."
        )

    return count_matches(mol, patt)


def calculate_descriptors(
    mol: Chem.Mol,
) -> dict[str, float]:
    values = {
        "MolWt": float(
            Descriptors.MolWt(mol)
        ),
        "MolLogP": float(
            Crippen.MolLogP(mol)
        ),
        "MolMR": float(
            Crippen.MolMR(mol)
        ),
        "TPSA": float(
            rdMolDescriptors.CalcTPSA(mol)
        ),
        "HBD": float(
            Lipinski.NumHDonors(mol)
        ),
        "HBA": float(
            Lipinski.NumHAcceptors(mol)
        ),
        "NumRotatableBonds": float(
            rdMolDescriptors.CalcNumRotatableBonds(
                mol
            )
        ),
        "RingCount": float(
            rdMolDescriptors.CalcNumRings(mol)
        ),
        "NumAromaticRings": float(
            rdMolDescriptors.CalcNumAromaticRings(
                mol
            )
        ),
        "FractionCSP3": float(
            rdMolDescriptors.CalcFractionCSP3(
                mol
            )
        ),
        "HeavyAtomCount": float(
            rdMolDescriptors.CalcNumHeavyAtoms(
                mol
            )
        ),
        "HeteroAtomCount": float(
            rdMolDescriptors.CalcNumHeteroatoms(
                mol
            )
        ),
        "LabuteASA": float(
            rdMolDescriptors.CalcLabuteASA(
                mol
            )
        ),
        "BertzCT": float(
            GraphDescriptors.BertzCT(mol)
        ),
        "NHOHCount": float(
            Lipinski.NHOHCount(mol)
        ),
        "NOCount": float(
            Lipinski.NOCount(mol)
        ),
        "AmideBondCount": float(
            calc_amide_bonds(mol)
        ),
        "FreeCarboxylCount": float(
            count_matches(
                mol,
                FREE_CARBOXYL_SMARTS,
            )
        ),
        "AmineCount": float(
            count_matches(
                mol,
                AMINE_SMARTS,
            )
        ),
        "AromaticAtomCount": float(
            aromatic_atom_count(mol)
        ),
        "DisulfideBondCount": float(
            count_matches(
                mol,
                DISULFIDE_SMARTS,
            )
        ),
        "FormalCharge": float(
            Chem.GetFormalCharge(mol)
        ),
    }

    if list(values) != DESCRIPTOR_NAMES:
        raise RuntimeError(
            "Descriptor implementation order differs "
            "from DESCRIPTOR_NAMES."
        )

    return values


# =============================================================================
# Deterministic split
# =============================================================================

def build_stratum(row: pd.Series) -> str:
    scaffold = first_pipe_token(
        row.get("scaffold_families", "")
    ) or "UNSPECIFIED"

    length = first_integer_pipe_token(
        row.get("sequence_lengths", "")
    )

    return f"{scaffold}|L{length}"


def stratified_synthetic_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    observed_train_only: bool,
) -> pd.DataFrame:
    if not math.isclose(
        train_fraction
        + val_fraction
        + test_fraction,
        1.0,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "train/val/test fractions must sum to 1."
        )

    if min(
        train_fraction,
        val_fraction,
        test_fraction,
    ) < 0:
        raise ValueError(
            "Split fractions must be non-negative."
        )

    out = frame.copy()
    out["observed_in_training_bool"] = out[
        "observed_in_training"
    ].map(bool_from_any)

    out["split_stratum"] = out.apply(
        build_stratum,
        axis=1,
    )

    out["split"] = ""

    if observed_train_only:
        observed_mask = out[
            "observed_in_training_bool"
        ]

        out.loc[
            observed_mask,
            "split",
        ] = "train"

        pool = out.loc[
            ~observed_mask
        ].copy()
    else:
        pool = out.copy()

    for stratum, group in pool.groupby(
        "split_stratum",
        sort=True,
    ):
        indices = list(group.index)

        rng = random.Random(
            stable_seed(seed, stratum)
        )

        rng.shuffle(indices)

        n = len(indices)

        if n == 0:
            continue

        # Rare synthetic strata remain train-only.
        if n < 5:
            n_val = 0
            n_test = 0
        else:
            n_val = int(round(n * val_fraction))
            n_test = int(round(n * test_fraction))

            if val_fraction > 0:
                n_val = max(1, n_val)

            if test_fraction > 0:
                n_test = max(1, n_test)

            # Preserve at least one train sample in every synthetic stratum.
            while (
                n_val + n_test
                >= n
            ):
                if n_test >= n_val and n_test > 0:
                    n_test -= 1
                elif n_val > 0:
                    n_val -= 1
                else:
                    break

        val_idx = indices[:n_val]
        test_idx = indices[
            n_val:n_val + n_test
        ]
        train_idx = indices[
            n_val + n_test:
        ]

        out.loc[val_idx, "split"] = "val"
        out.loc[test_idx, "split"] = "test"
        out.loc[train_idx, "split"] = "train"

    if out["split"].eq("").any():
        missing = out.loc[
            out["split"].eq(""),
            ["stage2c_id", "split_stratum"],
        ]

        raise RuntimeError(
            "Unassigned split rows remain:\n"
            + missing.head(20).to_string(
                index=False
            )
        )

    return out


# =============================================================================
# Morgan fingerprints
# =============================================================================

def calculate_morgan_matrix(
    mols: list[Chem.Mol],
    *,
    radius: int,
    n_bits: int,
) -> np.ndarray:
    generator = (
        rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
            includeChirality=False,
        )
    )

    matrix = np.zeros(
        (len(mols), n_bits),
        dtype=np.uint8,
    )

    for i, mol in enumerate(mols):
        fp = generator.GetFingerprint(mol)

        arr = np.zeros(
            (n_bits,),
            dtype=np.uint8,
        )

        DataStructs.ConvertToNumpyArray(
            fp,
            arr,
        )

        matrix[i] = arr

    return matrix


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: generate descriptor + Morgan-FP pretraining targets "
            "from the frozen Stage-2C library."
        )
    )

    parser.add_argument(
        "--input-library",
        type=Path,
        required=True,
        help=(
            "Stage-2C stage2c_pretraining_molecular_library.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--morgan-radius",
        type=int,
        default=DEFAULT_FP_RADIUS,
    )

    parser.add_argument(
        "--morgan-bits",
        type=int,
        default=DEFAULT_FP_BITS,
    )

    parser.add_argument(
        "--expect-rows",
        type=int,
        default=10097,
        help=(
            "Expected Stage-2C graph count; <=0 disables."
        ),
    )

    parser.add_argument(
        "--expect-observed-training-graphs",
        type=int,
        default=63,
        help=(
            "Expected observed-training graph count; <=0 disables."
        ),
    )

    parser.add_argument(
        "--split-observed-training",
        action="store_true",
        help=(
            "By default every observed downstream-training Fifth graph is "
            "forced into pretraining train. Set this flag only if you want "
            "observed graphs to participate in the ordinary split."
        ),
    )

    args = parser.parse_args()

    input_path = args.input_library.resolve()
    outdir = args.output_dir.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Read / validate Stage 2C
    # ------------------------------------------------------------------

    frame = pd.read_csv(
        input_path,
        dtype={
            "stage2c_id": str,
        },
    )

    required = {
        "stage2c_id",
        "Fifth",
        "Fifth_SMILE",
        "canonical_connectivity",
        "scaffold_families",
        "sequence_lengths",
        "sampling_sources",
        "observed_in_training",
        "synthetic_only",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Stage-2C library misses required columns: "
            + ", ".join(sorted(missing))
        )

    if (
        args.expect_rows > 0
        and len(frame) != args.expect_rows
    ):
        raise ValueError(
            f"Expected {args.expect_rows} Stage-2C rows, "
            f"found {len(frame)}."
        )

    if frame["stage2c_id"].duplicated().any():
        raise ValueError(
            "Stage-2C contains duplicate stage2c_id."
        )

    if frame[
        "canonical_connectivity"
    ].duplicated().any():
        raise ValueError(
            "Stage-2C contains duplicate model-visible graph identity."
        )

    if frame[
        "Fifth_SMILE"
    ].fillna("").astype(str).eq("[Fr]").any():
        raise ValueError(
            "[Fr] placeholder is not allowed in Stage 3."
        )

    observed_flags = frame[
        "observed_in_training"
    ].map(bool_from_any)

    observed_count = int(
        observed_flags.sum()
    )

    if (
        args.expect_observed_training_graphs > 0
        and observed_count
        != args.expect_observed_training_graphs
    ):
        raise ValueError(
            "Expected "
            f"{args.expect_observed_training_graphs} observed training graphs, "
            f"found {observed_count}."
        )

    # ------------------------------------------------------------------
    # Parse molecules once.
    # ------------------------------------------------------------------

    mols: list[Chem.Mol] = []
    recalculated_keys: list[str] = []

    for row in frame.itertuples(
        index=False
    ):
        mol = mol_or_fail(
            clean(row.Fifth_SMILE),
            clean(row.stage2c_id),
        )

        mols.append(mol)
        recalculated_keys.append(
            nonisomeric_key(mol)
        )

    if recalculated_keys != frame[
        "canonical_connectivity"
    ].astype(str).tolist():
        mismatch = frame.copy()

        mismatch[
            "recalculated_canonical_connectivity"
        ] = recalculated_keys

        mismatch = mismatch.loc[
            mismatch[
                "recalculated_canonical_connectivity"
            ]
            != mismatch[
                "canonical_connectivity"
            ].astype(str)
        ]

        mismatch.to_csv(
            outdir
            / "stage3_canonical_connectivity_mismatch.csv",
            index=False,
        )

        raise ValueError(
            "Stage-2C canonical connectivity does not match RDKit "
            "recalculation. See stage3_canonical_connectivity_mismatch.csv."
        )

    # ------------------------------------------------------------------
    # Split before any normalization.
    # ------------------------------------------------------------------

    split_frame = stratified_synthetic_split(
        frame,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        observed_train_only=(
            not args.split_observed_training
        ),
    )

    if (
        not args.split_observed_training
        and not split_frame.loc[
            split_frame[
                "observed_in_training_bool"
            ],
            "split",
        ].eq("train").all()
    ):
        raise RuntimeError(
            "Observed training graphs escaped pretraining train."
        )

    split_counts = (
        split_frame[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    if sum(
        int(v)
        for v in split_counts.values()
    ) != len(frame):
        raise RuntimeError(
            "Split row counts do not sum to Stage-3 row count."
        )

    split_output_cols = [
        "stage2c_id",
        "split",
        "split_stratum",
        "observed_in_training_bool",
        "synthetic_only",
        "scaffold_families",
        "sequence_lengths",
        "sampling_sources",
    ]

    split_frame[
        split_output_cols
    ].rename(
        columns={
            "observed_in_training_bool":
                "observed_in_training",
        }
    ).to_csv(
        outdir
        / "pretraining_split.csv",
        index=False,
    )

    train_mask = split_frame[
        "split"
    ].eq("train").to_numpy()

    val_mask = split_frame[
        "split"
    ].eq("val").to_numpy()

    test_mask = split_frame[
        "split"
    ].eq("test").to_numpy()

    if not (
        train_mask.any()
        and val_mask.any()
        and test_mask.any()
    ):
        raise ValueError(
            "Train/val/test must all contain at least one structure."
        )

    # ------------------------------------------------------------------
    # Descriptor calculation.
    # ------------------------------------------------------------------

    descriptor_rows = []

    for stage2c_id, mol in zip(
        frame["stage2c_id"],
        mols,
    ):
        values = calculate_descriptors(mol)

        descriptor_rows.append(
            {
                "stage2c_id": stage2c_id,
                **values,
            }
        )

    descriptor_df = pd.DataFrame(
        descriptor_rows
    )

    descriptor_matrix = descriptor_df[
        DESCRIPTOR_NAMES
    ].to_numpy(
        dtype=np.float64,
    )

    if not np.isfinite(
        descriptor_matrix
    ).all():
        bad = np.argwhere(
            ~np.isfinite(
                descriptor_matrix
            )
        )

        raise ValueError(
            "Descriptor matrix contains non-finite values. "
            f"First bad indices: {bad[:20].tolist()}"
        )

    train_values = descriptor_matrix[
        train_mask
    ]

    means = train_values.mean(
        axis=0,
    )

    # Population std (ddof=0), matching common neural-network scaling.
    stds = train_values.std(
        axis=0,
        ddof=0,
    )

    constant_mask = stds < 1e-12

    safe_stds = stds.copy()
    safe_stds[
        constant_mask
    ] = 1.0

    scaled_matrix = (
        descriptor_matrix - means
    ) / safe_stds

    if not np.isfinite(
        scaled_matrix
    ).all():
        raise RuntimeError(
            "Scaled descriptor matrix contains non-finite values."
        )

    descriptor_df.insert(
        1,
        "split",
        split_frame["split"].tolist(),
    )

    descriptor_df.to_csv(
        outdir
        / "descriptor_targets_raw.csv",
        index=False,
    )

    np.savez_compressed(
        outdir
        / "descriptor_targets_scaled.npz",
        stage2c_id=np.asarray(
        frame["stage2c_id"].astype(str).tolist(),
        dtype=np.str_,
        ),
        descriptor_names=np.array(
            DESCRIPTOR_NAMES,
            dtype=str,
        ),
        targets=scaled_matrix.astype(
            np.float32
        ),
        split=np.asarray(
    split_frame["split"].astype(str).tolist(),
    dtype=np.str_,
),
    )

    descriptor_stats_rows = []

    for i, name in enumerate(
        DESCRIPTOR_NAMES
    ):
        descriptor_stats_rows.append(
            {
                "descriptor": name,
                "train_mean": float(
                    means[i]
                ),
                "train_std_raw": float(
                    stds[i]
                ),
                "train_scale_used": float(
                    safe_stds[i]
                ),
                "constant_train": bool(
                    constant_mask[i]
                ),
                "global_min": float(
                    descriptor_matrix[
                        :,
                        i,
                    ].min()
                ),
                "global_max": float(
                    descriptor_matrix[
                        :,
                        i,
                    ].max()
                ),
                "train_min": float(
                    train_values[
                        :,
                        i,
                    ].min()
                ),
                "train_max": float(
                    train_values[
                        :,
                        i,
                    ].max()
                ),
            }
        )

    descriptor_statistics = pd.DataFrame(
        descriptor_stats_rows
    )

    descriptor_statistics.to_csv(
        outdir
        / "descriptor_train_statistics.csv",
        index=False,
    )

    scaler_payload = {
        "type": "standard_scaler",
        "fit_split": "train",
        "ddof": 0,
        "descriptors": [
            {
                "name": name,
                "mean": float(
                    means[i]
                ),
                "std_raw": float(
                    stds[i]
                ),
                "scale_used": float(
                    safe_stds[i]
                ),
                "constant_train": bool(
                    constant_mask[i]
                ),
            }
            for i, name in enumerate(
                DESCRIPTOR_NAMES
            )
        ],
    }

    with (
        outdir
        / "descriptor_scaler.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            scaler_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    # ------------------------------------------------------------------
    # Morgan fingerprints.
    # ------------------------------------------------------------------

    fp_matrix = calculate_morgan_matrix(
        mols,
        radius=args.morgan_radius,
        n_bits=args.morgan_bits,
    )

    if fp_matrix.shape != (
        len(frame),
        args.morgan_bits,
    ):
        raise RuntimeError(
            "Unexpected Morgan matrix shape: "
            f"{fp_matrix.shape}"
        )

    unique_fp_values = np.unique(
        fp_matrix
    )

    if not set(
        unique_fp_values.tolist()
    ).issubset({0, 1}):
        raise RuntimeError(
            "Morgan fingerprint matrix is not binary."
        )

    np.savez_compressed(
        outdir
        / f"morgan_fp_{args.morgan_bits}.npz",
        stage2c_id=np.asarray(
    frame["stage2c_id"].astype(str).tolist(),
    dtype=np.str_,
),
        fingerprints=fp_matrix,
        split=np.asarray(
    split_frame["split"].astype(str).tolist(),
    dtype=np.str_,
),
        radius=np.array(
            [args.morgan_radius],
            dtype=np.int32,
        ),
        n_bits=np.array(
            [args.morgan_bits],
            dtype=np.int32,
        ),
    )

    train_fp = fp_matrix[
        train_mask
    ].astype(
        np.int64
    )

    positive_counts = train_fp.sum(
        axis=0
    )

    train_n = int(
        train_fp.shape[0]
    )

    negative_counts = (
        train_n - positive_counts
    )

    prevalence = (
        positive_counts
        / max(train_n, 1)
    )

    active_mask = (
        positive_counts > 0
    )

    nonconstant_mask = (
        active_mask
        & (
            positive_counts
            < train_n
        )
    )

    # Raw class imbalance statistic.
    # For bits with zero positives, pos_weight is inf and should not
    # be used directly by the future training runner.
    pos_weight_raw = np.full(
        args.morgan_bits,
        np.inf,
        dtype=np.float64,
    )

    has_positive = (
        positive_counts > 0
    )

    pos_weight_raw[
        has_positive
    ] = (
        negative_counts[
            has_positive
        ]
        / positive_counts[
            has_positive
        ]
    )

    # A practical finite suggestion for future BCE loss.
    pos_weight_clipped = np.ones(
        args.morgan_bits,
        dtype=np.float32,
    )

    pos_weight_clipped[
        nonconstant_mask
    ] = np.clip(
        pos_weight_raw[
            nonconstant_mask
        ],
        1.0,
        20.0,
    ).astype(
        np.float32
    )

    np.savez_compressed(
        outdir
        / "morgan_fp_train_statistics.npz",
        positive_counts=positive_counts.astype(
            np.int64
        ),
        negative_counts=negative_counts.astype(
            np.int64
        ),
        prevalence=prevalence.astype(
            np.float32
        ),
        active_mask=active_mask.astype(
            np.bool_
        ),
        nonconstant_mask=nonconstant_mask.astype(
            np.bool_
        ),
        pos_weight_clipped_1_20=(
            pos_weight_clipped
        ),
    )

    fp_stats = pd.DataFrame(
        {
            "bit": np.arange(
                args.morgan_bits,
                dtype=int,
            ),
            "train_positive_count": (
                positive_counts
            ),
            "train_negative_count": (
                negative_counts
            ),
            "train_prevalence": (
                prevalence
            ),
            "active_in_train": (
                active_mask
            ),
            "nonconstant_in_train": (
                nonconstant_mask
            ),
            "suggested_pos_weight_clipped_1_20": (
                pos_weight_clipped
            ),
        }
    )

    fp_stats.to_csv(
        outdir
        / "morgan_fp_bit_statistics.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Schemas and manifest.
    # ------------------------------------------------------------------

    active_fp_bits = int(
        active_mask.sum()
    )

    nonconstant_fp_bits = int(
        nonconstant_mask.sum()
    )

    constant_descriptors = [
        DESCRIPTOR_NAMES[i]
        for i in range(
            len(DESCRIPTOR_NAMES)
        )
        if constant_mask[i]
    ]

    target_schema = {
        "descriptor_regression": {
            "number_of_descriptors": len(
                DESCRIPTOR_NAMES
            ),
            "descriptor_names": (
                DESCRIPTOR_NAMES
            ),
            "dtype": "float32",
            "normalization": {
                "type": "z_score",
                "fit_split": "train",
                "ddof": 0,
                "constant_train_policy": (
                    "retain target, scale=1.0, expose constant_train "
                    "flag so the pretraining runner may mask it"
                ),
            },
            "recommended_loss": (
                "SmoothL1Loss on nonconstant descriptor targets"
            ),
        },
        "morgan_fingerprint": {
            "radius": int(
                args.morgan_radius
            ),
            "n_bits": int(
                args.morgan_bits
            ),
            "use_chirality": False,
            "dtype": "uint8",
            "active_bits_in_train": (
                active_fp_bits
            ),
            "nonconstant_bits_in_train": (
                nonconstant_fp_bits
            ),
            "recommended_loss": (
                "BCEWithLogitsLoss; optionally mask train-constant bits "
                "and use clipped train-only pos_weight statistics"
            ),
        },
        "id_key": "stage2c_id",
    }

    with (
        outdir
        / "target_schema.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            target_schema,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    stratum_summary = (
        split_frame.groupby(
            [
                "split_stratum",
                "split",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="rows"
        )
    )

    stratum_summary.to_csv(
        outdir
        / "pretraining_split_stratum_summary.csv",
        index=False,
    )

    manifest = {
        "stage": "3_pretraining_targets",
        "input": {
            "library": str(
                input_path
            ),
            "library_sha256": sha256(
                input_path
            ),
            "rows": int(
                len(frame)
            ),
        },
        "split": {
            "seed": int(
                args.seed
            ),
            "requested_fractions": {
                "train": float(
                    args.train_fraction
                ),
                "val": float(
                    args.val_fraction
                ),
                "test": float(
                    args.test_fraction
                ),
            },
            "actual_counts": {
                key: int(
                    split_counts.get(
                        key,
                        0,
                    )
                )
                for key in (
                    "train",
                    "val",
                    "test",
                )
            },
            "actual_fractions": {
                key: float(
                    split_counts.get(
                        key,
                        0,
                    )
                    / len(frame)
                )
                for key in (
                    "train",
                    "val",
                    "test",
                )
            },
            "strategy": (
                "deterministic stratified synthetic split by "
                "primary scaffold family + sequence length"
            ),
            "observed_training_policy": (
                "forced_to_pretraining_train"
                if not args.split_observed_training
                else "participates_in_regular_split"
            ),
        },
        "descriptor_targets": {
            "count": int(
                len(DESCRIPTOR_NAMES)
            ),
            "names": (
                DESCRIPTOR_NAMES
            ),
            "constant_train_descriptors": (
                constant_descriptors
            ),
            "scaler_fit_split": (
                "train"
            ),
        },
        "morgan_fingerprint": {
            "radius": int(
                args.morgan_radius
            ),
            "n_bits": int(
                args.morgan_bits
            ),
            "use_chirality": False,
            "active_bits_in_train": (
                active_fp_bits
            ),
            "nonconstant_bits_in_train": (
                nonconstant_fp_bits
            ),
        },
        "hard_gates": {
            "unique_stage2c_id": True,
            "unique_model_visible_graph": True,
            "all_rdkit_valid": True,
            "no_Fr": True,
            "descriptor_finite": True,
            "fingerprint_binary": True,
            "split_disjoint_complete": True,
            "observed_training_graphs_in_train": (
                True
                if not args.split_observed_training
                else None
            ),
            "train_only_descriptor_scaler": True,
        },
        "next_stage": {
            "name": (
                "Stage 4 GraphGPS pretraining"
            ),
            "recommended_first_ablation": [
                "descriptor_only",
                "descriptor_plus_morgan",
            ],
        },
    }

    with (
        outdir
        / "stage3_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    # ------------------------------------------------------------------
    # Terminal report.
    # ------------------------------------------------------------------

    print("=" * 92)
    print(
        "STAGE 3 — PRETRAINING TARGET GENERATION"
    )
    print("=" * 92)

    print(
        f"Input unique molecular graphs:         {len(frame)}"
    )
    print(
        f"Observed downstream-training graphs:   {observed_count}"
    )
    print()

    print("Pretraining split:")
    for split_name in (
        "train",
        "val",
        "test",
    ):
        count = int(
            split_counts.get(
                split_name,
                0,
            )
        )

        print(
            f"  {split_name:<5}: "
            f"{count:5d} "
            f"({count / len(frame):.3f})"
        )

    print()
    print(
        f"Descriptor targets:                    {len(DESCRIPTOR_NAMES)}"
    )
    print(
        "Train-constant descriptors:            "
        + (
            ", ".join(
                constant_descriptors
            )
            if constant_descriptors
            else "none"
        )
    )

    print()
    print(
        f"Morgan fingerprint:                    radius={args.morgan_radius}, "
        f"bits={args.morgan_bits}"
    )
    print(
        f"Active Morgan bits in train:           {active_fp_bits}/{args.morgan_bits}"
    )
    print(
        f"Nonconstant Morgan bits in train:      {nonconstant_fp_bits}/{args.morgan_bits}"
    )

    if not args.split_observed_training:
        observed_train_count = int(
            (
                split_frame[
                    "observed_in_training_bool"
                ]
                & split_frame[
                    "split"
                ].eq("train")
            ).sum()
        )

        print()
        print(
            "Observed training graphs in pretrain train: "
            f"{observed_train_count}/{observed_count}"
        )

    print()
    print(
        f"Results written to:\n  {outdir}"
    )

    print()
    print("Inspect next:")
    for filename in (
        "pretraining_split.csv",
        "descriptor_train_statistics.csv",
        "descriptor_scaler.json",
        f"morgan_fp_{args.morgan_bits}.npz",
        "morgan_fp_bit_statistics.csv",
        "target_schema.json",
        "stage3_manifest.json",
    ):
        print(
            f"  {outdir / filename}"
        )

    print()
    print(
        "STAGE 3 PASSED all split, descriptor, fingerprint, "
        "normalization, and provenance gates."
    )


if __name__ == "__main__":
    main()
