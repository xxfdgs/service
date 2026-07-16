"""Shared, leakage-aware utilities for external generalization diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "generalization_diagnostics"
TARGET_COLUMNS = [
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
]
COMPONENTS = (
    ("IL", "IL_SMILE", "mol%_IL"),
    ("HL", "HL_SMILE", "mol%_HL"),
    ("Chol", "Chol_SMILE", "mol%_Chol"),
    ("PEG", "PEG_SMILE", "mol%_PEG"),
    ("Fifth", "Fifth_SMILE", "mol%_Fifth"),
)


@dataclass(frozen=True)
class DataSchema:
    """Detected schema and discovered project resources."""

    train_path: Path
    feedback_path: Path
    config_path: Path
    targets: list[str]
    id_column: str
    components: list[dict[str, str]]
    descriptor_paths: list[Path]
    nominal_split: list[float]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add command-line options shared by all diagnostic scripts."""
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--feedback-csv", type=Path, default=None)


def canonical_smiles(value: object) -> str:
    """Return a canonical RDKit SMILES, retaining a marked fallback if invalid."""
    if pd.isna(value) or not str(value).strip():
        return "<missing>"
    raw_value = str(value).strip()
    molecule = Chem.MolFromSmiles(raw_value)
    if molecule is None:
        return f"<invalid:{raw_value}>"
    return Chem.MolToSmiles(molecule, canonical=True)


def normalized_text(value: object) -> str:
    """Normalize a component name when no valid structure is present."""
    if pd.isna(value):
        return "<missing>"
    return " ".join(str(value).strip().lower().split()) or "<missing>"


def discover_schema(
    train_csv: Path | None = None, feedback_csv: Path | None = None
) -> DataSchema:
    """Discover the actual GraphGPS training CSV from the best-result config."""
    config_path = ROOT / "results/coarse_mordred/direct_train_coarse_noaux/config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            "Cannot find the current coarse+Mordred result configuration at "
            f"{config_path}."
        )
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    configured_csv = config.get("read_csv")
    if not configured_csv:
        raise ValueError(f"{config_path} does not define read_csv.")
    discovered_train = ROOT / "datasets_lrx/raw" / configured_csv
    discovered_feedback = ROOT / "datasets_lrx/raw/feedback/20260703_validation.csv"
    resolved_train = (train_csv or discovered_train).resolve()
    resolved_feedback = (feedback_csv or discovered_feedback).resolve()
    for csv_path in (resolved_train, resolved_feedback):
        if not csv_path.is_file():
            raise FileNotFoundError(f"Required CSV does not exist: {csv_path}")

    train_frame = pd.read_csv(resolved_train, nrows=3)
    feedback_frame = pd.read_csv(resolved_feedback, nrows=3)
    missing_targets = [column for column in TARGET_COLUMNS if column not in train_frame]
    missing_targets += [
        column for column in TARGET_COLUMNS if column not in feedback_frame
    ]
    if missing_targets:
        raise ValueError(f"Missing required target columns: {sorted(set(missing_targets))}")
    id_column = next(
        (column for column in ("ID", "id", "sample_id", "Sample_ID") if column in train_frame),
        "",
    )
    if not id_column:
        raise ValueError("No supported sample identifier column (ID/id/sample_id) found.")
    components: list[dict[str, str]] = []
    for name_column, smiles_column, ratio_column in COMPONENTS:
        required_columns = (name_column, smiles_column, ratio_column)
        missing_columns = [
            column for column in required_columns if column not in train_frame
        ]
        if missing_columns:
            raise ValueError(
                f"Component {name_column} is missing expected fields: {missing_columns}"
            )
        components.append(
            {"name_column": name_column, "smiles_column": smiles_column,
             "ratio_column": ratio_column}
        )

    descriptor_paths = sorted(
        path for path in (ROOT / "results").glob("mordred_train_feedback*/"
                                             "mordred_descriptors_unique_smiles.csv")
        if path.is_file()
    )
    return DataSchema(
        train_path=resolved_train,
        feedback_path=resolved_feedback,
        config_path=config_path.resolve(),
        targets=list(TARGET_COLUMNS),
        id_column=id_column,
        components=components,
        descriptor_paths=descriptor_paths,
        nominal_split=list(config.get("dataset", {}).get("split", [])),
    )


def load_frames(schema: DataSchema) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load source CSVs and append internal immutable row identifiers."""
    train_frame = pd.read_csv(schema.train_path).copy()
    feedback_frame = pd.read_csv(schema.feedback_path).copy()
    for domain, frame in (("train", train_frame), ("feedback", feedback_frame)):
        if frame[schema.id_column].isna().any():
            raise ValueError(f"{domain} contains missing values in {schema.id_column}.")
        frame["diagnostic_sample_id"] = frame[schema.id_column].astype(str)
        if frame["diagnostic_sample_id"].duplicated().any():
            frame["diagnostic_sample_id"] = (
                domain + ":" + frame[schema.id_column].astype(str) + ":" +
                frame.index.astype(str)
            )
    add_normalized_keys(train_frame, schema)
    add_normalized_keys(feedback_frame, schema)
    return train_frame, feedback_frame


def add_normalized_keys(frame: pd.DataFrame, schema: DataSchema) -> None:
    """Add component and formulation keys in-place without changing raw fields."""
    component_keys: list[str] = []
    for position, component in enumerate(schema.components, start=1):
        smiles_column = component["smiles_column"]
        name_column = component["name_column"]
        component_column = f"component_{position}_key"
        canonical_values = frame[smiles_column].map(canonical_smiles)
        fallback_values = frame[name_column].map(normalized_text)
        frame[component_column] = np.where(
            canonical_values.str.startswith("<"), fallback_values, canonical_values
        )
        component_keys.append(component_column)
    frame["fifth_component_key"] = frame["component_5_key"]
    frame["formula_identity_key"] = frame[component_keys].astype(str).agg("|".join, axis=1)
    ratio_columns = [component["ratio_column"] for component in schema.components]
    formatted_ratios = frame[ratio_columns].apply(
        lambda series: series.map(lambda value: "<missing>" if pd.isna(value)
        else f"{float(value):.6f}")
    )
    frame["formula_ratio_key"] = (
        frame["formula_identity_key"] + "|" + formatted_ratios.astype(str).agg("|".join, axis=1)
    )


def load_mordred_table(schema: DataSchema) -> pd.DataFrame:
    """Load the full cached 2D Mordred table, requiring a canonical SMILES key."""
    if not schema.descriptor_paths:
        raise FileNotFoundError(
            "No mordred_descriptors_unique_smiles.csv cache was discovered under results/."
        )
    descriptor_path = schema.descriptor_paths[0]
    descriptor_frame = pd.read_csv(descriptor_path)
    if "smiles" not in descriptor_frame:
        raise ValueError(f"Mordred cache lacks a smiles column: {descriptor_path}")
    descriptor_frame = descriptor_frame.copy()
    descriptor_frame["canonical_smiles"] = descriptor_frame["smiles"].map(canonical_smiles)
    descriptor_frame = descriptor_frame.drop_duplicates("canonical_smiles", keep="first")
    numeric_columns = [
        column for column in descriptor_frame.columns
        if column not in {"smiles", "canonical_smiles"}
        and pd.api.types.is_numeric_dtype(descriptor_frame[column])
    ]
    descriptor_frame[numeric_columns] = descriptor_frame[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    return descriptor_frame[["canonical_smiles", *numeric_columns]]


def _rdkit_feature_row(smiles_value: object) -> dict[str, float]:
    molecule = Chem.MolFromSmiles(str(smiles_value)) if not pd.isna(smiles_value) else None
    if molecule is None:
        return {
            "mol_weight": np.nan, "mol_logp": np.nan, "heavy_atom_count": np.nan,
            "ring_count": np.nan, "h_bond_donor_count": np.nan,
            "h_bond_acceptor_count": np.nan,
        }
    return {
        "mol_weight": Descriptors.MolWt(molecule),
        "mol_logp": Crippen.MolLogP(molecule),
        "heavy_atom_count": float(Lipinski.HeavyAtomCount(molecule)),
        "ring_count": float(Lipinski.RingCount(molecule)),
        "h_bond_donor_count": float(Lipinski.NumHDonors(molecule)),
        "h_bond_acceptor_count": float(Lipinski.NumHAcceptors(molecule)),
    }


def build_feature_frames(
    frame: pd.DataFrame, schema: DataSchema, mordred_frame: pd.DataFrame | None = None,
    max_mordred_features: int | None = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build numerical and categorical formulation features from reusable caches.

    Mordred columns are retained in cache order. The optional cap controls only
    diagnostics runtime; variance and scaling remain fit within each split.
    """
    numeric_parts: list[pd.DataFrame] = []
    category_features = pd.DataFrame(index=frame.index)
    ratio_columns = [component["ratio_column"] for component in schema.components]
    numeric_parts.append(frame[ratio_columns].copy().rename(
        columns={column: f"ratio_component_{index + 1}" for index, column in enumerate(ratio_columns)}
    ))
    numeric_parts.append(pd.DataFrame({
        "ratio_total": frame[ratio_columns].sum(axis=1, min_count=1),
        "ratio_missing_count": frame[ratio_columns].isna().sum(axis=1),
    }, index=frame.index))
    for position, component in enumerate(schema.components, start=1):
        category_features[f"component_{position}_identity"] = frame[
            f"component_{position}_key"
        ].astype(str)
        rdkit_features = pd.DataFrame(
            [_rdkit_feature_row(value) for value in frame[component["smiles_column"]]],
            index=frame.index,
        ).add_prefix(f"component_{position}_")
        numeric_parts.append(rdkit_features)

    if mordred_frame is not None:
        numeric_mordred_columns = [
            column for column in mordred_frame.columns if column != "canonical_smiles"
        ]
        if max_mordred_features is not None:
            numeric_mordred_columns = numeric_mordred_columns[:max_mordred_features]
        lookup = mordred_frame.set_index("canonical_smiles")[numeric_mordred_columns]
        for position, component in enumerate(schema.components, start=1):
            keys = frame[f"component_{position}_key"].copy()
            joined = lookup.reindex(keys).set_axis(frame.index)
            joined = joined.add_prefix(f"mordred_component_{position}__")
            numeric_parts.append(joined)

    numeric_features = pd.concat(numeric_parts, axis=1)
    numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
    return numeric_features, category_features


def safe_json_dump(data: dict[str, Any], output_path: Path) -> None:
    """Write UTF-8 JSON with parent directory creation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False, default=str)


def metric_dict(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Calculate robust regression metrics for a non-empty finite target subset."""
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, median_absolute_error, r2_score,
    )

    true_values = np.asarray(list(y_true), dtype=float)
    predicted_values = np.asarray(list(y_pred), dtype=float)
    valid_mask = np.isfinite(true_values) & np.isfinite(predicted_values)
    true_values = true_values[valid_mask]
    predicted_values = predicted_values[valid_mask]
    if len(true_values) == 0:
        return {metric_name: np.nan for metric_name in (
            "mae", "rmse", "r2", "median_absolute_error", "pearson", "spearman"
        )}
    pearson_value = pearsonr(true_values, predicted_values).statistic \
        if len(true_values) > 1 and np.std(true_values) > 0 and np.std(predicted_values) > 0 else np.nan
    spearman_value = spearmanr(true_values, predicted_values).statistic \
        if len(true_values) > 1 else np.nan
    return {
        "mae": float(mean_absolute_error(true_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(true_values, predicted_values))),
        "r2": float(r2_score(true_values, predicted_values)) if len(true_values) > 1 else np.nan,
        "median_absolute_error": float(median_absolute_error(true_values, predicted_values)),
        "pearson": float(pearson_value),
        "spearman": float(spearman_value),
    }
