"""Low-dimensional, chemically interpretable formulation feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from common import DataSchema


PHYSICOCHEM_NAMES = [
    "molecular_weight", "heavy_atom_count", "formal_charge", "logp", "tpsa",
    "hbd", "hba", "rotatable_bond_count", "ring_count", "aromatic_atom_count",
    "fraction_c", "fraction_hetero",
]


def molecule_physicochemical_features(smiles_value: object) -> dict[str, float]:
    """Calculate only the compact stable RDKit properties specified for stage two."""
    molecule = Chem.MolFromSmiles(str(smiles_value)) if not pd.isna(smiles_value) else None
    if molecule is None:
        return {name: np.nan for name in PHYSICOCHEM_NAMES}
    atoms = list(molecule.GetAtoms())
    atom_count = max(len(atoms), 1)
    carbon_count = sum(atom.GetAtomicNum() == 6 for atom in atoms)
    hetero_count = sum(atom.GetAtomicNum() not in {1, 6} for atom in atoms)
    return {
        "molecular_weight": float(Descriptors.MolWt(molecule)),
        "heavy_atom_count": float(Lipinski.HeavyAtomCount(molecule)),
        "formal_charge": float(sum(atom.GetFormalCharge() for atom in atoms)),
        "logp": float(Crippen.MolLogP(molecule)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
        "hbd": float(Lipinski.NumHDonors(molecule)),
        "hba": float(Lipinski.NumHAcceptors(molecule)),
        "rotatable_bond_count": float(Lipinski.NumRotatableBonds(molecule)),
        "ring_count": float(Lipinski.RingCount(molecule)),
        "aromatic_atom_count": float(sum(atom.GetIsAromatic() for atom in atoms)),
        "fraction_c": float(carbon_count / atom_count),
        "fraction_hetero": float(hetero_count / atom_count),
    }


def _ratio_features(frame: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    ratio_columns = [component["ratio_column"] for component in schema.components]
    ratios = frame[ratio_columns].astype(float).copy()
    ratios.columns = [f"ratio_component_{position}" for position in range(1, 6)]
    nonzero = ratios.where(ratios > 0)
    proportions = ratios.div(ratios.sum(axis=1).replace(0, np.nan), axis=0)
    entropy = -(proportions * np.log(proportions.where(proportions > 0))).sum(axis=1).fillna(0.0)
    return pd.concat([ratios, pd.DataFrame({
        "ratio_sum": ratios.sum(axis=1), "ratio_max": ratios.max(axis=1),
        "ratio_min_nonzero": nonzero.min(axis=1), "formulation_entropy": entropy,
        "effective_component_count": np.exp(entropy),
    }, index=frame.index)], axis=1)


def build_stable_feature_sets(
    frame: pd.DataFrame, schema: DataSchema,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, object]]:
    """Build F1–F4 without high-dimensional descriptors or target-derived encodings."""
    ratio_frame = _ratio_features(frame, schema)
    component_tables: list[pd.DataFrame] = []
    identity_frame = pd.DataFrame(index=frame.index)
    for position, component in enumerate(schema.components, start=1):
        features = pd.DataFrame(
            [molecule_physicochemical_features(value) for value in frame[component["smiles_column"]]],
            index=frame.index,
        ).add_prefix(f"component_{position}__")
        component_tables.append(features)
        identity_frame[f"component_{position}_identity"] = frame[
            f"component_{position}_key"
        ].astype(str)
    component_feature_frame = pd.concat(component_tables, axis=1)
    ratios = ratio_frame[[f"ratio_component_{position}" for position in range(1, 6)]].to_numpy(dtype=float)
    normalized_ratios = ratios / np.clip(ratios.sum(axis=1, keepdims=True), 1e-12, None)
    summary_data: dict[str, np.ndarray] = {}
    for property_name in PHYSICOCHEM_NAMES:
        values = np.column_stack([
            component_feature_frame[f"component_{position}__{property_name}"].to_numpy(dtype=float)
            for position in range(1, 6)
        ])
        weighted_mean = np.nansum(values * normalized_ratios, axis=1)
        weighted_variance = np.nansum(
            normalized_ratios * (values - weighted_mean[:, None]) ** 2, axis=1
        )
        first_four_ratios = normalized_ratios[:, :4]
        first_four_mean = np.nansum(values[:, :4] * first_four_ratios, axis=1) / np.clip(
            first_four_ratios.sum(axis=1), 1e-12, None
        )
        summary_data[f"weighted_mean__{property_name}"] = weighted_mean
        summary_data[f"weighted_std__{property_name}"] = np.sqrt(np.maximum(weighted_variance, 0.0))
        summary_data[f"max__{property_name}"] = np.nanmax(values, axis=1)
        summary_data[f"min__{property_name}"] = np.nanmin(values, axis=1)
        summary_data[f"fifth_minus_first4_mean__{property_name}"] = values[:, 4] - first_four_mean
    summary_frame = pd.DataFrame(summary_data, index=frame.index)
    interaction_data = {
        "interaction_ratio_5_x_logp_gap": ratio_frame["ratio_component_5"] * summary_frame["fifth_minus_first4_mean__logp"],
        "interaction_ratio_5_x_tpsa_gap": ratio_frame["ratio_component_5"] * summary_frame["fifth_minus_first4_mean__tpsa"],
        "interaction_ratio_5_x_mw_gap": ratio_frame["ratio_component_5"] * summary_frame["fifth_minus_first4_mean__molecular_weight"],
        "interaction_ratio_1_x_ratio_5": ratio_frame["ratio_component_1"] * ratio_frame["ratio_component_5"],
        "interaction_ratio_2_x_ratio_5": ratio_frame["ratio_component_2"] * ratio_frame["ratio_component_5"],
        "interaction_ratio_3_x_ratio_5": ratio_frame["ratio_component_3"] * ratio_frame["ratio_component_5"],
        "interaction_ratio_4_x_ratio_5": ratio_frame["ratio_component_4"] * ratio_frame["ratio_component_5"],
        "interaction_logp_x_tpsa": summary_frame["weighted_mean__logp"] * summary_frame["weighted_mean__tpsa"],
        "interaction_mw_x_ratio_entropy": summary_frame["weighted_mean__molecular_weight"] * ratio_frame["formulation_entropy"],
        "interaction_ring_x_aromatic": summary_frame["weighted_mean__ring_count"] * summary_frame["weighted_mean__aromatic_atom_count"],
    }
    interaction_frame = pd.DataFrame(interaction_data, index=frame.index)
    feature_sets = {
        "F1_ratio_only": ratio_frame,
        "F2_identity_ratio": pd.concat([ratio_frame, identity_frame], axis=1),
        "F3_physchem_weighted": pd.concat([ratio_frame, summary_frame], axis=1),
        "F4_physchem_interactions": pd.concat([ratio_frame, summary_frame, interaction_frame], axis=1),
    }
    schema_payload = {
        "physicochemical_properties": PHYSICOCHEM_NAMES,
        "feature_dimensions": {name: int(features.shape[1]) for name, features in feature_sets.items()},
        "identity_encoding": "component canonical identity is one-hot encoded within each training fold only",
        "f4_dimension_limit": 100,
        "f4_dimension_with_raw_categories": int(feature_sets["F4_physchem_interactions"].shape[1]),
    }
    return feature_sets, component_feature_frame, schema_payload
