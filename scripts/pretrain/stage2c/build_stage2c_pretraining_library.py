#!/usr/bin/env python3
"""
Stage 2C — merge and freeze the final Fifth-component pretraining molecular library.

Inputs
------
1. Final Stage-1 row-level Fifth audit.
2. Frozen Stage-1.5 scaffold inventory (recommended; optional).
3. Stage-2A single-residue library (80 targets by default).
4. Stage-2B DOPE-peptide library (10,000 targets by default).

Purpose
-------
Create ONE graph-level pretraining molecular library that contains:

    A. Stage 2A:
       validated single-residue scaffold × 20-AA coverage.

    B. Stage 2B:
       controlled DOPE-peptide sequence coverage.

    C. Every observed nonempty Fifth identity from the original training data,
       including observed-only / conditional structures that were deliberately
       not synthetically expanded, e.g.:
           DOPE-OMe variants
           HA-DSSC
           SQWS-DSSC
           S-C4 / S-C6 / ...
           Phg-containing structures
           any other observed structure not already represented by A/B

Deduplication identity
----------------------
The downstream GraphGPS in this project does NOT encode stereochemistry.
Therefore Stage 2C deduplicates by:

    RDKit canonical SMILES with isomericSmiles=False

That is, by model-visible molecular connectivity rather than stereochemical
identity.

Important consequence:
----------------------
63 observed Stage-1 canonical Fifth identities do NOT have to remain 63
different Stage-2C graph rows if some differ only in model-invisible
stereochemistry.

However, the hard coverage gate is identity-level:
    every observed nonempty training Fifth identity must map to a Stage-2C row.

Representative structure policy
-------------------------------
When multiple provenance records collapse to one graph, choose the
representative SMILES in this order:

    1. exact observed training structure from Stage 1
    2. Stage-2A observed-training structure
    3. Stage-2B observed-training structure
    4. Stage-2A generated structure
    5. Stage-2B generated structure

Thus synthetic chemistry never overwrites an observed training structure.

Outputs
-------
stage2c_pretraining_molecular_library.csv
    Final one-row-per-model-visible-graph pretraining library.

stage2c_training_identity_coverage.csv
    One row per observed nonempty Stage-1 Fifth identity, proving that it maps
    to the final library.

stage2c_dedup_groups.csv
    Provenance records for graph keys that had >1 source record before merge.

stage2c_source_summary.csv
    Source/provenance counts.

stage2c_scaffold_summary.csv
    Final graph counts by scaffold family.

stage2c_manifest.json
    Frozen provenance, hashes, gates, and policies.

Hard gates
----------
- all input structures must be RDKit-valid;
- Stage-2A and Stage-2B must each be internally unique under the same
  non-isomeric graph identity;
- no [Fr] placeholder may enter the pretraining molecular library;
- every expected observed nonempty Stage-1 Fifth identity must be covered;
- every observed nonempty training row must be covered;
- final library must be exactly one row per non-isomeric graph identity;
- every final row must carry provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


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
    if isinstance(value, bool):
        return value

    text = clean(value).lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n", ""}:
        return False

    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def mol_or_fail(smiles: str, label: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"RDKit failed to parse {label}: {smiles}"
        )

    Chem.SanitizeMol(mol)
    return mol


def graph_key(mol: Chem.Mol) -> str:
    """
    Model-visible graph identity: stereochemistry removed.
    """
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )


def canonical_isomeric(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


def pipe_join(values: Iterable[Any]) -> str:
    cleaned = {
        clean(value)
        for value in values
        if clean(value)
    }
    return "|".join(sorted(cleaned))


def pipe_join_ordered(values: Iterable[Any]) -> str:
    out: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = clean(value)

        if not text or text in seen:
            continue

        out.append(text)
        seen.add(text)

    return "|".join(out)


def integer_pipe(values: Iterable[Any]) -> str:
    ints: set[int] = set()

    for value in values:
        if value is None or pd.isna(value):
            continue

        text = clean(value)
        if not text:
            continue

        ints.add(int(float(text)))

    return "|".join(str(v) for v in sorted(ints))


# =============================================================================
# Optional Stage-1.5 scaffold metadata
# =============================================================================

def load_scaffold_inventory(
    path: Path | None,
) -> dict[str, dict[str, str]]:
    if path is None:
        return {}

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "canonical_fifth",
        "scaffold_family",
        "scaffold_variant",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            "Scaffold inventory misses required columns: "
            + ", ".join(sorted(missing))
        )

    mapping: dict[str, dict[str, str]] = {}

    for row in frame.itertuples(index=False):
        canonical = clean(row.canonical_fifth)

        if not canonical:
            continue

        record = {
            "scaffold_family": clean(row.scaffold_family),
            "scaffold_variant": clean(row.scaffold_variant),
            "inventory_role": clean(
                getattr(row, "inventory_role", "")
            ),
            "augmentation_eligible": clean(
                getattr(row, "augmentation_eligible", "")
            ),
        }

        if canonical in mapping and mapping[canonical] != record:
            raise ValueError(
                "Conflicting scaffold-inventory metadata for "
                f"{canonical}"
            )

        mapping[canonical] = record

    return mapping


# =============================================================================
# Candidate record constructors
# =============================================================================

def make_candidate(
    *,
    source_category: str,
    source_id: str,
    name: str,
    smiles: str,
    scaffold_family: str = "",
    scaffold_variant: str = "",
    sequence: str = "",
    sequence_length: Any = "",
    sampling_source: str = "",
    structure_source: str = "",
    is_observed_training: bool = False,
    training_identity: str = "",
    training_aliases: str = "",
    training_class: str = "",
    inventory_role: str = "",
    augmentation_eligible: str = "",
) -> dict[str, Any]:
    if not smiles:
        raise ValueError(
            f"Missing SMILES for candidate {source_category}:{source_id}"
        )

    mol = mol_or_fail(
        smiles,
        f"{source_category}:{source_id or name}",
    )

    return {
        "source_category": source_category,
        "source_id": source_id,
        "name": name,
        "input_smiles": smiles,
        "graph_key": graph_key(mol),
        "canonical_isomeric": canonical_isomeric(mol),
        "scaffold_family": scaffold_family,
        "scaffold_variant": scaffold_variant,
        "sequence": sequence,
        "sequence_length": sequence_length,
        "sampling_source": sampling_source,
        "structure_source": structure_source,
        "is_observed_training": bool(is_observed_training),
        "training_identity": training_identity,
        "training_aliases": training_aliases,
        "training_class": training_class,
        "inventory_role": inventory_role,
        "augmentation_eligible": augmentation_eligible,
    }


def candidates_from_stage2a(
    path: Path,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)

    required = {
        "stage2a_id",
        "Fifth",
        "Fifth_SMILE",
        "scaffold_family",
        "aa1",
        "structure_source",
        "observed_in_training",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            "Stage-2A library misses required columns: "
            + ", ".join(sorted(missing))
        )

    candidates = []

    for row in frame.itertuples(index=False):
        candidates.append(
            make_candidate(
                source_category="stage2a",
                source_id=clean(row.stage2a_id),
                name=clean(row.Fifth),
                smiles=clean(row.Fifth_SMILE),
                scaffold_family=clean(row.scaffold_family),
                scaffold_variant="single_residue",
                sequence=clean(row.aa1),
                sequence_length=1,
                sampling_source="stage2a_cartesian_AA",
                structure_source=clean(row.structure_source),
                is_observed_training=bool_from_any(
                    row.observed_in_training
                ),
            )
        )

    return candidates


def candidates_from_stage2b(
    path: Path,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)

    required = {
        "stage2b_id",
        "Fifth",
        "Fifth_SMILE",
        "scaffold_family",
        "scaffold_variant",
        "sequence",
        "sequence_length",
        "sampling_source",
        "structure_source",
        "observed_in_training",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            "Stage-2B library misses required columns: "
            + ", ".join(sorted(missing))
        )

    candidates = []

    for row in frame.itertuples(index=False):
        candidates.append(
            make_candidate(
                source_category="stage2b",
                source_id=clean(row.stage2b_id),
                name=clean(row.Fifth),
                smiles=clean(row.Fifth_SMILE),
                scaffold_family=clean(row.scaffold_family),
                scaffold_variant=clean(row.scaffold_variant),
                sequence=clean(row.sequence),
                sequence_length=row.sequence_length,
                sampling_source=clean(row.sampling_source),
                structure_source=clean(row.structure_source),
                is_observed_training=bool_from_any(
                    row.observed_in_training
                ),
            )
        )

    return candidates


def observed_training_candidates(
    row_audit_path: Path,
    scaffold_metadata: dict[str, dict[str, str]],
) -> tuple[
    list[dict[str, Any]],
    pd.DataFrame,
    int,
]:
    """
    Return:
        one candidate per unique observed nonempty Stage-1 identity,
        an identity table for later coverage auditing,
        count of nonempty observed training rows.
    """
    frame = pd.read_csv(
        row_audit_path,
        dtype={"ID": str},
    )

    required = {
        "ID",
        "Fifth",
        "Fifth_SMILE",
        "canonical_fifth",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            "Stage-1 row audit misses required columns: "
            + ", ".join(sorted(missing))
        )

    if "rdkit_valid" in frame.columns:
        bad = frame.loc[
            ~frame["rdkit_valid"].map(bool_from_any)
        ]

        if not bad.empty:
            raise ValueError(
                f"Stage-1 row audit contains {len(bad)} invalid RDKit rows."
            )

    # [Fr] is the project's no-Fifth placeholder and must not be pretrained.
    nonempty = frame.loc[
        frame["canonical_fifth"]
        .fillna("")
        .astype(str)
        .ne("[Fr]")
        &
        frame["Fifth_SMILE"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    candidates: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []

    for canonical_identity, group in nonempty.groupby(
        "canonical_fifth",
        sort=True,
        dropna=False,
    ):
        canonical_identity = clean(canonical_identity)

        smiles_values = [
            clean(value)
            for value in group["Fifth_SMILE"]
            if clean(value)
        ]

        if not smiles_values:
            raise ValueError(
                "Observed nonempty Fifth identity has no SMILES: "
                f"{canonical_identity}"
            )

        # Different textual SMILES for one Stage-1 canonical identity are okay
        # only if they collapse to one model-visible graph.
        mols = [
            mol_or_fail(
                smiles,
                f"Stage1:{canonical_identity}",
            )
            for smiles in smiles_values
        ]

        graph_keys = {
            graph_key(mol)
            for mol in mols
        }

        if len(graph_keys) != 1:
            raise ValueError(
                "One Stage-1 canonical identity maps to multiple non-isomeric "
                f"graphs: {canonical_identity}"
            )

        aliases = pipe_join_ordered(
            group["Fifth"].tolist()
        )

        classes = (
            pipe_join_ordered(
                group["Fifth_class_canonical"].tolist()
            )
            if "Fifth_class_canonical" in group.columns
            else ""
        )

        sequences = (
            pipe_join_ordered(group["sequence"].tolist())
            if "sequence" in group.columns
            else ""
        )

        sequence_lengths = (
            integer_pipe(group["sequence_length"].tolist())
            if "sequence_length" in group.columns
            else ""
        )

        metadata = scaffold_metadata.get(
            canonical_identity,
            {},
        )

        representative_name = (
            clean(group["Fifth"].iloc[0])
            or aliases
            or canonical_identity
        )

        representative_smiles = smiles_values[0]

        candidate = make_candidate(
            source_category="stage1_observed",
            source_id=canonical_identity,
            name=representative_name,
            smiles=representative_smiles,
            scaffold_family=metadata.get(
                "scaffold_family",
                "",
            ),
            scaffold_variant=metadata.get(
                "scaffold_variant",
                "",
            ),
            sequence=sequences,
            sequence_length=sequence_lengths,
            sampling_source="observed_training_identity",
            structure_source="observed_training",
            is_observed_training=True,
            training_identity=canonical_identity,
            training_aliases=aliases,
            training_class=classes,
            inventory_role=metadata.get(
                "inventory_role",
                "",
            ),
            augmentation_eligible=metadata.get(
                "augmentation_eligible",
                "",
            ),
        )

        candidates.append(candidate)

        identity_rows.append(
            {
                "training_identity": canonical_identity,
                "Fifth_names": aliases,
                "Fifth_class_values": classes,
                "training_rows": int(len(group)),
                "sequence_values": sequences,
                "sequence_length_values": sequence_lengths,
                "scaffold_family": metadata.get(
                    "scaffold_family",
                    "",
                ),
                "scaffold_variant": metadata.get(
                    "scaffold_variant",
                    "",
                ),
                "inventory_role": metadata.get(
                    "inventory_role",
                    "",
                ),
                "augmentation_eligible": metadata.get(
                    "augmentation_eligible",
                    "",
                ),
                "observed_graph_key": next(
                    iter(graph_keys)
                ),
                "example_ID": clean(group["ID"].iloc[0]),
                "example_smiles": representative_smiles,
            }
        )

    return (
        candidates,
        pd.DataFrame(identity_rows),
        int(len(nonempty)),
    )


# =============================================================================
# Input validation
# =============================================================================

def validate_internal_uniqueness(
    candidates: list[dict[str, Any]],
    *,
    source_category: str,
    output_dir: Path,
) -> None:
    subset = [
        record
        for record in candidates
        if record["source_category"] == source_category
    ]

    graph_to_records: defaultdict[str, list[dict[str, Any]]] = (
        defaultdict(list)
    )

    for record in subset:
        graph_to_records[record["graph_key"]].append(record)

    duplicates = {
        key: records
        for key, records in graph_to_records.items()
        if len(records) > 1
    }

    if not duplicates:
        return

    rows = []

    for key, records in duplicates.items():
        for record in records:
            rows.append(
                {
                    "graph_key": key,
                    "source_category": source_category,
                    "source_id": record["source_id"],
                    "name": record["name"],
                    "sequence": record["sequence"],
                    "input_smiles": record["input_smiles"],
                }
            )

    path = (
        output_dir
        / f"stage2c_{source_category}_internal_duplicates.csv"
    )

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )

    raise ValueError(
        f"{source_category} contains duplicate non-isomeric graph identities. "
        f"See {path}"
    )


# =============================================================================
# Merge / representative policy
# =============================================================================

def representative_priority(
    record: dict[str, Any],
) -> tuple[int, str]:
    source = record["source_category"]
    observed = bool(record["is_observed_training"])

    if source == "stage1_observed":
        rank = 0
    elif source == "stage2a" and observed:
        rank = 1
    elif source == "stage2b" and observed:
        rank = 2
    elif source == "stage2a":
        rank = 3
    elif source == "stage2b":
        rank = 4
    else:
        rank = 99

    return rank, record["source_id"]


def merge_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups: defaultdict[str, list[dict[str, Any]]] = (
        defaultdict(list)
    )

    for record in candidates:
        groups[record["graph_key"]].append(record)

    library_rows = []
    duplicate_rows = []

    for graph_identity in sorted(groups):
        records = groups[graph_identity]
        ordered = sorted(
            records,
            key=representative_priority,
        )

        representative = ordered[0]

        representative_mol = mol_or_fail(
            representative["input_smiles"],
            f"representative:{graph_identity}",
        )

        source_categories = pipe_join(
            record["source_category"]
            for record in records
        )

        source_ids = pipe_join(
            record["source_id"]
            for record in records
        )

        names = pipe_join(
            record["name"]
            for record in records
        )

        scaffold_families = pipe_join(
            record["scaffold_family"]
            for record in records
        )

        scaffold_variants = pipe_join(
            record["scaffold_variant"]
            for record in records
        )

        sequences = pipe_join(
            record["sequence"]
            for record in records
        )

        sequence_lengths = integer_pipe(
            record["sequence_length"]
            for record in records
        )

        sampling_sources = pipe_join(
            record["sampling_source"]
            for record in records
        )

        structure_sources = pipe_join(
            record["structure_source"]
            for record in records
        )

        training_identities = pipe_join(
            record["training_identity"]
            for record in records
        )

        training_aliases = pipe_join(
            record["training_aliases"]
            for record in records
        )

        training_classes = pipe_join(
            record["training_class"]
            for record in records
        )

        inventory_roles = pipe_join(
            record["inventory_role"]
            for record in records
        )

        augmentation_eligible_values = pipe_join(
            record["augmentation_eligible"]
            for record in records
        )

        has_training_observed = any(
            bool(record["is_observed_training"])
            for record in records
        )

        has_generated_source = any(
            record["source_category"] in {"stage2a", "stage2b"}
            and not bool(record["is_observed_training"])
            for record in records
        )

        library_rows.append(
            {
                # Assigned deterministically after sorting below.
                "stage2c_id": "",
                "Fifth": representative["name"],
                "Fifth_SMILE": representative["input_smiles"],
                "canonical_connectivity": graph_identity,
                "representative_canonical_isomeric": (
                    canonical_isomeric(representative_mol)
                ),
                "preferred_source_category": representative[
                    "source_category"
                ],
                "preferred_source_id": representative["source_id"],
                "source_categories": source_categories,
                "source_ids": source_ids,
                "source_record_count": int(len(records)),
                "all_names": names,
                "scaffold_families": scaffold_families,
                "scaffold_variants": scaffold_variants,
                "sequences": sequences,
                "sequence_lengths": sequence_lengths,
                "sampling_sources": sampling_sources,
                "structure_sources": structure_sources,
                "observed_in_training": bool(
                    has_training_observed
                ),
                "has_generated_source": bool(
                    has_generated_source
                ),
                "synthetic_only": bool(
                    has_generated_source
                    and not has_training_observed
                ),
                "training_identity_count": int(
                    len(
                        {
                            record["training_identity"]
                            for record in records
                            if record["training_identity"]
                        }
                    )
                ),
                "training_identities": training_identities,
                "training_aliases": training_aliases,
                "training_classes": training_classes,
                "inventory_roles": inventory_roles,
                "augmentation_eligible_values": (
                    augmentation_eligible_values
                ),
                "formal_charge": int(
                    Chem.GetFormalCharge(representative_mol)
                ),
                "mol_wt": float(
                    Descriptors.MolWt(representative_mol)
                ),
                "heavy_atom_count": int(
                    rdMolDescriptors.CalcNumHeavyAtoms(
                        representative_mol
                    )
                ),
                "hbond_donors": int(
                    Lipinski.NumHDonors(representative_mol)
                ),
                "hbond_acceptors": int(
                    Lipinski.NumHAcceptors(representative_mol)
                ),
                "rdkit_valid": True,
            }
        )

        if len(records) > 1:
            for record in ordered:
                duplicate_rows.append(
                    {
                        "canonical_connectivity": graph_identity,
                        "group_size": int(len(records)),
                        "selected_as_representative": bool(
                            record is representative
                        ),
                        "source_category": record[
                            "source_category"
                        ],
                        "source_id": record["source_id"],
                        "name": record["name"],
                        "scaffold_family": record[
                            "scaffold_family"
                        ],
                        "sequence": record["sequence"],
                        "sampling_source": record[
                            "sampling_source"
                        ],
                        "structure_source": record[
                            "structure_source"
                        ],
                        "is_observed_training": bool(
                            record["is_observed_training"]
                        ),
                        "training_identity": record[
                            "training_identity"
                        ],
                        "input_smiles": record[
                            "input_smiles"
                        ],
                    }
                )

    library = pd.DataFrame(library_rows).sort_values(
        "canonical_connectivity"
    ).reset_index(drop=True)

    library["stage2c_id"] = [
        f"S2C_{i:05d}"
        for i in range(len(library))
    ]

    dedup = pd.DataFrame(duplicate_rows)

    return library, dedup


# =============================================================================
# Coverage and summaries
# =============================================================================

def make_training_identity_coverage(
    identity_table: pd.DataFrame,
    library: pd.DataFrame,
) -> pd.DataFrame:
    graph_to_stage2c = dict(
        zip(
            library["canonical_connectivity"],
            library["stage2c_id"],
        )
    )

    graph_to_sources = dict(
        zip(
            library["canonical_connectivity"],
            library["source_categories"],
        )
    )

    graph_to_preferred = dict(
        zip(
            library["canonical_connectivity"],
            library["preferred_source_category"],
        )
    )

    coverage = identity_table.copy()

    coverage["stage2c_id"] = coverage[
        "observed_graph_key"
    ].map(graph_to_stage2c)

    coverage["covered"] = coverage[
        "stage2c_id"
    ].notna()

    coverage["stage2c_source_categories"] = coverage[
        "observed_graph_key"
    ].map(graph_to_sources).fillna("")

    coverage["stage2c_preferred_source"] = coverage[
        "observed_graph_key"
    ].map(graph_to_preferred).fillna("")

    return coverage


def make_source_summary(
    candidates: list[dict[str, Any]],
    library: pd.DataFrame,
) -> pd.DataFrame:
    candidate_counts = Counter(
        record["source_category"]
        for record in candidates
    )

    rows = []

    for source in sorted(candidate_counts):
        final_graphs_containing_source = int(
            library["source_categories"]
            .fillna("")
            .str.split("|")
            .map(lambda parts: source in parts)
            .sum()
        )

        preferred_count = int(
            library["preferred_source_category"]
            .eq(source)
            .sum()
        )

        rows.append(
            {
                "source_category": source,
                "input_candidate_records": int(
                    candidate_counts[source]
                ),
                "final_graphs_containing_source": (
                    final_graphs_containing_source
                ),
                "final_graphs_preferred_from_source": (
                    preferred_count
                ),
            }
        )

    return pd.DataFrame(rows)


def make_scaffold_summary(
    library: pd.DataFrame,
) -> pd.DataFrame:
    counter: Counter[str] = Counter()

    observed_counter: Counter[str] = Counter()
    synthetic_counter: Counter[str] = Counter()

    for row in library.itertuples(index=False):
        families = [
            part
            for part in clean(
                row.scaffold_families
            ).split("|")
            if part
        ]

        if not families:
            families = ["UNSPECIFIED"]

        for family in families:
            counter[family] += 1

            if bool(row.observed_in_training):
                observed_counter[family] += 1

            if bool(row.synthetic_only):
                synthetic_counter[family] += 1

    rows = []

    for family in sorted(counter):
        rows.append(
            {
                "scaffold_family": family,
                "final_graphs": int(counter[family]),
                "graphs_with_training_observation": int(
                    observed_counter[family]
                ),
                "synthetic_only_graphs": int(
                    synthetic_counter[family]
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Stage 2A + Stage 2B + all observed training Fifth "
            "identities into the frozen Stage-2C pretraining molecular library."
        )
    )

    parser.add_argument(
        "--row-audit",
        type=Path,
        required=True,
        help="Final Stage-1 row_level_fifth_audit.csv",
    )

    parser.add_argument(
        "--scaffold-inventory",
        type=Path,
        default=None,
        help=(
            "Recommended frozen Stage-1.5 scaffold_inventory.csv. "
            "If omitted, observed structures are still included but some "
            "scaffold metadata fields will be blank."
        ),
    )

    parser.add_argument(
        "--stage2a-library",
        type=Path,
        required=True,
        help="Stage-2A stage2a_single_aa_library.csv",
    )

    parser.add_argument(
        "--stage2b-library",
        type=Path,
        required=True,
        help="Stage-2B stage2b_dope_peptide_library.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--expect-training-identities",
        type=int,
        default=63,
        help=(
            "Expected number of nonempty observed canonical Fifth identities. "
            "Set <=0 to disable."
        ),
    )

    parser.add_argument(
        "--expect-stage2a-records",
        type=int,
        default=80,
        help="Expected Stage-2A input rows; <=0 disables.",
    )

    parser.add_argument(
        "--expect-stage2b-records",
        type=int,
        default=10_000,
        help="Expected Stage-2B input rows; <=0 disables.",
    )

    args = parser.parse_args()

    row_audit_path = args.row_audit.resolve()
    stage2a_path = args.stage2a_library.resolve()
    stage2b_path = args.stage2b_library.resolve()
    inventory_path = (
        args.scaffold_inventory.resolve()
        if args.scaffold_inventory is not None
        else None
    )
    outdir = args.output_dir.resolve()

    for path in (
        row_audit_path,
        stage2a_path,
        stage2b_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if inventory_path is not None and not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Load all candidate sources
    # ------------------------------------------------------------------

    scaffold_metadata = load_scaffold_inventory(
        inventory_path
    )

    stage2a_candidates = candidates_from_stage2a(
        stage2a_path
    )

    stage2b_candidates = candidates_from_stage2b(
        stage2b_path
    )

    (
        stage1_candidates,
        identity_table,
        nonempty_training_rows,
    ) = observed_training_candidates(
        row_audit_path,
        scaffold_metadata,
    )

    if (
        args.expect_stage2a_records > 0
        and len(stage2a_candidates)
        != args.expect_stage2a_records
    ):
        raise ValueError(
            f"Expected {args.expect_stage2a_records} Stage-2A rows, "
            f"found {len(stage2a_candidates)}."
        )

    if (
        args.expect_stage2b_records > 0
        and len(stage2b_candidates)
        != args.expect_stage2b_records
    ):
        raise ValueError(
            f"Expected {args.expect_stage2b_records} Stage-2B rows, "
            f"found {len(stage2b_candidates)}."
        )

    if (
        args.expect_training_identities > 0
        and len(identity_table)
        != args.expect_training_identities
    ):
        raise ValueError(
            f"Expected {args.expect_training_identities} observed nonempty "
            f"training identities, found {len(identity_table)}."
        )

    # Stage 2A and 2B themselves should already be graph-unique.
    all_candidates = (
        stage2a_candidates
        + stage2b_candidates
        + stage1_candidates
    )

    validate_internal_uniqueness(
        all_candidates,
        source_category="stage2a",
        output_dir=outdir,
    )

    validate_internal_uniqueness(
        all_candidates,
        source_category="stage2b",
        output_dir=outdir,
    )

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    library, dedup = merge_candidates(
        all_candidates
    )

    # [Fr] must never enter molecular pretraining.
    if library["Fifth_SMILE"].eq("[Fr]").any():
        raise SystemExit(
            "Stage 2C BLOCKED: [Fr] placeholder entered final library."
        )

    if library["canonical_connectivity"].duplicated().any():
        raise SystemExit(
            "Stage 2C BLOCKED: final library still contains duplicate "
            "non-isomeric graph identities."
        )

    if not library["rdkit_valid"].all():
        raise SystemExit(
            "Stage 2C BLOCKED: final library contains invalid RDKit molecules."
        )

    if library["source_categories"].fillna("").eq("").any():
        raise SystemExit(
            "Stage 2C BLOCKED: final library contains rows without provenance."
        )

    # ------------------------------------------------------------------
    # Observed-training identity coverage
    # ------------------------------------------------------------------

    training_coverage = make_training_identity_coverage(
        identity_table,
        library,
    )

    missing_identities = training_coverage.loc[
        ~training_coverage["covered"]
    ].copy()

    if not missing_identities.empty:
        missing_identities.to_csv(
            outdir / "stage2c_missing_training_identities.csv",
            index=False,
        )

        raise SystemExit(
            "Stage 2C BLOCKED: not every observed nonempty training Fifth "
            "identity is represented in the final library. See "
            "stage2c_missing_training_identities.csv."
        )

    covered_training_rows = int(
        training_coverage["training_rows"].sum()
    )

    if covered_training_rows != nonempty_training_rows:
        raise SystemExit(
            "Stage 2C BLOCKED: identity coverage passes but observed training "
            f"row accounting disagrees: {covered_training_rows} vs "
            f"{nonempty_training_rows}."
        )

    # ------------------------------------------------------------------
    # Output summaries
    # ------------------------------------------------------------------

    source_summary = make_source_summary(
        all_candidates,
        library,
    )

    scaffold_summary = make_scaffold_summary(
        library
    )

    library.to_csv(
        outdir / "stage2c_pretraining_molecular_library.csv",
        index=False,
    )

    training_coverage.to_csv(
        outdir / "stage2c_training_identity_coverage.csv",
        index=False,
    )

    if dedup.empty:
        pd.DataFrame(
            columns=[
                "canonical_connectivity",
                "group_size",
                "selected_as_representative",
                "source_category",
                "source_id",
                "name",
                "scaffold_family",
                "sequence",
                "sampling_source",
                "structure_source",
                "is_observed_training",
                "training_identity",
                "input_smiles",
            ]
        ).to_csv(
            outdir / "stage2c_dedup_groups.csv",
            index=False,
        )
    else:
        dedup.to_csv(
            outdir / "stage2c_dedup_groups.csv",
            index=False,
        )

    source_summary.to_csv(
        outdir / "stage2c_source_summary.csv",
        index=False,
    )

    scaffold_summary.to_csv(
        outdir / "stage2c_scaffold_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Useful counts
    # ------------------------------------------------------------------

    raw_candidates = len(all_candidates)

    duplicate_records_removed = (
        raw_candidates - len(library)
    )

    observed_final_graphs = int(
        library["observed_in_training"].sum()
    )

    synthetic_only_graphs = int(
        library["synthetic_only"].sum()
    )

    observed_identity_graphs = int(
        training_coverage["observed_graph_key"].nunique()
    )

    # Graphs added only by Stage-1 observed supplementation:
    stage1_only_graphs = int(
        library["source_categories"].eq("stage1_observed").sum()
    )

    graphs_with_stage1_and_augmented_source = int(
        library["source_categories"]
        .map(
            lambda text: (
                "stage1_observed" in clean(text).split("|")
                and (
                    "stage2a" in clean(text).split("|")
                    or "stage2b" in clean(text).split("|")
                )
            )
        )
        .sum()
    )

    # ------------------------------------------------------------------
    # Frozen manifest
    # ------------------------------------------------------------------

    manifest = {
        "stage": "2C_final_pretraining_molecular_library",
        "inputs": {
            "row_audit": str(row_audit_path),
            "row_audit_sha256": sha256(row_audit_path),
            "stage2a_library": str(stage2a_path),
            "stage2a_sha256": sha256(stage2a_path),
            "stage2b_library": str(stage2b_path),
            "stage2b_sha256": sha256(stage2b_path),
            "scaffold_inventory": (
                str(inventory_path)
                if inventory_path is not None
                else None
            ),
            "scaffold_inventory_sha256": (
                sha256(inventory_path)
                if inventory_path is not None
                else None
            ),
        },
        "input_counts": {
            "stage2a_records": int(
                len(stage2a_candidates)
            ),
            "stage2b_records": int(
                len(stage2b_candidates)
            ),
            "stage1_observed_identity_records": int(
                len(stage1_candidates)
            ),
            "raw_candidate_records_before_cross_source_dedup": int(
                raw_candidates
            ),
        },
        "final_counts": {
            "unique_model_visible_graphs": int(
                len(library)
            ),
            "duplicate_candidate_records_merged": int(
                duplicate_records_removed
            ),
            "graphs_with_training_observation": int(
                observed_final_graphs
            ),
            "synthetic_only_graphs": int(
                synthetic_only_graphs
            ),
            "stage1_only_supplemental_graphs": int(
                stage1_only_graphs
            ),
            "graphs_shared_between_stage1_and_stage2a_or_stage2b": int(
                graphs_with_stage1_and_augmented_source
            ),
        },
        "training_coverage": {
            "observed_nonempty_training_rows": int(
                nonempty_training_rows
            ),
            "observed_nonempty_training_identities": int(
                len(identity_table)
            ),
            "covered_training_identities": int(
                training_coverage["covered"].sum()
            ),
            "training_identity_coverage_rate": float(
                training_coverage["covered"].mean()
            ),
            "model_visible_graphs_spanned_by_training_identities": int(
                observed_identity_graphs
            ),
        },
        "deduplication_policy": {
            "identity": "RDKit canonical SMILES, isomericSmiles=False",
            "reason": (
                "Downstream GraphGPS does not encode stereochemistry; "
                "Stage 2C freezes one row per model-visible molecular graph."
            ),
            "representative_priority": [
                "stage1_observed",
                "stage2a_observed_training",
                "stage2b_observed_training",
                "stage2a_generated",
                "stage2b_generated",
            ],
            "observed_structure_authority": (
                "If a graph is observed in training, an exact observed Stage-1 "
                "SMILES is preferred as the final representative."
            ),
        },
        "hard_gates": {
            "stage2a_internal_graph_unique": True,
            "stage2b_internal_graph_unique": True,
            "no_Fr_placeholder": True,
            "all_final_graphs_rdkit_valid": True,
            "final_graph_identity_unique": True,
            "all_final_rows_have_provenance": True,
            "all_observed_training_identities_covered": True,
            "all_observed_nonempty_training_rows_accounted_for": True,
        },
        "next_stage": {
            "name": "Stage 3 pretraining targets",
            "input": "stage2c_pretraining_molecular_library.csv",
            "planned_targets": (
                "RDKit molecular descriptors + Morgan fingerprint; "
                "descriptor normalization must be fit on the pretraining "
                "train split only."
            ),
        },
    }

    with (
        outdir / "stage2c_manifest.json"
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
    # Terminal report
    # ------------------------------------------------------------------

    print("=" * 92)
    print("STAGE 2C — FINAL PRETRAINING MOLECULAR LIBRARY")
    print("=" * 92)

    print(f"Stage-2A candidate records:              {len(stage2a_candidates)}")
    print(f"Stage-2B candidate records:              {len(stage2b_candidates)}")
    print(f"Observed Stage-1 identity records:       {len(stage1_candidates)}")
    print(f"Raw candidates before cross-source dedup:{raw_candidates:>10}")
    print()

    print(f"Final unique model-visible graphs:       {len(library)}")
    print(f"Candidate records merged by dedup:       {duplicate_records_removed}")
    print(f"Graphs with training observation:        {observed_final_graphs}")
    print(f"Synthetic-only graphs:                   {synthetic_only_graphs}")
    print(f"Stage-1-only supplemental graphs:        {stage1_only_graphs}")
    print()

    print(
        "Observed training Fifth identities covered: "
        f"{int(training_coverage['covered'].sum())}/"
        f"{len(training_coverage)} "
        f"({training_coverage['covered'].mean():.3f})"
    )

    print(
        "Observed nonempty training rows accounted:  "
        f"{covered_training_rows}/{nonempty_training_rows}"
    )

    print(
        "Model-visible graphs spanned by observed identities: "
        f"{observed_identity_graphs}"
    )

    print()
    print("Source summary:")
    print(source_summary.to_string(index=False))

    print()
    print("Scaffold summary:")
    print(scaffold_summary.to_string(index=False))

    print()
    print(f"Results written to:\n  {outdir}")

    print()
    print("Inspect next:")
    print(
        f"  {outdir / 'stage2c_pretraining_molecular_library.csv'}"
    )
    print(
        f"  {outdir / 'stage2c_training_identity_coverage.csv'}"
    )
    print(
        f"  {outdir / 'stage2c_dedup_groups.csv'}"
    )
    print(
        f"  {outdir / 'stage2c_manifest.json'}"
    )

    print()
    print(
        "STAGE 2C PASSED all validity, deduplication, provenance, "
        "and training-coverage gates."
    )


if __name__ == "__main__":
    main()
