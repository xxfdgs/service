#!/usr/bin/env python3
"""
Build a frozen scaffold inventory from the Stage-1 Fifth audit.

This script is Stage 1.5 of the Fifth-component pretraining workflow.

Inputs
------
The recommended input is the FINAL row-level audit produced by:
    audit_fifth_amino_acid_coverage.py

Required columns:
    canonical_fifth
    Fifth
    Fifth_class_canonical
    sequence
    sequence_length
    parse_status
    peptide_category
    modification
    Fifth_SMILE
    ID

Outputs
-------
1. scaffold_inventory.csv
   One row per canonical Fifth identity.

2. scaffold_family_summary.csv
   One row per scaffold family / scaffold variant.

3. scaffold_generation_plan.csv
   The frozen high-level plan that Stage-2 generators should consume.

4. scaffold_review.csv
   Identities that must NOT be automatically expanded before review.

5. scaffold_manifest.json
   Frozen counts, policies, and provenance.

Design principles
-----------------
- Cover all training-derived scaffold identities in the pretraining corpus.
- Only perform amino-acid/peptide substitution when the substitution site is
  chemically interpretable from the current nomenclature.
- Keep "scaffold coverage" separate from "augmentation eligibility".
- Modified peptide scaffolds and DSSC-bearing special scaffolds are retained,
  but are conditional rather than blindly expanded.
- Unknown identities fail closed: they are retained as original structures
  but are excluded from automatic augmentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


AA3_TO_AA1 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}
CANONICAL_AA1 = set(AA3_TO_AA1.values())
RES3_PATTERN = "(?:" + "|".join(sorted(AA3_TO_AA1, key=len, reverse=True)) + "|Phg)"


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def split_pipe(values: pd.Series) -> list[str]:
    out: set[str] = set()
    for value in values:
        text = clean(value)
        if not text:
            continue
        for part in text.split("|"):
            part = part.strip()
            if part:
                out.add(part)
    return sorted(out)


def unique_nonempty(values: pd.Series) -> list[str]:
    return sorted({clean(v) for v in values if clean(v)})


def classify_identity(
    *,
    fifth_names: list[str],
    canonical_fifth: str,
    fifth_classes: list[str],
    sequences: list[str],
    parse_statuses: list[str],
    peptide_categories: list[str],
    modifications: list[str],
) -> dict[str, Any]:
    """
    Assign one canonical training-derived scaffold identity to a generation class.

    IMPORTANT:
    This is intentionally a nomenclature-aware inventory classifier, not a
    chemistry reaction engine. "augmentation_eligible=yes" means that Stage 2
    may construct a validated generator for this family; it does not authorize
    arbitrary graph edits.
    """

    nonempty_names = [n for n in fifth_names if n]
    name_blob = "|".join(nonempty_names)

    # ------------------------------------------------------------------
    # 0. Absent Fifth
    # ------------------------------------------------------------------
    if canonical_fifth == "[Fr]" or not nonempty_names:
        return {
            "scaffold_family": "ABSENT",
            "scaffold_variant": "none",
            "residue_or_sequence_site": "none",
            "replaceability": "no",
            "augmentation_eligible": "no",
            "generation_strategy": "exclude_from_pretraining_generation",
            "generation_scope": "none",
            "review_required": "no",
            "review_reason": "",
            "inventory_role": "absent_placeholder",
            "notes": "No Fifth component; keep only as downstream placeholder.",
        }

    # ------------------------------------------------------------------
    # 1. Explicit non-peptide S-* series
    # ------------------------------------------------------------------
    nonpeptide_match = None
    if len(nonempty_names) == 1:
        nonpeptide_match = re.fullmatch(
            r"S-(C4|C6|C8|C10|C12|NH2|Boc|COOH)",
            nonempty_names[0],
            flags=re.IGNORECASE,
        )

    if nonpeptide_match:
        variant = nonpeptide_match.group(1).upper()
        return {
            "scaffold_family": "S_nonpeptide_series",
            "scaffold_variant": variant,
            "residue_or_sequence_site": "none",
            "replaceability": "no",
            "augmentation_eligible": "no",
            "generation_strategy": "keep_original_identity_only",
            "generation_scope": "original_structure",
            "review_required": "no",
            "review_reason": "",
            "inventory_role": "nonpeptide_scaffold_coverage",
            "notes": (
                "Retain this observed structure in the pretraining corpus, "
                "but do not interpret leading S as serine and do not perform "
                "amino-acid substitution."
            ),
        }

    # ------------------------------------------------------------------
    # 2. Single-residue UC12 / UC18 families
    #    Canonical residues and Phg share the same replaceable scaffold family.
    # ------------------------------------------------------------------
    if len(nonempty_names) == 1:
        m = re.fullmatch(
            rf"(?P<res>{RES3_PATTERN})-UC(?P<n>12|18)",
            nonempty_names[0],
            flags=re.IGNORECASE,
        )
        if m:
            n = m.group("n")
            residue = m.group("res")
            return {
                "scaffold_family": f"UC{n}",
                "scaffold_variant": f"UC{n}",
                "residue_or_sequence_site": "single_residue",
                "replaceability": "yes",
                "augmentation_eligible": "yes",
                "generation_strategy": "enumerate_20_canonical_amino_acids",
                "generation_scope": "20_canonical_AA_cartesian_coverage",
                "review_required": "no",
                "review_reason": "",
                "inventory_role": "single_residue_template",
                "notes": (
                    f"Observed residue={residue}. Use the family as a scaffold "
                    "template; deduplicate generated structures by canonical SMILES."
                ),
            }

    # ------------------------------------------------------------------
    # 3. Single-residue C12-COOH / C18-COOH families
    # ------------------------------------------------------------------
    if len(nonempty_names) == 1:
        m = re.fullmatch(
            rf"(?P<res>{RES3_PATTERN})(?P<n>12|18)-COOH",
            nonempty_names[0],
            flags=re.IGNORECASE,
        )
        if m:
            n = m.group("n")
            residue = m.group("res")
            return {
                "scaffold_family": f"C{n}_COOH",
                "scaffold_variant": f"C{n}_COOH",
                "residue_or_sequence_site": "single_residue",
                "replaceability": "yes",
                "augmentation_eligible": "yes",
                "generation_strategy": "enumerate_20_canonical_amino_acids",
                "generation_scope": "20_canonical_AA_cartesian_coverage",
                "review_required": "no",
                "review_reason": "",
                "inventory_role": "single_residue_template",
                "notes": (
                    f"Observed residue={residue}. Use the family as a scaffold "
                    "template; deduplicate generated structures by canonical SMILES."
                ),
            }

    # ------------------------------------------------------------------
    # 4. Unmodified DOPE-peptide family
    #
    # Covers explicit sequences such as DSSC-DOPE, DRDRC-DOPE and repeat
    # notation such as 4DC-DOPE / 8RC-DOPE.
    # ------------------------------------------------------------------
    all_dope_suffix = all(
        re.fullmatch(
            r"(?:[ACDEFGHIKLMNPQRSTVWY]{2,}|\d+[ACDEFGHIKLMNPQRSTVWY]C)-DOPE",
            name,
            flags=re.IGNORECASE,
        )
        for name in nonempty_names
    )
    if all_dope_suffix and nonempty_names:
        return {
            "scaffold_family": "DOPE_peptide",
            "scaffold_variant": "unmodified",
            "residue_or_sequence_site": "peptide_sequence",
            "replaceability": "yes",
            "augmentation_eligible": "yes",
            "generation_strategy": "controlled_peptide_sampling",
            "generation_scope": "canonical_AA_sequences_length_2_to_9",
            "review_required": "no",
            "review_reason": "",
            "inventory_role": "peptide_template",
            "notes": (
                "Primary peptide scaffold for Stage 2. Sample lengths 2-9 with "
                "balanced residue and positional coverage; do not enumerate 20^L."
            ),
        }

    # ------------------------------------------------------------------
    # 5. DOPE OMe modified variants
    #
    # These count toward scaffold coverage, but OMe location depends on D/C
    # chemistry. They must not be blindly combined with arbitrary sequences.
    # ------------------------------------------------------------------
    all_dope_ome = all(
        re.fullmatch(
            r"DOPE-(?:C|D|DC)-(?:OMe|Ome)",
            name,
            flags=re.IGNORECASE,
        )
        for name in nonempty_names
    )
    if all_dope_ome and nonempty_names:
        variant_tokens = []
        for name in nonempty_names:
            m = re.fullmatch(
                r"DOPE-(C|D|DC)-(?:OMe|Ome)",
                name,
                flags=re.IGNORECASE,
            )
            assert m is not None
            variant_tokens.append(m.group(1).upper())

        variant = "+".join(sorted(set(variant_tokens)))
        return {
            "scaffold_family": "DOPE_peptide",
            "scaffold_variant": f"OMe_{variant}",
            "residue_or_sequence_site": "peptide_sequence_with_site_specific_OMe",
            "replaceability": "conditional",
            "augmentation_eligible": "conditional",
            "generation_strategy": "keep_original_then_validate_compatible_OMe_generator",
            "generation_scope": "original_structure_first",
            "review_required": "yes",
            "review_reason": (
                "OMe modification is residue/site dependent; arbitrary peptide "
                "replacement could destroy the intended modification chemistry."
            ),
            "inventory_role": "modified_peptide_scaffold_coverage",
            "notes": (
                "Include all observed OMe structures in pretraining. Expand only "
                "after a site-aware generator is validated."
            ),
        }

    # ------------------------------------------------------------------
    # 6. HA-DSSC / SQWS-DSSC
    #
    # Their DSSC sequence is established in Stage 1, but a generic sequence
    # replacement reaction/template has not yet been validated.
    # ------------------------------------------------------------------
    if len(nonempty_names) == 1:
        m = re.fullmatch(r"(HA|SQWS)-DSSC", nonempty_names[0], flags=re.IGNORECASE)
        if m:
            prefix = m.group(1).upper()
            return {
                "scaffold_family": f"{prefix}_peptide",
                "scaffold_variant": prefix,
                "residue_or_sequence_site": "DSSC_peptide_sequence",
                "replaceability": "conditional",
                "augmentation_eligible": "conditional",
                "generation_strategy": "keep_original_then_validate_peptide_template",
                "generation_scope": "original_structure_first",
                "review_required": "yes",
                "review_reason": (
                    "DSSC is identified, but general sequence substitution on this "
                    "scaffold has not yet been chemically/template validated."
                ),
                "inventory_role": "special_peptide_scaffold_coverage",
                "notes": (
                    "Retain observed scaffold in pretraining. Prefer adding this "
                    "family to Stage-2 augmentation only after template validation."
                ),
            }

    # ------------------------------------------------------------------
    # 7. Fail closed for anything unexpected
    # ------------------------------------------------------------------
    return {
        "scaffold_family": "UNCLASSIFIED",
        "scaffold_variant": "unknown",
        "residue_or_sequence_site": "unknown",
        "replaceability": "unknown",
        "augmentation_eligible": "no",
        "generation_strategy": "keep_original_identity_only",
        "generation_scope": "original_structure",
        "review_required": "yes",
        "review_reason": (
            "No frozen Stage-1.5 scaffold rule matched this identity."
        ),
        "inventory_role": "unclassified_scaffold_coverage",
        "notes": (
            f"Names={name_blob}; parse_statuses={parse_statuses}; "
            f"peptide_categories={peptide_categories}; modifications={modifications}"
        ),
    }


def aggregate_unique_identities(row_audit: pd.DataFrame) -> pd.DataFrame:
    required = {
        "canonical_fifth",
        "Fifth",
        "Fifth_class_canonical",
        "sequence",
        "sequence_length",
        "parse_status",
        "peptide_category",
        "modification",
        "Fifth_SMILE",
        "ID",
    }
    missing = required.difference(row_audit.columns)
    if missing:
        raise ValueError(
            "Row-level audit is missing required columns: "
            + ", ".join(sorted(missing))
        )

    identities = []

    for canonical, group in row_audit.groupby("canonical_fifth", dropna=False, sort=True):
        canonical = clean(canonical)
        fifth_names = unique_nonempty(group["Fifth"])
        classes = unique_nonempty(group["Fifth_class_canonical"])
        sequences = unique_nonempty(group["sequence"])
        statuses = unique_nonempty(group["parse_status"])
        peptide_categories = unique_nonempty(group["peptide_category"])
        modifications = unique_nonempty(group["modification"])

        if len(sequences) > 1:
            sequence = ""
            sequence_conflict = True
        else:
            sequence = sequences[0] if sequences else ""
            sequence_conflict = False

        classification = classify_identity(
            fifth_names=fifth_names,
            canonical_fifth=canonical,
            fifth_classes=classes,
            sequences=sequences,
            parse_statuses=statuses,
            peptide_categories=peptide_categories,
            modifications=modifications,
        )

        identities.append(
            {
                "canonical_fifth": canonical,
                "Fifth_names": "|".join(fifth_names),
                "Fifth_class_values": "|".join(classes),
                "rows": int(len(group)),
                "sequence": sequence,
                "sequence_length": len(sequence) if sequence else pd.NA,
                "parse_status_values": "|".join(statuses),
                "peptide_categories": "|".join(peptide_categories),
                "modifications": "|".join(modifications),
                "sequence_conflict": bool(sequence_conflict),
                "example_ID": clean(group["ID"].iloc[0]),
                "example_smiles": clean(group["Fifth_SMILE"].iloc[0]),
                **classification,
            }
        )

    return pd.DataFrame(identities)


def make_family_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []

    keys = ["scaffold_family", "scaffold_variant"]
    for (family, variant), group in inventory.groupby(keys, sort=True, dropna=False):
        names: set[str] = set()
        sequences: set[str] = set()
        modifications: set[str] = set()

        for value in group["Fifth_names"]:
            for part in clean(value).split("|"):
                if part:
                    names.add(part)
        for value in group["sequence"]:
            if clean(value):
                sequences.add(clean(value))
        for value in group["modifications"]:
            for part in clean(value).split("|"):
                if part:
                    modifications.add(part)

        replaceabilities = sorted(set(group["replaceability"].astype(str)))
        eligibilities = sorted(set(group["augmentation_eligible"].astype(str)))
        strategies = sorted(set(group["generation_strategy"].astype(str)))
        scopes = sorted(set(group["generation_scope"].astype(str)))
        review = bool((group["review_required"] == "yes").any())

        rows.append(
            {
                "scaffold_family": family,
                "scaffold_variant": variant,
                "unique_fifth_identities": int(len(group)),
                "training_rows": int(group["rows"].sum()),
                "observed_fifth_names": "|".join(sorted(names)),
                "observed_sequences": "|".join(sorted(sequences)),
                "observed_modifications": "|".join(sorted(modifications)),
                "replaceability": "|".join(replaceabilities),
                "augmentation_eligible": "|".join(eligibilities),
                "generation_strategy": "|".join(strategies),
                "generation_scope": "|".join(scopes),
                "review_required": bool_text(review),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["scaffold_family", "scaffold_variant"]
    ).reset_index(drop=True)


def make_generation_plan(family_summary: pd.DataFrame) -> pd.DataFrame:
    """
    High-level frozen plan, not a molecule generator.

    Stage 2 should consume these policies rather than rediscovering scaffold
    rules from names.
    """
    rows = []

    for row in family_summary.itertuples(index=False):
        family = row.scaffold_family
        variant = row.scaffold_variant

        if family in {"UC12", "UC18", "C12_COOH", "C18_COOH"}:
            plan = {
                "include_observed_structures": "yes",
                "automatic_augmentation": "yes",
                "target_residue_universe": "20 canonical AA",
                "target_sequence_lengths": "1",
                "sampling_policy": "full_cartesian_AA_coverage",
                "dedup_policy": "canonical_smiles",
                "priority": "P0",
            }
        elif family == "DOPE_peptide" and variant == "unmodified":
            plan = {
                "include_observed_structures": "yes",
                "automatic_augmentation": "yes",
                "target_residue_universe": "20 canonical AA",
                "target_sequence_lengths": "2-9",
                "sampling_policy": (
                    "controlled_sampling_balanced_residue_position_length;"
                    "mix_training_near_and_broad_coverage"
                ),
                "dedup_policy": "canonical_smiles",
                "priority": "P0",
            }
        elif family == "DOPE_peptide":
            plan = {
                "include_observed_structures": "yes",
                "automatic_augmentation": "no",
                "target_residue_universe": "",
                "target_sequence_lengths": "",
                "sampling_policy": "site_aware_generator_required",
                "dedup_policy": "canonical_smiles",
                "priority": "P1",
            }
        elif family in {"HA_peptide", "SQWS_peptide"}:
            plan = {
                "include_observed_structures": "yes",
                "automatic_augmentation": "no",
                "target_residue_universe": "",
                "target_sequence_lengths": "",
                "sampling_policy": "validate_peptide_template_before_expansion",
                "dedup_policy": "canonical_smiles",
                "priority": "P1",
            }
        elif family == "S_nonpeptide_series":
            plan = {
                "include_observed_structures": "yes",
                "automatic_augmentation": "no",
                "target_residue_universe": "",
                "target_sequence_lengths": "",
                "sampling_policy": "observed_structures_only",
                "dedup_policy": "canonical_smiles",
                "priority": "P1",
            }
        elif family == "ABSENT":
            plan = {
                "include_observed_structures": "no",
                "automatic_augmentation": "no",
                "target_residue_universe": "",
                "target_sequence_lengths": "",
                "sampling_policy": "exclude",
                "dedup_policy": "",
                "priority": "none",
            }
        else:
            plan = {
                "include_observed_structures": "yes",
                "automatic_augmentation": "no",
                "target_residue_universe": "",
                "target_sequence_lengths": "",
                "sampling_policy": "manual_review",
                "dedup_policy": "canonical_smiles",
                "priority": "BLOCKED",
            }

        rows.append(
            {
                "scaffold_family": family,
                "scaffold_variant": variant,
                "unique_fifth_identities": row.unique_fifth_identities,
                "training_rows": row.training_rows,
                **plan,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["priority", "scaffold_family", "scaffold_variant"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen Stage-1.5 Fifth scaffold inventory."
    )
    parser.add_argument(
        "--row-audit",
        type=Path,
        required=True,
        help="Final row_level_fifth_audit.csv from Stage 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expect-identities",
        type=int,
        default=64,
        help="Expected number of canonical Fifth identities; <=0 disables check.",
    )
    parser.add_argument(
        "--expect-nonempty-identities",
        type=int,
        default=63,
        help="Expected non-[Fr] identities; <=0 disables check.",
    )
    args = parser.parse_args()

    audit_path = args.row_audit.resolve()
    outdir = args.output_dir.resolve()

    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    outdir.mkdir(parents=True, exist_ok=True)

    row_audit = pd.read_csv(audit_path, dtype={"ID": str})
    inventory = aggregate_unique_identities(row_audit)

    if args.expect_identities > 0 and len(inventory) != args.expect_identities:
        raise ValueError(
            f"Expected {args.expect_identities} canonical identities, "
            f"found {len(inventory)}."
        )

    nonempty = inventory.loc[inventory["scaffold_family"] != "ABSENT"]
    if (
        args.expect_nonempty_identities > 0
        and len(nonempty) != args.expect_nonempty_identities
    ):
        raise ValueError(
            f"Expected {args.expect_nonempty_identities} nonempty identities, "
            f"found {len(nonempty)}."
        )

    if inventory["sequence_conflict"].any():
        conflicts = inventory.loc[inventory["sequence_conflict"]]
        conflicts.to_csv(outdir / "scaffold_sequence_conflicts.csv", index=False)
        raise ValueError(
            "Sequence conflicts detected. See scaffold_sequence_conflicts.csv."
        )

    inventory.to_csv(outdir / "scaffold_inventory.csv", index=False)

    family_summary = make_family_summary(inventory)
    family_summary.to_csv(
        outdir / "scaffold_family_summary.csv",
        index=False,
    )

    generation_plan = make_generation_plan(family_summary)
    generation_plan.to_csv(
        outdir / "scaffold_generation_plan.csv",
        index=False,
    )

    review = inventory.loc[
        (inventory["review_required"] == "yes")
        | (inventory["scaffold_family"] == "UNCLASSIFIED")
    ].copy()
    review.to_csv(outdir / "scaffold_review.csv", index=False)

    unclassified = inventory.loc[
        inventory["scaffold_family"] == "UNCLASSIFIED"
    ]

    auto_aug = inventory.loc[
        inventory["augmentation_eligible"] == "yes"
    ]
    conditional_aug = inventory.loc[
        inventory["augmentation_eligible"] == "conditional"
    ]
    keep_only = inventory.loc[
        inventory["augmentation_eligible"] == "no"
    ]

    manifest = {
        "stage": "1.5_scaffold_inventory",
        "input_row_audit": str(audit_path),
        "input_sha256": sha256(audit_path),
        "canonical_fifth_identities": int(len(inventory)),
        "nonempty_fifth_identities": int(len(nonempty)),
        "scaffold_families": sorted(
            set(nonempty["scaffold_family"].astype(str))
        ),
        "scaffold_family_variants": int(len(family_summary)),
        "automatic_augmentation_identities": int(len(auto_aug)),
        "conditional_augmentation_identities": int(len(conditional_aug)),
        "observed_only_or_excluded_identities": int(len(keep_only)),
        "review_identities": int(len(review)),
        "unclassified_identities": int(len(unclassified)),
        "policies": {
            "scaffold_coverage": (
                "Retain every observed nonempty training-derived Fifth identity "
                "in the pretraining corpus unless explicitly excluded for a "
                "later documented reason."
            ),
            "single_residue_augmentation": (
                "For validated UC12, UC18, C12_COOH and C18_COOH scaffold "
                "families, target all 20 canonical amino acids."
            ),
            "peptide_augmentation": (
                "For unmodified DOPE-peptide scaffold, use controlled sampling "
                "over canonical amino-acid sequences of length 2-9; do not "
                "enumerate 20^L."
            ),
            "modified_scaffolds": (
                "Modified/site-dependent and special peptide scaffolds count "
                "toward scaffold coverage but require a validated generator "
                "before sequence augmentation."
            ),
            "unknown_rule": (
                "Fail closed: retain original identity, block augmentation, "
                "and emit it to scaffold_review.csv."
            ),
        },
    }

    with (outdir / "scaffold_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("=" * 78)
    print("STAGE 1.5 — FIFTH SCAFFOLD INVENTORY")
    print("=" * 78)
    print(f"Canonical Fifth identities:       {len(inventory)}")
    print(f"Nonempty Fifth identities:        {len(nonempty)}")
    print(f"Family/variant rows:              {len(family_summary)}")
    print(f"Automatic augmentation identities:{len(auto_aug):>8}")
    print(f"Conditional identities:           {len(conditional_aug):>8}")
    print(f"Review identities:                {len(review):>8}")
    print(f"Unclassified identities:          {len(unclassified):>8}")
    print()
    print("Scaffold family summary:")
    show_cols = [
        "scaffold_family",
        "scaffold_variant",
        "unique_fifth_identities",
        "training_rows",
        "augmentation_eligible",
        "review_required",
    ]
    print(family_summary[show_cols].to_string(index=False))
    print()
    print(f"Results written to:\n  {outdir}")
    print()
    print("Inspect next:")
    print(f"  {outdir / 'scaffold_family_summary.csv'}")
    print(f"  {outdir / 'scaffold_generation_plan.csv'}")
    print(f"  {outdir / 'scaffold_review.csv'}")

    # Exit nonzero only for genuinely unknown identities.
    if len(unclassified) > 0:
        raise SystemExit(
            "Unclassified scaffold identities remain; automatic Stage 2 "
            "generation should not start yet."
        )


if __name__ == "__main__":
    main()
