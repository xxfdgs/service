#!/usr/bin/env python3
"""
Stage 2A — generate full 20-canonical-AA coverage for the four validated
single-residue Fifth scaffold families:

    UC12
    UC18
    C12_COOH
    C18_COOH

The script is deliberately fail-closed.

Core policy
-----------
1. Every family × canonical amino acid is generated (4 × 20 = 80 targets).
2. Existing training structures are NEVER overwritten by a generated
   stereoisomer. Their observed structure is retained exactly.
3. For existing targets:
      - Standard-InChI connectivity reconstruction MUST match 100%.
      - stereochemistry is audited separately.
4. Missing targets are generated from RDKit's standard L-amino-acid templates.
5. UC12/UC18:
      - N-terminus is converted to alkyl-urea:
            CnH(2n+1)-NH-C(=O)-N(AA)
      - all free carboxylic acids are ethyl-esterified.
        This is required to reconstruct the observed Asp/Glu UC structures.
6. C12_COOH/C18_COOH:
      - same alkyl-urea N-terminal scaffold
      - carboxyl groups remain free acids.
7. Output records explicitly distinguish:
      observed_training
      generated_L_canonical

Why stereochemistry is audited separately
------------------------------------------
The current training data contains residue-dependent alpha-carbon
stereochemistry under names that do not explicitly specify D/L. Therefore a
standard-L generator can be constitutionally correct while differing from an
observed stereoisomer or tautomer drawing. Such cases are reported, not silently "fixed".

Recommended inputs
------------------
--row-audit:
    final Stage-1 row_level_fifth_audit.csv

--generation-plan:
    frozen Stage-1.5 scaffold_generation_plan.csv

Outputs
-------
stage2a_single_aa_library.csv
stage2a_reconstruction_audit.csv
stage2a_stereochemistry_audit.csv
stage2a_family_coverage.csv
stage2a_manifest.json

The main pretraining structure library is stage2a_single_aa_library.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, inchi
from rdkit.Chem.rdchem import BondType


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
AA1_TO_AA3 = {aa1: aa3 for aa3, aa1 in AA3_TO_AA1.items()}
AA1_ORDER = list("ARNDCQEGHILKMFPSTWYV")

TARGET_FAMILIES = {
    "UC12": {
        "tail_carbons": 12,
        "esterify_carboxyls": True,
        "name_style": "uc",
    },
    "UC18": {
        "tail_carbons": 18,
        "esterify_carboxyls": True,
        "name_style": "uc",
    },
    "C12_COOH": {
        "tail_carbons": 12,
        "esterify_carboxyls": False,
        "name_style": "cooh",
    },
    "C18_COOH": {
        "tail_carbons": 18,
        "esterify_carboxyls": False,
        "name_style": "cooh",
    },
}


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_smiles(mol: Chem.Mol, *, isomeric: bool) -> str:
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=isomeric,
    )


def connectivity_inchikey_block(mol: Chem.Mol) -> str:
    """
    First InChIKey block from Standard InChI.

    This is used as the hard reconstruction gate because it is insensitive to
    stereochemistry and normalizes common prototropic tautomers (important for
    the two equivalent histidine imidazole tautomer drawings observed here).
    """
    text = inchi.MolToInchi(mol)
    if not text:
        raise ValueError(
            f"Could not create Standard InChI for {canonical_smiles(mol, isomeric=True)}"
        )
    key = inchi.InchiToInchiKey(text)
    if not key or "-" not in key:
        raise ValueError(f"Could not create InChIKey from {text}")
    return key.split("-", 1)[0]


def mol_from_smiles_or_fail(smiles: str, label: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse {label}: {smiles}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def expected_name(family: str, aa3: str) -> str:
    if family == "UC12":
        return f"{aa3}-UC12"
    if family == "UC18":
        return f"{aa3}-UC18"
    if family == "C12_COOH":
        return f"{aa3}12-COOH"
    if family == "C18_COOH":
        return f"{aa3}18-COOH"
    raise KeyError(family)


def parse_target_name(name: str) -> tuple[str, str] | None:
    """
    Return (family, aa1) only for canonical-AA members of Stage 2A families.
    Phg is intentionally excluded.
    """
    aa3_pattern = "|".join(AA3_TO_AA1)

    m = re.fullmatch(
        rf"(?P<aa>{aa3_pattern})-UC(?P<n>12|18)",
        name,
        flags=re.IGNORECASE,
    )
    if m:
        aa3 = next(
            key for key in AA3_TO_AA1
            if key.lower() == m.group("aa").lower()
        )
        return f"UC{m.group('n')}", AA3_TO_AA1[aa3]

    m = re.fullmatch(
        rf"(?P<aa>{aa3_pattern})(?P<n>12|18)-COOH",
        name,
        flags=re.IGNORECASE,
    )
    if m:
        aa3 = next(
            key for key in AA3_TO_AA1
            if key.lower() == m.group("aa").lower()
        )
        return f"C{m.group('n')}_COOH", AA3_TO_AA1[aa3]

    return None


def find_alpha_atoms(mol: Chem.Mol) -> tuple[int, int, int]:
    """
    Return:
        alpha_carbon_idx, alpha_amino_n_idx, alpha_carboxyl_c_idx

    Works for the 20 standard amino acids generated by RDKit MolFromFASTA,
    including glycine and proline.
    """
    candidates: list[tuple[int, int, int]] = []

    for carboxyl_c in mol.GetAtoms():
        if carboxyl_c.GetAtomicNum() != 6:
            continue

        double_o = []
        single_o = []

        for nbr in carboxyl_c.GetNeighbors():
            if nbr.GetAtomicNum() != 8:
                continue
            bond = mol.GetBondBetweenAtoms(
                carboxyl_c.GetIdx(), nbr.GetIdx()
            )
            if bond.GetBondType() == BondType.DOUBLE:
                double_o.append(nbr)
            elif bond.GetBondType() == BondType.SINGLE:
                single_o.append(nbr)

        if not double_o or not single_o:
            continue

        for alpha_c in carboxyl_c.GetNeighbors():
            if alpha_c.GetAtomicNum() != 6:
                continue

            alpha_ns = [
                nbr
                for nbr in alpha_c.GetNeighbors()
                if nbr.GetAtomicNum() == 7
            ]

            for alpha_n in alpha_ns:
                candidates.append(
                    (
                        alpha_c.GetIdx(),
                        alpha_n.GetIdx(),
                        carboxyl_c.GetIdx(),
                    )
                )

    # Asp/Glu contain another COOH but only the alpha-carboxyl carbon is
    # adjacent to a carbon that is also adjacent to the alpha-amino N.
    unique = sorted(set(candidates))

    if len(unique) != 1:
        raise ValueError(
            "Could not uniquely identify alpha C/N/carboxyl atoms. "
            f"Candidates={unique}; mol={canonical_smiles(mol, isomeric=True)}"
        )

    return unique[0]


def free_carboxyl_oxygen_indices(mol: Chem.Mol) -> list[int]:
    """
    Find neutral, degree-1 single-bond oxygens of carboxylic acids.

    These are esterified for UC12/UC18. This intentionally includes both
    alpha-COOH and Asp/Glu side-chain COOH groups.
    """
    targets: list[int] = []

    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6:
            continue

        has_double_o = False
        single_oxygens = []

        for nbr in carbon.GetNeighbors():
            if nbr.GetAtomicNum() != 8:
                continue
            bond = mol.GetBondBetweenAtoms(carbon.GetIdx(), nbr.GetIdx())
            if bond.GetBondType() == BondType.DOUBLE:
                has_double_o = True
            elif bond.GetBondType() == BondType.SINGLE:
                single_oxygens.append(nbr)

        if not has_double_o:
            continue

        for oxygen in single_oxygens:
            if (
                oxygen.GetDegree() == 1
                and oxygen.GetFormalCharge() == 0
            ):
                targets.append(oxygen.GetIdx())

    return sorted(set(targets))


def attach_alkyl_urea(
    aa_mol: Chem.Mol,
    *,
    tail_carbons: int,
) -> Chem.Mol:
    """
    Attach:
        AA-N-C(=O)-N-(CH2)n-CH3
    where tail_carbons is the number of carbons in the alkyl chain.
    """
    _, alpha_n_idx, _ = find_alpha_atoms(aa_mol)

    rw = Chem.RWMol(aa_mol)

    urea_c = rw.AddAtom(Chem.Atom(6))
    urea_o = rw.AddAtom(Chem.Atom(8))
    tail_n = rw.AddAtom(Chem.Atom(7))

    rw.AddBond(alpha_n_idx, urea_c, BondType.SINGLE)
    rw.AddBond(urea_c, urea_o, BondType.DOUBLE)
    rw.AddBond(urea_c, tail_n, BondType.SINGLE)

    previous = tail_n
    for _ in range(tail_carbons):
        carbon_idx = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(previous, carbon_idx, BondType.SINGLE)
        previous = carbon_idx

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def ethyl_esterify_all_free_carboxyls(mol: Chem.Mol) -> Chem.Mol:
    targets = free_carboxyl_oxygen_indices(mol)
    if not targets:
        raise ValueError(
            "Expected at least one free carboxyl group before UC esterification."
        )

    rw = Chem.RWMol(mol)

    for oxygen_idx in targets:
        ethyl_c1 = rw.AddAtom(Chem.Atom(6))
        ethyl_c2 = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(oxygen_idx, ethyl_c1, BondType.SINGLE)
        rw.AddBond(ethyl_c1, ethyl_c2, BondType.SINGLE)

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    Chem.AssignStereochemistry(out, cleanIt=True, force=True)
    return out


def generate_standard_l_target(family: str, aa1: str) -> Chem.Mol:
    cfg = TARGET_FAMILIES[family]

    aa = Chem.MolFromFASTA(aa1)
    if aa is None:
        raise ValueError(f"RDKit MolFromFASTA failed for amino acid {aa1}")

    Chem.SanitizeMol(aa)
    Chem.AssignStereochemistry(aa, cleanIt=True, force=True)

    mol = attach_alkyl_urea(
        aa,
        tail_carbons=int(cfg["tail_carbons"]),
    )

    if bool(cfg["esterify_carboxyls"]):
        mol = ethyl_esterify_all_free_carboxyls(mol)

    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def alpha_cip(mol: Chem.Mol) -> str:
    """
    Return alpha-carbon CIP label if present. Glycine returns "".
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    try:
        alpha_idx, _, _ = find_alpha_atoms(mol)
    except ValueError:
        # In final scaffold molecules the alpha-carboxyl motif remains
        # identifiable; this branch is defensive.
        return ""

    atom = mol.GetAtomWithIdx(alpha_idx)
    return atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""


def count_free_carboxyls(mol: Chem.Mol) -> int:
    return len(free_carboxyl_oxygen_indices(mol))


def build_observed_map(row_audit: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    required = {"Fifth", "Fifth_SMILE"}
    missing = required.difference(row_audit.columns)
    if missing:
        raise ValueError(
            "Row audit missing required columns: " + ", ".join(sorted(missing))
        )

    observed: dict[tuple[str, str], dict[str, str]] = {}

    for row in row_audit.itertuples(index=False):
        name = clean(getattr(row, "Fifth"))
        smiles = clean(getattr(row, "Fifth_SMILE"))

        parsed = parse_target_name(name)
        if parsed is None:
            continue

        family, aa1 = parsed
        if not smiles:
            raise ValueError(f"Observed Stage-2A target lacks SMILES: {name}")

        mol = mol_from_smiles_or_fail(smiles, name)
        stereo = canonical_smiles(mol, isomeric=True)
        connectivity = canonical_smiles(mol, isomeric=False)

        key = (family, aa1)

        record = {
            "observed_name": name,
            "observed_smiles": smiles,
            "observed_canonical_isomeric": stereo,
            "observed_canonical_connectivity": connectivity,
            "observed_connectivity_inchikey_block": connectivity_inchikey_block(mol),
        }

        if key in observed:
            old = observed[key]
            if old["observed_canonical_isomeric"] != stereo:
                raise ValueError(
                    "Same family/residue has conflicting observed stereochemical "
                    f"structures: key={key}, "
                    f"{old['observed_name']} vs {name}"
                )
        else:
            observed[key] = record

    return observed


def validate_generation_plan(plan_path: Path) -> None:
    plan = pd.read_csv(plan_path)

    required = {
        "scaffold_family",
        "automatic_augmentation",
        "target_residue_universe",
        "target_sequence_lengths",
    }
    missing = required.difference(plan.columns)
    if missing:
        raise ValueError(
            "Generation plan missing columns: " + ", ".join(sorted(missing))
        )

    by_family = plan.set_index("scaffold_family")

    for family in TARGET_FAMILIES:
        if family not in by_family.index:
            raise ValueError(
                f"Frozen generation plan lacks required Stage-2A family {family}"
            )

        row = by_family.loc[family]
        if isinstance(row, pd.DataFrame):
            if len(row) != 1:
                raise ValueError(
                    f"Expected one plan row for {family}, found {len(row)}"
                )
            row = row.iloc[0]

        if clean(row["automatic_augmentation"]).lower() != "yes":
            raise ValueError(
                f"Frozen plan does not authorize automatic augmentation for {family}"
            )

        if "20 canonical AA" not in clean(row["target_residue_universe"]):
            raise ValueError(
                f"Frozen plan residue universe mismatch for {family}: "
                f"{row['target_residue_universe']}"
            )

        length_value = clean(row["target_sequence_lengths"])
        if length_value not in {"1", "1.0"}:
            raise ValueError(
                f"Frozen plan sequence length mismatch for {family}: "
                f"{row['target_sequence_lengths']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and audit Stage-2A single-residue scaffold library."
    )
    parser.add_argument(
        "--row-audit",
        type=Path,
        required=True,
        help="Final Stage-1 row_level_fifth_audit.csv",
    )
    parser.add_argument(
        "--generation-plan",
        type=Path,
        required=True,
        help="Frozen Stage-1.5 scaffold_generation_plan.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expect-targets",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--require-connectivity-reconstruction-rate",
        type=float,
        default=1.0,
        help="Fail if observed connectivity reconstruction rate is below this.",
    )
    args = parser.parse_args()

    row_audit_path = args.row_audit.resolve()
    plan_path = args.generation_plan.resolve()
    outdir = args.output_dir.resolve()

    if not row_audit_path.is_file():
        raise FileNotFoundError(row_audit_path)
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)

    outdir.mkdir(parents=True, exist_ok=True)

    validate_generation_plan(plan_path)

    row_audit = pd.read_csv(row_audit_path, dtype={"ID": str})
    observed = build_observed_map(row_audit)

    library_rows = []
    reconstruction_rows = []
    stereo_rows = []

    generated_connectivity_to_target: dict[str, list[str]] = {}
    final_isomeric_to_target: dict[str, list[str]] = {}

    for family, cfg in TARGET_FAMILIES.items():
        for aa1 in AA1_ORDER:
            aa3 = AA1_TO_AA3[aa1]
            target_name = expected_name(family, aa3)

            generated_mol = generate_standard_l_target(family, aa1)
            generated_iso = canonical_smiles(
                generated_mol,
                isomeric=True,
            )
            generated_conn = canonical_smiles(
                generated_mol,
                isomeric=False,
            )
            generated_cip = alpha_cip(generated_mol)
            generated_inchi_block = connectivity_inchikey_block(generated_mol)

            key = (family, aa1)
            obs = observed.get(key)

            if obs is not None:
                observed_mol = mol_from_smiles_or_fail(
                    obs["observed_smiles"],
                    obs["observed_name"],
                )

                observed_iso = obs["observed_canonical_isomeric"]
                observed_conn = obs["observed_canonical_connectivity"]
                observed_cip = alpha_cip(observed_mol)
                observed_inchi_block = obs["observed_connectivity_inchikey_block"]

                exact_nonisomeric_smiles_match = generated_conn == observed_conn
                connectivity_match = generated_inchi_block == observed_inchi_block
                alpha_cip_match = observed_cip == generated_cip
                stereo_match = generated_iso == observed_iso

                # Existing training chemistry is authoritative for the final
                # library. Do not silently replace it with the standard-L form.
                final_smiles = obs["observed_smiles"]
                final_canonical = observed_iso
                source = "observed_training"
                observed_name = obs["observed_name"]

                reconstruction_rows.append(
                    {
                        "scaffold_family": family,
                        "aa1": aa1,
                        "aa3": aa3,
                        "target_name": target_name,
                        "observed_name": observed_name,
                        "observed_canonical_isomeric": observed_iso,
                        "generated_L_canonical_isomeric": generated_iso,
                        "observed_canonical_connectivity": observed_conn,
                        "generated_L_canonical_connectivity": generated_conn,
                        "exact_nonisomeric_smiles_match": bool(exact_nonisomeric_smiles_match),
                        "observed_connectivity_inchikey_block": observed_inchi_block,
                        "generated_connectivity_inchikey_block": generated_inchi_block,
                        "connectivity_match": bool(connectivity_match),
                        "alpha_cip_match": bool(alpha_cip_match),
                        "stereochemistry_match": bool(stereo_match),
                        "observed_alpha_cip": observed_cip,
                        "generated_L_alpha_cip": generated_cip,
                    }
                )

                stereo_rows.append(
                    {
                        "scaffold_family": family,
                        "aa1": aa1,
                        "aa3": aa3,
                        "observed_name": observed_name,
                        "observed_alpha_cip": observed_cip,
                        "standard_L_generated_alpha_cip": generated_cip,
                        "alpha_cip_match": bool(alpha_cip_match),
                        "strict_isomeric_match": bool(stereo_match),
                        "policy": (
                            "retain_observed_training_structure"
                            if not stereo_match
                            else "observed_matches_standard_L_generator"
                        ),
                    }
                )
            else:
                final_smiles = generated_iso
                final_canonical = generated_iso
                source = "generated_L_canonical"
                observed_name = ""
                connectivity_match = pd.NA
                exact_nonisomeric_smiles_match = pd.NA
                alpha_cip_match = pd.NA
                stereo_match = pd.NA
                observed_cip = ""

            final_mol = mol_from_smiles_or_fail(final_smiles, target_name)
            final_conn = canonical_smiles(final_mol, isomeric=False)
            final_iso = canonical_smiles(final_mol, isomeric=True)

            generated_connectivity_to_target.setdefault(
                generated_conn, []
            ).append(f"{family}:{aa1}")
            final_isomeric_to_target.setdefault(
                final_iso, []
            ).append(f"{family}:{aa1}")

            library_rows.append(
                {
                    "stage2a_id": f"S2A_{family}_{aa1}",
                    "Fifth": target_name,
                    "Fifth_SMILE": final_smiles,
                    "canonical_smiles": final_iso,
                    "canonical_connectivity": final_conn,
                    "scaffold_family": family,
                    "aa1": aa1,
                    "aa3": aa3,
                    "amino_acid_source": (
                        "canonical_20_standard_amino_acids"
                    ),
                    "structure_source": source,
                    "observed_in_training": bool(obs is not None),
                    "observed_name": observed_name,
                    "generated_L_reference_smiles": generated_iso,
                    "generated_L_alpha_cip": generated_cip,
                    "observed_alpha_cip": observed_cip,
                    "connectivity_reconstruction_match": connectivity_match,
                    "exact_nonisomeric_smiles_match": exact_nonisomeric_smiles_match,
                    "alpha_cip_match": alpha_cip_match,
                    "strict_stereo_reconstruction_match": stereo_match,
                    "formal_charge": int(
                        Chem.GetFormalCharge(final_mol)
                    ),
                    "free_carboxyl_count": int(
                        count_free_carboxyls(final_mol)
                    ),
                    "mol_wt": float(Descriptors.MolWt(final_mol)),
                    "heavy_atom_count": int(
                        rdMolDescriptors.CalcNumHeavyAtoms(final_mol)
                    ),
                    "rdkit_valid": True,
                }
            )

    library = pd.DataFrame(library_rows)
    reconstruction = pd.DataFrame(reconstruction_rows)
    stereo = pd.DataFrame(stereo_rows)

    # ------------------------------------------------------------------
    # Hard validation
    # ------------------------------------------------------------------

    if len(library) != args.expect_targets:
        raise ValueError(
            f"Expected {args.expect_targets} Stage-2A targets, found {len(library)}"
        )

    if library["stage2a_id"].duplicated().any():
        dup = library.loc[
            library["stage2a_id"].duplicated(keep=False)
        ]
        dup.to_csv(outdir / "stage2a_duplicate_ids.csv", index=False)
        raise ValueError("Duplicate Stage-2A IDs detected.")

    # Generated connectivity should identify unique family×AA targets.
    duplicate_generated_connectivity = {
        smiles: targets
        for smiles, targets in generated_connectivity_to_target.items()
        if len(targets) > 1
    }
    if duplicate_generated_connectivity:
        with (
            outdir / "stage2a_duplicate_generated_connectivity.json"
        ).open("w", encoding="utf-8") as f:
            json.dump(
                duplicate_generated_connectivity,
                f,
                indent=2,
                ensure_ascii=False,
            )
        raise ValueError(
            "Different family×AA targets collapsed to identical generated "
            "connectivity. Inspect stage2a_duplicate_generated_connectivity.json."
        )

    # Existing target connectivity must be reconstructed perfectly.
    if reconstruction.empty:
        raise ValueError(
            "No observed Stage-2A targets were found in the row audit."
        )

    connectivity_rate = float(
        reconstruction["connectivity_match"].mean()
    )
    stereo_rate = float(
        reconstruction["stereochemistry_match"].mean()
    )

    reconstruction.to_csv(
        outdir / "stage2a_reconstruction_audit.csv",
        index=False,
    )
    stereo.to_csv(
        outdir / "stage2a_stereochemistry_audit.csv",
        index=False,
    )

    failed_connectivity = reconstruction.loc[
        ~reconstruction["connectivity_match"]
    ].copy()

    if not failed_connectivity.empty:
        failed_connectivity.to_csv(
            outdir / "stage2a_failed_connectivity_reconstruction.csv",
            index=False,
        )

    # ------------------------------------------------------------------
    # Family coverage
    # ------------------------------------------------------------------

    family_rows = []

    for family, group in library.groupby("scaffold_family", sort=True):
        family_rows.append(
            {
                "scaffold_family": family,
                "targets": int(len(group)),
                "canonical_aa_covered": int(group["aa1"].nunique()),
                "observed_training_targets": int(
                    group["observed_in_training"].sum()
                ),
                "new_generated_targets": int(
                    (~group["observed_in_training"]).sum()
                ),
                "all_rdkit_valid": bool(group["rdkit_valid"].all()),
                "all_20_aa_present": bool(group["aa1"].nunique() == 20),
            }
        )

    family_coverage = pd.DataFrame(family_rows)

    library.to_csv(
        outdir / "stage2a_single_aa_library.csv",
        index=False,
    )
    family_coverage.to_csv(
        outdir / "stage2a_family_coverage.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    observed_count = int(library["observed_in_training"].sum())
    generated_count = int((~library["observed_in_training"]).sum())
    stereo_mismatch_count = int(
        (~reconstruction["stereochemistry_match"]).sum()
    )
    connectivity_mismatch_count = int(
        (~reconstruction["connectivity_match"]).sum()
    )
    exact_nonisomeric_mismatch_count = int(
        (~reconstruction["exact_nonisomeric_smiles_match"]).sum()
    )
    alpha_cip_mismatch_count = int(
        (~reconstruction["alpha_cip_match"]).sum()
    )

    manifest = {
        "stage": "2A_single_residue_scaffold_coverage",
        "row_audit": str(row_audit_path),
        "row_audit_sha256": sha256(row_audit_path),
        "generation_plan": str(plan_path),
        "generation_plan_sha256": sha256(plan_path),
        "target_families": list(TARGET_FAMILIES),
        "canonical_amino_acids": AA1_ORDER,
        "expected_targets": args.expect_targets,
        "actual_targets": int(len(library)),
        "observed_training_targets": observed_count,
        "new_generated_targets": generated_count,
        "connectivity_reconstruction": {
            "observed_targets_tested": int(len(reconstruction)),
            "matches": int(reconstruction["connectivity_match"].sum()),
            "mismatches": connectivity_mismatch_count,
            "rate": connectivity_rate,
            "required_rate": args.require_connectivity_reconstruction_rate,
        },
        "exact_nonisomeric_smiles_reconstruction": {
            "matches": int(reconstruction["exact_nonisomeric_smiles_match"].sum()),
            "mismatches": exact_nonisomeric_mismatch_count,
            "interpretation": (
                "Diagnostic only; common histidine prototropic tautomer drawings "
                "can differ as SMILES while Standard InChI connectivity agrees."
            ),
        },
        "alpha_cip_audit": {
            "matches_standard_L_generator": int(reconstruction["alpha_cip_match"].sum()),
            "mismatches": alpha_cip_mismatch_count,
        },
        "strict_isomeric_reconstruction": {
            "matches": int(reconstruction["stereochemistry_match"].sum()),
            "mismatches": stereo_mismatch_count,
            "rate": stereo_rate,
            "interpretation": (
                "Reported separately because the training dataset contains "
                "residue-dependent stereochemistry under names without explicit "
                "D/L specification and histidine can be drawn as equivalent "
                "prototropic tautomers. Existing structures are retained unchanged."
            ),
        },
        "generation_policy": {
            "missing_targets": (
                "Generate from RDKit standard L-amino-acid templates."
            ),
            "observed_targets": (
                "Retain the exact observed training structure; generated L form "
                "is stored only as a reference/audit field."
            ),
            "UC12_UC18": (
                "Attach C12/C18 alkyl urea to alpha amino N and ethyl-esterify "
                "all free carboxyl groups."
            ),
            "C12_COOH_C18_COOH": (
                "Attach C12/C18 alkyl urea to alpha amino N and retain free "
                "carboxylic acids."
            ),
            "hard_gate": (
                "Standard-InChI connectivity block reconstruction of all "
                "observed canonical-AA targets must meet the configured required rate."
            ),
        },
    }

    with (outdir / "stage2a_manifest.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # ------------------------------------------------------------------
    # Terminal report
    # ------------------------------------------------------------------

    print("=" * 80)
    print("STAGE 2A — SINGLE-RESIDUE SCAFFOLD GENERATION")
    print("=" * 80)
    print(f"Targets:                         {len(library)}")
    print(f"Observed training targets:       {observed_count}")
    print(f"New generated L-AA targets:      {generated_count}")
    print()
    print(
        "Connectivity reconstruction:    "
        f"{int(reconstruction['connectivity_match'].sum())}/"
        f"{len(reconstruction)} "
        f"({connectivity_rate:.3f})"
    )
    print(
        "Exact non-isomeric SMILES:       "
        f"{int(reconstruction['exact_nonisomeric_smiles_match'].sum())}/"
        f"{len(reconstruction)}"
    )
    print(
        "Alpha-C CIP vs standard L:       "
        f"{int(reconstruction['alpha_cip_match'].sum())}/"
        f"{len(reconstruction)}"
    )
    print(
        "Strict stereo reconstruction:    "
        f"{int(reconstruction['stereochemistry_match'].sum())}/"
        f"{len(reconstruction)} "
        f"({stereo_rate:.3f})"
    )
    print(
        "Stereo/tautomer mismatches retained as observed structures: "
        f"{stereo_mismatch_count}"
    )
    print()
    print("Family coverage:")
    print(family_coverage.to_string(index=False))
    print()
    print(f"Results written to:\n  {outdir}")
    print()
    print("Inspect next:")
    print(f"  {outdir / 'stage2a_reconstruction_audit.csv'}")
    print(f"  {outdir / 'stage2a_stereochemistry_audit.csv'}")
    print(f"  {outdir / 'stage2a_single_aa_library.csv'}")

    # Hard gate is evaluated after all diagnostics are written.
    if connectivity_rate < args.require_connectivity_reconstruction_rate:
        raise SystemExit(
            "Stage 2A BLOCKED: connectivity reconstruction rate "
            f"{connectivity_rate:.6f} is below required "
            f"{args.require_connectivity_reconstruction_rate:.6f}. "
            "Do not use the generated library for pretraining."
        )

    if not family_coverage["all_20_aa_present"].all():
        raise SystemExit(
            "Stage 2A BLOCKED: at least one family does not cover all 20 AA."
        )

    print()
    print("STAGE 2A PASSED the connectivity and coverage gates.")


if __name__ == "__main__":
    main()
