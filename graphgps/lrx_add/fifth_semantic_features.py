"""Deterministic, auditable chemical-series features for Fifth structures.

This module intentionally uses only RDKit molecular graphs.  In particular it
never reads the ``Fifth`` display-name column, targets, split membership, or
external validation tables.  Unrecognised residue chemistry is reported as an
explicit parse warning rather than guessed from a molecule name.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from graphgps.lrx_add.fifth_mechanistic_descriptors import _tail_components


ABSENT_SMILES = {"nan", "[Fr]"}
AA_ORDER = ("A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V")
SEMANTIC_NUMERIC_NAMES = (
    "has_DOPE_tail", "disulfide_bridge_count", "peptide_length",
    *(f"AA_count_{code}" for code in AA_ORDER),
    "positive_AA_count", "negative_AA_count", "peptide_net_charge_proxy",
    "UC_tail_carbon_count", "UC_terminal_carboxyl", "UC_terminal_ester",
    "UC_terminal_amide",
)
FAMILY_VALUES = ("UC_series", "DOPE_SS_peptide_series", "other")
UC_AA_VALUES = (*AA_ORDER, "PHG", "UNK", "none")


@dataclass(frozen=True)
class SemanticResult:
    family_type: str
    has_dope_tail: int
    disulfide_bridge_count: int
    peptide_length: int
    aa_counts: dict[str, int]
    positive_aa_count: int
    negative_aa_count: int
    peptide_net_charge_proxy: int
    uc_amino_acid_type: str
    uc_tail_carbon_count: int
    uc_terminal_carboxyl: int
    uc_terminal_ester: int
    uc_terminal_amide: int
    parse_status: str
    parse_warnings: tuple[str, ...]

    def numeric_vector(self) -> np.ndarray:
        return np.asarray([
            self.has_dope_tail, self.disulfide_bridge_count, self.peptide_length,
            *(self.aa_counts[code] for code in AA_ORDER),
            self.positive_aa_count, self.negative_aa_count,
            self.peptide_net_charge_proxy, self.uc_tail_carbon_count,
            self.uc_terminal_carboxyl, self.uc_terminal_ester,
            self.uc_terminal_amide,
        ], dtype=np.float32)

    def as_row(self) -> dict[str, object]:
        return {
            "family_type": self.family_type,
            "has_DOPE_tail": self.has_dope_tail,
            "disulfide_bridge_count": self.disulfide_bridge_count,
            "peptide_length": self.peptide_length,
            **{f"AA_count_{code}": self.aa_counts[code] for code in AA_ORDER},
            "positive_AA_count": self.positive_aa_count,
            "negative_AA_count": self.negative_aa_count,
            "peptide_net_charge_proxy": self.peptide_net_charge_proxy,
            "UC_amino_acid_type": self.uc_amino_acid_type,
            "UC_tail_carbon_count": self.uc_tail_carbon_count,
            "UC_terminal_carboxyl": self.uc_terminal_carboxyl,
            "UC_terminal_ester": self.uc_terminal_ester,
            "UC_terminal_amide": self.uc_terminal_amide,
            "parse_status": self.parse_status,
            "parse_warnings": ";".join(self.parse_warnings),
        }


def _is_carbonyl_carbon(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() == 6 and any(
        bond.GetBondTypeAsDouble() == 2.0
        and bond.GetOtherAtom(atom).GetAtomicNum() == 8
        for bond in atom.GetBonds())


def _alpha_carbons(molecule: Chem.Mol) -> list[int]:
    """Find N--C(alpha)--C(=O) residue centres by an explicit graph rule."""
    centers = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetIsAromatic() or atom.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        neighbours = list(atom.GetNeighbors())
        if any(neighbour.GetAtomicNum() == 7 for neighbour in neighbours) and any(
                _is_carbonyl_carbon(neighbour) for neighbour in neighbours):
            centers.append(atom.GetIdx())
    return centers


# Ordered from specific to broad.  Each SMARTS starts at C(alpha), which makes
# the mapping an auditable structural rule rather than a lookup by sample name.
_AA_SMARTS = {
    "R": "[C;X4]([N])([C](=O))[CH2][CH2][CH2]N[C](N)=N",
    "Y": "[C;X4]([N])([C](=O))[CH2][c]1[c][c][c]([O])[c][c]1",
    "F": "[C;X4]([N])([C](=O))[CH2][c]1[c][c][c][c][c]1",
    "H": "[C;X4]([N])([C](=O))[CH2][c]1[n,c][c,n][n,c][c,n]1",
    "M": "[C;X4]([N])([C](=O))[CH2][CH2]S[CH3]",
    "E": "[C;X4]([N])([C](=O))[CH2][CH2][C](=O)[O]",
    "D": "[C;X4]([N])([C](=O))[CH2][C](=O)[O]",
    "L": "[C;X4]([N])([C](=O))[CH2][CH]([CH3])[CH3]",
    "I": "[C;X4]([N])([C](=O))[CH]([CH3])[CH2][CH3]",
    "V": "[C;X4]([N])([C](=O))[CH]([CH3])[CH3]",
    "S": "[C;X4]([N])([C](=O))[CH2][O]",
    "C": "[C;X4]([N])([C](=O))[CH2]S",
    "PHG": "[C;X4]([N])([C](=O))[c]1[c][c][c][c][c]1",
    "A": "[C;X4]([N])([C](=O))[CH3]",
}
_AA_PATTERNS = tuple((code, Chem.MolFromSmarts(pattern)) for code, pattern in _AA_SMARTS.items())


def _classify_residue(molecule: Chem.Mol, alpha_index: int) -> str | None:
    """Return a supported amino-acid code, PHG, or ``None`` if unsupported."""
    alpha = molecule.GetAtomWithIdx(alpha_index)
    heavy_neighbours = [atom for atom in alpha.GetNeighbors() if atom.GetAtomicNum() > 1]
    if len(heavy_neighbours) == 2:
        # Glycine has only backbone N and carbonyl C attached to C(alpha).
        return "G"
    for code, pattern in _AA_PATTERNS:
        for match in molecule.GetSubstructMatches(pattern, uniquify=True):
            if match[0] == alpha_index:
                return code
    return None


def _disulfide_count(molecule: Chem.Mol) -> int:
    return sum(
        bond.GetBondTypeAsDouble() == 1.0
        and bond.GetBeginAtom().GetAtomicNum() == 16
        and bond.GetEndAtom().GetAtomicNum() == 16
        for bond in molecule.GetBonds())


def _is_phosphatidylethanolamine_tail(molecule: Chem.Mol) -> bool:
    """Identify a PE core with >=2 long acyl tails (DOPE structural proxy).

    The rule requires a phosphate with an O--C--C--N ethanolamine arm, at
    least two ester linkages, and at least two >=14-carbon hydrocarbon tail
    components.  It does not inspect a formulation name.
    """
    has_ethanolamine_arm = False
    for phosphorus in molecule.GetAtoms():
        if phosphorus.GetAtomicNum() != 15:
            continue
        for oxygen in phosphorus.GetNeighbors():
            if oxygen.GetAtomicNum() != 8:
                continue
            carbons = [atom for atom in oxygen.GetNeighbors()
                       if atom.GetIdx() != phosphorus.GetIdx() and atom.GetAtomicNum() == 6]
            for first in carbons:
                for second in first.GetNeighbors():
                    if second.GetIdx() == oxygen.GetIdx() or second.GetAtomicNum() != 6:
                        continue
                    if any(atom.GetAtomicNum() == 7 for atom in second.GetNeighbors()
                           if atom.GetIdx() != first.GetIdx()):
                        has_ethanolamine_arm = True
    ester_count = 0
    for atom in molecule.GetAtoms():
        if not _is_carbonyl_carbon(atom):
            continue
        if any(neighbour.GetAtomicNum() == 8 and neighbour.GetDegree() >= 2
               for neighbour in atom.GetNeighbors()):
            ester_count += 1
    long_tails = sum(len(component) >= 14 for component in _tail_components(molecule))
    return has_ethanolamine_arm and ester_count >= 2 and long_tails >= 2


def _uc_alpha_and_tail(molecule: Chem.Mol, alpha_indices: list[int]) -> tuple[int | None, int]:
    """Find an N-tail-C(=O)-N-C(alpha) urea motif and its attached tail size."""
    tails = _tail_components(molecule)
    tail_by_atom = {atom: len(component) for component in tails for atom in component}
    for alpha_index in alpha_indices:
        alpha = molecule.GetAtomWithIdx(alpha_index)
        for peptide_n in alpha.GetNeighbors():
            if peptide_n.GetAtomicNum() != 7:
                continue
            for carbonyl in peptide_n.GetNeighbors():
                if carbonyl.GetIdx() == alpha_index or not _is_carbonyl_carbon(carbonyl):
                    continue
                for tail_n in carbonyl.GetNeighbors():
                    if tail_n.GetIdx() == peptide_n.GetIdx() or tail_n.GetAtomicNum() != 7:
                        continue
                    tail_size = max((tail_by_atom.get(neighbour.GetIdx(), 0)
                                     for neighbour in tail_n.GetNeighbors()
                                     if neighbour.GetIdx() != carbonyl.GetIdx()), default=0)
                    if tail_size >= 8:
                        return alpha_index, tail_size
    return None, 0


def _terminal_state(molecule: Chem.Mol, alpha_index: int) -> tuple[int, int, int]:
    """Classify the C(alpha)-attached carbonyl as free acid, ester, or amide."""
    alpha = molecule.GetAtomWithIdx(alpha_index)
    carbonyls = [atom for atom in alpha.GetNeighbors() if _is_carbonyl_carbon(atom)]
    if len(carbonyls) != 1:
        return 0, 0, 0
    carbonyl = carbonyls[0]
    for neighbour in carbonyl.GetNeighbors():
        if neighbour.GetIdx() == alpha_index or neighbour.GetAtomicNum() == 8 and any(
                bond.GetBondTypeAsDouble() == 2.0 and bond.GetOtherAtom(carbonyl).GetIdx() == neighbour.GetIdx()
                for bond in carbonyl.GetBonds()):
            continue
        if neighbour.GetAtomicNum() == 7:
            return 0, 0, 1
        if neighbour.GetAtomicNum() == 8:
            return (1, 0, 0) if neighbour.GetDegree() == 1 else (0, 1, 0)
    return 0, 0, 0


def _empty(status: str, warning: str = "") -> SemanticResult:
    return SemanticResult(
        family_type="other", has_dope_tail=0, disulfide_bridge_count=0,
        peptide_length=0, aa_counts={code: 0 for code in AA_ORDER},
        positive_aa_count=0, negative_aa_count=0, peptide_net_charge_proxy=0,
        uc_amino_acid_type="none", uc_tail_carbon_count=0,
        uc_terminal_carboxyl=0, uc_terminal_ester=0, uc_terminal_amide=0,
        parse_status=status, parse_warnings=(warning,) if warning else (),
    )


def semantic_features_from_mol(molecule: Chem.Mol | None) -> SemanticResult:
    if molecule is None:
        return _empty("absent")
    alpha_indices = _alpha_carbons(molecule)
    residue_codes = [_classify_residue(molecule, index) for index in alpha_indices]
    unsupported = sum(code is None for code in residue_codes)
    aa_counts = {code: 0 for code in AA_ORDER}
    for code in residue_codes:
        if code in aa_counts:
            aa_counts[code] += 1
    dope = int(_is_phosphatidylethanolamine_tail(molecule))
    disulfides = _disulfide_count(molecule)
    uc_alpha, uc_tail = _uc_alpha_and_tail(molecule, alpha_indices)
    uc_type = "none"
    terminal = (0, 0, 0)
    warnings = []
    if unsupported:
        warnings.append(f"unsupported_residue_count={unsupported}")
    if uc_alpha is not None:
        code = _classify_residue(molecule, uc_alpha)
        uc_type = code if code in UC_AA_VALUES else "UNK"
        terminal = _terminal_state(molecule, uc_alpha)
        if code is None:
            warnings.append("UC_alpha_residue_unsupported")
        family = "UC_series"
    elif dope and disulfides > 0 and alpha_indices:
        family = "DOPE_SS_peptide_series"
    else:
        family = "other"
    positive = aa_counts["R"] + aa_counts["K"]
    negative = aa_counts["D"] + aa_counts["E"]
    status = "ok" if not warnings else "partial_unsupported"
    return SemanticResult(
        family_type=family, has_dope_tail=dope, disulfide_bridge_count=disulfides,
        peptide_length=len(alpha_indices), aa_counts=aa_counts,
        positive_aa_count=positive, negative_aa_count=negative,
        peptide_net_charge_proxy=positive - negative,
        uc_amino_acid_type=uc_type, uc_tail_carbon_count=uc_tail,
        uc_terminal_carboxyl=terminal[0], uc_terminal_ester=terminal[1],
        uc_terminal_amide=terminal[2], parse_status=status,
        parse_warnings=tuple(warnings),
    )


def semantic_features(smiles: object) -> SemanticResult:
    if str(smiles) in ABSENT_SMILES:
        return _empty("absent")
    molecule = Chem.MolFromSmiles(str(smiles))
    return semantic_features_from_mol(molecule) if molecule is not None else _empty("invalid_smiles", "invalid_smiles")


SEMANTIC_DEFINITIONS = {
    "family_type": "UC_series if a >=8-carbon tail-N-C(=O)-N-C(alpha) urea motif is present; DOPE_SS_peptide_series if PE core + >=2 long ester tails + S-S + >=1 residue are present; otherwise other.",
    "has_DOPE_tail": "PE phosphate O-C-C-N arm, >=2 ester linkages, and >=2 >=14-carbon hydrocarbon components.",
    "disulfide_bridge_count": "Number of covalent single S-S bonds.",
    "peptide_length": "Count of structural N-C(alpha)-C(=O) centres.",
    "AA_composition_20": "Counts of structurally matched standard residues; unrecognised centres are reported in parse_warnings and never assigned a residue identity.",
    "positive_AA_count": "Count of R and K residues. Histidine is neutral in this fixed no-pKa proxy.",
    "negative_AA_count": "Count of D and E residues.",
    "peptide_net_charge_proxy": "positive_AA_count - negative_AA_count; a residue-identity proxy, not pKa or charge prediction.",
    "UC_tail_carbon_count": "Carbon count of the >=8-carbon component directly attached to the urea tail nitrogen.",
    "UC_terminal_state": "The C(alpha)-attached carbonyl is free acid if its O has degree 1, ester if that O has degree >=2, and amide if the single hetero substituent is N.",
}
