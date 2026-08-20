"""Auditable, structure-only descriptors for the fifth formulation component.

These are deliberately simple graph rules rather than pKa predictions or
hand-assigned lipid classes.  Every value is determined from the supplied
RDKit molecular graph; absent/invalid components return an all-zero vector.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen


MECHANISTIC_DESCRIPTOR_NAMES = (
    "MolLogP",
    "tertiary_amine_count",
    "ionizable_N_count",
    "tail_count",
    "total_tail_carbon_count",
    "max_tail_length",
    "tail_length_asymmetry",
    "branch_density",
    "double_bond_count",
    "ester_count",
    "head_to_linker_distance",
    "head_tail_size_ratio",
)

ABSENT_SMILES = {"nan", "[Fr]"}


def _is_carbonyl_carbon(atom: Chem.Atom) -> bool:
    """True for a carbon with a double-bonded oxygen neighbour."""
    return atom.GetAtomicNum() == 6 and any(
        bond.GetBondTypeAsDouble() == 2.0
        and bond.GetOtherAtom(atom).GetAtomicNum() == 8
        for bond in atom.GetBonds()
    )


def _has_deactivating_acyl_neighbour(atom: Chem.Atom) -> bool:
    """Identify amide/sulfonamide/phosphoramide-like nitrogen attachment."""
    for neighbour in atom.GetNeighbors():
        if _is_carbonyl_carbon(neighbour):
            return True
        if neighbour.GetAtomicNum() in (15, 16) and any(
            bond.GetBondTypeAsDouble() == 2.0
            and bond.GetOtherAtom(neighbour).GetAtomicNum() == 8
            for bond in neighbour.GetBonds()
        ):
            return True
    return False


def _ionizable_nitrogen_indices(molecule: Chem.Mol) -> set[int]:
    """A reproducible structural proxy for potentially basic nitrogen sites.

    Count neutral/protonated non-aromatic nitrogens with heavy-atom degree
    below four unless directly acyl/sulfonyl/phosphoryl deactivated.  Also
    count aromatic ``n`` sites with no attached H (pyridine-like); aromatic
    ``[nH]`` (pyrrole-like) and quaternary nitrogen are excluded.  This is an
    operational graph definition, not a predicted pKa or protonation model.
    """
    indices: set[int] = set()
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        if atom.GetIsAromatic():
            if atom.GetFormalCharge() == 0 and atom.GetTotalNumHs() == 0:
                indices.add(atom.GetIdx())
            continue
        if atom.GetDegree() < 4 and not _has_deactivating_acyl_neighbour(atom):
            indices.add(atom.GetIdx())
    return indices


def _tertiary_amine_count(molecule: Chem.Mol) -> int:
    """Count neutral, non-aromatic, non-deactivated tertiary amines."""
    return sum(
        atom.GetAtomicNum() == 7
        and not atom.GetIsAromatic()
        and atom.GetFormalCharge() == 0
        and atom.GetDegree() == 3
        and atom.GetTotalNumHs() == 0
        and not _has_deactivating_acyl_neighbour(atom)
        for atom in molecule.GetAtoms()
    )


def _tail_components(molecule: Chem.Mol) -> list[set[int]]:
    """Return qualifying hydrocarbon-tail components using a fixed graph rule.

    Candidate nodes are all non-aromatic carbon atoms except carbonyl carbon.
    Connected components are formed through C--C bonds, then retained only at
    six or more carbon atoms.  Thus rings/aromatic head groups and short
    amino-acid side chains are not called tails, while straight, branched and
    unsaturated aliphatic chains use the same deterministic rule.
    """
    candidates = {
        atom.GetIdx() for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic()
        and not _is_carbonyl_carbon(atom)
    }
    adjacency = {index: set() for index in candidates}
    for bond in molecule.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if first in candidates and second in candidates:
            adjacency[first].add(second)
            adjacency[second].add(first)
    components: list[set[int]] = []
    unseen = set(candidates)
    while unseen:
        start = unseen.pop()
        component = {start}
        queue = [start]
        while queue:
            node = queue.pop()
            new = adjacency[node] & unseen
            unseen.difference_update(new)
            component.update(new)
            queue.extend(new)
        if len(component) >= 6:
            components.append(component)
    return components


def _component_diameter(component: set[int], molecule: Chem.Mol) -> int:
    """Maximum shortest C--C path length in atoms for one tail component."""
    if not component:
        return 0
    adjacency = {index: [] for index in component}
    for bond in molecule.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if first in component and second in component:
            adjacency[first].append(second)
            adjacency[second].append(first)
    diameter = 1
    for start in component:
        distance = {start: 0}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in adjacency[node]:
                if neighbour not in distance:
                    distance[neighbour] = distance[node] + 1
                    queue.append(neighbour)
        diameter = max(diameter, max(distance.values()) + 1)
    return diameter


def _ester_count(molecule: Chem.Mol) -> int:
    """Count unique carbonyl carbons in R-C(=O)-O-C ester motifs."""
    count = 0
    for atom in molecule.GetAtoms():
        if not _is_carbonyl_carbon(atom):
            continue
        for bond in atom.GetBonds():
            neighbour = bond.GetOtherAtom(atom)
            if (bond.GetBondTypeAsDouble() == 1.0 and neighbour.GetAtomicNum() == 8
                    and any(other.GetAtomicNum() == 6 for other in neighbour.GetNeighbors()
                            if other.GetIdx() != atom.GetIdx())):
                count += 1
                break
    return count


def _linker_carbonyl_indices(molecule: Chem.Mol) -> set[int]:
    """Carbonyl C in ester/amide/thioester-like C(=O)-hetero linkers."""
    indices: set[int] = set()
    for atom in molecule.GetAtoms():
        if not _is_carbonyl_carbon(atom):
            continue
        if any(bond.GetBondTypeAsDouble() == 1.0 and bond.GetOtherAtom(atom).GetAtomicNum() in (7, 8, 16)
               for bond in atom.GetBonds()):
            indices.add(atom.GetIdx())
    return indices


def _head_to_linker_distance(molecule: Chem.Mol, heads: set[int]) -> int:
    """Shortest heavy-atom path from an ionizable-N head proxy to a linker C."""
    linkers = _linker_carbonyl_indices(molecule)
    if not heads or not linkers:
        return 0
    distances = Chem.GetDistanceMatrix(molecule)
    return int(min(distances[head, linker] for head in heads for linker in linkers))


def descriptor_vector_from_mol(molecule: Chem.Mol | None) -> np.ndarray:
    """Compute the 12 O13-E raw descriptors from one RDKit molecule."""
    if molecule is None:
        return np.zeros(len(MECHANISTIC_DESCRIPTOR_NAMES), dtype=np.float32)
    tails = _tail_components(molecule)
    tail_atoms = set().union(*tails) if tails else set()
    tail_lengths = [_component_diameter(component, molecule) for component in tails]
    total_tail_carbons = len(tail_atoms)
    tail_adjacency_degree = {index: 0 for index in tail_atoms}
    double_bonds = 0
    for bond in molecule.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if first in tail_atoms and second in tail_atoms:
            tail_adjacency_degree[first] += 1
            tail_adjacency_degree[second] += 1
            if bond.GetBondTypeAsDouble() == 2.0:
                double_bonds += 1
    heads = _ionizable_nitrogen_indices(molecule)
    max_tail = max(tail_lengths, default=0)
    min_tail = min(tail_lengths, default=0)
    asymmetry = (max_tail - min_tail) / max_tail if len(tail_lengths) >= 2 and max_tail else 0.0
    branch_density = (sum(degree >= 3 for degree in tail_adjacency_degree.values()) / total_tail_carbons
                      if total_tail_carbons else 0.0)
    head_tail_ratio = ((molecule.GetNumHeavyAtoms() - total_tail_carbons) / total_tail_carbons
                       if total_tail_carbons else 0.0)
    return np.asarray([
        Crippen.MolLogP(molecule),
        _tertiary_amine_count(molecule),
        len(heads),
        len(tails),
        total_tail_carbons,
        max_tail,
        asymmetry,
        branch_density,
        double_bonds,
        _ester_count(molecule),
        _head_to_linker_distance(molecule, heads),
        head_tail_ratio,
    ], dtype=np.float32)


def descriptor_vector(smiles: object) -> np.ndarray:
    """Canonical public entry point; absent/invalid values are all zeros."""
    if str(smiles) in ABSENT_SMILES:
        return np.zeros(len(MECHANISTIC_DESCRIPTOR_NAMES), dtype=np.float32)
    return descriptor_vector_from_mol(Chem.MolFromSmiles(str(smiles)))


DESCRIPTOR_DEFINITIONS = {
    "MolLogP": "RDKit Crippen MolLogP.",
    "tertiary_amine_count": "Neutral non-aromatic N, degree 3, zero H, not directly acyl/sulfonyl/phosphoryl deactivated.",
    "ionizable_N_count": "Graph proxy: non-quaternary non-aromatic N not directly acyl/sulfonyl/phosphoryl deactivated, plus aromatic n with zero H; not a pKa prediction.",
    "tail_count": "Number of >=6-carbon connected components of non-aromatic, non-carbonyl carbon joined by C-C bonds.",
    "total_tail_carbon_count": "Total atoms in the qualifying tail components.",
    "max_tail_length": "Largest all-pairs shortest-path diameter (carbon atoms) among qualifying tail components.",
    "tail_length_asymmetry": "(max_tail_length-min_tail_length)/max_tail_length across >=2 qualifying tails; otherwise 0.",
    "branch_density": "Fraction of qualifying tail carbon atoms with C-C degree >=3 within the tail graph.",
    "double_bond_count": "C=C bonds whose two carbon atoms belong to qualifying tails.",
    "ester_count": "Unique R-C(=O)-O-C ester carbonyls.",
    "head_to_linker_distance": "Shortest heavy-atom distance from ionizable-N proxy to ester/amide/thioester linker carbonyl; 0 if either set is absent.",
    "head_tail_size_ratio": "(heavy_atom_count-total_tail_carbon_count)/total_tail_carbon_count; 0 when no qualifying tail.",
}
