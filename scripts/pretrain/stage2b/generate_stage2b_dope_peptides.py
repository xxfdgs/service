#!/usr/bin/env python3
"""
Stage 2B — controlled DOPE-peptide generation for Fifth-component pretraining.

Target chemistry
----------------
The observed unmodified DOPE-peptide Fifth structures share one consistent
architecture:

    DOPE-head -- linker-S-S-Cys(peptide)

All observed peptide names end in C (Cys), e.g.
    DC-DOPE
    DSSC-DOPE
    DRDRC-DOPE
    8DC-DOPE  -> DDDDDDDDC

Therefore Stage 2B samples the chemically compatible sequence universe:

    X1 X2 ... X(L-1) C
    Xi in the 20 canonical amino acids
    L = 2 ... 9

The terminal C is fixed because its side-chain sulfur forms the disulfide
bond to the DOPE linker. Internal residues may contain any canonical AA,
including additional Cys.

Default library
---------------
10,000 unique DOPE-peptide structures:
    - all observed unmodified DOPE-peptides are retained exactly;
    - 90% of NEW structures are broad coverage samples;
    - 10% of NEW structures are training-near mutations.

The broad-heavy default is intentional. A 50/50 pilot strongly reintroduced
the D/R-rich bias of the observed DOPE peptides, whereas the purpose of this
pretraining corpus is to broaden amino-acid representation.

Length-aware OOD-bridge coverage:
    - uses an explicit FINAL target count for every peptide length;
    - default 10k profile:
          L2=20, L3=400,
          L4=1980, L5=1980,
          L6=2520, L7=2520,
          L8=0,
          L9=580
    - L2 and L3 are exhaustive;
    - L4/L5 represent the dense training-supported regime;
    - L6/L7 are deliberately emphasized as intermediate-length OOD bridges;
    - L8 is omitted by default;
    - L9 is retained only as a sparse anchor because training contains
      special long repeat peptides rather than broad 9-mer diversity.

Training-near:
    - starts from an observed sequence;
    - keeps the same length and terminal C;
    - mutates 1 or 2 non-terminal residues;
    - excludes observed/broad/duplicate sequences.

Hard gates
----------
1. The frozen Stage-1.5 generation plan must authorize unmodified
   DOPE_peptide automatic augmentation over lengths 2-9.
2. A DOPE linker template is extracted directly from an observed unmodified
   structure by cutting its unique disulfide bond and retaining the
   phosphorus-containing fragment.
3. Every observed unmodified DOPE-peptide must be reconstructed with identical
   NON-ISOMERIC canonical connectivity.
4. Exactly --n-targets unique sequences and unique non-isomeric structures
   must be produced.
5. The final count at every length must exactly match the frozen length
   target profile. Lengths with target 0 (L8 by default) must remain absent.
6. For every TARGETED length L and every variable position 1..L-1, all
   20 canonical amino acids must be represented.
7. The terminal position must be C in every generated sequence.
8. Every generated structure must pass RDKit sanitization.
9. Across all non-terminal (variable) residue positions, the maximum/minimum
   canonical-AA occurrence ratio must not exceed --max-variable-aa-imbalance
   (default 2.0), preventing the D/R-rich observed distribution from dominating
   the pretraining corpus.

Stereochemistry
---------------
The downstream GraphGPS used in this project does not encode stereochemistry,
so Stage 2B uses non-isomeric canonical SMILES as the reconstruction and
deduplication identity. Observed training SMILES are nevertheless retained
unchanged in the output library.

Inputs
------
--row-audit
    Final Stage-1 row_level_fifth_audit.csv.

--generation-plan
    Frozen Stage-1.5 scaffold_generation_plan.csv.

Outputs
-------
stage2b_dope_peptide_library.csv
stage2b_reconstruction_audit.csv
stage2b_length_coverage.csv
stage2b_amino_acid_coverage.csv
stage2b_position_coverage.csv
stage2b_sampling_summary.csv
stage2b_manifest.json

Optional diagnostics on failure:
stage2b_duplicate_structures.csv
stage2b_failed_reconstruction.csv
stage2b_position_coverage_failures.csv
stage2b_length_target_failures.csv
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


AA1 = "ARNDCQEGHILKMFPSTWYV"
AA_SET = set(AA1)
MIN_LENGTH = 2
MAX_LENGTH = 9

# Final structure counts for the default 10k Stage-2B library.
#
# Rationale:
#   - L2/L3: complete sequence-space coverage.
#   - L4/L5: dense coverage of training-supported lengths.
#   - L6/L7: strongest emphasis as intermediate-length OOD bridges.
#   - L8: omitted in the default profile.
#   - L9: sparse anchor only; observed training 9-mers are highly repetitive.
DEFAULT_10K_FINAL_LENGTH_TARGETS = {
    2: 20,
    3: 400,
    4: 1980,
    5: 1980,
    6: 2520,
    7: 2520,
    8: 0,
    9: 580,
}

if sum(DEFAULT_10K_FINAL_LENGTH_TARGETS.values()) != 10_000:
    raise RuntimeError("DEFAULT_10K_FINAL_LENGTH_TARGETS must sum to 10,000.")

# A deterministic mapping is useful in reports.
AA1_TO_AA3 = {
    "A": "Ala",
    "R": "Arg",
    "N": "Asn",
    "D": "Asp",
    "C": "Cys",
    "Q": "Gln",
    "E": "Glu",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Leu",
    "K": "Lys",
    "M": "Met",
    "F": "Phe",
    "P": "Pro",
    "S": "Ser",
    "T": "Thr",
    "W": "Trp",
    "Y": "Tyr",
    "V": "Val",
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


def mol_or_fail(smiles: str, label: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse {label}: {smiles}")
    Chem.SanitizeMol(mol)
    return mol


def canonical_nonisomeric(mol: Chem.Mol) -> str:
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


# =============================================================================
# Stage-1 nomenclature parsing for unmodified DOPE peptides
# =============================================================================

def parse_unmodified_dope_name(name: str) -> str | None:
    """
    Parse ONLY the high-confidence unmodified DOPE nomenclature already frozen
    in Stage 1.

    Examples
    --------
    DC-DOPE      -> DC
    DRDRC-DOPE   -> DRDRC
    4DC-DOPE     -> DDDDC
    8RC-DOPE     -> RRRRRRRRC
    """
    name = clean(name)

    m = re.fullmatch(
        rf"(?P<n>\d+)(?P<aa>[{''.join(sorted(AA_SET))}])C-DOPE",
        name,
        flags=re.IGNORECASE,
    )
    if m:
        repeat_n = int(m.group("n"))
        aa = m.group("aa").upper()
        sequence = aa * repeat_n + "C"
        if MIN_LENGTH <= len(sequence) <= MAX_LENGTH:
            return sequence
        return None

    m = re.fullmatch(
        rf"(?P<seq>[{''.join(sorted(AA_SET))}]{{2,9}})-DOPE",
        name,
        flags=re.IGNORECASE,
    )
    if m:
        sequence = m.group("seq").upper()
        if sequence.endswith("C"):
            return sequence

    return None


def build_observed_dope_map(
    row_audit: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    required = {"Fifth", "Fifth_SMILE"}
    missing = required.difference(row_audit.columns)
    if missing:
        raise ValueError(
            "Row audit missing required columns: "
            + ", ".join(sorted(missing))
        )

    observed: dict[str, dict[str, str]] = {}

    for row in row_audit.itertuples(index=False):
        name = clean(getattr(row, "Fifth"))
        sequence = parse_unmodified_dope_name(name)
        if sequence is None:
            continue

        smiles = clean(getattr(row, "Fifth_SMILE"))
        if not smiles:
            raise ValueError(
                f"Observed unmodified DOPE peptide lacks SMILES: {name}"
            )

        mol = mol_or_fail(smiles, name)
        conn = canonical_nonisomeric(mol)
        iso = canonical_isomeric(mol)

        record = {
            "sequence": sequence,
            "observed_name": name,
            "observed_smiles": smiles,
            "observed_canonical_connectivity": conn,
            "observed_canonical_isomeric": iso,
        }

        if sequence in observed:
            old = observed[sequence]
            if old["observed_canonical_connectivity"] != conn:
                raise ValueError(
                    "One peptide sequence has conflicting observed connectivity: "
                    f"{sequence}: {old['observed_name']} vs {name}"
                )
        else:
            observed[sequence] = record

    if not observed:
        raise ValueError(
            "No observed unmodified DOPE-peptide structures were found."
        )

    return observed


# =============================================================================
# Frozen generation-plan validation
# =============================================================================

def validate_generation_plan(plan_path: Path) -> None:
    plan = pd.read_csv(plan_path)

    required = {
        "scaffold_family",
        "scaffold_variant",
        "automatic_augmentation",
        "target_residue_universe",
        "target_sequence_lengths",
        "sampling_policy",
    }
    missing = required.difference(plan.columns)
    if missing:
        raise ValueError(
            "Generation plan missing columns: "
            + ", ".join(sorted(missing))
        )

    rows = plan.loc[
        plan["scaffold_family"].eq("DOPE_peptide")
        & plan["scaffold_variant"].eq("unmodified")
    ]

    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one frozen generation-plan row for "
            "DOPE_peptide/unmodified; found "
            f"{len(rows)}."
        )

    row = rows.iloc[0]

    if clean(row["automatic_augmentation"]).lower() != "yes":
        raise ValueError(
            "Frozen plan does not authorize automatic augmentation for "
            "DOPE_peptide/unmodified."
        )

    if "20 canonical AA" not in clean(row["target_residue_universe"]):
        raise ValueError(
            "Frozen plan residue universe is not '20 canonical AA': "
            f"{row['target_residue_universe']}"
        )

    if clean(row["target_sequence_lengths"]) != "2-9":
        raise ValueError(
            "Frozen plan sequence-length policy is not 2-9: "
            f"{row['target_sequence_lengths']}"
        )


# =============================================================================
# DOPE-linker extraction and molecule construction
# =============================================================================

def extract_dope_linker_template(
    observed: dict[str, dict[str, str]],
) -> tuple[Chem.Mol, str]:
    """
    Extract the constant DOPE-side linker directly from an observed molecule.

    Prefer DC-DOPE because it has the shortest peptide, but any observed
    unmodified DOPE-peptide with one S-S bond is chemically sufficient.

    The S-S bond is cut without dummies. The phosphorus-containing fragment is
    retained. Its terminal sulfur becomes the attachment sulfur for generated
    terminal Cys.
    """
    if "DC" in observed:
        template_record = observed["DC"]
    else:
        template_record = min(
            observed.values(),
            key=lambda x: len(x["sequence"]),
        )

    mol = mol_or_fail(
        template_record["observed_smiles"],
        template_record["observed_name"],
    )

    ss_bonds = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if a.GetAtomicNum() == 16 and b.GetAtomicNum() == 16:
            ss_bonds.append(bond.GetIdx())

    if len(ss_bonds) != 1:
        raise ValueError(
            "Expected exactly one disulfide bond in DOPE template "
            f"{template_record['observed_name']}; found {len(ss_bonds)}."
        )

    fragmented = Chem.FragmentOnBonds(
        mol,
        ss_bonds,
        addDummies=False,
    )

    fragments = Chem.GetMolFrags(
        fragmented,
        asMols=True,
        sanitizeFrags=True,
    )

    phosphorus_fragments = [
        frag
        for frag in fragments
        if any(atom.GetAtomicNum() == 15 for atom in frag.GetAtoms())
    ]

    if len(phosphorus_fragments) != 1:
        raise ValueError(
            "Could not uniquely identify phosphorus-containing DOPE linker "
            f"fragment; found {len(phosphorus_fragments)}."
        )

    linker = phosphorus_fragments[0]

    terminal_s = [
        atom
        for atom in linker.GetAtoms()
        if atom.GetAtomicNum() == 16 and atom.GetDegree() == 1
    ]

    if len(terminal_s) != 1:
        raise ValueError(
            "Expected exactly one degree-1 terminal sulfur in extracted "
            f"DOPE linker; found {len(terminal_s)}."
        )

    Chem.SanitizeMol(linker)

    return linker, template_record["observed_name"]


def terminal_cys_sulfur_idx(
    peptide: Chem.Mol,
    sequence: str,
) -> int:
    """
    Identify the terminal Cys SG atom using RDKit PDB residue metadata
    generated by MolFromFASTA.
    """
    if not sequence.endswith("C"):
        raise ValueError(
            f"DOPE peptide sequence must terminate in Cys: {sequence}"
        )

    residue_number = len(sequence)
    hits = []

    for atom in peptide.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue

        if (
            info.GetResidueNumber() == residue_number
            and info.GetResidueName().strip().upper() == "CYS"
            and info.GetName().strip().upper() == "SG"
        ):
            hits.append(atom.GetIdx())

    if len(hits) != 1:
        raise ValueError(
            "Could not uniquely locate terminal Cys SG for sequence "
            f"{sequence}; hits={hits}"
        )

    return hits[0]


def linker_terminal_s_idx(linker: Chem.Mol) -> int:
    hits = [
        atom.GetIdx()
        for atom in linker.GetAtoms()
        if atom.GetAtomicNum() == 16 and atom.GetDegree() == 1
    ]
    if len(hits) != 1:
        raise ValueError(
            f"Expected one terminal linker sulfur; found {hits}"
        )
    return hits[0]


def generate_dope_peptide(
    linker: Chem.Mol,
    sequence: str,
) -> Chem.Mol:
    if not (
        MIN_LENGTH <= len(sequence) <= MAX_LENGTH
        and set(sequence).issubset(AA_SET)
        and sequence.endswith("C")
    ):
        raise ValueError(f"Invalid Stage-2B peptide sequence: {sequence}")

    peptide = Chem.MolFromFASTA(sequence)
    if peptide is None:
        raise ValueError(
            f"RDKit MolFromFASTA failed for sequence {sequence}"
        )

    Chem.SanitizeMol(peptide)

    peptide_s = terminal_cys_sulfur_idx(peptide, sequence)
    linker_s = linker_terminal_s_idx(linker)

    combo = Chem.CombineMols(linker, peptide)
    rw = Chem.RWMol(combo)

    rw.AddBond(
        linker_s,
        linker.GetNumAtoms() + peptide_s,
        Chem.BondType.SINGLE,
    )

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


# =============================================================================
# Reconstruction audit
# =============================================================================

def reconstruction_audit(
    linker: Chem.Mol,
    observed: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows = []

    for sequence in sorted(
        observed,
        key=lambda s: (len(s), s),
    ):
        record = observed[sequence]

        generated = generate_dope_peptide(linker, sequence)
        generated_conn = canonical_nonisomeric(generated)
        observed_conn = record["observed_canonical_connectivity"]

        rows.append(
            {
                "sequence": sequence,
                "sequence_length": len(sequence),
                "observed_name": record["observed_name"],
                "observed_canonical_connectivity": observed_conn,
                "generated_canonical_connectivity": generated_conn,
                "connectivity_match": bool(
                    generated_conn == observed_conn
                ),
                "generated_canonical_isomeric": canonical_isomeric(
                    generated
                ),
                "observed_canonical_isomeric": record[
                    "observed_canonical_isomeric"
                ],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Sequence-space sampling
# =============================================================================

def sequence_capacity(length: int) -> int:
    # Last residue is fixed C.
    return len(AA1) ** (length - 1)


def _largest_remainder_allocation(
    total: int,
    weights: dict[int, float],
) -> dict[int, int]:
    """
    Deterministically allocate an integer total according to positive weights.
    """
    if total < 0:
        raise ValueError("Allocation total must be >= 0.")

    positive = {
        int(k): float(v)
        for k, v in weights.items()
        if float(v) > 0
    }

    if total == 0:
        return {int(k): 0 for k in weights}

    if not positive:
        raise ValueError("No positive weights available for nonzero allocation.")

    weight_sum = sum(positive.values())

    raw = {
        k: total * w / weight_sum
        for k, w in positive.items()
    }

    allocated = {
        k: int(math.floor(value))
        for k, value in raw.items()
    }

    remainder = total - sum(allocated.values())

    ranking = sorted(
        positive,
        key=lambda k: (
            -(raw[k] - allocated[k]),
            k,
        ),
    )

    for k in ranking[:remainder]:
        allocated[k] += 1

    return {
        int(k): int(allocated.get(k, 0))
        for k in weights
    }


def resolve_final_length_targets(
    n_targets: int,
    custom_targets: str | None,
) -> dict[int, int]:
    """
    Resolve the FINAL number of structures wanted at each length.

    Default n_targets=10,000 uses the frozen OOD-bridge profile exactly.

    For other totals, preserve:
      - exhaustive L2 when possible;
      - exhaustive L3 when possible;
      - L8=0;
      - sparse L9 anchor (~5.8%);
      - remaining mass across L4/L5/L6/L7 with weights 0.22/0.22/0.28/0.28.

    A custom profile can be supplied as:
        --length-targets "2:20,3:400,4:1980,5:1980,6:2520,7:2520,8:0,9:580"
    """
    if custom_targets:
        parsed: dict[int, int] = {}

        for token in custom_targets.split(","):
            token = token.strip()
            if not token:
                continue

            if ":" not in token:
                raise ValueError(
                    "--length-targets entries must be LENGTH:COUNT, got "
                    f"{token!r}"
                )

            left, right = token.split(":", 1)
            length = int(left)
            count = int(right)

            if length < MIN_LENGTH or length > MAX_LENGTH:
                raise ValueError(
                    f"Unsupported length in --length-targets: {length}"
                )

            if count < 0:
                raise ValueError(
                    f"Negative target count for length {length}: {count}"
                )

            if length in parsed:
                raise ValueError(
                    f"Duplicate length in --length-targets: {length}"
                )

            parsed[length] = count

        targets = {
            length: int(parsed.get(length, 0))
            for length in range(MIN_LENGTH, MAX_LENGTH + 1)
        }

        if sum(targets.values()) != n_targets:
            raise ValueError(
                "--length-targets must sum exactly to --n-targets. "
                f"Got {sum(targets.values())} vs {n_targets}."
            )

        return targets

    if n_targets == 10_000:
        return dict(DEFAULT_10K_FINAL_LENGTH_TARGETS)

    if n_targets < 1_000:
        raise ValueError(
            "The OOD-bridge profile is intended for >=1000 targets. "
            "Use --length-targets explicitly for smaller pilot libraries."
        )

    targets = {
        length: 0
        for length in range(MIN_LENGTH, MAX_LENGTH + 1)
    }

    # Preserve exhaustive short-sequence spaces whenever the library is large
    # enough to make that meaningful.
    targets[2] = min(20, n_targets)

    remaining = n_targets - targets[2]

    l3 = min(400, max(0, remaining))
    targets[3] = l3
    remaining -= l3

    if remaining <= 0:
        return targets

    # Sparse 9-mer anchor: same fraction as the frozen 10k design.
    l9 = int(round(n_targets * 0.058))
    l9 = min(l9, remaining)
    targets[9] = l9
    remaining -= l9

    # Explicitly omit L8 in this profile.
    targets[8] = 0

    bridge_weights = {
        4: 0.22,
        5: 0.22,
        6: 0.28,
        7: 0.28,
    }

    allocation = _largest_remainder_allocation(
        remaining,
        bridge_weights,
    )

    for length, count in allocation.items():
        targets[length] = count

    if sum(targets.values()) != n_targets:
        raise RuntimeError(
            "Scaled OOD-bridge target allocation does not sum to n_targets."
        )

    return targets


def validate_final_length_targets(
    final_targets: dict[int, int],
    observed_sequences: set[str],
) -> None:
    """
    Validate target counts against chemistry/sequence-space constraints.
    """
    observed_by_length = Counter(
        len(seq) for seq in observed_sequences
    )

    for length in range(MIN_LENGTH, MAX_LENGTH + 1):
        target = int(final_targets.get(length, 0))
        observed_count = int(observed_by_length.get(length, 0))
        capacity = sequence_capacity(length)

        if target < observed_count:
            raise ValueError(
                f"Length {length}: final target {target} is smaller than "
                f"{observed_count} observed training sequences."
            )

        if target > capacity:
            raise ValueError(
                f"Length {length}: final target {target} exceeds sequence-space "
                f"capacity {capacity}."
            )


def allocate_near_quotas(
    *,
    n_near: int,
    final_targets: dict[int, int],
    observed_sequences: set[str],
) -> dict[int, int]:
    """
    Allocate training-near sequences ONLY to lengths that have observed
    parents and are not intentionally exhaustive short spaces.

    L2/L3 are reserved for exhaustive broad coverage.
    L6/L7 have no observed parents and are therefore broad-only.
    L8 is omitted by default.
    L9 receives a small near component because real length-9 anchors exist.
    """
    observed_by_length = Counter(
        len(seq) for seq in observed_sequences
    )

    quotas = {
        length: 0
        for length in range(MIN_LENGTH, MAX_LENGTH + 1)
    }

    eligible = {}

    for length in range(MIN_LENGTH, MAX_LENGTH + 1):
        if length in {2, 3}:
            continue

        observed_count = observed_by_length.get(length, 0)
        if observed_count <= 0:
            continue

        headroom = (
            int(final_targets.get(length, 0))
            - int(observed_count)
        )

        if headroom <= 0:
            continue

        # Weight by the target population at this length.
        eligible[length] = float(headroom)

    if n_near == 0:
        return quotas

    if not eligible:
        raise ValueError(
            "Training-near samples requested but no eligible observed lengths "
            "have available target headroom."
        )

    preliminary = _largest_remainder_allocation(
        n_near,
        eligible,
    )

    # Clip to per-length headroom and redistribute any overflow.
    remaining = n_near

    for length in sorted(preliminary):
        headroom = (
            int(final_targets[length])
            - int(observed_by_length.get(length, 0))
        )
        take = min(int(preliminary[length]), headroom)
        quotas[length] = take
        remaining -= take

    while remaining > 0:
        candidates = []

        for length in sorted(eligible):
            headroom = (
                int(final_targets[length])
                - int(observed_by_length.get(length, 0))
                - int(quotas[length])
            )

            if headroom > 0:
                candidates.append((length, headroom))

        if not candidates:
            raise ValueError(
                "Requested training-near count exceeds total eligible "
                "per-length headroom."
            )

        for length, _ in candidates:
            if remaining == 0:
                break
            quotas[length] += 1
            remaining -= 1

    return quotas


def derive_broad_quotas(
    *,
    final_targets: dict[int, int],
    near_quotas: dict[int, int],
    observed_sequences: set[str],
) -> dict[int, int]:
    """
    Broad count is whatever remains after observed + near samples at each
    length. This guarantees the FINAL length distribution exactly matches the
    frozen profile.
    """
    observed_by_length = Counter(
        len(seq) for seq in observed_sequences
    )

    broad = {}

    for length in range(MIN_LENGTH, MAX_LENGTH + 1):
        value = (
            int(final_targets.get(length, 0))
            - int(observed_by_length.get(length, 0))
            - int(near_quotas.get(length, 0))
        )

        if value < 0:
            raise ValueError(
                f"Length {length}: negative broad quota after observed/near "
                f"allocation: {value}"
            )

        broad[length] = value

    return broad


def random_sequence(
    rng: random.Random,
    length: int,
) -> str:
    return "".join(
        rng.choice(AA1)
        for _ in range(length - 1)
    ) + "C"


def enumerate_sequence_space(length: int) -> Iterable[str]:
    for prefix in itertools.product(
        AA1,
        repeat=length - 1,
    ):
        yield "".join(prefix) + "C"


def sample_broad_sequences(
    *,
    rng: random.Random,
    quotas: dict[int, int],
    blocked: set[str],
) -> list[str]:
    output: list[str] = []
    used = set(blocked)

    for length in range(MIN_LENGTH, MAX_LENGTH + 1):
        target = quotas[length]
        if target <= 0:
            continue

        total_capacity = sequence_capacity(length)

        # Exhaustive candidate construction is cheap for L2/L3 and is also
        # useful whenever we request most of a small sequence space.
        if total_capacity <= 10_000 and target >= 0.5 * total_capacity:
            candidates = [
                seq
                for seq in enumerate_sequence_space(length)
                if seq not in used
            ]
            rng.shuffle(candidates)

            if len(candidates) < target:
                raise ValueError(
                    f"Length {length}: only {len(candidates)} broad candidates "
                    f"available for requested {target}."
                )

            chosen = candidates[:target]
            output.extend(chosen)
            used.update(chosen)
            continue

        # Otherwise rejection-sample unique sequences.
        chosen: list[str] = []
        attempts = 0
        max_attempts = max(100_000, target * 200)

        while len(chosen) < target and attempts < max_attempts:
            attempts += 1
            sequence = random_sequence(rng, length)

            if sequence in used:
                continue

            chosen.append(sequence)
            used.add(sequence)

        if len(chosen) != target:
            raise RuntimeError(
                f"Length {length}: generated {len(chosen)}/{target} broad "
                f"sequences after {attempts} attempts."
            )

        output.extend(chosen)

    if len(output) != sum(quotas.values()):
        raise RuntimeError(
            "Broad sequence output size does not equal allocated quota."
        )

    return output


def mutate_training_near(
    *,
    rng: random.Random,
    parent: str,
) -> tuple[str, int]:
    """
    Mutate one or two NON-terminal positions.
    Terminal C remains fixed.
    """
    mutable_positions = list(range(len(parent) - 1))

    if not mutable_positions:
        raise ValueError(f"No mutable positions in parent {parent}")

    if len(mutable_positions) == 1:
        n_mutations = 1
    else:
        n_mutations = 1 if rng.random() < 0.70 else 2

    positions = rng.sample(
        mutable_positions,
        k=min(n_mutations, len(mutable_positions)),
    )

    chars = list(parent)

    for pos in positions:
        original = chars[pos]
        choices = [aa for aa in AA1 if aa != original]
        chars[pos] = rng.choice(choices)

    child = "".join(chars)

    if not child.endswith("C"):
        raise AssertionError(
            "Training-near mutation changed terminal Cys."
        )

    return child, len(positions)


def sample_training_near_sequences(
    *,
    rng: random.Random,
    quotas_by_length: dict[int, int],
    observed_sequences: set[str],
    blocked: set[str],
) -> list[dict[str, Any]]:
    """
    Generate exact per-length training-near quotas.

    Each child:
      - has the same length as its observed parent;
      - preserves terminal C;
      - mutates 1 or 2 non-terminal positions;
      - is globally unique and absent from the blocked set.
    """
    parents_by_length: dict[int, list[str]] = defaultdict(list)

    for sequence in observed_sequences:
        parents_by_length[len(sequence)].append(sequence)

    for length in parents_by_length:
        parents_by_length[length] = sorted(
            parents_by_length[length]
        )

    used = set(blocked)
    output: list[dict[str, Any]] = []

    for length in range(MIN_LENGTH, MAX_LENGTH + 1):
        target = int(quotas_by_length.get(length, 0))

        if target <= 0:
            continue

        parents = parents_by_length.get(length, [])

        if not parents:
            raise ValueError(
                f"Length {length}: near quota {target} requested but no "
                "observed parent sequence exists."
            )

        produced = 0
        attempts = 0
        max_attempts = max(200_000, target * 1000)

        while produced < target and attempts < max_attempts:
            attempts += 1

            parent = rng.choice(parents)

            child, n_mut = mutate_training_near(
                rng=rng,
                parent=parent,
            )

            if child in used:
                continue

            used.add(child)
            produced += 1

            output.append(
                {
                    "sequence": child,
                    "parent_sequence": parent,
                    "mutation_count": n_mut,
                    "parent_hamming_distance": hamming_distance(
                        child,
                        parent,
                    ),
                }
            )

        if produced != target:
            raise RuntimeError(
                f"Length {length}: generated only {produced}/{target} unique "
                f"training-near sequences after {attempts} attempts."
            )

    expected = sum(
        int(v)
        for v in quotas_by_length.values()
    )

    if len(output) != expected:
        raise RuntimeError(
            f"Training-near output size {len(output)} != expected {expected}."
        )

    return output


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError(
            "Hamming distance requires equal-length sequences."
        )
    return sum(x != y for x, y in zip(a, b))


def nearest_observed_hamming(
    sequence: str,
    observed_sequences: set[str],
) -> int | None:
    same_length = [
        obs
        for obs in observed_sequences
        if len(obs) == len(sequence)
    ]

    if not same_length:
        return None

    return min(
        hamming_distance(sequence, obs)
        for obs in same_length
    )


# =============================================================================
# Coverage reports
# =============================================================================

def make_length_coverage(
    library: pd.DataFrame,
    final_targets: dict[int, int],
) -> pd.DataFrame:
    rows = []

    for length in range(MIN_LENGTH, MAX_LENGTH + 1):
        subset = library.loc[
            library["sequence_length"].eq(length)
        ]

        target = int(final_targets.get(length, 0))

        rows.append(
            {
                "sequence_length": length,
                "target_sequences": target,
                "total_sequences": int(len(subset)),
                "target_match": bool(len(subset) == target),
                "observed_training": int(
                    subset["sampling_source"]
                    .eq("observed_training")
                    .sum()
                ),
                "broad_coverage": int(
                    subset["sampling_source"]
                    .eq("broad_coverage")
                    .sum()
                ),
                "training_near_mutation": int(
                    subset["sampling_source"]
                    .eq("training_near_mutation")
                    .sum()
                ),
                "full_theoretical_sequence_space": int(
                    sequence_capacity(length)
                ),
                "coverage_fraction_of_full_space": float(
                    len(subset) / sequence_capacity(length)
                ),
            }
        )

    return pd.DataFrame(rows)


def make_aa_coverage(library: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for aa in AA1:
        row = {
            "aa1": aa,
            "aa3": AA1_TO_AA3[aa],
        }

        for source in (
            "observed_training",
            "broad_coverage",
            "training_near_mutation",
            "all",
        ):
            if source == "all":
                subset = library
            else:
                subset = library.loc[
                    library["sampling_source"].eq(source)
                ]

            all_occ = sum(
                sequence.count(aa)
                for sequence in subset["sequence"]
            )

            variable_occ = sum(
                sequence[:-1].count(aa)
                for sequence in subset["sequence"]
            )

            row[f"{source}_all_residue_occurrences"] = int(all_occ)
            row[f"{source}_variable_position_occurrences"] = int(
                variable_occ
            )

        rows.append(row)

    return pd.DataFrame(rows)


def make_position_coverage(
    library: pd.DataFrame,
    final_targets: dict[int, int],
) -> pd.DataFrame:
    """
    Positional coverage is required only for TARGETED lengths.

    Lengths whose final target is zero (L8 by default) are intentionally absent
    and therefore do not create a false coverage failure.
    """
    rows = []

    for length in range(MIN_LENGTH, MAX_LENGTH + 1):
        target = int(final_targets.get(length, 0))

        if target <= 0:
            continue

        subset = library.loc[
            library["sequence_length"].eq(length)
        ]

        if subset.empty:
            rows.append(
                {
                    "sequence_length": length,
                    "position": pd.NA,
                    "is_terminal_position": pd.NA,
                    "unique_aa_count": 0,
                    "observed_amino_acids": "",
                    "expected_amino_acids": "",
                    "coverage_pass": False,
                    "min_count_among_expected": 0,
                    "max_count_among_expected": 0,
                }
            )
            continue

        for position in range(1, length + 1):
            counts = Counter(
                sequence[position - 1]
                for sequence in subset["sequence"]
            )

            is_terminal = position == length
            expected = {"C"} if is_terminal else set(AA1)
            observed = set(counts)

            rows.append(
                {
                    "sequence_length": length,
                    "position": position,
                    "is_terminal_position": bool(is_terminal),
                    "unique_aa_count": int(len(observed)),
                    "observed_amino_acids": "".join(
                        aa for aa in AA1 if aa in observed
                    ),
                    "expected_amino_acids": (
                        "C" if is_terminal else AA1
                    ),
                    "coverage_pass": bool(observed == expected),
                    "min_count_among_expected": int(
                        min(
                            counts.get(aa, 0)
                            for aa in expected
                        )
                        if expected
                        else 0
                    ),
                    "max_count_among_expected": int(
                        max(
                            counts.get(aa, 0)
                            for aa in expected
                        )
                        if expected
                        else 0
                    ),
                }
            )

    return pd.DataFrame(rows)


def make_sampling_summary(library: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for source, subset in library.groupby(
        "sampling_source",
        sort=True,
    ):
        rows.append(
            {
                "sampling_source": source,
                "sequences": int(len(subset)),
                "min_length": int(
                    subset["sequence_length"].min()
                ),
                "max_length": int(
                    subset["sequence_length"].max()
                ),
                "mean_length": float(
                    subset["sequence_length"].mean()
                ),
                "mean_nearest_observed_hamming_same_length": float(
                    pd.to_numeric(
                        subset[
                            "nearest_observed_hamming_same_length"
                        ],
                        errors="coerce",
                    ).mean()
                )
                if subset[
                    "nearest_observed_hamming_same_length"
                ].notna().any()
                else None,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Stage-2B controlled DOPE-peptide pretraining library."
        )
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
        "--n-targets",
        type=int,
        default=10_000,
        help=(
            "Total unique DOPE-peptide structures INCLUDING observed "
            "unmodified training structures."
        ),
    )

    parser.add_argument(
        "--length-targets",
        type=str,
        default=None,
        help=(
            "Optional explicit FINAL per-length target counts, e.g. "
            "'2:20,3:400,4:1980,5:1980,6:2520,7:2520,8:0,9:580'. "
            "Counts must sum to --n-targets. If omitted, use the frozen "
            "OOD-bridge profile (exactly the values above for 10k)."
        ),
    )

    parser.add_argument(
        "--near-fraction",
        type=float,
        default=0.10,
        help=(
            "Fraction of NEW generated structures assigned to "
            "training-near mutation sampling. Default 0.10 to preserve "
            "local realism without reintroducing the D/R-heavy training bias."
        ),
    )

    parser.add_argument(
        "--max-variable-aa-imbalance",
        type=float,
        default=2.0,
        help=(
            "Maximum allowed ratio of the most frequent to least frequent "
            "canonical AA across all non-terminal peptide positions."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260818,
    )

    args = parser.parse_args()

    if args.n_targets < 100:
        raise ValueError("--n-targets must be at least 100.")

    if not 0.0 <= args.near_fraction <= 1.0:
        raise ValueError("--near-fraction must be in [0, 1].")

    if args.max_variable_aa_imbalance < 1.0:
        raise ValueError("--max-variable-aa-imbalance must be >= 1.0.")

    row_audit_path = args.row_audit.resolve()
    plan_path = args.generation_plan.resolve()
    outdir = args.output_dir.resolve()

    if not row_audit_path.is_file():
        raise FileNotFoundError(row_audit_path)

    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)

    outdir.mkdir(parents=True, exist_ok=True)

    validate_generation_plan(plan_path)

    row_audit = pd.read_csv(
        row_audit_path,
        dtype={"ID": str},
    )

    observed = build_observed_dope_map(row_audit)
    observed_sequences = set(observed)

    # Training chemistry check: all observed unmodified DOPE peptides must
    # terminate in Cys.
    invalid_observed_terminal = sorted(
        seq
        for seq in observed_sequences
        if not seq.endswith("C")
    )

    if invalid_observed_terminal:
        raise ValueError(
            "Observed unmodified DOPE peptides violate terminal-Cys rule: "
            + ", ".join(invalid_observed_terminal)
        )

    linker, linker_source_name = extract_dope_linker_template(
        observed
    )

    reconstruction = reconstruction_audit(
        linker,
        observed,
    )

    reconstruction.to_csv(
        outdir / "stage2b_reconstruction_audit.csv",
        index=False,
    )

    failed_reconstruction = reconstruction.loc[
        ~reconstruction["connectivity_match"]
    ]

    if not failed_reconstruction.empty:
        failed_reconstruction.to_csv(
            outdir / "stage2b_failed_reconstruction.csv",
            index=False,
        )
        raise SystemExit(
            "Stage 2B BLOCKED: observed DOPE-peptide connectivity "
            "reconstruction is not 100%."
        )

    n_observed = len(observed_sequences)

    if args.n_targets <= n_observed:
        raise ValueError(
            f"--n-targets={args.n_targets} must exceed observed unmodified "
            f"DOPE sequence count {n_observed}."
        )

    final_length_targets = resolve_final_length_targets(
        args.n_targets,
        args.length_targets,
    )

    validate_final_length_targets(
        final_length_targets,
        observed_sequences,
    )

    n_new = args.n_targets - n_observed
    n_near_requested = int(
        round(n_new * args.near_fraction)
    )

    near_quotas = allocate_near_quotas(
        n_near=n_near_requested,
        final_targets=final_length_targets,
        observed_sequences=observed_sequences,
    )

    broad_quotas = derive_broad_quotas(
        final_targets=final_length_targets,
        near_quotas=near_quotas,
        observed_sequences=observed_sequences,
    )

    n_near = sum(near_quotas.values())
    n_broad = sum(broad_quotas.values())

    if n_observed + n_near + n_broad != args.n_targets:
        raise RuntimeError(
            "Observed + near + broad counts do not sum to --n-targets."
        )

    rng = random.Random(args.seed)

    # ------------------------------------------------------------------
    # Broad coverage.
    #
    # The per-length broad quota is derived from the FINAL target profile
    # after subtracting observed and training-near samples. Short spaces
    # (L2/L3) become exhaustive. L6/L7 are broad-only OOD bridges.
    # ------------------------------------------------------------------

    broad_sequences = sample_broad_sequences(
        rng=rng,
        quotas=broad_quotas,
        blocked=set(observed_sequences),
    )

    broad_set = set(broad_sequences)

    # ------------------------------------------------------------------
    # Training-near mutations.
    #
    # Exact per-length quotas prevent the near sampler from distorting the
    # frozen final length distribution.
    # ------------------------------------------------------------------

    near_records = sample_training_near_sequences(
        rng=rng,
        quotas_by_length=near_quotas,
        observed_sequences=observed_sequences,
        blocked=observed_sequences | broad_set,
    )

    near_sequences = {
        record["sequence"]
        for record in near_records
    }

    if len(near_sequences) != len(near_records):
        raise AssertionError(
            "Training-near sampler returned duplicate sequences."
        )

    near_by_sequence = {
        record["sequence"]: record
        for record in near_records
    }

    # ------------------------------------------------------------------
    # Build final sequence specification
    # ------------------------------------------------------------------

    specs: list[dict[str, Any]] = []

    for sequence in sorted(
        observed_sequences,
        key=lambda s: (len(s), s),
    ):
        specs.append(
            {
                "sequence": sequence,
                "sampling_source": "observed_training",
                "parent_sequence": "",
                "mutation_count": 0,
                "parent_hamming_distance": 0,
            }
        )

    for sequence in sorted(
        broad_sequences,
        key=lambda s: (len(s), s),
    ):
        specs.append(
            {
                "sequence": sequence,
                "sampling_source": "broad_coverage",
                "parent_sequence": "",
                "mutation_count": pd.NA,
                "parent_hamming_distance": pd.NA,
            }
        )

    for sequence in sorted(
        near_sequences,
        key=lambda s: (len(s), s),
    ):
        record = near_by_sequence[sequence]
        specs.append(
            {
                "sequence": sequence,
                "sampling_source": "training_near_mutation",
                "parent_sequence": record["parent_sequence"],
                "mutation_count": record["mutation_count"],
                "parent_hamming_distance": record[
                    "parent_hamming_distance"
                ],
            }
        )

    if len(specs) != args.n_targets:
        raise RuntimeError(
            f"Sequence specification size {len(specs)} != "
            f"requested {args.n_targets}."
        )

    spec_sequences = [record["sequence"] for record in specs]

    if len(set(spec_sequences)) != len(spec_sequences):
        raise RuntimeError(
            "Duplicate sequences remain after Stage-2B sampling."
        )

    # ------------------------------------------------------------------
    # Molecule generation
    # ------------------------------------------------------------------

    library_rows = []
    canonical_to_sequences: defaultdict[str, list[str]] = defaultdict(list)

    for i, spec in enumerate(specs):
        sequence = spec["sequence"]
        source = spec["sampling_source"]

        generated_mol = generate_dope_peptide(
            linker,
            sequence,
        )

        generated_conn = canonical_nonisomeric(
            generated_mol
        )
        generated_iso = canonical_isomeric(
            generated_mol
        )

        if source == "observed_training":
            observed_record = observed[sequence]
            final_smiles = observed_record["observed_smiles"]
            final_mol = mol_or_fail(
                final_smiles,
                observed_record["observed_name"],
            )
            final_conn = canonical_nonisomeric(final_mol)

            if final_conn != generated_conn:
                raise RuntimeError(
                    "Observed structure passed reconstruction earlier but "
                    f"failed during final library build: {sequence}"
                )

            fifth_name = observed_record["observed_name"]
            observed_name = observed_record["observed_name"]
            structure_source = "observed_training"
        else:
            # Model ignores stereochemistry, but keep an isomeric valid
            # generated SMILES as the structure representation.
            final_smiles = generated_iso
            final_mol = generated_mol
            final_conn = generated_conn
            fifth_name = f"{sequence}-DOPE"
            observed_name = ""
            structure_source = "generated_standard_L_peptide"

        canonical_to_sequences[final_conn].append(sequence)

        nearest = nearest_observed_hamming(
            sequence,
            observed_sequences,
        )

        library_rows.append(
            {
                "stage2b_id": f"S2B_{i:05d}",
                "Fifth": fifth_name,
                "Fifth_SMILE": final_smiles,
                "canonical_connectivity": final_conn,
                "generated_reference_isomeric_smiles": generated_iso,
                "scaffold_family": "DOPE_peptide",
                "scaffold_variant": "unmodified",
                "sequence": sequence,
                "sequence_length": len(sequence),
                "terminal_residue": sequence[-1],
                "variable_sequence": sequence[:-1],
                "sampling_source": source,
                "structure_source": structure_source,
                "observed_in_training": bool(
                    source == "observed_training"
                ),
                "observed_name": observed_name,
                "parent_sequence": spec["parent_sequence"],
                "mutation_count": spec["mutation_count"],
                "parent_hamming_distance": spec[
                    "parent_hamming_distance"
                ],
                "nearest_observed_hamming_same_length": nearest,
                "contains_training_unseen_AA": False,  # filled later
                "internal_cysteine_count": int(
                    sequence[:-1].count("C")
                ),
                "formal_charge": int(
                    Chem.GetFormalCharge(final_mol)
                ),
                "mol_wt": float(
                    Descriptors.MolWt(final_mol)
                ),
                "heavy_atom_count": int(
                    rdMolDescriptors.CalcNumHeavyAtoms(
                        final_mol
                    )
                ),
                "hbond_donors": int(
                    Lipinski.NumHDonors(final_mol)
                ),
                "hbond_acceptors": int(
                    Lipinski.NumHAcceptors(final_mol)
                ),
                "rdkit_valid": True,
            }
        )

    library = pd.DataFrame(library_rows)

    # Two distinct novelty definitions are useful:
    # 1) AA absent from the observed DOPE-peptide VARIABLE positions.
    # 2) AA absent from all reliably parsed Stage-1 peptide/single sequences.
    observed_dope_variable_aas = set(
        "".join(seq[:-1] for seq in observed_sequences)
    )

    stage1_seen_aas: set[str] = set()
    if "sequence" in row_audit.columns:
        for value in row_audit["sequence"].dropna():
            sequence = clean(value).upper()
            if sequence and set(sequence).issubset(AA_SET):
                stage1_seen_aas.update(sequence)

    # The observed DOPE nomenclature parser is itself a frozen high-confidence
    # Stage-1 rule. Union these residues so novelty accounting remains correct
    # even if an accidentally stale row-audit file predates the one-letter
    # DOPE sequence parser.
    for sequence in observed_sequences:
        stage1_seen_aas.update(sequence)

    library["contains_dope_unseen_variable_AA"] = library[
        "variable_sequence"
    ].map(
        lambda seq: bool(
            set(seq).difference(observed_dope_variable_aas)
        )
    )

    library["contains_stage1_unseen_AA"] = library[
        "sequence"
    ].map(
        lambda seq: bool(
            set(seq).difference(stage1_seen_aas)
        )
    )

    # ------------------------------------------------------------------
    # Dedup / validity hard gates
    # ------------------------------------------------------------------

    if len(library) != args.n_targets:
        raise RuntimeError(
            f"Final library size {len(library)} != {args.n_targets}."
        )

    if library["sequence"].nunique() != args.n_targets:
        raise RuntimeError(
            "Final library does not contain unique sequences."
        )

    duplicate_structures = {
        canonical: sequences
        for canonical, sequences in canonical_to_sequences.items()
        if len(sequences) > 1
    }

    if duplicate_structures:
        rows = []
        for canonical, sequences in duplicate_structures.items():
            for sequence in sequences:
                rows.append(
                    {
                        "canonical_connectivity": canonical,
                        "sequence": sequence,
                    }
                )

        pd.DataFrame(rows).to_csv(
            outdir / "stage2b_duplicate_structures.csv",
            index=False,
        )

        raise SystemExit(
            "Stage 2B BLOCKED: distinct sequences collapsed to duplicate "
            "non-isomeric molecular connectivity."
        )

    if not library["sequence"].str.endswith("C").all():
        raise SystemExit(
            "Stage 2B BLOCKED: at least one sequence does not terminate in C."
        )

    if not library["rdkit_valid"].all():
        raise SystemExit(
            "Stage 2B BLOCKED: at least one generated structure is invalid."
        )

    # ------------------------------------------------------------------
    # Coverage tables / hard gates
    # ------------------------------------------------------------------

    length_coverage = make_length_coverage(
        library,
        final_length_targets,
    )
    aa_coverage = make_aa_coverage(library)
    position_coverage = make_position_coverage(
        library,
        final_length_targets,
    )
    sampling_summary = make_sampling_summary(library)

    length_coverage.to_csv(
        outdir / "stage2b_length_coverage.csv",
        index=False,
    )
    aa_coverage.to_csv(
        outdir / "stage2b_amino_acid_coverage.csv",
        index=False,
    )
    position_coverage.to_csv(
        outdir / "stage2b_position_coverage.csv",
        index=False,
    )
    sampling_summary.to_csv(
        outdir / "stage2b_sampling_summary.csv",
        index=False,
    )

    length_target_failures = length_coverage.loc[
        ~length_coverage["target_match"]
    ]

    if not length_target_failures.empty:
        length_target_failures.to_csv(
            outdir / "stage2b_length_target_failures.csv",
            index=False,
        )
        raise SystemExit(
            "Stage 2B BLOCKED: final per-length counts do not match the "
            "frozen target profile. See stage2b_length_target_failures.csv."
        )

    position_failures = position_coverage.loc[
        ~position_coverage["coverage_pass"]
    ]

    if not position_failures.empty:
        position_failures.to_csv(
            outdir / "stage2b_position_coverage_failures.csv",
            index=False,
        )
        raise SystemExit(
            "Stage 2B BLOCKED: amino-acid positional coverage gate failed. "
            "See stage2b_position_coverage_failures.csv."
        )

    # Every canonical AA must occur in at least one NON-terminal position.
    variable_counts = aa_coverage.set_index("aa1")[
        "all_variable_position_occurrences"
    ]

    missing_variable_aa = [
        aa
        for aa in AA1
        if int(variable_counts.loc[aa]) == 0
    ]

    if missing_variable_aa:
        raise SystemExit(
            "Stage 2B BLOCKED: canonical AA absent from variable peptide "
            "positions: "
            + ", ".join(missing_variable_aa)
        )

    min_variable_count = int(variable_counts.min())
    max_variable_count = int(variable_counts.max())
    variable_aa_imbalance_ratio = (
        float(max_variable_count / min_variable_count)
        if min_variable_count > 0
        else float("inf")
    )

    if variable_aa_imbalance_ratio > args.max_variable_aa_imbalance:
        raise SystemExit(
            "Stage 2B BLOCKED: variable-position AA imbalance ratio "
            f"{variable_aa_imbalance_ratio:.3f} exceeds configured maximum "
            f"{args.max_variable_aa_imbalance:.3f}. Reduce --near-fraction "
            "or increase broad coverage."
        )

    # ------------------------------------------------------------------
    # Save final library only after all gates pass.
    # ------------------------------------------------------------------

    library.to_csv(
        outdir / "stage2b_dope_peptide_library.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    source_counts = (
        library["sampling_source"]
        .value_counts()
        .to_dict()
    )

    observed_length_set = sorted(
        {len(seq) for seq in observed_sequences}
    )

    generated_length_set = sorted(
        set(library["sequence_length"])
    )

    manifest = {
        "stage": "2B_DOPE_peptide_controlled_sampling",
        "row_audit": str(row_audit_path),
        "row_audit_sha256": sha256(row_audit_path),
        "generation_plan": str(plan_path),
        "generation_plan_sha256": sha256(plan_path),
        "seed": args.seed,
        "n_targets": args.n_targets,
        "observed_unmodified_dope_sequences": n_observed,
        "new_generated_sequences": n_new,
        "near_fraction_of_new_requested": args.near_fraction,
        "near_fraction_of_new_actual": float(
            n_near / n_new
        ) if n_new else 0.0,
        "final_length_targets": {
            str(k): int(v)
            for k, v in final_length_targets.items()
        },
        "near_length_quotas": {
            str(k): int(v)
            for k, v in near_quotas.items()
        },
        "max_variable_aa_imbalance": args.max_variable_aa_imbalance,
        "actual_variable_aa_imbalance_ratio": variable_aa_imbalance_ratio,
        "stage1_seen_canonical_AA": "".join(
            aa for aa in AA1 if aa in stage1_seen_aas
        ),
        "stage1_unseen_canonical_AA": "".join(
            aa for aa in AA1 if aa not in stage1_seen_aas
        ),
        "observed_dope_variable_AA": "".join(
            aa for aa in AA1 if aa in observed_dope_variable_aas
        ),
        "sampling_source_counts": {
            str(k): int(v)
            for k, v in source_counts.items()
        },
        "broad_length_quotas": {
            str(k): int(v)
            for k, v in broad_quotas.items()
        },
        "sequence_policy": {
            "canonical_amino_acids": AA1,
            "length_min": MIN_LENGTH,
            "length_max": MAX_LENGTH,
            "targeted_lengths": [
                int(length)
                for length, count in final_length_targets.items()
                if int(count) > 0
            ],
            "omitted_lengths": [
                int(length)
                for length, count in final_length_targets.items()
                if int(count) == 0
            ],
            "terminal_residue": "C",
            "terminal_reason": (
                "Observed unmodified DOPE-peptide scaffold attaches through "
                "a disulfide bond to the terminal Cys side-chain sulfur."
            ),
            "internal_positions": (
                "All 20 canonical amino acids are allowed."
            ),
            "observed_lengths": observed_length_set,
            "generated_lengths": generated_length_set,
        },
        "sampling_policy": {
            "observed": (
                "Retain all observed unmodified DOPE-peptide structures."
            ),
            "broad": (
                "Explicit OOD-bridge final-length profile. L2/L3 are "
                "exhaustive; L4/L5 remain dense; L6/L7 receive the largest "
                "quotas; L8 is omitted by default; L9 is a sparse anchor. "
                "Within each length, broad samples use canonical-AA sampling "
                "in variable positions."
            ),
            "training_near": (
                "Mutate 1 residue with probability 0.70 or 2 residues with "
                "probability 0.30 among non-terminal positions of observed "
                "sequences; preserve sequence length and terminal C."
            ),
        },
        "linker_template": {
            "source_observed_name": linker_source_name,
            "extraction": (
                "Cut unique S-S bond; retain phosphorus-containing fragment; "
                "reconnect terminal linker sulfur to terminal Cys SG."
            ),
        },
        "reconstruction_gate": {
            "observed_sequences_tested": int(len(reconstruction)),
            "connectivity_matches": int(
                reconstruction["connectivity_match"].sum()
            ),
            "rate": float(
                reconstruction["connectivity_match"].mean()
            ),
            "identity": "canonical_nonisomeric_smiles",
        },
        "coverage_gates": {
            "final_length_targets_exact": True,
            "all_20_AA_in_each_variable_position_by_targeted_length": True,
            "terminal_position_is_C": True,
            "unique_sequences": True,
            "unique_nonisomeric_connectivity": True,
            "all_rdkit_valid": True,
            "variable_AA_imbalance_within_limit": True,
        },
        "length_profile_methodology": {
            "profile_name": "OOD_bridge_2to7_sparse9",
            "default_10k_targets": {
                str(k): int(v)
                for k, v in DEFAULT_10K_FINAL_LENGTH_TARGETS.items()
            },
            "rationale": (
                "Do not interpret training Lmax=9 as a reason for uniform "
                "2-9 sampling. Concentrate pretraining on dense L4/L5 and "
                "intermediate L6/L7 bridge lengths; omit L8 by default and "
                "retain only a sparse L9 anchor."
            ),
            "external_validation_informed": True,
            "note": (
                "The profile choice was informed by the project's observed "
                "downstream validation length regime. No validation sequence "
                "or validation label is read by this script. For a strictly "
                "training-only external benchmark, use an independently frozen "
                "--length-targets policy."
            ),
        },
        "stereochemistry_policy": (
            "Not used as a hard gate because the downstream GraphGPS does "
            "not encode stereochemistry."
        ),
    }

    with (outdir / "stage2b_manifest.json").open(
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

    print("=" * 88)
    print("STAGE 2B — DOPE-PEPTIDE CONTROLLED GENERATION")
    print("=" * 88)
    print(f"Observed unmodified DOPE sequences: {n_observed}")
    print(
        "Connectivity reconstruction:      "
        f"{int(reconstruction['connectivity_match'].sum())}/"
        f"{len(reconstruction)} "
        f"({reconstruction['connectivity_match'].mean():.3f})"
    )
    print(f"Total target structures:           {len(library)}")
    print(f"Observed structures retained:      {source_counts.get('observed_training', 0)}")
    print(f"Broad coverage generated:          {source_counts.get('broad_coverage', 0)}")
    print(f"Training-near generated:           {source_counts.get('training_near_mutation', 0)}")
    print()
    print("Final length target profile:")
    for length in range(MIN_LENGTH, MAX_LENGTH + 1):
        print(
            f"  L={length}: target={final_length_targets[length]:4d} "
            f"observed={sum(len(seq) == length for seq in observed_sequences):3d} "
            f"near={near_quotas[length]:4d} "
            f"broad={broad_quotas[length]:4d}"
        )
    print()
    print("Final length coverage:")
    print(
        length_coverage[
            [
                "sequence_length",
                "target_sequences",
                "total_sequences",
                "observed_training",
                "broad_coverage",
                "training_near_mutation",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        "Variable-position canonical AA coverage: "
        f"{int((aa_coverage['all_variable_position_occurrences'] > 0).sum())}/20"
    )
    print(
        "Position-by-length coverage gates:       "
        f"{int(position_coverage['coverage_pass'].sum())}/"
        f"{len(position_coverage)} passed"
    )
    print(
        "Variable-position AA imbalance ratio:     "
        f"{variable_aa_imbalance_ratio:.3f} "
        f"(limit {args.max_variable_aa_imbalance:.3f})"
    )
    print(
        "Sequences containing AA absent from observed DOPE variable positions: "
        f"{int(library['contains_dope_unseen_variable_AA'].sum())}"
    )
    print(
        "Sequences containing AA unseen anywhere in Stage 1: "
        f"{int(library['contains_stage1_unseen_AA'].sum())}"
    )
    print()
    print(f"Results written to:\n  {outdir}")
    print()
    print("Inspect next:")
    print(f"  {outdir / 'stage2b_reconstruction_audit.csv'}")
    print(f"  {outdir / 'stage2b_length_coverage.csv'}")
    print(f"  {outdir / 'stage2b_amino_acid_coverage.csv'}")
    print(f"  {outdir / 'stage2b_position_coverage.csv'}")
    print(f"  {outdir / 'stage2b_dope_peptide_library.csv'}")
    print()
    print("STAGE 2B PASSED all reconstruction, validity, uniqueness, and coverage gates.")


if __name__ == "__main__":
    main()
