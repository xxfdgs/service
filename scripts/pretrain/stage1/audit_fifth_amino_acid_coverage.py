#!/usr/bin/env python3
"""
Stage 1 audit for Fifth-component amino-acid / peptide coverage.

This version is tailored to the nomenclature observed in the current
20260812-sum-700.csv dataset while remaining conservative.

Automatically supported high-confidence rules
----------------------------------------------
1) Standard 3-letter amino-acid names, e.g.
       Phe-UC18     -> F
       Tyr-UC12     -> Y
       Asp12-COOH   -> D

2) One-letter peptide + DOPE names, e.g.
       DC-DOPE      -> DC
       DDSC-DOPE    -> DDSC
       DRDRC-DOPE   -> DRDRC
       RSSC-DOPE    -> RSSC

3) Repeat notation observed in the dataset, e.g.
       4DC-DOPE     -> DDDDC
       8DC-DOPE     -> DDDDDDDDC
       4RC-DOPE     -> RRRRC
       8RC-DOPE     -> RRRRRRRRC

4) Explicit fixed DSSC scaffolds observed in the dataset:
       HA-DSSC      -> DSSC
       SQWS-DSSC    -> DSSC

Explicitly excluded from peptide parsing
----------------------------------------
Names such as S-C4, S-C6, S-C8, S-C10, S-C12, S-Boc, S-COOH and
S-NH2 are treated as non-peptide naming conventions.  Their "S" is not
interpreted as serine.

Conservative unresolved policy
------------------------------
Potentially ambiguous names such as DOPE-C-Ome / DOPE-D-OMe /
DOPE-DC-Ome are NOT assigned a peptide sequence automatically in this
version.  If their sequence is confirmed, provide an exact manual mapping
CSV with columns:

    Fifth,sequence
    DOPE-C-Ome,DSSC

Only non-empty, 20-standard-AA one-letter sequences are loaded from the
manual mapping file. Blank template rows are ignored.

Outputs
-------
row_level_fifth_audit.csv
unique_fifth_inventory.csv
amino_acid_coverage.csv
sequence_coverage.csv
peptide_length_coverage.csv
position_coverage.csv
unresolved_fifths.csv
manual_sequence_mapping_template.csv
audit_summary.json

Optional outputs if problems are found
--------------------------------------
invalid_fifth_smiles.csv
sequence_conflicts.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem


# =============================================================================
# Amino-acid vocabulary
# =============================================================================

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

AA1_TO_AA3 = {v: k for k, v in AA3_TO_AA1.items()}
STANDARD_AA1 = set(AA1_TO_AA3)

AA_FULL_NAME = {
    "A": "Alanine",
    "R": "Arginine",
    "N": "Asparagine",
    "D": "Aspartic acid",
    "C": "Cysteine",
    "Q": "Glutamine",
    "E": "Glutamic acid",
    "G": "Glycine",
    "H": "Histidine",
    "I": "Isoleucine",
    "L": "Leucine",
    "K": "Lysine",
    "M": "Methionine",
    "F": "Phenylalanine",
    "P": "Proline",
    "S": "Serine",
    "T": "Threonine",
    "W": "Tryptophan",
    "Y": "Tyrosine",
    "V": "Valine",
}

# Noncanonical residue names that must not be silently mapped to a canonical AA.
KNOWN_NONCANONICAL_TOKENS = {
    "Phg",  # phenylglycine; not canonical phenylalanine
}

# Explicit high-confidence peptide mappings from the current dataset's naming
# conventions. Keep this small and auditable.
FIXED_SEQUENCE_MAPPING = {
    "HA-DSSC": "DSSC",
    "SQWS-DSSC": "DSSC",
}

# Names whose apparent one-letter amino-acid letters are part of another
# chemical naming convention. These are explicitly non-peptide for this audit.
EXPLICIT_NONPEPTIDE_PATTERNS = [
    re.compile(r"^S-C(?:4|6|8|10|12)$", re.IGNORECASE),
    re.compile(r"^S-(?:Boc|COOH|NH2)$", re.IGNORECASE),
]

# Names that are known to need manual confirmation before assigning a sequence.
# They remain unresolved unless supplied in --manual-mapping.
AMBIGUOUS_SPECIAL_PATTERNS = [
    re.compile(r"^DOPE-(?:C|D|DC)-(?:OMe|Ome)$", re.IGNORECASE),
]

# Canonical one-letter sequence alphabet in a regex-safe deterministic order.
AA1_REGEX = "ACDEFGHIKLMNPQRSTVWY"


# =============================================================================
# Generic utilities
# =============================================================================

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_robust(path: Path) -> pd.DataFrame:
    failures = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            frame = pd.read_csv(path, encoding=encoding, dtype={"ID": str})
            print(f"[INFO] Read CSV with encoding={encoding}")
            return frame
        except UnicodeDecodeError as exc:
            failures.append(f"{encoding}: {exc}")
    raise UnicodeError("Unable to decode input CSV.\n" + "\n".join(failures))


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )


def is_absent_name(name: str) -> bool:
    return name.strip().lower() in {"", "0", "0.0", "nan", "none"}


def canonical_fifth_smiles(
    fifth_name: object,
    fifth_smiles: object,
) -> tuple[str, bool, str]:
    """Return canonical_smiles, rdkit_valid, error."""
    name = clean_text(fifth_name)
    smiles = clean_text(fifth_smiles)

    if is_absent_name(name):
        return "[Fr]", True, ""

    # A named Fifth with no SMILES is not equivalent to an absent Fifth.
    if smiles.lower() in {"", "nan", "none"}:
        return "", False, f"Named Fifth {name!r} has missing Fifth_SMILE"

    if smiles == "[Fr]":
        return "[Fr]", True, ""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", False, f"RDKit failed to parse: {smiles}"

    canonical = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )
    return canonical, True, ""


def normalize_class(value: object) -> str:
    return clean_text(value).lower()


# =============================================================================
# Manual mapping
# =============================================================================

def validate_manual_sequence(sequence: object) -> str:
    sequence = clean_text(sequence).upper().replace("-", "").replace(" ", "")
    if not sequence:
        return ""

    bad = sorted(set(sequence).difference(STANDARD_AA1))
    if bad:
        raise ValueError(
            f"Manual sequence {sequence!r} contains unsupported residue codes: "
            f"{bad}. Only the 20 standard amino acids are accepted."
        )
    return sequence


def load_manual_mapping(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    required = {"Fifth", "sequence"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manual mapping file misses columns: {sorted(missing)}")

    mapping: dict[str, str] = {}
    for _, row in frame.iterrows():
        name = clean_text(row["Fifth"])
        sequence = validate_manual_sequence(row["sequence"])

        # Blank template rows are intentionally ignored.
        if not name or not sequence:
            continue

        if name in mapping and mapping[name] != sequence:
            raise ValueError(
                f"Conflicting manual mappings for Fifth={name!r}: "
                f"{mapping[name]!r} vs {sequence!r}"
            )
        mapping[name] = sequence

    print(f"[INFO] Loaded {len(mapping)} manual Fifth->sequence mappings")
    return mapping


# =============================================================================
# Nomenclature parsing helpers
# =============================================================================

def detect_noncanonical_tokens(name: str) -> list[str]:
    hits = []
    for token in sorted(KNOWN_NONCANONICAL_TOKENS):
        pattern = re.compile(
            rf"(?<![A-Za-z]){re.escape(token)}(?![A-Za-z])",
            flags=re.IGNORECASE,
        )
        if pattern.search(name):
            hits.append(token)
    return hits


def extract_standard_three_letter_tokens(name: str) -> list[tuple[int, str, str]]:
    """
    Extract ordered standard amino-acid 3-letter tokens.

    Letter boundaries are required so Phg is not mistaken for Phe, and arbitrary
    scaffold text is not interpreted as an amino-acid sequence.
    """
    hits: list[tuple[int, str, str]] = []
    for aa3, aa1 in AA3_TO_AA1.items():
        pattern = re.compile(
            rf"(?<![A-Za-z]){re.escape(aa3)}(?![A-Za-z])",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(name):
            hits.append((match.start(), aa1, aa3))
    hits.sort(key=lambda item: item[0])
    return hits


def infer_name_metadata(name: str) -> tuple[str, str]:
    """Best-effort name-only scaffold/tail family and terminal modification."""
    lower = name.lower()

    tail_family = ""
    modification = ""

    if "dope" in lower:
        tail_family = "DOPE"
    elif name in FIXED_SEQUENCE_MAPPING:
        tail_family = name.split("-", 1)[0]
    else:
        m = re.search(r"-(UC\d+)$", name, flags=re.IGNORECASE)
        if m:
            tail_family = m.group(1).upper()
        else:
            m = re.match(r"^S-(C\d+)$", name, flags=re.IGNORECASE)
            if m:
                tail_family = m.group(1).upper()

    if re.search(r"-(?:OMe|Ome)$", name, flags=re.IGNORECASE):
        modification = "OMe"
    elif re.search(r"-COOH$", name, flags=re.IGNORECASE):
        modification = "COOH"
    elif re.search(r"-NH2$", name, flags=re.IGNORECASE):
        modification = "NH2"
    elif re.search(r"-Boc$", name, flags=re.IGNORECASE):
        modification = "Boc"

    return tail_family, modification


def make_parse_result(
    *,
    sequence: str = "",
    parse_status: str,
    parse_reason: str,
    parser_rule: str,
    parser_confidence: str,
    noncanonical_tokens: str = "",
    peptide_category: str,
    tail_family: str = "",
    modification: str = "",
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "sequence_length": len(sequence) if sequence else pd.NA,
        "parse_status": parse_status,
        "parse_reason": parse_reason,
        "parser_rule": parser_rule,
        "parser_confidence": parser_confidence,
        "peptide_category": peptide_category,
        "noncanonical_tokens": noncanonical_tokens,
        "tail_family": tail_family,
        "modification": modification,
    }


def parse_fifth_name(
    name: str,
    fifth_class: str,
    manual_mapping: dict[str, str],
) -> dict[str, object]:
    """Conservative, ordered parser for the current Fifth nomenclature."""
    tail_family, modification = infer_name_metadata(name)

    # ------------------------------------------------------------------
    # Rule 0: exact manual mapping always wins.
    # ------------------------------------------------------------------
    if name in manual_mapping:
        seq = manual_mapping[name]
        return make_parse_result(
            sequence=seq,
            parse_status="manual",
            parse_reason="exact manual Fifth-name mapping",
            parser_rule="manual_exact",
            parser_confidence="manual",
            peptide_category="canonical_peptide",
            tail_family=tail_family,
            modification=modification,
        )

    # ------------------------------------------------------------------
    # Rule 1: explicit known noncanonical residue names.
    # ------------------------------------------------------------------
    noncanonical = detect_noncanonical_tokens(name)
    if noncanonical:
        return make_parse_result(
            parse_status="noncanonical",
            parse_reason=(
                "known noncanonical residue token(s): " + "|".join(noncanonical)
            ),
            parser_rule="known_noncanonical_token",
            parser_confidence="high",
            peptide_category="noncanonical_residue",
            noncanonical_tokens="|".join(noncanonical),
            tail_family=tail_family,
            modification=modification,
        )

    # ------------------------------------------------------------------
    # Rule 2: explicit non-peptide names where S must NOT mean serine.
    # ------------------------------------------------------------------
    if any(pattern.fullmatch(name) for pattern in EXPLICIT_NONPEPTIDE_PATTERNS):
        return make_parse_result(
            parse_status="nonpeptide",
            parse_reason=(
                "explicit non-peptide naming convention; leading S is not "
                "interpreted as serine"
            ),
            parser_rule="explicit_nonpeptide_S_series",
            parser_confidence="high",
            peptide_category="nonpeptide",
            tail_family=tail_family,
            modification=modification,
        )

    # ------------------------------------------------------------------
    # Rule 3: special names needing confirmation stay unresolved.
    # ------------------------------------------------------------------
    if any(pattern.fullmatch(name) for pattern in AMBIGUOUS_SPECIAL_PATTERNS):
        return make_parse_result(
            parse_status="unresolved",
            parse_reason=(
                "special DOPE/OMe nomenclature requires explicit manual "
                "sequence confirmation"
            ),
            parser_rule="ambiguous_dope_ome",
            parser_confidence="none",
            peptide_category="unresolved",
            tail_family=tail_family,
            modification=modification,
        )

    # ------------------------------------------------------------------
    # Rule 4: exact fixed DSSC scaffold names.
    # ------------------------------------------------------------------
    if name in FIXED_SEQUENCE_MAPPING:
        seq = FIXED_SEQUENCE_MAPPING[name]
        return make_parse_result(
            sequence=seq,
            parse_status="auto",
            parse_reason=f"exact fixed scaffold mapping: {name} -> {seq}",
            parser_rule="fixed_DSSC_scaffold",
            parser_confidence="high",
            peptide_category="canonical_peptide",
            tail_family=tail_family,
            modification=modification,
        )

    # ------------------------------------------------------------------
    # Rule 5: repeat notation, e.g. 4DC-DOPE -> DDDDC.
    # Pattern is intentionally restricted to '<N><AA>C-DOPE'.
    # ------------------------------------------------------------------
    repeat_match = re.fullmatch(
        rf"(?P<n>\d+)(?P<aa>[{AA1_REGEX}])C-DOPE",
        name,
        flags=re.IGNORECASE,
    )
    if repeat_match:
        repeat_n = int(repeat_match.group("n"))
        aa = repeat_match.group("aa").upper()

        if repeat_n < 1 or repeat_n > 50:
            return make_parse_result(
                parse_status="ambiguous",
                parse_reason=f"implausible repeat count {repeat_n}; manual review required",
                parser_rule="repeat_AAC_DOPE",
                parser_confidence="none",
                peptide_category="unresolved",
                tail_family=tail_family,
                modification=modification,
            )

        seq = aa * repeat_n + "C"
        return make_parse_result(
            sequence=seq,
            parse_status="auto",
            parse_reason=(
                f"repeat notation: {repeat_n}{aa}C-DOPE -> {aa}*{repeat_n}+C"
            ),
            parser_rule="repeat_AAC_DOPE",
            parser_confidence="high",
            peptide_category="canonical_peptide",
            tail_family="DOPE",
            modification=modification,
        )

    # ------------------------------------------------------------------
    # Rule 6: explicit one-letter peptide sequence followed by -DOPE.
    # This is safe because the suffix anchors the grammar and S-C4 etc. do not
    # match this rule.
    # ------------------------------------------------------------------
    peptide_dope_match = re.fullmatch(
        rf"(?P<seq>[{AA1_REGEX}]{{2,}})-DOPE",
        name,
        flags=re.IGNORECASE,
    )
    if peptide_dope_match:
        seq = peptide_dope_match.group("seq").upper()
        return make_parse_result(
            sequence=seq,
            parse_status="auto",
            parse_reason="explicit one-letter peptide sequence before -DOPE",
            parser_rule="one_letter_peptide_DOPE",
            parser_confidence="high",
            peptide_category="canonical_peptide",
            tail_family="DOPE",
            modification=modification,
        )

    # ------------------------------------------------------------------
    # Rule 7: standard 3-letter amino-acid tokens.
    # Handles Phe-UC18, Tyr-UC12, Asp12-COOH, etc.
    # ------------------------------------------------------------------
    hits = extract_standard_three_letter_tokens(name)
    if hits:
        seq = "".join(aa1 for _, aa1, _ in hits)

        # Multiple explicit 3-letter tokens are accepted in name order. This
        # remains much safer than arbitrary one-letter parsing.
        return make_parse_result(
            sequence=seq,
            parse_status="auto",
            parse_reason=(
                "ordered standard three-letter amino-acid token(s): "
                + "-".join(aa3 for _, _, aa3 in hits)
            ),
            parser_rule="three_letter_AA_tokens",
            parser_confidence="high",
            peptide_category="canonical_peptide",
            tail_family=tail_family,
            modification=modification,
        )

    # ------------------------------------------------------------------
    # No reliable rule matched.
    # ------------------------------------------------------------------
    return make_parse_result(
        parse_status="unresolved",
        parse_reason="no high-confidence nomenclature rule matched",
        parser_rule="none",
        parser_confidence="none",
        peptide_category="unresolved",
        tail_family=tail_family,
        modification=modification,
    )


# =============================================================================
# Coverage summaries
# =============================================================================

def valid_sequence_rows(parsed: pd.DataFrame) -> pd.DataFrame:
    return parsed.loc[
        parsed["sequence"].fillna("").astype(str).ne("")
    ].copy()


def make_aa_coverage(parsed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = valid_sequence_rows(parsed)

    for aa1 in sorted(STANDARD_AA1):
        row: dict[str, object] = {
            "aa1": aa1,
            "aa3": AA1_TO_AA3[aa1],
            "full_name": AA_FULL_NAME[aa1],
        }

        for fifth_class in ("single", "double"):
            subset = valid.loc[
                valid["Fifth_class_canonical"].eq(fifth_class)
            ]
            contains = subset["sequence"].astype(str).str.contains(
                aa1, regex=False, na=False
            )

            row[f"{fifth_class}_rows_containing"] = int(contains.sum())
            row[f"{fifth_class}_unique_fifths_containing"] = int(
                subset.loc[contains, "canonical_fifth"].nunique()
            )
            # Python sum avoids pandas object/empty-series dtype corner cases.
            row[f"{fifth_class}_residue_occurrences"] = sum(
                str(seq).count(aa1)
                for seq in subset["sequence"].fillna("")
            )

        contains_all = valid["sequence"].astype(str).str.contains(
            aa1, regex=False, na=False
        )
        row["total_rows_containing"] = int(contains_all.sum())
        row["total_unique_fifths_containing"] = int(
            valid.loc[contains_all, "canonical_fifth"].nunique()
        )
        row["total_residue_occurrences"] = sum(
            str(seq).count(aa1)
            for seq in valid["sequence"].fillna("")
        )
        row["seen"] = bool(row["total_residue_occurrences"] > 0)
        rows.append(row)

    return pd.DataFrame(rows)


def make_sequence_coverage(parsed: pd.DataFrame) -> pd.DataFrame:
    valid = valid_sequence_rows(parsed)
    columns = [
        "Fifth_class",
        "sequence",
        "sequence_length",
        "rows",
        "unique_fifth_identities",
        "unique_fifth_names",
        "Fifth_names",
        "parser_rules",
    ]
    if valid.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (fifth_class, sequence), group in valid.groupby(
        ["Fifth_class_canonical", "sequence"],
        sort=True,
        dropna=False,
    ):
        rows.append(
            {
                "Fifth_class": fifth_class,
                "sequence": sequence,
                "sequence_length": len(sequence),
                "rows": len(group),
                "unique_fifth_identities": group["canonical_fifth"].nunique(),
                "unique_fifth_names": group["Fifth"].nunique(),
                "Fifth_names": "|".join(sorted(group["Fifth"].astype(str).unique())),
                "parser_rules": "|".join(sorted(group["parser_rule"].astype(str).unique())),
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["Fifth_class", "sequence_length", "sequence"]
    )


def make_length_coverage(parsed: pd.DataFrame) -> pd.DataFrame:
    valid = valid_sequence_rows(parsed)
    rows = []

    # Class-specific summaries.
    for fifth_class in ("single", "double"):
        subset = valid.loc[valid["Fifth_class_canonical"].eq(fifth_class)]
        for length, group in subset.groupby("sequence_length", sort=True):
            rows.append(
                {
                    "scope": fifth_class,
                    "sequence_length": int(length),
                    "rows": len(group),
                    "unique_sequences": group["sequence"].nunique(),
                    "unique_fifth_identities": group["canonical_fifth"].nunique(),
                }
            )

    # Overall peptide summary, because Fifth_class != peptide-length category.
    for length, group in valid.groupby("sequence_length", sort=True):
        rows.append(
            {
                "scope": "all",
                "sequence_length": int(length),
                "rows": len(group),
                "unique_sequences": group["sequence"].nunique(),
                "unique_fifth_identities": group["canonical_fifth"].nunique(),
            }
        )

    return pd.DataFrame(rows)


def make_position_coverage(parsed: pd.DataFrame) -> pd.DataFrame:
    """
    Position coverage for reliably parsed canonical sequences.

    Both 'double' and 'all' scopes are reported because some dataset 'single'
    Fifth entries (e.g. HA-DSSC) contain a multi-residue peptide sequence.
    """
    valid = valid_sequence_rows(parsed)
    rows = []

    for scope in ("double", "all"):
        subset = (
            valid.loc[valid["Fifth_class_canonical"].eq("double")].copy()
            if scope == "double"
            else valid.copy()
        )
        if subset.empty:
            continue

        max_length = int(subset["sequence"].str.len().max())
        for position in range(1, max_length + 1):
            for aa1 in sorted(STANDARD_AA1):
                mask = subset["sequence"].map(
                    lambda seq: len(str(seq)) >= position and str(seq)[position - 1] == aa1
                )
                group = subset.loc[mask]
                rows.append(
                    {
                        "scope": scope,
                        "position": position,
                        "aa1": aa1,
                        "aa3": AA1_TO_AA3[aa1],
                        "rows": len(group),
                        "unique_sequences": group["sequence"].nunique(),
                        "unique_fifth_identities": group["canonical_fifth"].nunique(),
                        "seen": bool(len(group) > 0),
                    }
                )

    return pd.DataFrame(rows)


def make_unique_inventory(parsed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for canonical, group in parsed.groupby("canonical_fifth", sort=True):
        sequences = sorted(
            {
                seq
                for seq in group["sequence"].fillna("").astype(str)
                if seq
            }
        )
        statuses = sorted(set(group["parse_status"].astype(str)))
        parser_rules = sorted(set(group["parser_rule"].astype(str)))
        peptide_categories = sorted(set(group["peptide_category"].astype(str)))
        tail_families = sorted(
            {x for x in group["tail_family"].fillna("").astype(str) if x}
        )
        modifications = sorted(
            {x for x in group["modification"].fillna("").astype(str) if x}
        )

        rows.append(
            {
                "canonical_fifth": canonical,
                "Fifth_names": "|".join(sorted(set(group["Fifth"].astype(str)))),
                "Fifth_class_values": "|".join(
                    sorted(set(group["Fifth_class_canonical"].astype(str)))
                ),
                "rows": len(group),
                "sequence": sequences[0] if len(sequences) == 1 else "",
                "sequence_candidates": "|".join(sequences),
                "sequence_length": len(sequences[0]) if len(sequences) == 1 else pd.NA,
                "parse_status_values": "|".join(statuses),
                "parser_rules": "|".join(parser_rules),
                "peptide_categories": "|".join(peptide_categories),
                "tail_families": "|".join(tail_families),
                "modifications": "|".join(modifications),
                "identity_sequence_conflict": bool(
                    group["identity_sequence_conflict"].any()
                ),
                "example_smiles": group["Fifth_SMILE"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Fifth amino-acid and peptide coverage."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manual-mapping",
        type=Path,
        default=None,
        help="Optional CSV with columns Fifth,sequence. Exact mappings override automatic rules.",
    )
    parser.add_argument(
        "--expect-rows",
        type=int,
        default=700,
        help="Expected row count; set <=0 to disable the check.",
    )
    args = parser.parse_args()

    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()

    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_csv_robust(input_csv)

    required = {"ID", "Fifth", "Fifth_SMILE", "Fifth_class"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input CSV misses required columns: {sorted(missing)}")

    if args.expect_rows > 0 and len(data) != args.expect_rows:
        raise ValueError(f"Expected {args.expect_rows} rows, found {len(data)}")

    if data["ID"].isna().any():
        raise ValueError("Input contains missing ID values.")
    if data["ID"].duplicated().any():
        duplicates = data.loc[data["ID"].duplicated(keep=False), "ID"].tolist()
        raise ValueError(f"Input contains duplicate IDs: {duplicates[:20]}")

    manual_mapping = load_manual_mapping(args.manual_mapping)

    data = data.copy()
    data["Fifth_class_canonical"] = data["Fifth_class"].map(normalize_class)

    allowed_classes = {"single", "double", ""}
    unexpected = sorted(set(data["Fifth_class_canonical"]).difference(allowed_classes))
    if unexpected:
        raise ValueError(f"Unexpected Fifth_class values: {unexpected}")

    # -------------------------------------------------------------------------
    # Row-level parse
    # -------------------------------------------------------------------------
    row_results = []
    for source_index, row in data.iterrows():
        name = clean_text(row["Fifth"])
        fifth_class = row["Fifth_class_canonical"]
        canonical, valid, smiles_error = canonical_fifth_smiles(
            row["Fifth"], row["Fifth_SMILE"]
        )

        if canonical == "[Fr]" and is_absent_name(name):
            parsed = make_parse_result(
                parse_status="absent",
                parse_reason="no Fifth component / [Fr]",
                parser_rule="absent",
                parser_confidence="high",
                peptide_category="absent",
            )
        elif not valid:
            parsed = make_parse_result(
                parse_status="invalid_smiles",
                parse_reason=smiles_error,
                parser_rule="not_parsed_due_to_invalid_smiles",
                parser_confidence="none",
                peptide_category="unresolved",
            )
        else:
            parsed = parse_fifth_name(
                name=name,
                fifth_class=fifth_class,
                manual_mapping=manual_mapping,
            )

        row_results.append(
            {
                "source_row_index": int(source_index),
                "ID": str(row["ID"]),
                "Fifth": name,
                "Fifth_SMILE": clean_text(row["Fifth_SMILE"]),
                "Fifth_class": clean_text(row["Fifth_class"]),
                "Fifth_class_canonical": fifth_class,
                "canonical_fifth": canonical,
                "rdkit_valid": valid,
                "rdkit_error": smiles_error,
                **parsed,
            }
        )

    parsed = pd.DataFrame(row_results)

    # Write an early row-level audit even if chemistry validation fails.
    parsed.to_csv(output_dir / "row_level_fifth_audit.csv", index=False)

    invalid = parsed.loc[~parsed["rdkit_valid"]].copy()
    if not invalid.empty:
        invalid.to_csv(output_dir / "invalid_fifth_smiles.csv", index=False)
        raise ValueError(
            f"{len(invalid)} rows contain invalid/missing named Fifth SMILES. "
            "See invalid_fifth_smiles.csv"
        )

    # -------------------------------------------------------------------------
    # Identity-level sequence consistency
    # -------------------------------------------------------------------------
    sequence_sets = (
        parsed.loc[
            parsed["sequence"].fillna("").astype(str).ne("")
            & parsed["canonical_fifth"].ne("[Fr]")
        ]
        .groupby("canonical_fifth")["sequence"]
        .agg(lambda x: sorted(set(x)))
    )
    conflicts = sequence_sets[sequence_sets.map(len) > 1]
    conflict_identities = set(conflicts.index)

    parsed["identity_sequence_conflict"] = parsed["canonical_fifth"].isin(
        conflict_identities
    )
    parsed.to_csv(output_dir / "row_level_fifth_audit.csv", index=False)

    if conflict_identities:
        parsed.loc[parsed["identity_sequence_conflict"]].to_csv(
            output_dir / "sequence_conflicts.csv", index=False
        )

    # -------------------------------------------------------------------------
    # Inventory and unresolved list
    # -------------------------------------------------------------------------
    inventory = make_unique_inventory(parsed)
    inventory.to_csv(output_dir / "unique_fifth_inventory.csv", index=False)

    review_statuses = {"unresolved", "ambiguous", "noncanonical"}
    review_rows = parsed.loc[
        parsed["parse_status"].isin(review_statuses)
        & parsed["canonical_fifth"].ne("[Fr]")
    ].copy()

    if review_rows.empty:
        unresolved = pd.DataFrame(
            columns=[
                "canonical_fifth",
                "Fifth",
                "Fifth_class_canonical",
                "parse_status",
                "parse_reason",
                "parser_rule",
                "peptide_category",
                "tail_family",
                "modification",
                "rows",
                "example_ID",
                "example_smiles",
            ]
        )
    else:
        unresolved = (
            review_rows.groupby(
                [
                    "canonical_fifth",
                    "Fifth",
                    "Fifth_class_canonical",
                    "parse_status",
                    "parse_reason",
                    "parser_rule",
                    "peptide_category",
                    "tail_family",
                    "modification",
                ],
                dropna=False,
            )
            .agg(
                rows=("ID", "size"),
                example_ID=("ID", "first"),
                example_smiles=("Fifth_SMILE", "first"),
            )
            .reset_index()
            .sort_values(
                ["Fifth_class_canonical", "parse_status", "Fifth"]
            )
        )

    unresolved.to_csv(output_dir / "unresolved_fifths.csv", index=False)

    # Manual template includes only entries that truly need review. Explicit
    # non-peptide S-series entries are intentionally excluded.
    template_cols = [
        "Fifth",
        "Fifth_class_canonical",
        "parse_status",
        "parse_reason",
        "parser_rule",
        "example_smiles",
    ]
    manual_template = unresolved[template_cols].drop_duplicates("Fifth").copy()
    manual_template = manual_template.rename(
        columns={"Fifth_class_canonical": "Fifth_class"}
    )
    manual_template.insert(1, "sequence", "")
    manual_template.to_csv(
        output_dir / "manual_sequence_mapping_template.csv", index=False
    )

    # -------------------------------------------------------------------------
    # Coverage outputs
    # -------------------------------------------------------------------------
    aa_coverage = make_aa_coverage(parsed)
    aa_coverage.to_csv(output_dir / "amino_acid_coverage.csv", index=False)

    sequence_coverage = make_sequence_coverage(parsed)
    sequence_coverage.to_csv(output_dir / "sequence_coverage.csv", index=False)

    length_coverage = make_length_coverage(parsed)
    length_coverage.to_csv(output_dir / "peptide_length_coverage.csv", index=False)

    position_coverage = make_position_coverage(parsed)
    position_coverage.to_csv(output_dir / "position_coverage.csv", index=False)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    class_counts = parsed["Fifth_class_canonical"].value_counts(dropna=False).to_dict()
    parse_counts = parsed["parse_status"].value_counts(dropna=False).to_dict()
    rule_counts = parsed["parser_rule"].value_counts(dropna=False).to_dict()

    seen_aas = aa_coverage.loc[aa_coverage["seen"], "aa1"].tolist()
    unseen_aas = aa_coverage.loc[~aa_coverage["seen"], "aa1"].tolist()

    valid = valid_sequence_rows(parsed)
    parsed_double = valid.loc[valid["Fifth_class_canonical"].eq("double")]

    max_double_length = (
        int(parsed_double["sequence"].str.len().max())
        if not parsed_double.empty
        else None
    )
    max_any_length = (
        int(valid["sequence"].str.len().max())
        if not valid.empty
        else None
    )

    seen_by_class = {}
    for fifth_class in ("single", "double"):
        column = f"{fifth_class}_residue_occurrences"
        seen_by_class[fifth_class] = aa_coverage.loc[
            aa_coverage[column] > 0, "aa1"
        ].tolist()

    nonpeptide_unique = int(
        parsed.loc[parsed["parse_status"].eq("nonpeptide"), "canonical_fifth"].nunique()
    )

    summary = {
        "input_csv": str(input_csv),
        "input_sha256": sha256(input_csv),
        "rows": len(parsed),
        "unique_ids": int(parsed["ID"].nunique()),
        "class_row_counts": {str(k): int(v) for k, v in class_counts.items()},
        "unique_fifth_identities": int(parsed["canonical_fifth"].nunique()),
        "unique_named_nonempty_fifths": int(
            parsed.loc[parsed["canonical_fifth"].ne("[Fr]"), "canonical_fifth"].nunique()
        ),
        "parse_status_row_counts": {str(k): int(v) for k, v in parse_counts.items()},
        "parser_rule_row_counts": {str(k): int(v) for k, v in rule_counts.items()},
        "standard_amino_acids_seen": seen_aas,
        "standard_amino_acids_unseen": unseen_aas,
        "number_standard_amino_acids_seen": len(seen_aas),
        "number_standard_amino_acids_unseen": len(unseen_aas),
        "standard_amino_acids_seen_by_fifth_class": seen_by_class,
        "maximum_reliably_parsed_double_sequence_length": max_double_length,
        "maximum_reliably_parsed_sequence_length_any_fifth_class": max_any_length,
        "unresolved_unique_name_entries": int(
            unresolved["Fifth"].nunique() if not unresolved.empty else 0
        ),
        "explicit_nonpeptide_unique_identities": nonpeptide_unique,
        "sequence_conflicting_identities": len(conflict_identities),
        "manual_mapping_used": (
            str(args.manual_mapping.resolve())
            if args.manual_mapping is not None
            else None
        ),
        "parser_policy": (
            "Conservative nomenclature parser: standard 3-letter AA tokens; "
            "anchored one-letter peptide-DOPE grammar; observed repeat notation; "
            "fixed HA-DSSC/SQWS-DSSC mapping; explicit exclusion of S-Cn/S-Boc/"
            "S-COOH/S-NH2; ambiguous DOPE-*-OMe names require manual mapping."
        ),
    }

    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    # -------------------------------------------------------------------------
    # Terminal report
    # -------------------------------------------------------------------------
    print()
    print("=" * 78)
    print("FIFTH AMINO-ACID / PEPTIDE COVERAGE AUDIT")
    print("=" * 78)
    print(f"Input rows:                    {len(parsed)}")
    print(f"Unique Fifth identities:       {parsed['canonical_fifth'].nunique()}")
    print(f"single rows:                   {class_counts.get('single', 0)}")
    print(f"double rows:                   {class_counts.get('double', 0)}")
    print(f"absent/blank-class rows:       {class_counts.get('', 0)}")

    print("\nParse status:")
    for key, value in sorted(parse_counts.items()):
        print(f"  {key:<18} {value}")

    print("\nParser rules:")
    for key, value in sorted(rule_counts.items()):
        print(f"  {key:<34} {value}")

    print()
    print(f"Standard AA seen:              {len(seen_aas)}/20 : {' '.join(seen_aas)}")
    print(f"Standard AA unseen:            {len(unseen_aas)}/20 : {' '.join(unseen_aas)}")
    print(f"Seen in single:                {' '.join(seen_by_class['single'])}")
    print(f"Seen in double:                {' '.join(seen_by_class['double'])}")

    print("\nAA coverage:")
    print(
        aa_coverage[
            [
                "aa1",
                "aa3",
                "single_unique_fifths_containing",
                "double_unique_fifths_containing",
                "single_residue_occurrences",
                "double_residue_occurrences",
                "total_residue_occurrences",
                "seen",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Maximum reliably parsed double sequence length: "
        f"{max_double_length if max_double_length is not None else 'NOT DETERMINED'}"
    )
    print(
        "Maximum reliably parsed sequence length (all classes): "
        f"{max_any_length if max_any_length is not None else 'NOT DETERMINED'}"
    )
    print(f"Explicit non-peptide identities: {nonpeptide_unique}")
    print(
        "Unresolved/review unique Fifth names: "
        f"{summary['unresolved_unique_name_entries']}"
    )
    print(f"Sequence-conflicting identities: {len(conflict_identities)}")

    print(f"\nResults written to:\n  {output_dir}")
    print(
        "\nInspect next:\n"
        f"  {output_dir / 'audit_summary.json'}\n"
        f"  {output_dir / 'amino_acid_coverage.csv'}\n"
        f"  {output_dir / 'peptide_length_coverage.csv'}\n"
        f"  {output_dir / 'unresolved_fifths.csv'}"
    )


if __name__ == "__main__":
    main()
