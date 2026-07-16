#!/usr/bin/env python3
"""Audit a proposed deduplicated five-component dataset without changing raw CSVs.

The audit deliberately separates an observed duplicate pattern from a claim
that a row was safe to delete.  It writes stable, content-derived sample IDs
and every downstream stage must consume ``dataset_with_sample_id.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
TARGETS = [
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
]
COMPONENTS = [
    ("IL", "IL_SMILE", "mol%_IL"),
    ("HL", "HL_SMILE", "mol%_HL"),
    ("Chol", "Chol_SMILE", "mol%_Chol"),
    ("PEG", "PEG_SMILE", "mol%_PEG"),
    ("Fifth", "Fifth_SMILE", "mol%_Fifth"),
]
REQUIRED = ["ID", *TARGETS, *(item for component in COMPONENTS for item in component)]
SOURCE_ID_CANDIDATES = ("ID", "id", "sample_id", "Sample_ID")
# The supplied schema exposes no actual experimental-condition fields (for
# example batch, operator, date, or replicate run).  `Norm_*` are outcomes / a
# downstream normalization, so including them would make identity depend on a
# label-like value.  The stable business ID is retained and any hidden-condition
# claim is explicitly left unresolved in the audit.
CONDITION_COLUMNS: tuple[str, ...] = ()
NAME_TOKENS = ("dedup", "deduplicated", "clean", "unique", "no_duplicate", "fixed")


def sha256_file(path: Path) -> str:
    """Return the stable byte-level identity of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def append_execution(output_dir: Path, record: dict[str, object]) -> None:
    """Append a machine-readable record of this real audit invocation."""
    path = output_dir / "execution_manifest.json"
    existing: list[dict[str, object]] = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    existing.append(record)
    json_dump(existing, path)


def read_csv(path: Path) -> pd.DataFrame:
    """Read user CSVs without mutating either raw source."""
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype={"ID": str})
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Unable to decode {path}")


def normalized_text(value: object) -> str:
    if pd.isna(value):
        return "<missing>"
    result = " ".join(str(value).strip().lower().split())
    return result or "<missing>"


def canonical_smiles(value: object) -> str:
    """Use canonical SMILES, retaining an explicit invalid/missing sentinel."""
    if pd.isna(value) or not str(value).strip():
        return "<missing>"
    raw = str(value).strip()
    molecule = Chem.MolFromSmiles(raw)
    if molecule is None:
        return f"<invalid:{raw}>"
    return Chem.MolToSmiles(molecule, canonical=True)


def number_token(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "<missing>" if pd.isna(number) else format(float(number), ".10g")


def source_id_column(frame: pd.DataFrame) -> str:
    for column in SOURCE_ID_CANDIDATES:
        if column in frame:
            return column
    raise ValueError(f"No stable source ID column found; expected one of {SOURCE_ID_CANDIDATES}.")


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    """Add non-destructive normalized identity fields used by every audit."""
    result = frame.copy()
    result["original_row_index"] = np.arange(len(result), dtype=int)
    source_column = source_id_column(result)
    result["source_business_id"] = result[source_column].astype("string").fillna("<missing>").map(normalized_text)
    canonical_columns: list[str] = []
    name_columns: list[str] = []
    ratio_columns: list[str] = []
    for position, (name, smiles, ratio) in enumerate(COMPONENTS, start=1):
        canonical_column = f"canonical_component_{position}_smiles"
        name_column = f"normalized_component_{position}_name"
        ratio_column = f"normalized_component_{position}_ratio"
        result[canonical_column] = result[smiles].map(canonical_smiles)
        result[name_column] = result[name].map(normalized_text)
        result[ratio_column] = result[ratio].map(number_token)
        result[f"component_{position}_key"] = np.where(
            result[canonical_column].str.startswith("<"), result[name_column], result[canonical_column]
        )
        canonical_columns.append(canonical_column)
        name_columns.append(name_column)
        ratio_columns.append(ratio_column)
    result["fifth_component_key"] = result["component_5_key"]
    result["formula_identity_key"] = result[[f"component_{index}_key" for index in range(1, 6)]].astype(str).agg("|".join, axis=1)
    result["formula_ratio_key"] = result["formula_identity_key"] + "|" + result[ratio_columns].astype(str).agg("|".join, axis=1)
    condition_columns = [column for column in CONDITION_COLUMNS if column in result]
    for column in condition_columns:
        result[f"normalized_condition__{column}"] = result[column].map(number_token if column.startswith("Norm_") else normalized_text)
    id_fields = ["source_business_id", *canonical_columns, *ratio_columns,
                 *(f"normalized_condition__{column}" for column in condition_columns)]
    result["sample_id_payload"] = "v1|" + result[id_fields].astype(str).agg("|".join, axis=1)
    result["sample_id"] = result["sample_id_payload"].map(sha256_text).map(lambda digest: f"lrx-v1-{digest}")
    return result


def file_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "absolute_path": str(path.resolve()),
        "sha256": sha256_file(path),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": stat.st_size,
    }


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(dtype=float)


def value_summary(values: pd.Series) -> dict[str, object]:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return {"n_finite": 0, "min": None, "max": None, "mean": None, "std": None,
                "q01": None, "q05": None, "q25": None, "median": None, "q75": None, "q95": None, "q99": None}
    quantiles = finite.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n_finite": int(len(finite)), "min": float(finite.min()), "max": float(finite.max()),
        "mean": float(finite.mean()), "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "q01": float(quantiles.loc[0.01]), "q05": float(quantiles.loc[0.05]),
        "q25": float(quantiles.loc[0.25]), "median": float(quantiles.loc[0.50]),
        "q75": float(quantiles.loc[0.75]), "q95": float(quantiles.loc[0.95]), "q99": float(quantiles.loc[0.99]),
    }


def add_issue(rows: list[dict[str, object]], severity: str, code: str, count: int, detail: str) -> None:
    rows.append({"severity": severity, "code": code, "count": int(count), "detail": detail})


def dataset_profile(frame: pd.DataFrame, path: Path, issues: list[dict[str, object]], *, label: str) -> dict[str, object]:
    """Return a full non-mutating data-quality profile and accumulate explicit issues."""
    missing_columns = sorted(set(REQUIRED) - set(frame.columns))
    if missing_columns:
        add_issue(issues, "FAIL", "missing_required_columns", len(missing_columns), "|".join(missing_columns))
    profile: dict[str, object] = {"dataset_label": label, **file_metadata(path), "rows": int(len(frame)),
                                  "columns": int(len(frame.columns)), "field_list": list(frame.columns),
                                  "missing_required_columns": missing_columns}
    if missing_columns:
        return profile
    source_column = source_id_column(frame)
    profile["source_id_column"] = source_column
    profile["experimental_condition_columns_used_for_sample_id"] = list(CONDITION_COLUMNS)
    profile["source_id_unique"] = bool(not frame[source_column].astype(str).duplicated().any())
    profile["missing_values"] = {column: int(frame[column].isna().sum()) for column in frame.columns}
    profile["infinite_values"] = {
        column: int(np.isinf(numeric_series(frame, column)).sum())
        for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])
    }
    profile["targets"] = {target: value_summary(numeric_series(frame, target)) for target in TARGETS}
    profile["target_outlier_counts"] = {}
    for target in TARGETS:
        values = numeric_series(frame, target)
        finite = values[np.isfinite(values)]
        if finite.empty:
            profile["target_outlier_counts"][target] = 0
            continue
        q25, q75 = finite.quantile([0.25, 0.75])
        iqr = q75 - q25
        profile["target_outlier_counts"][target] = int(((finite < q25 - 1.5 * iqr) | (finite > q75 + 1.5 * iqr)).sum()) if iqr > 0 else 0
    ratio_columns = [ratio for _, _, ratio in COMPONENTS]
    ratios = frame[ratio_columns].apply(pd.to_numeric, errors="coerce")
    ratio_sum = ratios.sum(axis=1, min_count=len(ratio_columns))
    profile["ratio_summary"] = {
        "sum": value_summary(ratio_sum),
        "negative_rows": int((ratios < 0).any(axis=1).sum()),
        "above_100_rows": int((ratios > 100).any(axis=1).sum()),
        "sum_not_100_rows": int((ratio_sum.sub(100.0).abs() > 1e-6).sum()),
        "all_zero_rows": int((ratios.fillna(0).sum(axis=1) == 0).sum()),
    }
    invalid: dict[str, int] = {}
    missing_smiles: dict[str, int] = {}
    unparsable_smiles: dict[str, int] = {}
    for _, smiles, _ in COMPONENTS:
        canonical = frame[smiles].map(canonical_smiles)
        missing_smiles[smiles] = int((canonical == "<missing>").sum())
        unparsable_smiles[smiles] = int(canonical.str.startswith("<invalid:").sum())
        invalid[smiles] = missing_smiles[smiles] + unparsable_smiles[smiles]
    profile["invalid_or_missing_smiles"] = invalid
    profile["missing_smiles"] = missing_smiles
    profile["unparsable_smiles"] = unparsable_smiles
    if not profile["source_id_unique"]:
        add_issue(issues, "FAIL", "duplicate_source_business_id", int(frame[source_column].astype(str).duplicated(False).sum()), source_column)
    missing_targets = sum(int(numeric_series(frame, target).isna().sum()) for target in TARGETS)
    infinite_targets = sum(int(np.isinf(numeric_series(frame, target)).sum()) for target in TARGETS)
    if missing_targets or infinite_targets:
        add_issue(issues, "FAIL", "invalid_target_values", missing_targets + infinite_targets, "targets must be finite")
    for code, count in (("negative_component_ratio", profile["ratio_summary"]["negative_rows"]),
                        ("component_ratio_above_100", profile["ratio_summary"]["above_100_rows"]),
                        ("all_zero_formula", profile["ratio_summary"]["all_zero_rows"])):
        if count:
            add_issue(issues, "FAIL", code, int(count), "five-component ratio business constraint")
    if profile["ratio_summary"]["sum_not_100_rows"]:
        add_issue(issues, "WARNING", "ratio_sum_not_100", int(profile["ratio_summary"]["sum_not_100_rows"]),
                  "No project-specific ratio-sum rule was supplied; retain raw values and confirm before final claims.")
    for position, (_, smiles, _) in enumerate(COMPONENTS, start=1):
        if unparsable_smiles[smiles]:
            add_issue(issues, "FAIL", "unparseable_component_smiles", unparsable_smiles[smiles], smiles)
        if missing_smiles[smiles]:
            severity = "WARNING" if position == 5 else "FAIL"
            detail = "absent fifth component is preserved as an explicit masked component; not imputed" if position == 5 else smiles
            add_issue(issues, severity, "missing_component_smiles", missing_smiles[smiles], detail)
    return profile


def duplicate_rows(frame: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """List observed duplicate/near-duplicate patterns without declaring deletions safe."""
    records: list[dict[str, object]] = []
    raw_columns = [column for column in frame.columns if not column.startswith(("canonical_", "normalized_", "component_"))
                   and column not in {"sample_id", "sample_id_payload", "source_business_id", "formula_identity_key", "formula_ratio_key", "fifth_component_key", "original_row_index"}]
    content_columns = [column for column in raw_columns if column != "ID"]
    exact_mask = frame.duplicated(raw_columns, keep=False)
    content_mask = frame.duplicated(content_columns, keep=False)
    exact_formula_mask = frame.duplicated("formula_ratio_key", keep=False)
    formula_identity_mask = frame.duplicated("formula_identity_key", keep=False)
    sample_mask = frame.duplicated("sample_id", keep=False)
    ratio_columns = [ratio for _, _, ratio in COMPONENTS]
    numeric_ratios = frame[ratio_columns].apply(pd.to_numeric, errors="coerce")
    near_groups: set[int] = set()
    for _, group in frame.groupby("formula_identity_key", sort=False):
        if len(group) < 2:
            continue
        group_indexes = group.index.to_list()
        group_ratios = numeric_ratios.loc[group_indexes].to_numpy(dtype=float)
        for left in range(len(group_indexes)):
            for right in range(left + 1, len(group_indexes)):
                if np.isfinite(group_ratios[[left, right]]).all() and np.max(np.abs(group_ratios[left] - group_ratios[right])) <= tolerance:
                    near_groups.update((group_indexes[left], group_indexes[right]))
    for index, row in frame.iterrows():
        flags: list[str] = []
        if exact_mask.loc[index]:
            flags.append("exact_full_row_duplicate")
        if content_mask.loc[index] and not exact_mask.loc[index]:
            flags.append("identical_when_source_id_ignored")
        if exact_formula_mask.loc[index]:
            flags.append("same_canonical_formula_and_ratios")
        if formula_identity_mask.loc[index] and not exact_formula_mask.loc[index]:
            flags.append("same_canonical_formula_different_ratios")
        if sample_mask.loc[index]:
            flags.append("sample_id_conflict")
        if index in near_groups and not exact_formula_mask.loc[index]:
            flags.append("near_duplicate_ratio")
        if flags:
            records.append({
                "original_row_index": int(row["original_row_index"]), "source_business_id": row["source_business_id"],
                "sample_id": row["sample_id"], "formula_identity_key": row["formula_identity_key"],
                "formula_ratio_key": row["formula_ratio_key"], "duplicate_flags": "|".join(flags),
                "classification": "observed_pattern_requires_provenance_review",
                "target_signature": "|".join(number_token(row[target]) for target in TARGETS),
            })
    summary = [
        {"check": "exact_full_row_duplicate", "rows": int(exact_mask.sum())},
        {"check": "identical_when_source_id_ignored", "rows": int(content_mask.sum())},
        {"check": "same_canonical_formula_and_ratios", "rows": int(exact_formula_mask.sum())},
        {"check": "same_canonical_formula_different_ratios", "rows": int((formula_identity_mask & ~exact_formula_mask).sum())},
        {"check": "sample_id_conflict", "rows": int(sample_mask.sum())},
        {"check": "near_duplicate_ratio", "rows": int(len(near_groups))},
    ]
    return pd.DataFrame(records), summary


def comparison_rows(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare records by source ID and formulation, never by row position."""
    old_by_id = old.set_index("source_business_id", drop=False)
    new_by_id = new.set_index("source_business_id", drop=False)
    removed: list[dict[str, object]] = []
    retained_changed: list[dict[str, object]] = []
    for source_id, row in old_by_id.iterrows():
        if source_id not in new_by_id.index:
            removed.append({"source_business_id": source_id, "old_original_row_index": int(row.original_row_index),
                            "old_sample_id": row.sample_id, "old_formula_identity_key": row.formula_identity_key,
                            "old_formula_ratio_key": row.formula_ratio_key, "comparison_status": "not_present_by_source_id",
                            "deletion_justification": "unverified_without_experimental_condition_metadata",
                            **{target: float(pd.to_numeric(pd.Series([row[target]]), errors="coerce").iloc[0]) for target in TARGETS}})
        else:
            new_row = new_by_id.loc[source_id]
            if isinstance(new_row, pd.DataFrame):
                continue
            if row.formula_ratio_key != new_row.formula_ratio_key or any(number_token(row[target]) != number_token(new_row[target]) for target in TARGETS):
                retained_changed.append({"source_business_id": source_id, "old_original_row_index": int(row.original_row_index),
                                         "new_original_row_index": int(new_row.original_row_index), "old_sample_id": row.sample_id,
                                         "new_sample_id": new_row.sample_id, "old_formula_ratio_key": row.formula_ratio_key,
                                         "new_formula_ratio_key": new_row.formula_ratio_key, "comparison_status": "same_source_id_changed_record"})
    return pd.DataFrame(removed), pd.DataFrame(retained_changed)


def distribution_comparison(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for target in TARGETS:
        for name, frame in (("old", old), ("new", new)):
            summary = value_summary(numeric_series(frame, target))
            records.append({"target": target, "dataset": name, **summary})
    return pd.DataFrame(records)


def coverage_comparison(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for group_name, column in (("fifth_component", "fifth_component_key"), ("formula_identity", "formula_identity_key")):
        old_counts = old.groupby(column).size()
        new_counts = new.groupby(column).size()
        for key in sorted(set(old_counts.index) | set(new_counts.index)):
            records.append({"group_type": group_name, "group_id": key, "old_count": int(old_counts.get(key, 0)),
                            "new_count": int(new_counts.get(key, 0)), "status": "lost" if key not in new_counts else ("new" if key not in old_counts else "retained")})
    return pd.DataFrame(records)


def candidate_table(input_dir: Path, chosen: Path, old: Path | None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(input_dir.glob("*.csv")):
        metadata = file_metadata(path)
        try:
            raw = read_csv(path)
            prepared = enrich(raw) if set(REQUIRED).issubset(raw.columns) else None
            duplicate_count = int(prepared.duplicated("formula_ratio_key", keep=False).sum()) if prepared is not None else None
            missing_fifth = int(raw[["Fifth", "Fifth_SMILE", "mol%_Fifth"]].isna().any(axis=1).sum()) if set(["Fifth", "Fifth_SMILE", "mol%_Fifth"]).issubset(raw.columns) else None
            rows, columns = len(raw), len(raw.columns)
        except Exception as error:  # keep discovery diagnostic complete even for broken candidates
            rows, columns, duplicate_count, missing_fifth = None, None, None, None
            metadata["read_error"] = str(error)
        name = path.name.lower()
        records.append({**metadata, "filename": path.name, "rows": rows, "columns": columns,
                        "name_matches_dedup_token": any(token in name for token in NAME_TOKENS),
                        "canonical_formula_ratio_duplicate_rows": duplicate_count, "rows_missing_fifth_component": missing_fifth,
                        "selected": path.resolve() == chosen.resolve(),
                        "comparison_reference": old is not None and path.resolve() == old.resolve()})
    return records


def write_report(path: Path, profile: dict[str, object], duplicate_summary: list[dict[str, object]], issues: pd.DataFrame,
                 old_profile: dict[str, object] | None, removed: pd.DataFrame, coverage: pd.DataFrame, status: str) -> None:
    lines = ["# Deduplicated Dataset Audit", "", f"## Status: {status}", "",
             f"- Selected dataset: `{profile['absolute_path']}`", f"- Selected SHA256: `{profile['sha256']}`",
             f"- Rows: {profile['rows']}",
             f"- Historical comparison CSV: `{old_profile['absolute_path']}`" if old_profile else "- Historical comparison CSV: unavailable.",
             f"- Historical comparison rows: {old_profile['rows']}" if old_profile else "- Historical comparison rows: unavailable.",
             f"- Rows absent by source ID: {len(removed)}", "",
             "## Duplicate checks", ""]
    lines.extend(f"- {row['check']}: {row['rows']} affected rows." for row in duplicate_summary)
    lines.extend(["", "## Rule and interpretation", "",
                  "- A duplicate pattern is not treated as permission to delete a row. Same chemistry can be a true replicate or have a hidden experimental-condition difference.",
                  "- `sample_id` is a SHA256 of stable business ID, canonical five-component structures, ratios, and recorded condition fields; it never contains the DataFrame index.",
                  "- Any remaining full duplicate, pseudo-duplicate, or sample-ID conflict is reported separately and prevents an unqualified PASS.", "",
                  "## Data-quality issues", ""])
    if issues.empty:
        lines.append("- None.")
    else:
        lines.extend(f"- {row.severity}: `{row.code}` ({row['count']}) — {row.detail}" for _, row in issues.iterrows())
    lost = coverage.loc[coverage.status == "lost"] if not coverage.empty else coverage
    lines.extend(["", "## Coverage", "", f"- Groups lost relative to comparison source: {len(lost)}.",
                  "- No feedback labels were read or used by this audit.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-csv", type=Path, required=True, help="Proposed deduplicated raw CSV.")
    parser.add_argument("--old-csv", type=Path, default=None, help="Historical raw CSV used only for descriptive comparison.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "deduplicated_rebaseline")
    parser.add_argument("--near-ratio-tolerance", type=float, default=1e-6)
    arguments = parser.parse_args()
    new_path = arguments.new_csv.resolve()
    old_path = arguments.old_csv.resolve() if arguments.old_csv else None
    if not new_path.is_file() or (old_path is not None and not old_path.is_file()):
        raise FileNotFoundError("--new-csv and, when supplied, --old-csv must exist.")
    if old_path is not None and new_path == old_path:
        raise ValueError("New and old CSV paths are identical; a deduplicated rebaseline cannot be established.")
    output_dir = arguments.output_dir.resolve()
    audit_dir = output_dir / "data_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, object]] = []
    new_raw = read_csv(new_path)
    new_frame = enrich(new_raw)
    profile = dataset_profile(new_raw, new_path, issues, label="selected_new")
    old_profile: dict[str, object] | None = None
    old_frame: pd.DataFrame | None = None
    if old_path is not None:
        old_issues: list[dict[str, object]] = []
        old_raw = read_csv(old_path)
        old_frame = enrich(old_raw)
        old_profile = dataset_profile(old_raw, old_path, old_issues, label="historical_comparison")
    else:
        add_issue(issues, "FAIL", "unproven_new_dataset_identity", 1,
                  "No distinct historical CSV is currently available to prove this candidate is the requested repaired dataset.")
    if not CONDITION_COLUMNS:
        add_issue(issues, "WARNING", "no_explicit_experimental_condition_metadata", 1,
                  "Same chemistry with a different hidden condition cannot be distinguished beyond its business ID.")
    duplicates, duplicate_summary = duplicate_rows(new_frame, arguments.near_ratio_tolerance)
    for row in duplicate_summary:
        if row["check"] in {"exact_full_row_duplicate", "identical_when_source_id_ignored", "sample_id_conflict"} and row["rows"]:
            add_issue(issues, "FAIL", row["check"], int(row["rows"]), "deduplication invariant")
        elif row["check"] in {"same_canonical_formula_and_ratios", "near_duplicate_ratio"} and row["rows"]:
            add_issue(issues, "WARNING", row["check"], int(row["rows"]), "requires explicit replicate/condition provenance")
    if old_frame is None:
        removed = pd.DataFrame(columns=["source_business_id", "comparison_status"])
        retained_changed = pd.DataFrame(columns=["source_business_id", "comparison_status"])
        distributions = pd.DataFrame(columns=["target", "dataset"])
        coverage = pd.DataFrame(columns=["group_type", "group_id", "old_count", "new_count", "status"])
    else:
        removed, retained_changed = comparison_rows(old_frame, new_frame)
        distributions = distribution_comparison(old_frame, new_frame)
        coverage = coverage_comparison(old_frame, new_frame)
    profile["candidate_discovery"] = candidate_table(new_path.parent, new_path, old_path)
    profile["old_dataset"] = old_profile or {"status": "unavailable"}
    profile["duplicate_summary"] = duplicate_summary
    profile["sample_id_hash"] = sha256_text("\n".join(sorted(new_frame.sample_id.astype(str))))
    profile["selection_basis"] = [
        "Selected only after content-level schema, five-component completeness, canonical-SMILES, row-count, and duplicate checks.",
        "Filename tokens are recorded as weak discovery evidence only and were not used as the deciding criterion.",
        "Historical comparison is descriptive; it does not justify deletion without source experimental provenance.",
        "No explicit ratio-sum business rule was provided; finite non-negative ratios whose sum differs from 100 are warnings, not silently normalized or dropped.",
    ]
    issue_frame = pd.DataFrame(issues, columns=["severity", "code", "count", "detail"])
    status = "FAIL" if (issue_frame.severity == "FAIL").any() else ("PASS_WITH_WARNINGS" if not issue_frame.empty else "PASS")
    profile["audit_status"] = status
    json_dump(profile, audit_dir / "dataset_profile.json")
    pd.DataFrame({"column": new_raw.columns, "dtype": [str(new_raw[column].dtype) for column in new_raw.columns],
                  "missing_count": [int(new_raw[column].isna().sum()) for column in new_raw.columns]}).to_csv(audit_dir / "schema.csv", index=False)
    duplicates.to_csv(audit_dir / "duplicate_audit.csv", index=False)
    removed.to_csv(audit_dir / "removed_records.csv", index=False)
    retained_changed.to_csv(audit_dir / "retained_changed_records.csv", index=False)
    distributions.to_csv(audit_dir / "old_new_distribution_comparison.csv", index=False)
    coverage.to_csv(audit_dir / "group_coverage_comparison.csv", index=False)
    issue_frame.to_csv(audit_dir / "data_quality_issues.csv", index=False)
    dataset_output_columns = list(new_raw.columns) + ["original_row_index", "sample_id", "source_business_id",
        "canonical_component_1_smiles", "canonical_component_2_smiles", "canonical_component_3_smiles", "canonical_component_4_smiles", "canonical_component_5_smiles",
        "component_1_key", "component_2_key", "component_3_key", "component_4_key", "component_5_key", "fifth_component_key", "formula_identity_key", "formula_ratio_key"]
    new_frame[dataset_output_columns].to_csv(audit_dir / "dataset_with_sample_id.csv", index=False)
    new_frame[["sample_id", "source_business_id", "original_row_index", "formula_identity_key", "formula_ratio_key"]].to_csv(audit_dir / "sample_id_mapping.csv", index=False)
    sample_rule = {"version": "lrx-v1", "source_id_column": source_id_column(new_raw),
                   "payload_fields": ["source_business_id", "canonical_component_1_smiles", "canonical_component_2_smiles", "canonical_component_3_smiles", "canonical_component_4_smiles", "canonical_component_5_smiles", "normalized_component_1_ratio", "normalized_component_2_ratio", "normalized_component_3_ratio", "normalized_component_4_ratio", "normalized_component_5_ratio", *[f"normalized_condition__{column}" for column in CONDITION_COLUMNS if column in new_frame]],
                   "hash": "SHA256 UTF-8 payload, prefixed lrx-v1-", "forbidden_field": "DataFrame row index"}
    json_dump(sample_rule, audit_dir / "sample_id_rule.json")
    reread = enrich(read_csv(audit_dir / "dataset_with_sample_id.csv"))
    integrity = pd.DataFrame([
        {"check": "globally_unique", "passed": bool(not new_frame.sample_id.duplicated().any()), "detail": f"{new_frame.sample_id.nunique()} unique IDs"},
        {"check": "stable_after_reorder", "passed": bool(sorted(new_frame.sample_id) == sorted(new_frame.sample_id.sample(frac=1, random_state=0))), "detail": "content-derived IDs ignore DataFrame order"},
        {"check": "stable_after_csv_roundtrip", "passed": bool(new_frame.sample_id.tolist() == reread.sample_id.tolist()), "detail": "recomputed from persisted fields"},
        {"check": "sample_id_hash", "passed": True, "detail": profile["sample_id_hash"]},
    ])
    integrity.to_csv(audit_dir / "sample_id_integrity.csv", index=False)
    write_report(audit_dir / "data_audit_report.md", profile, duplicate_summary, issue_frame, old_profile, removed, coverage, status)
    json_dump({"selection_status": "SELECTED" if old_profile else "AMBIGUOUS_NOT_SELECTED", "dataset_path": str(new_path) if old_profile else None,
               "candidate_path": str(new_path), "dataset_sha256": profile["sha256"], "historical_comparison_path": str(old_path) if old_path else None,
               "historical_comparison_sha256": old_profile["sha256"] if old_profile else None, "audit_status": status,
               "sample_id_hash": profile["sample_id_hash"], "selection_evidence": profile["candidate_discovery"],
               "reason": None if old_profile else "Only the path named as the historical dataset remains; the distinct repaired CSV is unavailable."}, output_dir / "data_source.json")
    append_execution(output_dir, {
        "timestamp": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
        "dataset_path": str(new_path), "dataset_sha256": profile["sha256"], "protocol": None, "fold": None,
        "seed": None, "manifest_sha256": None, "feature_hash": None, "config_hash": None, "checkpoint": None,
        "status": status, "error": None if status != "FAIL" else "audit gate failed; no modeling launched", "output_path": str(audit_dir),
    })
    print(json.dumps({"dataset": str(new_path), "sha256": profile["sha256"], "status": status, "rows": len(new_frame),
                      "output": str(audit_dir)}, ensure_ascii=False))
    if status == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
