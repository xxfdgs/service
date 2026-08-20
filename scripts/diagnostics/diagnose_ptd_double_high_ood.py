#!/usr/bin/env python3
"""Attribute strict Fifth-OOD double-high errors without opening external data.

The labels used here are exclusively those in the locked 700-row source's
outer test partitions.  The tool records overlapping mechanism flags rather
than pretending that representation distance, shrinkage, and uncertainty are
mutually exclusive causes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def finite_spearman(y: pd.Series, prediction: pd.Series) -> float:
    if len(y) < 2 or y.nunique() < 2 or prediction.nunique() < 2:
        return math.nan
    return float(spearmanr(y, prediction).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--similarity-far-cutoff", type=float, default=0.50)
    args = parser.parse_args()
    if not 0 <= args.similarity_far_cutoff <= 1:
        raise ValueError("--similarity-far-cutoff must be in [0, 1].")

    audit_dir, output = args.audit_dir.resolve(), args.output_dir.resolve()
    errors = pd.read_csv(audit_dir / "p1_ptd_double_gt1_error_audit.csv")
    per_split = pd.read_csv(audit_dir / "p1_ptd_internal_ood_metrics_per_split.csv")
    required = {"seed", "sample_id", "y_true", "y_pred", "signed_error", "false_negative",
                "unseen_fifth_identity", "nearest_train_fifth_tanimoto", "ensemble_std"}
    missing = required.difference(errors.columns)
    if missing:
        raise ValueError(f"Audit is missing required columns: {sorted(missing)}")
    errors = errors.copy()
    errors["false_negative"] = errors.false_negative.astype(bool)
    errors["representation_far"] = (
        errors.unseen_fifth_identity.astype(bool)
        & errors.nearest_train_fifth_tanimoto.lt(args.similarity_far_cutoff)
    )
    fn = errors.loc[errors.false_negative].copy()
    uncertainty_cutoff = float(fn.ensemble_std.dropna().median()) if fn.ensemble_std.notna().any() else math.nan
    errors["high_epistemic_uncertainty"] = (
        errors.ensemble_std.ge(uncertainty_cutoff)
        if np.isfinite(uncertainty_cutoff) else False
    )

    split_high = per_split.loc[per_split.subset.eq("double_gt1"), ["seed", "mean_signed_error", "spearman"]].copy()
    split_high["split_systematic_underprediction"] = split_high.mean_signed_error.lt(0)
    split_high["split_rank_preserved_but_biased"] = (
        split_high.split_systematic_underprediction & split_high.spearman.gt(0)
    )
    errors = errors.merge(split_high, on="seed", how="left", validate="many_to_one")
    errors["mechanism_A_representation_ood"] = errors.representation_far
    errors["mechanism_B_shrinkage"] = errors.split_systematic_underprediction
    errors["mechanism_C_uncertainty"] = errors.high_epistemic_uncertainty
    errors["mechanism_D_objective_mismatch"] = errors.split_rank_preserved_but_biased
    errors["mechanism_flags"] = errors.apply(
        lambda row: ";".join(letter for letter, column in (
            ("A", "mechanism_A_representation_ood"),
            ("B", "mechanism_B_shrinkage"),
            ("C", "mechanism_C_uncertainty"),
            ("D", "mechanism_D_objective_mismatch"),
        ) if bool(row[column])) or "none", axis=1,
    )

    output.mkdir(parents=True, exist_ok=True)
    fn = errors.loc[errors.false_negative].copy()
    all_rows, fn_rows = len(errors), len(fn)
    summary = {
        "scope": "Locked source / frozen Fifth-OOD test predictions only; no new_validation access.",
        "double_gt1_rows": int(all_rows),
        "false_negative_rows": int(fn_rows),
        "fn_uncertainty_median_cutoff": uncertainty_cutoff,
        "similarity_far_cutoff": float(args.similarity_far_cutoff),
        "flag_definitions": {
            "A_representation_ood": "unseen Fifth identity and nearest train Fifth Morgan Tanimoto below cutoff",
            "B_shrinkage": "the row's split has negative double-high mean signed error",
            "C_uncertainty": "across-seed std at or above the false-negative median",
            "D_objective_mismatch": "the row's split has negative signed bias and positive double-high Spearman",
        },
        "fn_flag_fraction": {
            label: float(fn[column].mean()) if fn_rows else math.nan
            for label, column in (
                ("A_representation_ood", "mechanism_A_representation_ood"),
                ("B_shrinkage", "mechanism_B_shrinkage"),
                ("C_uncertainty", "mechanism_C_uncertainty"),
                ("D_objective_mismatch", "mechanism_D_objective_mismatch"),
            )
        },
        "fn_unseen_identity_fraction": float(fn.unseen_fifth_identity.mean()) if fn_rows else math.nan,
        "fn_nearest_similarity_mean": float(fn.nearest_train_fifth_tanimoto.mean()) if fn_rows else math.nan,
        "double_high_mean_signed_error": float(errors.signed_error.mean()) if all_rows else math.nan,
        "double_high_spearman": finite_spearman(errors.y_true, errors.y_pred),
    }
    output_rows = errors.sort_values(["false_negative", "absolute_error", "seed"], ascending=[False, False, True])
    output_rows.to_csv(output / "double_gt1_mechanism_attribution.csv", index=False)
    split_high.to_csv(output / "double_gt1_split_shrinkage_rank_diagnostics.csv", index=False)
    (output / "double_gt1_mechanism_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
