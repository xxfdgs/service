#!/usr/bin/env python3
"""Audit strict PT-D Fifth-OOD predictions without using external labels.

This script is deliberately an *internal-proxy* diagnostic.  It only reads the
locked 700-row source, frozen manifest memberships, and saved selected-
checkpoint predictions.  In particular, it never opens ``new_validation``.

It exports per-split subset metrics and a row-level audit for the scientifically
important ``Fifth_class=double & Norm_before > 1`` population.  Structural
support fields are calculated against that split's outer-train rows only.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SUBSETS = {
    "all": lambda frame: np.ones(len(frame), dtype=bool),
    "single": lambda frame: frame["Fifth_class"].eq("single").to_numpy(),
    "double": lambda frame: frame["Fifth_class"].eq("double").to_numpy(),
    "double_le1": lambda frame: (
        frame["Fifth_class"].eq("double") & frame["y_true"].le(1.0)
    ).to_numpy(),
    "double_gt1": lambda frame: (
        frame["Fifth_class"].eq("double") & frame["y_true"].gt(1.0)
    ).to_numpy(),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Continuous and threshold metrics in physical Norm units."""
    y = frame.y_true.to_numpy(float)
    p = frame.y_pred.to_numpy(float)
    error = p - y
    absolute = np.abs(error)
    true_high, pred_high = y > 1.0, p > 1.0
    tp = int(np.sum(true_high & pred_high))
    tn = int(np.sum(~true_high & ~pred_high))
    fp = int(np.sum(~true_high & pred_high))
    fn = int(np.sum(true_high & ~pred_high))
    under = error < 0
    result: dict[str, float | int] = {
        "n": int(len(frame)),
        "mae": float(mean_absolute_error(y, p)) if len(frame) else math.nan,
        "rmse": float(mean_squared_error(y, p) ** 0.5) if len(frame) else math.nan,
        "median_ae": float(np.median(absolute)) if len(frame) else math.nan,
        "mean_signed_error": float(np.mean(error)) if len(frame) else math.nan,
        "underprediction_mae": float(np.mean(absolute[under])) if np.any(under) else 0.0,
        "r2": float(r2_score(y, p)) if len(frame) >= 2 and np.std(y) else math.nan,
        "spearman": (
            float(spearmanr(y, p).statistic)
            if len(frame) >= 2 and np.std(y) and np.std(p) else math.nan
        ),
        "prediction_mean": float(np.mean(p)) if len(frame) else math.nan,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision_gt1": _rate(tp, tp + fp),
        "recall_gt1": _rate(tp, tp + fn),
        "f2_gt1": 0.0 if (tp + fn) and not tp else (
            5.0 * tp / (5.0 * tp + 4.0 * fn + fp) if (5 * tp + 4 * fn + fp) else math.nan
        ),
    }
    return result


@functools.lru_cache(maxsize=None)
def _canonical_fifth(value: str) -> str | None:
    if not value.strip() or value.strip() == "[Fr]":
        return None
    molecule = Chem.MolFromSmiles(value)
    return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule is not None else None


def canonical_fifth(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip() or value.strip() == "[Fr]":
        return None
    return _canonical_fifth(value)


@functools.lru_cache(maxsize=None)
def _fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048) if molecule else None


def fingerprint(smiles: str | None):
    if not isinstance(smiles, str):
        return None
    return _fingerprint(smiles)


def train_support(source: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Return target-free structural support, except neighbor target summary."""
    joined = manifest[["sample_id", "split"]].merge(
        source, left_on="sample_id", right_on="ID", how="left", validate="one_to_one"
    )
    if joined.ID.isna().any():
        raise ValueError("Manifest IDs are missing from the locked input source.")
    train = joined.loc[joined.split.eq("train")].copy()
    test = joined.loc[joined.split.eq("test")].copy()
    train["canonical_fifth"] = train.Fifth_SMILE.map(canonical_fifth)
    test["canonical_fifth"] = test.Fifth_SMILE.map(canonical_fifth)
    train["fingerprint"] = train.canonical_fifth.map(fingerprint)
    test["fingerprint"] = test.canonical_fifth.map(fingerprint)
    rows = []
    for item in test.itertuples(index=False):
        exact = train.loc[train.canonical_fifth.eq(item.canonical_fifth)] if item.canonical_fifth else train.iloc[0:0]
        candidates = train.loc[train.fingerprint.notna()]
        query_fp = item.fingerprint
        if query_fp is None or candidates.empty:
            similarity, neighbor_target_mean, neighbor_target_std = math.nan, math.nan, math.nan
        else:
            scores = np.asarray([
                DataStructs.TanimotoSimilarity(query_fp, candidate)
                for candidate in candidates.fingerprint
            ])
            top = candidates.iloc[np.argsort(scores)[-min(5, len(scores)):]]
            similarity = float(scores.max())
            neighbor_target_mean = float(top.Norm_before.mean())
            neighbor_target_std = float(top.Norm_before.std(ddof=0))
        rows.append({
            "sample_id": item.sample_id,
            "train_same_fifth_count": int(len(exact)),
            "unseen_fifth_identity": bool(len(exact) == 0),
            "nearest_train_fifth_tanimoto": similarity,
            "nearest_neighbor_norm_before_mean": neighbor_target_mean,
            "nearest_neighbor_norm_before_std": neighbor_target_std,
        })
    return pd.DataFrame(rows)


def validate_fifth_ood_membership(source: pd.DataFrame, manifest: pd.DataFrame,
                                  prediction_test_ids: pd.Series) -> dict[str, int]:
    """Prove exact saved-test membership and no real Fifth identity crossing."""
    required = {"sample_id", "split"}
    if required.difference(manifest.columns) or manifest.sample_id.duplicated().any():
        raise ValueError("Frozen manifest is malformed or has duplicate sample IDs.")
    if not set(manifest.split.unique()).issubset({"train", "val", "test"}):
        raise ValueError("Frozen manifest has an unsupported split value.")
    expected_test = set(manifest.loc[manifest.split.eq("test"), "sample_id"].astype(str))
    saved_test = set(prediction_test_ids.astype(str))
    if saved_test != expected_test:
        missing, extra = sorted(expected_test - saved_test), sorted(saved_test - expected_test)
        raise ValueError(
            "Saved selected-checkpoint test predictions do not exactly match the frozen manifest; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    joined = manifest.merge(source, left_on="sample_id", right_on="ID",
                            how="left", validate="one_to_one")
    if joined.ID.isna().any():
        raise ValueError("Frozen manifest contains an ID absent from the locked source.")
    identities = {
        split: {value for value in joined.loc[joined.split.eq(split), "Fifth_SMILE"].map(canonical_fifth)
                if value is not None}
        for split in ("train", "val", "test")
    }
    crossings = {
        "train_val": identities["train"] & identities["val"],
        "train_test": identities["train"] & identities["test"],
        "val_test": identities["val"] & identities["test"],
    }
    if any(crossings.values()):
        detail = {name: sorted(values)[:3] for name, values in crossings.items() if values}
        raise ValueError(f"Fifth identity crosses frozen OOD partitions: {detail}")
    formulation_columns = [
        "IL_SMILE", "HL_SMILE", "Chol_SMILE", "PEG_SMILE", "Fifth_SMILE",
        "mol%_IL", "mol%_HL", "mol%_Chol", "mol%_PEG", "mol%_Fifth",
    ]
    duplicate_crossing = {"train_val": math.nan, "train_test": math.nan, "val_test": math.nan}
    if not set(formulation_columns).difference(joined.columns):
        # This is a non-destructive leakage check, not a deduplication rule:
        # real experimental replicates remain records, but their split overlap
        # is made explicit in the audit manifest.
        formulation = joined[formulation_columns].fillna("<missing>").astype(str).agg("|".join, axis=1)
        key_sets = {
            split: set(formulation.loc[joined.split.eq(split)])
            for split in ("train", "val", "test")
        }
        duplicate_crossing = {
            "train_val": len(key_sets["train"] & key_sets["val"]),
            "train_test": len(key_sets["train"] & key_sets["test"]),
            "val_test": len(key_sets["val"] & key_sets["test"]),
        }
    return {
        "manifest_rows": int(len(manifest)), "test_rows": int(len(expected_test)),
        "train_real_fifth_identities": int(len(identities["train"])),
        "val_real_fifth_identities": int(len(identities["val"])),
        "test_real_fifth_identities": int(len(identities["test"])),
        "exact_formulation_keys_crossing_train_val": duplicate_crossing["train_val"],
        "exact_formulation_keys_crossing_train_test": duplicate_crossing["train_test"],
        "exact_formulation_keys_crossing_val_test": duplicate_crossing["val_test"],
    }


def aggregate(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "n", "mae", "rmse", "median_ae", "mean_signed_error", "underprediction_mae", "r2",
        "spearman", "prediction_mean", "precision_gt1", "recall_gt1", "f2_gt1", "tp", "tn", "fp", "fn",
    ]
    records = []
    for subset, group in metrics_frame.groupby("subset", sort=False):
        record = {"subset": subset, "completed_splits": int(len(group))}
        for column in value_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            record[f"{column}_mean"] = float(values.mean())
            record[f"{column}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else math.nan
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, nargs="+", required=True,
                        help="One or more run roots collectively containing split100...split109.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 110)))
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.warning")

    source = pd.read_csv(args.input_csv, dtype={"ID": str})
    required = {"ID", "Fifth", "Fifth_SMILE", "Fifth_class", "Norm_before"}
    missing = required.difference(source.columns)
    if missing or source.ID.duplicated().any():
        raise ValueError(f"Locked source invalid; missing={sorted(missing)}, duplicate_id={source.ID.duplicated().any()}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_split, audit_rows, membership_checks = [], [], []
    for seed in args.seeds:
        manifest_path = args.manifest_dir / f"fifth_identity_manifest_seed{seed}.csv"
        prediction_candidates = [
            root / f"split{seed}" / "predictions.csv"
            for root in args.prediction_root
        ]
        existing_predictions = [path for path in prediction_candidates if path.is_file()]
        if len(existing_predictions) != 1:
            raise FileNotFoundError(
                f"Expected exactly one predictions.csv for split{seed}; found: "
                + ", ".join(str(path) for path in existing_predictions)
            )
        prediction_path = existing_predictions[0]
        manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
        prediction = pd.read_csv(prediction_path, dtype={"sample_id": str})
        test = prediction.loc[prediction.split.eq("test")].copy()
        if test.sample_id.duplicated().any():
            raise ValueError(f"Saved P1 test prediction has duplicate IDs for split{seed}.")
        membership_checks.append({"seed": seed, **validate_fifth_ood_membership(source, manifest, test.sample_id)})
        metadata = source[["ID", "Fifth", "Fifth_SMILE", "Fifth_class"]]
        test = test.merge(metadata, left_on="sample_id", right_on="ID", how="left", validate="one_to_one")
        if test.ID.isna().any():
            raise ValueError(f"P1 prediction ID absent from source in split{seed}.")
        support = train_support(source, manifest)
        test = test.merge(support, on="sample_id", how="left", validate="one_to_one")
        for subset, selector in SUBSETS.items():
            selected = test.loc[selector(test)]
            per_split.append({"seed": seed, "subset": subset, **metrics(selected)})
        high = test.loc[(test.Fifth_class.eq("double")) & (test.y_true.gt(1.0))].copy()
        high["seed"] = seed
        high["absolute_error"] = (high.y_pred - high.y_true).abs()
        high["signed_error"] = high.y_pred - high.y_true
        high["underprediction"] = high.signed_error.lt(0)
        high["false_negative"] = high.y_pred.lt(1.0)
        audit_rows.append(high)
    per_split_frame = pd.DataFrame(per_split)
    audit = pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame()
    # Variation across independently trained Fifth-OOD members is descriptive
    # uncertainty only; it is not used to select a checkpoint or a candidate.
    if not audit.empty:
        uncertainty = audit.groupby("sample_id", as_index=False).agg(
            ensemble_prediction_mean=("y_pred", "mean"),
            ensemble_std=("y_pred", "std"),
            prediction_count=("y_pred", "size"),
        )
        audit = audit.merge(uncertainty, on="sample_id", how="left", validate="many_to_one")
        audit = audit.sort_values(["false_negative", "absolute_error", "seed"], ascending=[False, False, True])
    per_split_frame.to_csv(output / "p1_ptd_internal_ood_metrics_per_split.csv", index=False)
    aggregate(per_split_frame).to_csv(output / "p1_ptd_internal_ood_metrics_summary.csv", index=False)
    audit.to_csv(output / "p1_ptd_double_gt1_error_audit.csv", index=False)
    if not audit.empty:
        # A single sample can occur in several frozen outer-test folds.  Keep
        # both the long table above and this explicit seed-wide view so a
        # reviewer can inspect every available checkpoint prediction without
        # reconstructing a pivot themselves.
        identity_columns = [
            "sample_id", "Fifth", "Fifth_SMILE", "Fifth_class", "y_true",
            "train_same_fifth_count", "unseen_fifth_identity",
            "nearest_train_fifth_tanimoto", "nearest_neighbor_norm_before_mean",
            "nearest_neighbor_norm_before_std", "ensemble_prediction_mean",
            "ensemble_std", "prediction_count",
        ]
        identity = audit[identity_columns].drop_duplicates("sample_id")
        wide = identity.merge(
            audit.pivot(index="sample_id", columns="seed", values="y_pred").reset_index(),
            on="sample_id", how="left", validate="one_to_one")
        wide.columns = [
            f"prediction_seed{column}" if isinstance(column, (int, np.integer)) else column
            for column in wide.columns
        ]
        wide.to_csv(output / "p1_ptd_double_gt1_error_audit_wide.csv", index=False)
    fn = audit.loc[audit.false_negative] if not audit.empty else audit
    diagnosis = {
        "scope": "Locked 700-row Fifth-identity OOD proxy only; new_validation was not read.",
        "baseline": "P1_PT_D_strict_no_mordred / Norm_before / validation-selected checkpoints",
        "input_sha256": sha256(args.input_csv),
        "seeds": args.seeds,
        "fifth_ood_membership_checks": membership_checks,
        "double_gt1_test_rows_across_splits": int(len(audit)),
        "false_negative_rows": int(len(fn)),
        "false_negative_unseen_fifth_fraction": float(fn.unseen_fifth_identity.mean()) if len(fn) else math.nan,
        "false_negative_mean_nearest_train_tanimoto": float(fn.nearest_train_fifth_tanimoto.mean()) if len(fn) else math.nan,
        "false_negative_mean_seed_prediction_std": float(fn.ensemble_std.mean()) if len(fn) else math.nan,
        "classification_rule": {
            "A_representation_ood": "FN with unseen Fifth identity and nearest train Fifth Tanimoto < 0.50",
            "B_shrinkage": "all double>1 rows summarized by mean signed error and true-to-prediction slope in report",
            "C_epistemic_uncertainty": "FN whose across-seed prediction std is at or above the FN median",
            "D_objective_mismatch": "suggested only if double>1 Spearman remains positive while mean signed error is negative",
        },
    }
    (output / "p1_ptd_internal_ood_audit_manifest.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    print(aggregate(per_split_frame).to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
