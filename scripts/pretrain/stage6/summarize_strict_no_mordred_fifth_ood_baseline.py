#!/usr/bin/env python3
"""Summarize the strict No-Mordred Fifth-identity OOD P0/P1/P2 baseline.

This evaluation-only script reads validation-selected test predictions from
the isolated strict No-Mordred output root.  It fail-closes if a run does not
prove ``use_mordred_features == false`` in its persisted provenance.

Outputs include per-split metrics, model mean/std/median over the requested
splits, and all pairwise P0/P1/P2 comparisons on exactly matching test rows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


MODELS = {
    "P0_random": {
        "run_label": "P0_random_strict_no_mordred",
        "init_mode": "random",
    },
    "P1_PT_D": {
        "run_label": "P1_PT_D_strict_no_mordred",
        "init_mode": "stage4_pretrained_full_finetune",
    },
    "P2_PT_DF": {
        "run_label": "P2_PT_DF_strict_no_mordred",
        "init_mode": "stage4_pretrained_full_finetune",
    },
}

METRIC_COLUMNS = (
    "n",
    "mae",
    "r2",
    "spearman",
    "precision_gt1",
    "recall_gt1",
    "f2_gt1",
    "tp_gt1",
    "fn_gt1",
    "fp_gt1",
    "tn_gt1",
    "prediction_mean",
    "prediction_std",
)


def selected_test_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"split", "target", "y_true", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    frame = frame.loc[
        frame["split"].astype(str).eq("test")
        & frame["target"].astype(str).eq("Norm_before")
    ].copy()
    if "checkpoint" in frame.columns:
        checkpoint = frame["checkpoint"].fillna("").astype(str)
        selected = checkpoint.str.contains("selected_best", regex=False)
        if selected.any():
            frame = frame.loc[selected].copy()
        elif checkpoint.nunique() > 1:
            raise ValueError(
                f"{path}: multiple checkpoint labels but no selected_best row"
            )
    if frame.empty:
        raise ValueError(f"{path}: no selected-best test Norm_before predictions")
    if "sample_id" in frame.columns and frame["sample_id"].duplicated().any():
        duplicate = frame.loc[
            frame["sample_id"].duplicated(keep=False), "sample_id"
        ].tolist()[:10]
        raise ValueError(f"{path}: duplicate selected test sample IDs: {duplicate}")

    for column in ("y_true", "y_pred"):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite {column} values")
    return frame.reset_index(drop=True)


def validate_strict_provenance(run_dir: Path, logical_model: str) -> dict:
    spec = MODELS[logical_model]
    settings_path = run_dir / "run_settings.json"
    init_path = run_dir / "comp5_initialization.json"
    effective_path = run_dir / "effective_config.yaml"
    for path in (settings_path, init_path, effective_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    init = json.loads(init_path.read_text(encoding="utf-8"))
    if settings.get("use_mordred_features") is not False:
        raise ValueError(f"{settings_path}: not a No-Mordred run")
    if settings.get("mordred_feature_path") not in ("", None):
        raise ValueError(f"{settings_path}: unexpected Mordred lookup path")
    if int(settings.get("mordred_feature_dim", -1)) != 0:
        raise ValueError(f"{settings_path}: unexpected nonzero Mordred dimension")
    if "use_mordred_features: false" not in effective_path.read_text(encoding="utf-8"):
        raise ValueError(f"{effective_path}: does not persist disabled Mordred")
    if init.get("label") != spec["run_label"]:
        raise ValueError(f"{init_path}: wrong model label {init.get('label')!r}")
    if init.get("mode") != spec["init_mode"]:
        raise ValueError(f"{init_path}: wrong initialization mode {init.get('mode')!r}")
    if spec["init_mode"] != "random":
        transfer = init.get("strict_transfer_report") or {}
        if transfer.get("strict") is not True or not init.get("checkpoint_sha256"):
            raise ValueError(f"{init_path}: missing strict Stage-4 transfer proof")
    return init


def threshold_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    actual = y > 1.0
    predicted = p > 1.0
    tp = int(np.sum(actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    if np.isfinite(precision) and np.isfinite(recall):
        denominator = 4.0 * precision + recall
        f2 = 5.0 * precision * recall / denominator if denominator else 0.0
    else:
        f2 = math.nan
    return {
        "precision_gt1": precision,
        "recall_gt1": recall,
        "f2_gt1": f2,
        "tp_gt1": tp,
        "fn_gt1": fn,
        "fp_gt1": fp,
        "tn_gt1": tn,
    }


def run_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame["y_true"].to_numpy(dtype=float)
    p = frame["y_pred"].to_numpy(dtype=float)
    spearman = (
        float(spearmanr(y, p).statistic)
        if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0
        else math.nan
    )
    result = {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)) if len(y) > 1 and np.std(y) > 0 else math.nan,
        "spearman": spearman,
        "prediction_mean": float(np.mean(p)),
        "prediction_std": float(np.std(p, ddof=0)),
    }
    result.update(threshold_metrics(y, p))
    return result


def assert_paired_rows(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
    candidate_name: str,
    comparator_name: str,
    split_seed: int,
) -> None:
    if "sample_id" in candidate.columns and "sample_id" in comparator.columns:
        merged = candidate[["sample_id", "y_true"]].merge(
            comparator[["sample_id", "y_true"]],
            on="sample_id",
            how="outer",
            suffixes=("_candidate", "_comparator"),
            indicator=True,
            validate="one_to_one",
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(
                f"split{split_seed}: {candidate_name}/{comparator_name} test IDs differ"
            )
        if not np.allclose(
            merged["y_true_candidate"], merged["y_true_comparator"], rtol=0, atol=0
        ):
            raise ValueError(
                f"split{split_seed}: {candidate_name}/{comparator_name} labels differ"
            )
        return
    if len(candidate) != len(comparator) or not np.allclose(
        candidate["y_true"], comparator["y_true"], rtol=0, atol=0
    ):
        raise ValueError(
            f"split{split_seed}: {candidate_name}/{comparator_name} cannot be paired"
        )


def group_summary(per_split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        group = per_split.loc[per_split["model"].eq(model)]
        if group.empty:
            raise ValueError(f"No metrics for {model}")
        row: dict[str, object] = {"model": model, "splits": int(len(group))}
        for column in METRIC_COLUMNS:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else math.nan
            )
            row[f"{column}_median"] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_rows(
    per_split: pd.DataFrame,
    frames: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for candidate_name, comparator_name in (
        ("P1_PT_D", "P0_random"),
        ("P2_PT_DF", "P0_random"),
        ("P2_PT_DF", "P1_PT_D"),
    ):
        candidate_metrics = per_split.loc[
            per_split["model"].eq(candidate_name)
        ].set_index("split_seed")
        comparator_metrics = per_split.loc[
            per_split["model"].eq(comparator_name)
        ].set_index("split_seed")
        if set(candidate_metrics.index) != set(comparator_metrics.index):
            raise ValueError(
                f"Split mismatch: {candidate_name} and {comparator_name} are not paired"
            )
        for split_seed in sorted(candidate_metrics.index):
            assert_paired_rows(
                frames[(candidate_name, int(split_seed))],
                frames[(comparator_name, int(split_seed))],
                candidate_name,
                comparator_name,
                int(split_seed),
            )
            c, b = candidate_metrics.loc[split_seed], comparator_metrics.loc[split_seed]
            rows.append({
                "candidate_model": candidate_name,
                "comparator_model": comparator_name,
                "split_seed": int(split_seed),
                # Positive values always mean the candidate is preferable.
                "improvement_mae": float(b["mae"] - c["mae"]),
                "improvement_r2": float(c["r2"] - b["r2"]),
                "improvement_spearman": float(c["spearman"] - b["spearman"]),
                "improvement_precision_gt1": float(c["precision_gt1"] - b["precision_gt1"]),
                "improvement_recall_gt1": float(c["recall_gt1"] - b["recall_gt1"]),
                "improvement_f2_gt1": float(c["f2_gt1"] - b["f2_gt1"]),
                "tp_increase_gt1": float(c["tp_gt1"] - b["tp_gt1"]),
                "fn_reduction_gt1": float(b["fn_gt1"] - c["fn_gt1"]),
                "fp_reduction_gt1": float(b["fp_gt1"] - c["fp_gt1"]),
                "tn_increase_gt1": float(c["tn_gt1"] - b["tn_gt1"]),
                "prediction_mean_change": float(c["prediction_mean"] - b["prediction_mean"]),
                "prediction_std_change": float(c["prediction_std"] - b["prediction_std"]),
            })
    return pd.DataFrame(rows)


def paired_summary(per_pair: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column for column in per_pair.columns
        if column not in {"candidate_model", "comparator_model", "split_seed"}
    ]
    rows = []
    for (candidate, comparator), group in per_pair.groupby(
        ["candidate_model", "comparator_model"], sort=False
    ):
        row: dict[str, object] = {
            "candidate_model": candidate,
            "comparator_model": comparator,
            "splits": int(len(group)),
        }
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else math.nan
            )
            row[f"{column}_median"] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/fifth_pretraining/stage6_strict_no_mordred_fifth_ood_baseline"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        default=list(range(100, 110)),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "analysis_strict_no_mordred"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for model, spec in MODELS.items():
        for split_seed in args.splits:
            run_dir = root / spec["run_label"] / f"split{split_seed}"
            prediction_path = run_dir / "predictions.csv"
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            init = validate_strict_provenance(run_dir, model)
            frame = selected_test_predictions(prediction_path)
            frames[(model, int(split_seed))] = frame
            rows.append({
                "baseline": "strict_no_mordred_fifth_identity_ood",
                "model": model,
                "run_label": spec["run_label"],
                "split_seed": int(split_seed),
                "initialization_mode": init.get("mode"),
                "checkpoint_sha256": init.get("checkpoint_sha256"),
                **run_metrics(frame),
            })

    per_split = pd.DataFrame(rows)
    summary = group_summary(per_split)
    per_pair = paired_rows(per_split, frames)
    pair_summary = paired_summary(per_pair)

    provenance = {
        "baseline": "strict_no_mordred_fifth_identity_ood",
        "mordred_policy": "disabled in every run; no Mordred lookup accepted",
        "target": "Norm_before",
        "threshold": "y_true > 1 and y_pred > 1",
        "split_seeds": [int(seed) for seed in args.splits],
        "models": MODELS,
    }
    (output_dir / "strict_no_mordred_summary_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    per_split.to_csv(output_dir / "strict_no_mordred_per_split_metrics.csv", index=False)
    summary.to_csv(output_dir / "strict_no_mordred_10split_summary.csv", index=False)
    per_pair.to_csv(output_dir / "strict_no_mordred_paired_per_split.csv", index=False)
    pair_summary.to_csv(output_dir / "strict_no_mordred_paired_summary.csv", index=False)

    print("=" * 112)
    print("STRICT NO-MORDRED FIFTH-IDENTITY OOD BASELINE")
    print("=" * 112)
    print("All values are split-level summaries; std uses ddof=1 and selection remains validation-only.")
    show = [
        "model", "splits", "mae_mean", "r2_mean", "spearman_mean",
        "precision_gt1_mean", "recall_gt1_mean", "f2_gt1_mean",
        "tp_gt1_mean", "fn_gt1_mean", "fp_gt1_mean", "tn_gt1_mean",
        "prediction_mean_mean", "prediction_std_mean",
    ]
    print(summary[show].to_string(index=False))
    print()
    print("Pairwise changes (positive means candidate is preferable):")
    print(pair_summary.to_string(index=False))
    print()
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
