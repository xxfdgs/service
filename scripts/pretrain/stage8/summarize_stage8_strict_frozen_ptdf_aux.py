#!/usr/bin/env python3
"""Evaluate strict No-Mordred Stage8 P3 against strict P0/P1/P2 controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


CONTROL_LABELS = {
    "P0_random_NoMordred": "P0_random_strict_no_mordred",
    "P1_PT_D_NoMordred": "P1_PT_D_strict_no_mordred",
    "P2_PT_DF_NoMordred": "P2_PT_DF_strict_no_mordred",
}
P3_MODEL = "P3_PT_DF_FrozenAux_NoMordred"
METRICS = (
    "n", "mae", "r2", "spearman", "precision_gt1", "recall_gt1",
    "f2_gt1", "tp_gt1", "fn_gt1", "fp_gt1", "tn_gt1",
    "prediction_mean", "prediction_std",
)


def selected_test_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"split", "target", "y_true", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
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
            raise ValueError(f"{path}: ambiguous checkpoint rows")
    if frame.empty:
        raise ValueError(f"{path}: no selected-best Norm_before test rows")
    if "sample_id" in frame and frame["sample_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate selected test sample_id values")
    for column in ("y_true", "y_pred"):
        if not np.isfinite(frame[column].to_numpy(float)).all():
            raise ValueError(f"{path}: non-finite {column}")
    return frame.reset_index(drop=True)


def assert_no_mordred(run_dir: Path) -> tuple[dict, dict]:
    settings_path = run_dir / "run_settings.json"
    init_path = run_dir / "comp5_initialization.json"
    if not settings_path.is_file() or not init_path.is_file():
        raise FileNotFoundError(f"Missing strict No-Mordred provenance under {run_dir}")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    init = json.loads(init_path.read_text(encoding="utf-8"))
    if settings.get("use_mordred_features") is not False:
        raise ValueError(f"{run_dir}: Mordred was not disabled")
    if settings.get("mordred_feature_path") not in ("", None):
        raise ValueError(f"{run_dir}: unexpected Mordred lookup path")
    if int(settings.get("mordred_feature_dim", -1)) != 0:
        raise ValueError(f"{run_dir}: unexpected Mordred feature dimension")
    return settings, init


def assert_p3_provenance(run_dir: Path, pt_df_checkpoint: Path) -> dict:
    settings, task = assert_no_mordred(run_dir)
    frozen_path = run_dir / "frozen_comp5_aux_initialization.json"
    selected_path = run_dir / "checkpoints" / "selected_best.pt"
    if not frozen_path.is_file() or not selected_path.is_file():
        raise FileNotFoundError(f"Missing Stage8 frozen provenance under {run_dir}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    expected_sha = hashlib.sha256(pt_df_checkpoint.read_bytes()).hexdigest()
    if task.get("mode") != "random":
        raise ValueError(f"{run_dir}: Stage8 task Comp5 must be random/trainable")
    required = {
        "enabled": True,
        "checkpoint_sha256": expected_sha,
        "trainable_parameter_count": 0,
        "optimizer_includes_frozen_parameters": False,
        "optimizer_exact_trainable_partition": True,
        "frozen_training_after_model_train": False,
    }
    for key, expected in required.items():
        if frozen.get(key) != expected:
            raise ValueError(f"{run_dir}: frozen audit {key}={frozen.get(key)!r}, expected {expected!r}")
    if (frozen.get("strict_transfer_report") or {}).get("strict") is not True:
        raise ValueError(f"{run_dir}: frozen encoder is not proven strict-loaded")
    if frozen.get("task_comp5_trainable_parameter_count", 0) <= 0:
        raise ValueError(f"{run_dir}: task Comp5 encoder is not trainable")
    if (frozen.get("topology") or {}).get("task_and_frozen_encoder_distinct") is not True:
        raise ValueError(f"{run_dir}: Stage8 encoders are not proven distinct")
    if settings.get("frozen_comp5_aux_checkpoint_sha256") != expected_sha:
        raise ValueError(f"{run_dir}: run settings frozen checkpoint hash mismatch")
    return frozen


def threshold_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    actual, predicted = y > 1.0, p > 1.0
    tp = int(np.sum(actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    if np.isfinite(precision) and np.isfinite(recall):
        denominator = 4 * precision + recall
        f2 = 5 * precision * recall / denominator if denominator else 0.0
    else:
        f2 = math.nan
    return {
        "precision_gt1": precision, "recall_gt1": recall, "f2_gt1": f2,
        "tp_gt1": tp, "fn_gt1": fn, "fp_gt1": fp, "tn_gt1": tn,
    }


def metrics(frame: pd.DataFrame) -> dict:
    y, p = frame["y_true"].to_numpy(float), frame["y_pred"].to_numpy(float)
    result = {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)) if len(y) > 1 and np.std(y) else math.nan,
        "spearman": float(spearmanr(y, p).statistic) if len(y) > 1 and np.std(y) and np.std(p) else math.nan,
        "prediction_mean": float(np.mean(p)),
        "prediction_std": float(np.std(p, ddof=0)),
    }
    result.update(threshold_metrics(y, p))
    return result


def assert_pair(left: pd.DataFrame, right: pd.DataFrame, description: str) -> None:
    if "sample_id" in left and "sample_id" in right:
        merged = left[["sample_id", "y_true"]].merge(
            right[["sample_id", "y_true"]], on="sample_id", how="outer",
            suffixes=("_left", "_right"), indicator=True, validate="one_to_one",
        )
        if not merged["_merge"].eq("both").all() or not np.allclose(
            merged["y_true_left"], merged["y_true_right"], rtol=0, atol=0
        ):
            raise ValueError(f"Paired test membership/labels differ: {description}")
    elif len(left) != len(right) or not np.allclose(left["y_true"], right["y_true"], rtol=0, atol=0):
        raise ValueError(f"Cannot pair test rows: {description}")


def summarize(frame: pd.DataFrame, group_columns: list[str], metrics_to_summarize: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=False):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        row = dict(zip(group_columns, keys))
        row["splits"] = int(len(group))
        for column in metrics_to_summarize:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
            row[f"{column}_median"] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_metrics(per_split: pd.DataFrame, frames: dict, candidate: str, comparator: str) -> pd.DataFrame:
    candidate_rows = per_split.loc[per_split.model.eq(candidate)].set_index("split_seed")
    comparator_rows = per_split.loc[per_split.model.eq(comparator)].set_index("split_seed")
    if set(candidate_rows.index) != set(comparator_rows.index):
        raise ValueError(f"Unpaired splits: {candidate} vs {comparator}")
    rows = []
    for split in sorted(candidate_rows.index):
        assert_pair(frames[(candidate, int(split))], frames[(comparator, int(split))], f"{candidate}/{comparator}/split{split}")
        c, b = candidate_rows.loc[split], comparator_rows.loc[split]
        rows.append({
            "candidate_model": candidate, "comparator_model": comparator, "split_seed": int(split),
            "improvement_mae": float(b.mae - c.mae),
            "improvement_r2": float(c.r2 - b.r2),
            "improvement_spearman": float(c.spearman - b.spearman),
            "improvement_precision_gt1": float(c.precision_gt1 - b.precision_gt1),
            "improvement_recall_gt1": float(c.recall_gt1 - b.recall_gt1),
            "improvement_f2_gt1": float(c.f2_gt1 - b.f2_gt1),
            "tp_increase_gt1": float(c.tp_gt1 - b.tp_gt1),
            "fn_reduction_gt1": float(b.fn_gt1 - c.fn_gt1),
            "fp_reduction_gt1": float(b.fp_gt1 - c.fp_gt1),
            "tn_increase_gt1": float(c.tn_gt1 - b.tn_gt1),
            "prediction_mean_change": float(c.prediction_mean - b.prediction_mean),
            "prediction_std_change": float(c.prediction_std - b.prediction_std),
        })
    return pd.DataFrame(rows)


def go_no_go(p3_vs_p0: pd.DataFrame, max_mae_regression: float, max_spearman_drop: float, max_mean_lift: float) -> dict:
    n = len(p3_vs_p0)
    majority = n // 2 + 1
    wins = {
        "recall_up": int((p3_vs_p0.improvement_recall_gt1 > 0).sum()),
        "f2_up": int((p3_vs_p0.improvement_f2_gt1 > 0).sum()),
        "fn_down": int((p3_vs_p0.fn_reduction_gt1 > 0).sum()),
    }
    means = p3_vs_p0.drop(columns=["split_seed"]).mean(numeric_only=True).to_dict()
    checks = {
        "majority_recall_up": wins["recall_up"] >= majority,
        "majority_f2_up": wins["f2_up"] >= majority,
        "majority_fn_down": wins["fn_down"] >= majority,
        "mae_not_materially_worse": means["improvement_mae"] >= -max_mae_regression,
        "spearman_not_materially_worse": means["improvement_spearman"] >= -max_spearman_drop,
        "prediction_mean_not_materially_lifted": means["prediction_mean_change"] <= max_mean_lift,
    }
    return {
        "decision": "GO_EXTEND_TO_SPLIT100_109" if all(checks.values()) else "NO_GO_NEGATIVE_ABLATION",
        "criteria": {"majority": majority, "max_mae_regression": max_mae_regression, "max_spearman_drop": max_spearman_drop, "max_prediction_mean_lift": max_mean_lift},
        "wins": wins, "mean_paired_changes": means, "checks": checks,
        "note": "The prediction-mean condition is a safeguard, not causal proof; inspect paired CSV before expansion.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-root", type=Path, default=Path("results/fifth_pretraining/stage6_strict_no_mordred_fifth_ood_baseline"))
    parser.add_argument("--p3-root", type=Path, default=Path("results/fifth_pretraining/stage8_strict_no_mordred_fifth_ood"))
    parser.add_argument("--p3-label", default=P3_MODEL)
    parser.add_argument("--pt-df-checkpoint", type=Path, default=Path("results/fifth_pretraining/stage4_graphgps_pretraining/PT_DF/checkpoints/best_comp5_encoder_state_dict.pt"))
    parser.add_argument("--splits", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-mae-regression", type=float, default=0.05)
    parser.add_argument("--max-spearman-drop", type=float, default=0.05)
    parser.add_argument("--max-prediction-mean-lift", type=float, default=0.10)
    args = parser.parse_args()

    controls_root, p3_root = args.controls_root.resolve(), args.p3_root.resolve()
    pt_df = args.pt_df_checkpoint.resolve()
    if not pt_df.is_file():
        raise FileNotFoundError(pt_df)
    output = args.output_dir.resolve() if args.output_dir else p3_root / "analysis_strict_no_mordred"
    output.mkdir(parents=True, exist_ok=True)

    rows, frames = [], {}
    sources = {**{name: (controls_root, label) for name, label in CONTROL_LABELS.items()}, P3_MODEL: (p3_root, args.p3_label)}
    for model, (root, label) in sources.items():
        for split in args.splits:
            run_dir = root / label / f"split{split}"
            prediction_path = run_dir / "predictions.csv"
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            provenance = assert_p3_provenance(run_dir, pt_df) if model == P3_MODEL else assert_no_mordred(run_dir)[1]
            frame = selected_test_predictions(prediction_path)
            frames[(model, int(split))] = frame
            rows.append({"baseline": "stage8_strict_no_mordred_fifth_identity_ood", "model": model, "run_label": label, "split_seed": int(split), "initialization_mode": provenance.get("mode", "frozen_aux"), **metrics(frame)})

    per_split = pd.DataFrame(rows)
    summary = summarize(per_split, ["model"], list(METRICS))
    p3_vs_p0 = paired_metrics(per_split, frames, P3_MODEL, "P0_random_NoMordred")
    p3_vs_p2 = paired_metrics(per_split, frames, P3_MODEL, "P2_PT_DF_NoMordred")
    paired = pd.concat([p3_vs_p0, p3_vs_p2], ignore_index=True)
    paired_summary = summarize(paired, ["candidate_model", "comparator_model"], [column for column in paired.columns if column not in {"candidate_model", "comparator_model", "split_seed"}])
    decision = go_no_go(p3_vs_p0, args.max_mae_regression, args.max_spearman_drop, args.max_prediction_mean_lift)

    per_split.to_csv(output / "stage8_strict_no_mordred_per_split_metrics.csv", index=False)
    summary.to_csv(output / "stage8_strict_no_mordred_summary.csv", index=False)
    paired.to_csv(output / "stage8_strict_no_mordred_paired_per_split.csv", index=False)
    paired_summary.to_csv(output / "stage8_strict_no_mordred_paired_summary.csv", index=False)
    (output / "stage8_strict_no_mordred_go_no_go.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    print("=" * 118)
    print("STAGE8 STRICT NO-MORDRED FROZEN PT-DF AUXILIARY SCREENING")
    print("=" * 118)
    print(summary[["model", "splits", "mae_mean", "r2_mean", "spearman_mean", "precision_gt1_mean", "recall_gt1_mean", "f2_gt1_mean", "tp_gt1_mean", "fn_gt1_mean", "fp_gt1_mean", "tn_gt1_mean", "prediction_mean_mean", "prediction_std_mean"]].to_string(index=False))
    print()
    print(f"GO/NO-GO: {decision['decision']}")
    print(json.dumps(decision["checks"], indent=2))
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
