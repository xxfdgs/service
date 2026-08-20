#!/usr/bin/env python3
"""Write the evidence-bound final Markdown/CSV summary for this objective."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


LOWER_BETTER = {"mae", "rmse", "median_ae", "underprediction_mae", "fn", "fp"}
METRICS = ("mae", "rmse", "median_ae", "mean_signed_error", "underprediction_mae",
           "recall_gt1", "f2_gt1", "fn", "fp", "r2", "spearman", "prediction_mean")


def markdown(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("" if not np.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def read_audit(directory: Path) -> pd.DataFrame:
    path = directory / "p1_ptd_internal_ood_metrics_per_split.csv"
    frame = pd.read_csv(path)
    if frame.duplicated(["seed", "subset"]).any():
        raise ValueError(f"Non-unique seed/subset metrics: {path}")
    return frame


def parse_variant(value: str) -> tuple[str, Path]:
    label, sep, path = value.partition("=")
    if not sep or not label or not path:
        raise ValueError(f"Variant must be LABEL=AUDIT_DIR, got {value!r}")
    return label, Path(path)


def summarize(label: str, candidate: pd.DataFrame, baseline: pd.DataFrame) -> tuple[dict, list[dict]]:
    high = candidate.loc[candidate.subset.eq("double_gt1")]
    double = candidate.loc[candidate.subset.eq("double")]
    if high.empty or len(high) != len(double):
        raise ValueError(f"{label}: incomplete double/high metric audit")
    row = {"variant": label, "completed_splits": int(len(high))}
    for prefix, subset in (("double_gt1", high), ("double", double)):
        for metric in METRICS:
            values = pd.to_numeric(subset[metric], errors="coerce")
            row[f"{prefix}_{metric}_mean"] = float(values.mean())
            row[f"{prefix}_{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else math.nan
    paired = []
    for subset in ("double_gt1", "double"):
        left = baseline.loc[baseline.subset.eq(subset)].set_index("seed")
        right = candidate.loc[candidate.subset.eq(subset)].set_index("seed")
        common = left.index.intersection(right.index).sort_values()
        for metric in METRICS:
            delta = right.loc[common, metric].astype(float) - left.loc[common, metric].astype(float)
            better = delta.lt(0) if metric in LOWER_BETTER else delta.gt(0)
            paired.append({
                "variant": label, "subset": subset, "metric": metric,
                "matched_splits": int(len(common)), "mean_delta": float(delta.mean()),
                "std_delta": float(delta.std(ddof=1)) if len(delta) > 1 else math.nan,
                "candidate_better_splits": int(better.sum()),
            })
    return row, paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--mechanism-diagnosis", type=Path, required=True)
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--variant", action="append", default=[], metavar="LABEL=AUDIT_DIR")
    parser.add_argument("--external-report", type=Path, required=True)
    parser.add_argument("--recommendation", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    baseline = read_audit(args.baseline_audit)
    base_row, _ = summarize("P1_PT_D_strict_no_mordred", baseline, baseline)
    rows, paired = [base_row], []
    for value in args.variant:
        label, directory = parse_variant(value)
        row, comparisons = summarize(label, read_audit(directory), baseline)
        rows.append(row); paired.extend(comparisons)
    table, pair_table = pd.DataFrame(rows), pd.DataFrame(paired)
    table.to_csv(output / "double_gt1_variant_summary.csv", index=False)
    pair_table.to_csv(output / "double_gt1_paired_deltas.csv", index=False)

    mechanism = json.loads(args.mechanism_diagnosis.read_text())
    runtime = json.loads(args.runtime_audit.read_text())
    errors = pd.read_csv(args.baseline_audit / "p1_ptd_double_gt1_error_audit.csv")
    false_negatives = errors.loc[errors.false_negative].copy()
    top = (false_negatives.loc[false_negatives.groupby("sample_id")["absolute_error"].idxmax()]
           .sort_values("absolute_error", ascending=False)
           .loc[:, ["sample_id", "Fifth", "y_true", "y_pred", "absolute_error",
                    "unseen_fifth_identity", "nearest_train_fifth_tanimoto",
                    "nearest_neighbor_norm_before_mean", "ensemble_std"]]
           .head(12))
    top.to_csv(output / "p1_top_double_gt1_false_negatives.csv", index=False)

    display = table[[
        "variant", "completed_splits", "double_gt1_mae_mean", "double_gt1_recall_gt1_mean",
        "double_gt1_f2_gt1_mean", "double_gt1_fn_mean", "double_mae_mean",
        "double_r2_mean", "double_spearman_mean", "double_fp_mean",
    ]].copy()
    display.columns = ["variant", "splits", "high_MAE", "high_recall", "high_F2", "high_FN",
                       "double_MAE", "double_R2", "double_Spearman", "double_FP"]
    high_deltas = pair_table.loc[(pair_table.subset.eq("double_gt1")) & pair_table.metric.isin(["mae", "fn", "recall_gt1", "f2_gt1"])]
    lines = [
        "# double > 1 optimization summary", "",
        "## Protocol", "",
        "- Baseline: strict No-Mordred P1 PT-D, one-output `Norm_before`, full-data training, validation-selected checkpoint.",
        "- Internal selection data: locked 700-row source and frozen Fifth-identity OOD manifests, seeds 100–109; no external labels are used in any audit or selection table below.",
        "- External evaluation: the earlier fixed H30 readout is retained only as a final descriptive result. No later Stage-10 candidate was scored or selected using `new_validation`.", "",
        "## Baseline failure mechanism", "",
        f"- High-double FN rows: {mechanism['false_negative_rows']}/{mechanism['double_gt1_rows']}; all had unseen Fifth identities, yet mean nearest Morgan similarity was {mechanism['fn_nearest_similarity_mean']:.3f}.",
        f"- Strong far-structure A evidence: {mechanism['fn_flag_fraction']['A_representation_ood']:.1%}; split-level negative-shrinkage B evidence: {mechanism['fn_flag_fraction']['B_shrinkage']:.1%}; uncertainty C flag: {mechanism['fn_flag_fraction']['C_uncertainty']:.1%}; rank-preserved-but-biased D evidence: {mechanism['fn_flag_fraction']['D_objective_mismatch']:.1%}.",
        f"- Aggregate high-double signed error is {mechanism['double_high_mean_signed_error']:.3f}; pooled Spearman is {mechanism['double_high_spearman']:.3f}. This supports shrinkage/objective imbalance as the immediate intervention target, while Fifth identity remains OOD by construction.", "",
        "## Fifth feature runtime check", "",
        f"- {runtime['status']}: selected P1 batch fifth GraphGPS gradient norm={runtime['fifth_graph_encoder_gradient_norm']:.3f}, Fifth-class embedding gradient norm={runtime['fifth_class_embedding_gradient_norm']:.3f}, auxiliary encoder gradient norm={runtime['component_aux_encoder_gradient_norm']:.3f}.",
        f"- Fifth ratios in the asserted real batch were [{runtime['fifth_ratio_min']:.3f}, {runtime['fifth_ratio_max']:.3f}] with {runtime['fifth_ratio_zero_rows']} zeros; this source batch does not exercise zero-ratio padding.", "",
        "## Internal Fifth-OOD variants", "", markdown(display), "",
        "## Paired high-double changes versus P1", "",
        markdown(high_deltas[["variant", "metric", "matched_splits", "mean_delta", "candidate_better_splits"]]) if not high_deltas.empty else "No candidate comparison available.", "",
        "## Most severe P1 high-double false negatives", "", markdown(top), "",
        "## Frozen external readout", "",
        f"See `{args.external_report.resolve()}` and its point-level CSV/figures. It must not be used to select or reweight later candidates.", "",
        "## Recommendation", "", args.recommendation, "",
        "## Reproducibility and safety checks", "",
        "- Every reported training run records frozen manifest membership, selected checkpoint, target scaler, effective config, strict PT-D transfer report, predictions, and threshold metrics.",
        "- The external inference controller replaces every property label with zero before loader construction and joins labels only after prediction for metrics.",
    ]
    (output / "double_gt1_optimization_summary.md").write_text("\n".join(lines) + "\n")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
