#!/usr/bin/env python3
"""Strict seed-paired O12 versus O13-B Fifth_class-embedding diagnostic."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


METRICS = ("mae", "r2", "pearson", "spearman")
KEYS = ["target_group", "split_seed", "split", "target"]
FOCUS = ("EE_before", "EE_after", "mRNA_Recovery_Efficiency")
CLASSES = ("single", "double")


def correlation(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or not np.std(truth) or not np.std(prediction):
        return math.nan
    return float(function(truth, prediction).statistic)


def class_value(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return "__unknown__"
    return str(value).strip().lower()


def prediction_dispersion(predictions: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for values, frame in predictions.groupby(KEYS, sort=False):
        truth_std = float(frame.y_true.std(ddof=0))
        if truth_std <= 0:
            raise ValueError(f"{label} has zero true standard deviation for {values}")
        rows.append({**dict(zip(KEYS, values)),
                     "prediction_std_to_true_std": float(frame.y_pred.std(ddof=0) / truth_std)})
    return pd.DataFrame(rows)


def per_seed(metrics_path: Path, predictions_path: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path, dtype={"sample_id": str})
    required_metric = {*KEYS, *METRICS}
    required_prediction = {*KEYS, "sample_id", "y_true", "y_pred"}
    if missing := required_metric.difference(metrics.columns):
        raise ValueError(f"{label} metrics miss columns: {sorted(missing)}")
    if missing := required_prediction.difference(predictions.columns):
        raise ValueError(f"{label} predictions miss columns: {sorted(missing)}")
    base = metrics[KEYS + list(METRICS)].copy()
    dispersion = prediction_dispersion(predictions, label)
    if base.duplicated(KEYS).any() or dispersion.duplicated(KEYS).any():
        raise ValueError(f"{label} has duplicate seed/split/target rows")
    output = base.merge(dispersion, on=KEYS, how="inner", validate="one_to_one")
    if len(output) != len(base):
        raise RuntimeError(f"{label} metrics and predictions do not align")
    return output, predictions


def paired(o12: pd.DataFrame, o13b: pd.DataFrame, protocol: str) -> pd.DataFrame:
    left, right = set(map(tuple, o12[KEYS].to_numpy())), set(map(tuple, o13b[KEYS].to_numpy()))
    if left != right:
        raise RuntimeError(f"{protocol} has non-identical O12/O13-B seed-target keys")
    return o12.merge(o13b, on=KEYS, suffixes=("_o12", "_o13b"), validate="one_to_one")


def summary(pairs: pd.DataFrame, protocol: str) -> pd.DataFrame:
    rows = []
    for (split, target), frame in pairs.groupby(["split", "target"], sort=True):
        row = {"protocol": protocol, "split": split, "target": target, "seeds": len(frame)}
        for metric in (*METRICS, "prediction_std_to_true_std"):
            o12, o13b = frame[f"{metric}_o12"], frame[f"{metric}_o13b"]
            delta = o13b - o12
            row.update({
                f"o12_{metric}_mean": float(o12.mean()),
                f"o12_{metric}_std": float(o12.std(ddof=1)),
                f"o13b_{metric}_mean": float(o13b.mean()),
                f"o13b_{metric}_std": float(o13b.std(ddof=1)),
                f"paired_delta_o13b_minus_o12_{metric}_mean": float(delta.mean()),
                f"paired_delta_o13b_minus_o12_{metric}_std": float(delta.std(ddof=1)),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def class_metrics(predictions: pd.DataFrame, classes: pd.DataFrame, label: str) -> pd.DataFrame:
    merged = predictions.merge(classes, on="sample_id", how="left", validate="many_to_one")
    if merged.fifth_class.isna().any():
        raise RuntimeError(f"{label} predictions include unknown source IDs")
    rows = []
    selected = merged.loc[merged.fifth_class.isin(CLASSES)]
    for values, frame in selected.groupby(KEYS + ["fifth_class"], sort=True):
        truth, prediction = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
        truth_std = float(truth.std(ddof=0))
        rows.append({
            **dict(zip(KEYS + ["fifth_class"], values)),
            "n": len(frame),
            "true_mean": float(truth.mean()),
            "predicted_mean": float(prediction.mean()),
            "mae": float(mean_absolute_error(truth, prediction)),
            "r2": float(r2_score(truth, prediction)) if truth_std else math.nan,
            "pearson": correlation(pearsonr, truth, prediction),
            "spearman": correlation(spearmanr, truth, prediction),
            "prediction_std_to_true_std": float(prediction.std(ddof=0) / truth_std) if truth_std else math.nan,
        })
    output = pd.DataFrame(rows)
    if output.empty or output.duplicated(KEYS + ["fifth_class"]).any():
        raise RuntimeError(f"{label} has incomplete or duplicate Fifth_class metrics")
    return output


def class_summary(o12: pd.DataFrame, o13b: pd.DataFrame, protocol: str) -> pd.DataFrame:
    class_keys = KEYS + ["fifth_class"]
    left, right = set(map(tuple, o12[class_keys].to_numpy())), set(map(tuple, o13b[class_keys].to_numpy()))
    if left != right:
        raise RuntimeError(f"{protocol} has inconsistent O12/O13-B Fifth_class rows")
    joined = o12.merge(o13b, on=class_keys, suffixes=("_o12", "_o13b"), validate="one_to_one")
    rows = []
    for (split, target, fifth_class), frame in joined.groupby(["split", "target", "fifth_class"], sort=True):
        row = {"protocol": protocol, "split": split, "target": target,
               "fifth_class": fifth_class, "seeds": len(frame)}
        for metric in (*METRICS, "prediction_std_to_true_std", "true_mean", "predicted_mean"):
            left_value, right_value = frame[f"{metric}_o12"], frame[f"{metric}_o13b"]
            delta = right_value - left_value
            row.update({
                f"o12_{metric}_mean": float(left_value.mean()),
                f"o12_{metric}_std": float(left_value.std(ddof=1)),
                f"o13b_{metric}_mean": float(right_value.mean()),
                f"o13b_{metric}_std": float(right_value.std(ddof=1)),
                f"paired_delta_o13b_minus_o12_{metric}_mean": float(delta.mean()),
                f"paired_delta_o13b_minus_o12_{metric}_std": float(delta.std(ddof=1)),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def decision(overall: pd.DataFrame) -> list[str]:
    ood = overall.query("protocol == 'fifth_identity_ood' and split == 'test'").set_index("target")
    random = overall.query("protocol == 'random' and split == 'test'").set_index("target")
    focus = ood.loc[list(FOCUS)]
    r2 = int(focus.paired_delta_o13b_minus_o12_r2_mean.gt(0).sum())
    rank = int(focus.paired_delta_o13b_minus_o12_spearman_mean.gt(0).sum())
    compression = int((
        (focus.o13b_prediction_std_to_true_std_mean - 1).abs()
        < (focus.o12_prediction_std_to_true_std_mean - 1).abs()
    ).sum())
    random_harm = int(random.loc[list(FOCUS), "paired_delta_o13b_minus_o12_mae_mean"].gt(0).sum())
    if r2 >= 2 and rank >= 2 and compression >= 2 and random_harm <= 1:
        verdict = "O13-B accepted"
    elif r2 == 0 and rank == 0:
        verdict = "O13-B rejected"
    else:
        verdict = "mixed / insufficient evidence"
    return [
        verdict,
        f"Focus targets: Fifth-OOD R² improves on {r2}/3; Spearman improves on {rank}/3; "
        f"dispersion moves closer to 1 on {compression}/3; random-split MAE worsens on {random_harm}/3.",
        "Proceed to pooling/Fifth molecular-representation changes only if the paired OOD results do not provide "
        "sufficient evidence that this coarse single/double embedding resolves the deficit.",
    ]


def table_lines(data: pd.DataFrame, protocol: str) -> list[str]:
    lines = [f"## {protocol}: test", "",
             "| target | O12 MAE | O13-B MAE | Δ MAE | O12 R² | O13-B R² | Δ R² | O12 Spearman | O13-B Spearman | Δ Spearman | O12 σpred/σtrue | O13-B σpred/σtrue | Δ ratio |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in data.query("protocol == @protocol and split == 'test'").itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.o12_mae_mean:.3f} ± {row.o12_mae_std:.3f} "
            f"| {row.o13b_mae_mean:.3f} ± {row.o13b_mae_std:.3f} "
            f"| {row.paired_delta_o13b_minus_o12_mae_mean:+.3f} "
            f"| {row.o12_r2_mean:.3f} | {row.o13b_r2_mean:.3f} | {row.paired_delta_o13b_minus_o12_r2_mean:+.3f} "
            f"| {row.o12_spearman_mean:.3f} | {row.o13b_spearman_mean:.3f} | {row.paired_delta_o13b_minus_o12_spearman_mean:+.3f} "
            f"| {row.o12_prediction_std_to_true_std_mean:.3f} | {row.o13b_prediction_std_to_true_std_mean:.3f} "
            f"| {row.paired_delta_o13b_minus_o12_prediction_std_to_true_std_mean:+.3f} |")
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    for model in ("o12", "o13b"):
        for protocol in ("random", "ood"):
            parser.add_argument(f"--{model}-{protocol}-metrics", type=Path, required=True)
            parser.add_argument(f"--{model}-{protocol}-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.input_csv, dtype={"ID": str})
    if source.ID.duplicated().any() or "Fifth_class" not in source:
        raise ValueError("input CSV requires unique ID and Fifth_class columns")
    classes = pd.DataFrame({"sample_id": source.ID, "fifth_class": source.Fifth_class.map(class_value)})

    pairs, class_summaries = {}, []
    for protocol, short in (("random", "random"), ("fifth_identity_ood", "ood")):
        o12, o12_predictions = per_seed(getattr(args, f"o12_{short}_metrics"),
                                         getattr(args, f"o12_{short}_predictions"), f"O12 {protocol}")
        o13b, o13b_predictions = per_seed(getattr(args, f"o13b_{short}_metrics"),
                                           getattr(args, f"o13b_{short}_predictions"), f"O13-B {protocol}")
        pairs[protocol] = paired(o12, o13b, protocol)
        class_summaries.append(class_summary(class_metrics(o12_predictions, classes, f"O12 {protocol}"),
                                             class_metrics(o13b_predictions, classes, f"O13-B {protocol}"), protocol))
    overall = pd.concat([summary(frame, protocol) for protocol, frame in pairs.items()], ignore_index=True)
    per_seed_pairs = pd.concat([frame.assign(protocol=protocol) for protocol, frame in pairs.items()], ignore_index=True)
    by_class = pd.concat(class_summaries, ignore_index=True)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_seed_pairs.to_csv(output / "paired_per_seed_metrics_and_dispersion.csv", index=False)
    overall.to_csv(output / "o12_vs_o13b_paired_summary.csv", index=False)
    by_class.to_csv(output / "o12_vs_o13b_by_fifth_class_summary.csv", index=False)

    lines = ["# O13-B Fifth_class embedding paired diagnostic", "",
             "O13-B retains complete O12 fusion and differs only by the input-derived Fifth_class embedding. "
             "Deltas are O13-B minus O12; negative MAE and positive R²/Spearman are improvements. "
             "A prediction-dispersion ratio closer to 1 is less regression-to-the-mean.", ""]
    lines.extend(table_lines(overall, "random"))
    lines.extend(table_lines(overall, "fifth_identity_ood"))
    lines.extend(["## Fifth-OOD within-class ranking", "",
                  "The following table is intentionally restricted to single/double test rows. It tests whether any "
                  "benefit remains within the coarse category, rather than merely separating category means.", "",
                  "| target | class | O12 Spearman | O13-B Spearman | Δ Spearman | O12 predicted mean | O13-B predicted mean |", "|---|---|---:|---:|---:|---:|---:|"])
    class_table = by_class.query("protocol == 'fifth_identity_ood' and split == 'test'")
    for row in class_table.itertuples(index=False):
        lines.append(f"| {row.target} | {row.fifth_class} | {row.o12_spearman_mean:.3f} | "
                     f"{row.o13b_spearman_mean:.3f} | {row.paired_delta_o13b_minus_o12_spearman_mean:+.3f} | "
                     f"{row.o12_predicted_mean_mean:.3f} | {row.o13b_predicted_mean_mean:.3f} |")
    lines.extend(["", "## Pre-specified verdict", "", *[f"- {item}" for item in decision(overall)], ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(overall.query("split == 'test'").to_string(index=False))


if __name__ == "__main__":
    main()
