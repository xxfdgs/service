#!/usr/bin/env python3
"""Strict seed-paired O12 vs O13-C (mean graph pooling) comparison."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


KEYS = ["target_group", "split_seed", "split", "target"]
METRICS = ("mae", "r2", "pearson", "spearman", "prediction_std", "true_std", "prediction_std_to_true_std")
FOCUS = ("EE_before", "EE_after", "mRNA_Recovery_Efficiency")


def corr(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or not np.std(truth) or not np.std(prediction):
        return math.nan
    return float(function(truth, prediction).statistic)


def canonical_class(value: object) -> str:
    return "__unknown__" if pd.isna(value) or not str(value).strip() else str(value).strip().lower()


def per_seed(metrics_path: Path, predictions_path: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric = pd.read_csv(metrics_path)
    prediction = pd.read_csv(predictions_path, dtype={"sample_id": str})
    required_metric = {*KEYS, "mae", "r2", "pearson", "spearman"}
    required_prediction = {*KEYS, "sample_id", "y_true", "y_pred"}
    if missing := required_metric.difference(metric.columns):
        raise ValueError(f"{label} metrics missing {sorted(missing)}")
    if missing := required_prediction.difference(prediction.columns):
        raise ValueError(f"{label} predictions missing {sorted(missing)}")
    rows = []
    for values, frame in prediction.groupby(KEYS, sort=False):
        true_std = float(frame.y_true.std(ddof=0))
        if true_std <= 0:
            raise RuntimeError(f"{label} has zero true standard deviation: {values}")
        rows.append({**dict(zip(KEYS, values)), "prediction_std": float(frame.y_pred.std(ddof=0)),
                     "true_std": true_std,
                     "prediction_std_to_true_std": float(frame.y_pred.std(ddof=0) / true_std)})
    dispersion = pd.DataFrame(rows)
    metric = metric[KEYS + ["mae", "r2", "pearson", "spearman"]]
    if metric.duplicated(KEYS).any() or dispersion.duplicated(KEYS).any():
        raise RuntimeError(f"{label} has duplicate seed/split/target values")
    output = metric.merge(dispersion, on=KEYS, how="inner", validate="one_to_one")
    if len(output) != len(metric):
        raise RuntimeError(f"{label} metrics/predictions do not align")
    return output, prediction


def pair(o12: pd.DataFrame, o13c: pd.DataFrame, protocol: str) -> pd.DataFrame:
    if set(map(tuple, o12[KEYS].to_numpy())) != set(map(tuple, o13c[KEYS].to_numpy())):
        raise RuntimeError(f"{protocol}: O12 and O13-C have different seed/split/target keys")
    return o12.merge(o13c, on=KEYS, suffixes=("_o12", "_o13c"), validate="one_to_one")


def summarize(paired: pd.DataFrame, protocol: str) -> pd.DataFrame:
    rows = []
    for (split, target), frame in paired.groupby(["split", "target"], sort=True):
        row = {"protocol": protocol, "split": split, "target": target, "seeds": len(frame)}
        for metric in METRICS:
            o12, o13c = frame[f"{metric}_o12"], frame[f"{metric}_o13c"]
            delta = o13c - o12
            row.update({
                f"o12_{metric}_mean": float(o12.mean()), f"o12_{metric}_std": float(o12.std(ddof=1)),
                f"o13c_{metric}_mean": float(o13c.mean()), f"o13c_{metric}_std": float(o13c.std(ddof=1)),
                f"paired_delta_o13c_minus_o12_{metric}_mean": float(delta.mean()),
                f"paired_delta_o13c_minus_o12_{metric}_std": float(delta.std(ddof=1)),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def within_class(predictions: pd.DataFrame, source: pd.DataFrame, label: str) -> pd.DataFrame:
    classes = source[["ID", "Fifth_class"]].rename(columns={"ID": "sample_id"}).copy()
    classes["fifth_class"] = classes.Fifth_class.map(canonical_class)
    joined = predictions.merge(classes[["sample_id", "fifth_class"]], on="sample_id", how="left",
                               validate="many_to_one")
    if joined.fifth_class.isna().any():
        raise RuntimeError(f"{label} has predictions outside the locked input CSV")
    rows = []
    for values, frame in joined.loc[joined.fifth_class.isin(("single", "double"))].groupby(KEYS + ["fifth_class"]):
        truth, prediction = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
        rows.append({**dict(zip(KEYS + ["fifth_class"], values)), "n": len(frame),
                     "mae": float(mean_absolute_error(truth, prediction)),
                     "r2": float(r2_score(truth, prediction)) if np.std(truth) else math.nan,
                     "pearson": corr(pearsonr, truth, prediction),
                     "spearman": corr(spearmanr, truth, prediction),
                     "true_mean": float(truth.mean()), "predicted_mean": float(prediction.mean())})
    return pd.DataFrame(rows)


def class_summary(o12: pd.DataFrame, o13c: pd.DataFrame, protocol: str) -> pd.DataFrame:
    keys = KEYS + ["fifth_class"]
    if set(map(tuple, o12[keys].to_numpy())) != set(map(tuple, o13c[keys].to_numpy())):
        raise RuntimeError(f"{protocol}: class-wise O12/O13-C keys differ")
    joined = o12.merge(o13c, on=keys, suffixes=("_o12", "_o13c"), validate="one_to_one")
    rows = []
    for values, frame in joined.groupby(["split", "target", "fifth_class"], sort=True):
        row = {"protocol": protocol, "split": values[0], "target": values[1],
               "fifth_class": values[2], "seeds": len(frame)}
        for metric in ("mae", "r2", "pearson", "spearman", "true_mean", "predicted_mean"):
            delta = frame[f"{metric}_o13c"] - frame[f"{metric}_o12"]
            row.update({f"o12_{metric}_mean": float(frame[f"{metric}_o12"].mean()),
                        f"o12_{metric}_std": float(frame[f"{metric}_o12"].std(ddof=1)),
                        f"o13c_{metric}_mean": float(frame[f"{metric}_o13c"].mean()),
                        f"o13c_{metric}_std": float(frame[f"{metric}_o13c"].std(ddof=1)),
                        f"paired_delta_o13c_minus_o12_{metric}_mean": float(delta.mean()),
                        f"paired_delta_o13c_minus_o12_{metric}_std": float(delta.std(ddof=1))})
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(summary: pd.DataFrame) -> list[str]:
    ood = summary.query("protocol == 'fifth_identity_ood' and split == 'test'").set_index("target").loc[list(FOCUS)]
    random = summary.query("protocol == 'random' and split == 'test'").set_index("target").loc[list(FOCUS)]
    r2 = int(ood.paired_delta_o13c_minus_o12_r2_mean.gt(0).sum())
    rank = int(ood.paired_delta_o13c_minus_o12_spearman_mean.gt(0).sum())
    compression = int(((ood.o13c_prediction_std_to_true_std_mean - 1).abs() <
                       (ood.o12_prediction_std_to_true_std_mean - 1).abs()).sum())
    harm = int(random.paired_delta_o13c_minus_o12_mae_mean.gt(0).sum())
    if r2 >= 2 and rank >= 2 and compression >= 2 and harm <= 1:
        label = "O13-C accepted"
    elif r2 == 0 and rank == 0:
        label = "O13-C rejected"
    else:
        label = "mixed / insufficient evidence"
    return [label, f"Focus targets: OOD R² improves {r2}/3; OOD Spearman improves {rank}/3; "
            f"dispersion moves closer to 1 for {compression}/3; random MAE worsens {harm}/3."]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    for model in ("o12", "o13c"):
        for protocol in ("random", "ood"):
            parser.add_argument(f"--{model}-{protocol}-metrics", type=Path, required=True)
            parser.add_argument(f"--{model}-{protocol}-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.input_csv, dtype={"ID": str})
    if source.ID.duplicated().any() or "Fifth_class" not in source:
        raise ValueError("Locked input must contain unique ID and Fifth_class")

    pairs, class_rows = {}, []
    for protocol, short in (("random", "random"), ("fifth_identity_ood", "ood")):
        o12, o12_pred = per_seed(getattr(args, f"o12_{short}_metrics"), getattr(args, f"o12_{short}_predictions"), f"O12 {protocol}")
        o13c, o13c_pred = per_seed(getattr(args, f"o13c_{short}_metrics"), getattr(args, f"o13c_{short}_predictions"), f"O13-C {protocol}")
        pairs[protocol] = pair(o12, o13c, protocol)
        class_rows.append(class_summary(within_class(o12_pred, source, f"O12 {protocol}"),
                                        within_class(o13c_pred, source, f"O13-C {protocol}"), protocol))
    summary_rows = pd.concat([summarize(value, key) for key, value in pairs.items()], ignore_index=True)
    paired_rows = pd.concat([value.assign(protocol=key) for key, value in pairs.items()], ignore_index=True)
    class_rows = pd.concat(class_rows, ignore_index=True)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    paired_rows.to_csv(output / "paired_per_seed_metrics_and_dispersion.csv", index=False)
    summary_rows.to_csv(output / "o12_vs_o13c_paired_summary.csv", index=False)
    class_rows.to_csv(output / "o12_vs_o13c_by_fifth_class_summary.csv", index=False)
    lines = ["# O13-C mean graph-pooling paired diagnostic", "",
             "O13-C differs from O12 only by component-5 GraphGPS graph pooling: add → mean. "
             "All deltas are O13-C minus O12.", ""]
    for protocol in ("random", "fifth_identity_ood"):
        lines += [f"## {protocol}: test", "", "| target | ΔMAE | ΔR² | ΔPearson | ΔSpearman | Δσpred | Δσtrue | Δ(σpred/σtrue) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in summary_rows.query("protocol == @protocol and split == 'test'").itertuples(index=False):
            lines.append(f"| {row.target} | {row.paired_delta_o13c_minus_o12_mae_mean:+.3f} | "
                         f"{row.paired_delta_o13c_minus_o12_r2_mean:+.3f} | "
                         f"{row.paired_delta_o13c_minus_o12_pearson_mean:+.3f} | "
                         f"{row.paired_delta_o13c_minus_o12_spearman_mean:+.3f} | "
                         f"{row.paired_delta_o13c_minus_o12_prediction_std_mean:+.3f} | "
                         f"{row.paired_delta_o13c_minus_o12_true_std_mean:+.3f} | "
                         f"{row.paired_delta_o13c_minus_o12_prediction_std_to_true_std_mean:+.3f} |")
        lines.append("")
    lines += ["## Fifth-OOD within-class Spearman", "", "| target | class | Δ Spearman |", "|---|---|---:|"]
    for row in class_rows.query("protocol == 'fifth_identity_ood' and split == 'test'").itertuples(index=False):
        lines.append(f"| {row.target} | {row.fifth_class} | {row.paired_delta_o13c_minus_o12_spearman_mean:+.3f} |")
    lines += ["", "## Verdict", "", *[f"- {line}" for line in verdict(summary_rows)], ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary_rows.query("split == 'test'").to_string(index=False))


if __name__ == "__main__":
    main()
