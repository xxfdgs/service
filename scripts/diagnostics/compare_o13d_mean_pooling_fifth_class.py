#!/usr/bin/env python3
"""Four-model, seed-paired O12/O13-B/O13-C/O13-D diagnostic comparison.

O13-D is tested primarily against O13-C.  O12-vs-D and O13-B-vs-D are
reported as supporting paired comparisons.  The script never retrains,
selects, calibrates, or changes a checkpoint.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


MODELS = ("o12", "o13b", "o13c", "o13d")
KEYS = ["target_group", "split_seed", "split", "target"]
METRICS = ("mae", "r2", "pearson", "spearman", "prediction_std", "true_std",
           "prediction_std_to_true_std")
CORE_METRICS = ("mae", "r2", "spearman", "prediction_std_to_true_std")
FOCUS = ("EE_before", "EE_after", "mRNA_Recovery_Efficiency")


def corr(function, true: np.ndarray, predicted: np.ndarray) -> float:
    if len(true) < 2 or not np.std(true) or not np.std(predicted):
        return math.nan
    return float(function(true, predicted).statistic)


def canonical_class(value: object) -> str:
    return "__unknown__" if pd.isna(value) or not str(value).strip() else str(value).strip().lower()


def load_model(metrics_path: Path, predictions_path: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric, prediction = pd.read_csv(metrics_path), pd.read_csv(predictions_path, dtype={"sample_id": str})
    required_metric, required_pred = {*KEYS, "mae", "r2", "pearson", "spearman"}, {*KEYS, "sample_id", "y_true", "y_pred"}
    if missing := required_metric.difference(metric.columns):
        raise ValueError(f"{label} metrics miss {sorted(missing)}")
    if missing := required_pred.difference(prediction.columns):
        raise ValueError(f"{label} predictions miss {sorted(missing)}")
    rows = []
    for values, frame in prediction.groupby(KEYS, sort=False):
        true_std = float(frame.y_true.std(ddof=0))
        if true_std <= 0:
            raise RuntimeError(f"{label} has zero true standard deviation at {values}")
        rows.append({**dict(zip(KEYS, values)), "prediction_std": float(frame.y_pred.std(ddof=0)),
                     "true_std": true_std, "prediction_std_to_true_std": float(frame.y_pred.std(ddof=0) / true_std)})
    dispersion = pd.DataFrame(rows)
    metric = metric[KEYS + ["mae", "r2", "pearson", "spearman"]]
    if metric.duplicated(KEYS).any() or dispersion.duplicated(KEYS).any():
        raise RuntimeError(f"{label} has duplicate seed/split/target rows")
    result = metric.merge(dispersion, on=KEYS, how="inner", validate="one_to_one")
    if len(result) != len(metric):
        raise RuntimeError(f"{label} metric/prediction alignment failed")
    return result, prediction


def ensure_same_keys(frames: dict[str, pd.DataFrame], protocol: str) -> None:
    reference = set(map(tuple, frames["o12"][KEYS].to_numpy()))
    for name, frame in frames.items():
        if set(map(tuple, frame[KEYS].to_numpy())) != reference:
            raise RuntimeError(f"{protocol}: {name} differs from O12 in seed/split/target membership")


def paired_summary(reference: pd.DataFrame, candidate: pd.DataFrame, ref_name: str,
                   cand_name: str, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = reference.merge(candidate, on=KEYS, suffixes=(f"_{ref_name}", f"_{cand_name}"),
                             validate="one_to_one")
    rows = []
    for (split, target), frame in joined.groupby(["split", "target"], sort=True):
        row = {"protocol": protocol, "comparison": f"{cand_name}_minus_{ref_name}",
               "reference": ref_name, "candidate": cand_name, "split": split,
               "target": target, "seeds": len(frame)}
        for metric in METRICS:
            left, right = frame[f"{metric}_{ref_name}"], frame[f"{metric}_{cand_name}"]
            delta = right - left
            row.update({
                f"{ref_name}_{metric}_mean": float(left.mean()), f"{ref_name}_{metric}_std": float(left.std(ddof=1)),
                f"{cand_name}_{metric}_mean": float(right.mean()), f"{cand_name}_{metric}_std": float(right.std(ddof=1)),
                f"paired_delta_{cand_name}_minus_{ref_name}_{metric}_mean": float(delta.mean()),
                f"paired_delta_{cand_name}_minus_{ref_name}_{metric}_std": float(delta.std(ddof=1)),
            })
        rows.append(row)
    return pd.DataFrame(rows), joined


def class_metrics(predictions: pd.DataFrame, source: pd.DataFrame, label: str) -> pd.DataFrame:
    classes = source[["ID", "Fifth_class"]].rename(columns={"ID": "sample_id"}).copy()
    classes["fifth_class"] = classes.Fifth_class.map(canonical_class)
    joined = predictions.merge(classes[["sample_id", "fifth_class"]], on="sample_id", how="left",
                               validate="many_to_one")
    if joined.fifth_class.isna().any():
        raise RuntimeError(f"{label} includes sample IDs outside locked source data")
    rows = []
    for values, frame in joined.loc[joined.fifth_class.isin(("single", "double"))].groupby(KEYS + ["fifth_class"], sort=True):
        true, prediction = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
        rows.append({**dict(zip(KEYS + ["fifth_class"], values)), "n": len(frame),
                     "mae": float(mean_absolute_error(true, prediction)),
                     "r2": float(r2_score(true, prediction)) if np.std(true) else math.nan,
                     "pearson": corr(pearsonr, true, prediction), "spearman": corr(spearmanr, true, prediction),
                     "true_mean": float(true.mean()), "predicted_mean": float(prediction.mean())})
    result = pd.DataFrame(rows)
    if result.empty or result.duplicated(KEYS + ["fifth_class"]).any():
        raise RuntimeError(f"{label} has incomplete class-wise metrics")
    return result


def class_four_model_summary(frames: dict[str, pd.DataFrame], protocol: str) -> pd.DataFrame:
    class_keys = KEYS + ["fifth_class"]
    reference = set(map(tuple, frames["o12"][class_keys].to_numpy()))
    for name, frame in frames.items():
        if set(map(tuple, frame[class_keys].to_numpy())) != reference:
            raise RuntimeError(f"{protocol}: class-wise rows differ for {name}")
    combined = frames["o12"]
    for name in ("o13b", "o13c", "o13d"):
        combined = combined.merge(frames[name], on=class_keys, suffixes=("", f"_{name}"), validate="one_to_one")
    # O12 columns lack a suffix after the first merge.
    combined = combined.rename(columns={column: f"{column}_o12" for column in
                                       ("mae", "r2", "pearson", "spearman", "true_mean", "predicted_mean", "n")})
    rows = []
    for values, frame in combined.groupby(["split", "target", "fifth_class"], sort=True):
        row = {"protocol": protocol, "split": values[0], "target": values[1], "fifth_class": values[2], "seeds": len(frame)}
        for metric in ("mae", "r2", "pearson", "spearman", "true_mean", "predicted_mean"):
            for model in MODELS:
                suffix = "_o12" if model == "o12" else f"_{model}"
                values_for_model = frame[f"{metric}{suffix}"]
                row[f"{model}_{metric}_mean"] = float(values_for_model.mean())
                row[f"{model}_{metric}_std"] = float(values_for_model.std(ddof=1))
            for reference_model in ("o12", "o13b", "o13c"):
                delta = frame[f"{metric}_o13d"] - frame[f"{metric}_{reference_model if reference_model != 'o12' else 'o12'}"]
                row[f"paired_delta_o13d_minus_{reference_model}_{metric}_mean"] = float(delta.mean())
                row[f"paired_delta_o13d_minus_{reference_model}_{metric}_std"] = float(delta.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def interaction(per_seed: dict[str, pd.DataFrame], protocol: str) -> pd.DataFrame:
    merged = per_seed["o12"].merge(per_seed["o13b"], on=KEYS, suffixes=("_o12", "_o13b"), validate="one_to_one")
    merged = merged.merge(per_seed["o13c"], on=KEYS, validate="one_to_one")
    merged = merged.merge(per_seed["o13d"], on=KEYS, suffixes=("_o13c", "_o13d"), validate="one_to_one")
    rows = []
    for (split, target), frame in merged.groupby(["split", "target"], sort=True):
        row = {"protocol": protocol, "split": split, "target": target, "seeds": len(frame)}
        for metric in ("mae", "r2", "pearson", "spearman", "prediction_std_to_true_std"):
            # Interaction: (D-O12) - (B-O12) - (C-O12) = D - B - C + O12.
            value = frame[f"{metric}_o13d"] - frame[f"{metric}_o13b"] - frame[f"{metric}_o13c"] + frame[f"{metric}_o12"]
            row[f"interaction_{metric}_mean"] = float(value.mean())
            row[f"interaction_{metric}_std"] = float(value.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(o13c_vs_d: pd.DataFrame, class_summary: pd.DataFrame) -> list[str]:
    test = o13c_vs_d.query("protocol == 'fifth_identity_ood' and split == 'test'").set_index("target").loc[list(FOCUS)]
    rank = int(test.paired_delta_o13d_minus_o13c_spearman_mean.gt(0).sum())
    r2 = int(test.paired_delta_o13d_minus_o13c_r2_mean.gt(0).sum())
    mae = int(test.paired_delta_o13d_minus_o13c_mae_mean.lt(0).sum())
    ratio = int(((test.o13d_prediction_std_to_true_std_mean - 1).abs() <
                 (test.o13c_prediction_std_to_true_std_mean - 1).abs()).sum())
    within = class_summary.query("protocol == 'fifth_identity_ood' and split == 'test' and target in @FOCUS")
    within_rank = int(within.paired_delta_o13d_minus_o13c_spearman_mean.gt(0).sum())
    if r2 >= 2 and rank >= 2 and mae >= 2 and ratio >= 2:
        result = "O13-D accepted over O13-C"
    elif r2 == 0 and rank == 0 and within_rank == 0:
        result = "O13-D no added value over O13-C"
    else:
        result = "mixed / insufficient evidence"
    return [result, f"Against O13-C on the three focus targets: MAE improves {mae}/3, R² improves {r2}/3, "
            f"Spearman improves {rank}/3, dispersion moves closer to 1 for {ratio}/3; "
            f"within-class Spearman improves {within_rank}/6 single/double rows."]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    for model in MODELS:
        for protocol in ("random", "ood"):
            parser.add_argument(f"--{model}-{protocol}-metrics", type=Path, required=True)
            parser.add_argument(f"--{model}-{protocol}-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.input_csv, dtype={"ID": str})
    if source.ID.duplicated().any() or "Fifth_class" not in source:
        raise ValueError("Locked source requires unique ID and Fifth_class")

    all_comparisons, all_per_seed, all_class, all_interactions = [], [], [], []
    d_vs_c_summary = []
    for protocol, short in (("random", "random"), ("fifth_identity_ood", "ood")):
        per_seed, predictions = {}, {}
        for model in MODELS:
            per_seed[model], predictions[model] = load_model(
                getattr(args, f"{model}_{short}_metrics"), getattr(args, f"{model}_{short}_predictions"),
                f"{model} {protocol}")
        ensure_same_keys(per_seed, protocol)
        all_per_seed.extend(frame.assign(protocol=protocol, model=model) for model, frame in per_seed.items())
        for reference in ("o12", "o13b", "o13c"):
            summary, _ = paired_summary(per_seed[reference], per_seed["o13d"], reference, "o13d", protocol)
            all_comparisons.append(summary)
            if reference == "o13c":
                d_vs_c_summary.append(summary)
        class_frames = {model: class_metrics(predictions[model], source, f"{model} {protocol}") for model in MODELS}
        all_class.append(class_four_model_summary(class_frames, protocol))
        all_interactions.append(interaction(per_seed, protocol))

    comparisons = pd.concat(all_comparisons, ignore_index=True)
    class_summary_frame = pd.concat(all_class, ignore_index=True)
    interaction_frame = pd.concat(all_interactions, ignore_index=True)
    per_seed_frame = pd.concat(all_per_seed, ignore_index=True)
    d_vs_c = pd.concat(d_vs_c_summary, ignore_index=True)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(output / "paired_comparisons_o12_o13b_o13c_vs_o13d.csv", index=False)
    per_seed_frame.to_csv(output / "per_seed_metrics_and_dispersion_all_models.csv", index=False)
    class_summary_frame.to_csv(output / "fifth_ood_within_class_four_model_summary.csv", index=False)
    interaction_frame.to_csv(output / "interaction_o13d_minus_o12_b_minus_c.csv", index=False)

    lines = ["# O13-D mean pooling + Fifth_class embedding", "",
             "The primary comparison is O13-D − O13-C; all results are strict seed-paired comparisons. "
             "For MAE negative is better; for R²/Pearson/Spearman positive is better; "
             "for σpred/σtrue, closeness to 1 is preferred.", ""]
    for protocol in ("random", "fifth_identity_ood"):
        lines += [f"## {protocol}: O13-D − O13-C test", "",
                  "| target | ΔMAE | ΔR² | ΔPearson | ΔSpearman | Δ(σpred/σtrue) |", "|---|---:|---:|---:|---:|---:|"]
        table = d_vs_c.query("protocol == @protocol and split == 'test'")
        for row in table.itertuples(index=False):
            lines.append(f"| {row.target} | {row.paired_delta_o13d_minus_o13c_mae_mean:+.3f} | "
                         f"{row.paired_delta_o13d_minus_o13c_r2_mean:+.3f} | "
                         f"{row.paired_delta_o13d_minus_o13c_pearson_mean:+.3f} | "
                         f"{row.paired_delta_o13d_minus_o13c_spearman_mean:+.3f} | "
                         f"{row.paired_delta_o13d_minus_o13c_prediction_std_to_true_std_mean:+.3f} |")
        lines.append("")
    lines += ["## Fifth-OOD within-class ranking: O13-D − O13-C", "", "| target | class | Δ Spearman |", "|---|---|---:|"]
    table = class_summary_frame.query("protocol == 'fifth_identity_ood' and split == 'test'")
    for row in table.itertuples(index=False):
        lines.append(f"| {row.target} | {row.fifth_class} | {row.paired_delta_o13d_minus_o13c_spearman_mean:+.3f} |")
    lines += ["", "## Interaction", "",
              "Interaction = (O13-D−O12)−(O13-B−O12)−(O13-C−O12). Positive values for R²/Spearman and negative values for MAE indicate super-additive benefit; the CSV reports every target.", "",
              "## Verdict", "", *[f"- {line}" for line in verdict(d_vs_c, class_summary_frame)], ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(d_vs_c.query("split == 'test'").to_string(index=False))


if __name__ == "__main__":
    main()
