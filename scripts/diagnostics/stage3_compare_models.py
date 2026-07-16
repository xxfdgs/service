#!/usr/bin/env python3
"""Fairly compare GraphGPS fold ensembles against stage-three tree OOF predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import metric_dict


def bootstrap_mae(values: pd.Series, seed: int, repeats: int = 2000) -> tuple[float, float]:
    """Bootstrap a 95% interval for sample-level absolute error."""
    array = values.to_numpy(dtype=float)
    generator = np.random.default_rng(seed)
    means = np.array([generator.choice(array, len(array), replace=True).mean() for _ in range(repeats)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def graph_predictions(graph_dir: Path) -> pd.DataFrame:
    """Read available fold ensembles and attach the common GraphGPS model name."""
    frames = [pd.read_csv(path, dtype={"sample_id": str}) for path in graph_dir.glob("fold_ensemble_predictions/*/fold_*.csv")]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame["model"] = "GraphGPS_coarse_mordred_ensemble"
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generalization_stage3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    result_dir = output_dir / "model_comparison"
    result_dir.mkdir(parents=True, exist_ok=True)
    graph = graph_predictions(output_dir / "graphgps_raw_cv")
    trees = pd.read_csv(output_dir / "tree_cv/pooled_oof_predictions.csv", dtype={"sample_id": str})
    trees = trees.loc[trees.model.isin(["ExtraTrees", "RandomForest", "TrainMean"])].copy()
    if graph.empty:
        for name in ("fold_pairwise_comparison.csv", "pooled_oof_comparison.csv", "paired_statistical_tests.csv",
                     "residual_comparison.csv", "hard_samples.csv"):
            pd.DataFrame().to_csv(result_dir / name, index=False)
        (result_dir / "model_comparison_report.md").write_text(
            "# Model Comparison\n\nGraphGPS three-seed ensembles are not yet available; no comparison was inferred.\n",
            encoding="utf-8")
        print(f"Wrote incomplete {result_dir}")
        return
    combined = pd.concat([graph, trees], ignore_index=True, sort=False)
    fold_rows: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    for (protocol, fold, target, model), group in combined.groupby(["protocol", "fold", "target", "model"]):
        fold_rows.append({"protocol": protocol, "fold": fold, "target": target, "model": model,
                          "n": len(group), **metric_dict(group.y_true, group.y_pred),
                          "prediction_std": float(group.y_pred.std(ddof=1)),
                          "residual_mean": float((group.y_true - group.y_pred).mean()),
                          "residual_std": float((group.y_true - group.y_pred).std(ddof=1))})
    for (protocol, target, model), group in combined.groupby(["protocol", "target", "model"]):
        lower, upper = bootstrap_mae(group.absolute_error, arguments.seed)
        pooled_rows.append({"protocol": protocol, "target": target, "model": model, "n": len(group),
                            **metric_dict(group.y_true, group.y_pred), "mae_ci95_low": lower, "mae_ci95_high": upper,
                            "completed_folds": group.fold.nunique(), "prediction_std": float(group.y_pred.std(ddof=1))})
        residual = group.y_true - group.y_pred
        residual_rows.append({"protocol": protocol, "target": target, "model": model,
                              "residual_mean": float(residual.mean()), "residual_std": float(residual.std(ddof=1)),
                              "prediction_std": float(group.y_pred.std(ddof=1)),
                              "true_std": float(group.y_true.std(ddof=1)),
                              "prediction_range_over_true_range": float((group.y_pred.max()-group.y_pred.min()) /
                                  max(group.y_true.max()-group.y_true.min(), 1e-12)),
                              "spearman": float(spearmanr(group.y_true, group.y_pred).statistic)})
    pair_rows: list[dict[str, object]] = []
    tests: list[dict[str, object]] = []
    hard_frames: list[pd.DataFrame] = []
    for (protocol, target), group in combined.groupby(["protocol", "target"]):
        graph_group = group.loc[group.model == "GraphGPS_coarse_mordred_ensemble"]
        for baseline in ("ExtraTrees", "RandomForest", "TrainMean"):
            base_group = group.loc[group.model == baseline]
            paired = graph_group.merge(base_group, on=["sample_id", "fold", "target", "protocol"], how="inner",
                                       validate="one_to_one", suffixes=("_graphgps", "_baseline"))
            if paired.empty:
                continue
            paired["absolute_error_difference"] = paired.absolute_error_graphgps - paired.absolute_error_baseline
            by_fold = paired.groupby("fold").absolute_error_difference.mean()
            try:
                wilcoxon_result = wilcoxon(paired.absolute_error_difference, alternative="two-sided", method="auto")
                statistic, pvalue = float(wilcoxon_result.statistic), float(wilcoxon_result.pvalue)
            except ValueError:
                statistic, pvalue = np.nan, np.nan
            pair_rows.append({"protocol": protocol, "target": target, "baseline": baseline,
                              "completed_paired_folds": paired.fold.nunique(),
                              "mean_graphgps_minus_baseline_mae": float(paired.absolute_error_difference.mean()),
                              "graphgps_win_folds": int((by_fold < 0).sum()),
                              "graphgps_win_sample_fraction": float((paired.absolute_error_difference < 0).mean()),
                              "max_fold_contribution": float(by_fold.abs().max())})
            tests.append({"protocol": protocol, "target": target, "baseline": baseline,
                          "wilcoxon_statistic": statistic, "wilcoxon_pvalue": pvalue, "n_samples": len(paired)})
            hard = paired.nlargest(10, "absolute_error_graphgps")[["sample_id", "protocol", "fold", "target",
                                                                      "y_true_graphgps", "y_pred_graphgps",
                                                                      "absolute_error_graphgps", "y_pred_baseline",
                                                                      "absolute_error_baseline"]].copy()
            hard["baseline"] = baseline
            hard_frames.append(hard)
    pd.DataFrame(fold_rows).to_csv(result_dir / "fold_pairwise_comparison.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(result_dir / "pooled_oof_comparison.csv", index=False)
    pd.DataFrame(tests).to_csv(result_dir / "paired_statistical_tests.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(result_dir / "residual_comparison.csv", index=False)
    (pd.concat(hard_frames, ignore_index=True) if hard_frames else pd.DataFrame()).to_csv(result_dir / "hard_samples.csv", index=False)
    complete = all(row["completed_folds"] == 5 for row in pair_rows) if pair_rows else False
    report = ["# Model Comparison", "", f"- Complete paired five-fold comparison: {complete}.",
              "- All pairings use sample_id, fold, protocol, and target keys; no position-based matching is used.",
              "- Interpret GraphGPS claims only when completed_paired_folds=5 for both group protocols."]
    (result_dir / "model_comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {result_dir}")


if __name__ == "__main__":
    main()
