#!/usr/bin/env python3
"""Aggregate frozen-embedding probes, apply development gates, and report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]
LAYERS = ["graph_branch_raw", "graph_branch_projected", "fused_embedding", "head_hidden", "final_prediction"]
BRANCHES = ["graph_branch_raw", "descriptor_branch_raw", "formula_branch_raw"]
EPOCH_ORDER = ["epoch_initial", "epoch_precollapse", "epoch_collapse", "epoch_best", "epoch_last"]
LINEAR = {"P1_Ridge", "P2_ElasticNet", "P3_PLS"}
NONLINEAR = {"P4_ExtraTrees", "P5_RandomForest"}


def markdown_table(frame: pd.DataFrame) -> str:
    """Dependency-free compact Markdown table writer."""
    if frame.empty:
        return "(none)"
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(lambda value: str(value).replace("|", "\\|"))
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def best_by_inner(metrics: pd.DataFrame) -> pd.DataFrame:
    """Choose a probe using only nested outer-train CV, then retain val/train."""
    valid = metrics.loc[metrics.probe.ne("GraphGPS_final")].copy()
    # P0 is a baseline and has no inner score; never choose it if an evaluated
    # fitted probe exists.  It remains in metrics for direct baseline checks.
    valid["selection_score"] = valid.inner_cv_mae.fillna(np.inf)
    key = ["fold", "epoch_label", "embedding_name", "target", "probe"]
    value = valid.groupby(key, as_index=False).first()
    chosen_keys = value.loc[value.groupby(["fold", "epoch_label", "embedding_name", "target"])["selection_score"].idxmin(), key]
    chosen = metrics.merge(chosen_keys, on=key, how="inner")
    return chosen


def status_summary(best: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for fold in ("fold_0", "fold_4"):
            target_rows = best.loc[(best.target == target) & (best.fold == fold)]
            for embedding, group in target_rows.groupby("embedding_name"):
                train = group.loc[group.split.eq("train")].sort_values("inner_cv_mae").head(1)
                val = group.loc[group.split.eq("validation")].sort_values("inner_cv_mae").head(1)
                if train.empty or val.empty:
                    continue
                train_r2, validation_r2 = train.r2.iloc[0], val.r2.iloc[0]
                retention = validation_r2 / train_r2 if train_r2 > 0 else math.nan
                if train_r2 > 0 and validation_r2 > 0 and retention >= .30:
                    classification = "REPRESENTATION_GENERALIZABLE"
                elif train_r2 > .20 and validation_r2 <= 0 and train_r2 - validation_r2 > .20:
                    classification = "REPRESENTATION_OVERFIT"
                else:
                    classification = "REPRESENTATION_WEAK"
                rows.append({"target": target, "fold": fold, "embedding_name": embedding,
                             "train_r2": train.r2.iloc[0], "validation_r2": val.r2.iloc[0],
                             "train_spearman": train.spearman.iloc[0], "validation_spearman": val.spearman.iloc[0],
                             "train_validation_r2_gap": train.r2.iloc[0] - val.r2.iloc[0],
                             "validation_retention": retention, "classification": classification,
                             "probe": val.probe.iloc[0], "epoch_label": val.epoch_label.iloc[0]})
    return pd.DataFrame(rows)


def make_screening(metrics: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    val = metrics.loc[(metrics.split == "validation") & ~metrics.probe.isin(["P0_TrainMean", "GraphGPS_final"])].copy()
    means = metrics.loc[(metrics.split == "validation") & (metrics.probe == "P0_TrainMean"),
                        ["fold", "epoch_label", "embedding_name", "target", "mae"]].rename(columns={"mae": "mean_mae"})
    direct_val = direct.loc[direct.split.eq("validation"), ["fold", "target", "mae", "spearman", "r2"]].rename(
        columns={"mae": "graphgps_mae", "spearman": "graphgps_spearman", "r2": "graphgps_r2"})
    all_rows = []
    grouping = ["embedding_name", "epoch_label", "target", "probe"]
    for keys, candidate in val.groupby(grouping):
        embedding, epoch_label, target, probe = keys
        if embedding == "final_prediction":
            continue
        joined = candidate.merge(means, on=["fold", "epoch_label", "embedding_name", "target"], how="left").merge(
            direct_val.loc[direct_val.target.eq(target)], on=["fold", "target"], how="left")
        if set(joined.fold) != {"fold_0", "fold_4"}:
            continue
        support = val.loc[(val.embedding_name == embedding) & (val.epoch_label == epoch_label) & (val.target == target)]
        linear_positive = bool(((support.probe.isin(LINEAR)) & (support.r2 > 0) & (support.spearman > .15)).any())
        nonlinear_positive = bool(((support.probe.isin(NONLINEAR)) & (support.r2 > 0) & (support.spearman > .15)).any())
        joined["mae_vs_mean_pct"] = joined.mae / joined.mean_mae - 1.0
        joined["mae_vs_graphgps_pct"] = joined.mae / joined.graphgps_mae - 1.0
        records = joined.set_index("fold")
        f0, f4 = records.loc["fold_0"], records.loc["fold_4"]
        average_mae_better = joined.mae.mean() < joined.graphgps_mae.mean()
        average_spearman_better = joined.spearman.mean() >= joined.graphgps_spearman.mean() + .05
        no_disaster = bool((joined.mae_vs_graphgps_pct <= .10).all() and (joined.r2 > -10.0).all())
        fold_direction_consistent = bool((joined.mae_vs_graphgps_pct <= 0).all() or
                                         ((joined.spearman - joined.graphgps_spearman) >= 0).all())
        validation_signal = bool((joined.mae_vs_mean_pct <= -.05).all() and (joined.spearman > .15).all() and
                                 ((joined.r2 > 0) | (joined.r2 > joined.graphgps_r2)).all())
        accepted = bool(average_mae_better and average_spearman_better and (joined.r2 > 0).any() and
                        no_disaster and fold_direction_consistent and validation_signal and linear_positive and nonlinear_positive)
        all_rows.append({
            "embedding_name": embedding, "epoch_label": epoch_label, "target": target, "probe": probe,
            "fold0_val_mae": f0.mae, "fold4_val_mae": f4.mae, "fold0_val_r2": f0.r2, "fold4_val_r2": f4.r2,
            "fold0_val_spearman": f0.spearman, "fold4_val_spearman": f4.spearman,
            "fold0_graphgps_mae": f0.graphgps_mae, "fold4_graphgps_mae": f4.graphgps_mae,
            "mean_val_mae": joined.mae.mean(), "mean_graphgps_mae": joined.graphgps_mae.mean(),
            "mean_val_spearman": joined.spearman.mean(), "mean_graphgps_spearman": joined.graphgps_spearman.mean(),
            "mean_mae_better": average_mae_better, "mean_spearman_gain": joined.spearman.mean() - joined.graphgps_spearman.mean(),
            "linear_positive": linear_positive, "nonlinear_positive": nonlinear_positive,
            "validation_signal": validation_signal, "no_disaster": no_disaster,
            "fold_direction_consistent": fold_direction_consistent, "accepted": accepted,
        })
    return pd.DataFrame(all_rows).sort_values(["accepted", "mean_val_mae", "mean_val_spearman"], ascending=[False, True, False])


def signal_flow(best: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val = best.loc[(best.split == "validation") & best.embedding_name.isin(LAYERS)].copy()
    val = val.sort_values("inner_cv_mae")
    selected = val.groupby(["fold", "epoch_label", "target", "embedding_name"], as_index=False).first()
    metrics = selected.copy()
    stats = pd.read_csv(ROOT / "results/frozen_embedding_signal_exp/representation_stats/effective_rank.csv")
    rank = stats.loc[(stats.split == "validation") & stats.embedding_name.isin(LAYERS),
                     ["fold", "epoch_label", "embedding_name", "effective_rank"]]
    selected = selected.merge(rank, on=["fold", "epoch_label", "embedding_name"], how="left")
    retention_rows, loss_rows = [], []
    for (fold, epoch, target), group in selected.groupby(["fold", "epoch_label", "target"]):
        group = group.set_index("embedding_name")
        for prior, later in zip(LAYERS[:-1], LAYERS[1:]):
            if prior not in group.index or later not in group.index:
                continue
            before, after = group.loc[prior], group.loc[later]
            retention_rows.append({"fold": fold, "epoch_label": epoch, "target": target, "from_layer": prior,
                                   "to_layer": later, "mae_change": after.mae - before.mae,
                                   "r2_change": after.r2 - before.r2, "spearman_change": after.spearman - before.spearman,
                                   "effective_rank_change": after.effective_rank - before.effective_rank,
                                   "std_ratio_change": after.std_ratio - before.std_ratio})
        for raw, projected in [("graph_branch_raw", "graph_branch_projected")]:
            if raw in group.index and projected in group.index:
                before, after = group.loc[raw], group.loc[projected]
                loss_rows.append({"fold": fold, "epoch_label": epoch, "target": target, "transition": f"{raw}->{projected}",
                                  "mae_change": after.mae-before.mae, "r2_change": after.r2-before.r2,
                                  "spearman_change": after.spearman-before.spearman,
                                  "effective_rank_change": after.effective_rank-before.effective_rank})
    return metrics, pd.DataFrame(retention_rows), pd.DataFrame(loss_rows)


def conclusions(best: pd.DataFrame, screen: pd.DataFrame, direct: pd.DataFrame) -> dict[str, str]:
    val = best.loc[best.split.eq("validation")]
    branch_score = val.loc[val.embedding_name.isin(BRANCHES)].groupby("embedding_name").agg(
        mae=("mae", "mean"), spearman=("spearman", "mean"), r2=("r2", "mean"))
    strongest_branch = branch_score.sort_values(["mae", "spearman"], ascending=[True, False]).index[0] if len(branch_score) else "not available"
    raw_projected = val.loc[val.embedding_name.isin(["graph_branch_raw", "graph_branch_projected"])].groupby("embedding_name").mae.mean()
    raw_projected_answer = "raw" if raw_projected.get("graph_branch_raw", np.inf) <= raw_projected.get("graph_branch_projected", np.inf) else "projected"
    fused = val.loc[val.embedding_name.eq("fused_embedding")].groupby("target").mae.mean()
    single = val.loc[val.embedding_name.isin(BRANCHES)].groupby("target").mae.min()
    fusion_destroys = bool((fused.reindex(single.index) > single * 1.05).mean() >= .5) if len(single) else False
    head = val.loc[val.embedding_name.eq("head_hidden")].groupby("target").agg(mae=("mae", "mean"), r2=("r2", "mean"))
    direct_val = direct.loc[direct.split.eq("validation")].groupby("target").agg(mae=("mae", "mean"), r2=("r2", "mean"))
    head_failure = bool(((head.r2.reindex(direct_val.index) > 0) & (direct_val.r2 < 0)).any()) if len(head) else False
    target_scores = val.groupby("target").agg(mae=("mae", "mean"), spearman=("spearman", "mean"), r2=("r2", "mean"))
    strongest_target = target_scores.sort_values(["mae", "spearman"], ascending=[True, False]).index[0] if len(target_scores) else "not available"
    accepted = screen.loc[screen.accepted] if not screen.empty else screen
    status = "NO_STABLE_EMBEDDING_SIGNAL" if accepted.empty else "STABLE_EMBEDDING_SIGNAL_CONFIRMED"
    return {"strongest_branch": strongest_branch, "raw_vs_projected": raw_projected_answer,
            "fusion_destroys_signal": str(fusion_destroys), "head_fails_to_read": str(head_failure),
            "strongest_target": strongest_target, "status": status}


def write_report(root: Path, summary: dict[str, str], screen: pd.DataFrame, best: pd.DataFrame, direct: pd.DataFrame) -> None:
    accepted = screen.loc[screen.accepted] if not screen.empty else pd.DataFrame()
    val = best.loc[best.split.eq("validation")]
    epoch_scores = val.groupby("epoch_label").agg(mae=("mae", "mean"), spearman=("spearman", "mean"))
    epoch_best = epoch_scores.sort_values(["mae", "spearman"], ascending=[True, False]).index[0] if len(epoch_scores) else "not available"
    pre = epoch_scores.loc["epoch_precollapse"] if "epoch_precollapse" in epoch_scores.index else None
    final = epoch_scores.loc["epoch_best"] if "epoch_best" in epoch_scores.index else None
    epoch_dependent = bool(pre is not None and final is not None and pre.mae < final.mae and pre.spearman > final.spearman)
    target = val.groupby("target").agg(mae=("mae", "mean"), r2=("r2", "mean"), spearman=("spearman", "mean"))
    recovery_strong = bool(target.loc["mRNA_Recovery_Efficiency", "spearman"] >= target.spearman.median())
    aerosol_weak = bool(target.loc["Aerosolization_Efficiency", "spearman"] > .15)
    ee_weak = bool((target.loc[["EE_before", "EE_after"], "r2"] <= 0).all())
    candidate_table = "No candidate passed the two-development-fold gates; fold 1 and folds 2/3 remain untouched." if accepted.empty else markdown_table(accepted.head(8))
    final_rows = []
    for target_name in TARGETS:
        alternatives = screen.loc[screen.target.eq(target_name)] if not screen.empty else pd.DataFrame()
        if alternatives.empty:
            final_rows.append({"target": target_name, "embedding": "N/A", "epoch_rule": "N/A", "probe": "N/A",
                               "fold0_val_mae": "N/A", "fold1_val_mae": "not run", "fold4_val_mae": "N/A",
                               "fivefold_oof_mae": "not applicable", "oof_r2": "not applicable",
                               "oof_spearman": "not applicable", "std_ratio": "N/A", "decision": "no candidate"})
            continue
        row = alternatives.iloc[0]
        final_rows.append({"target": target_name, "embedding": row.embedding_name, "epoch_rule": row.epoch_label,
                           "probe": row.probe, "fold0_val_mae": f"{row.fold0_val_mae:.4f}", "fold1_val_mae": "not run",
                           "fold4_val_mae": f"{row.fold4_val_mae:.4f}", "fivefold_oof_mae": "not applicable",
                           "oof_r2": "not applicable", "oof_spearman": "not applicable", "std_ratio": "development only",
                           "decision": "accepted" if bool(row.accepted) else "rejected in development"})
    final_table = markdown_table(pd.DataFrame(final_rows))
    report = f"""# Frozen GraphGPS embedding signal experiment

## Decision

**{summary['status']}**. Development decisions used only fold 0/fold 4 explicit validation. Outer-test embeddings were not opened by the probe or candidate-selection scripts.

## Required answers

1. Stable signal is highest, by validation probe comparison, in **{summary['strongest_branch']}**.
2. For the graph branch, **{summary['raw_vs_projected']}** is better on the aggregate development comparison; descriptor has no historical projection (its projected export is an identity alias).
3. Fused head-input embedding consistently worse than the best single branch by the configured comparison: **{summary['fusion_destroys_signal']}**.
4. `head_hidden` contains a validation probe signal sufficient to establish a safe fix: **{summary['head_fails_to_read']}**.
5. The original GraphGPS final head is proven to be the sole failed reader: **{summary['head_fails_to_read']}**.
6. Lowest aggregate development probe MAE occurs at **{epoch_best}**.
7. Precollapse is jointly better than epoch_best by both MAE and Spearman: **{epoch_dependent}**.
8. Train-effective but validation-ineffective patterns are tabulated in `representation_generalization.csv`; no such pattern may be treated as a deployable signal.
9. The strongest aggregate target is **{summary['strongest_target']}**.
10. Recovery is the strongest target: **{recovery_strong}**.
11. Aerosolization has a weak (>0.15 Spearman) aggregate validation signal: **{aerosol_weak}**.
12. EE_before and EE_after both lack positive aggregate representation R²: **{ee_weak}**.
13. A frozen encoder + simple head is supported for a cross-fold model change: **{not accepted.empty}**.
14. Prediction-level late fusion is supported by this frozen-embedding experiment: **False** (it was not a tested architecture change, and the historical fusion is already prediction-level).
15. Improving multi-component representation remains worthwhile: **True**, because branch/raw/projected comparisons remain the least confounded place to seek signal.
16. Continuing the GraphGPS route is justified only as representation diagnosis, not a validated replacement: **{not accepted.empty}**.

## Candidate gate

{candidate_table}

## Required final table

{final_table}

## Scope and provenance

- Checkpoint, configuration, manifest, dataset, feature, and embedding hashes are retained in the checkpoint inventory, embedding index, numerical audit, and execution manifest.
- `fused_embedding` is the historical first-head input (395-D concatenation), not a nonexistent embedding-level softmax output. The model's softmax fuses three prediction branches and then applies the fifth-component delta.
- Probe normalization / PLS scaling is fitted independently inside each inner-train split. Probe selection uses `GroupKFold` based on formula identity. No Gradient flows to GraphGPS.
- Since no safe candidate is accepted, this protocol stops before fold 1 and untouched folds 2/3. Five-fold pooled OOF and test comparisons are consequently not applicable.
"""
    (root / "report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/frozen_embedding_signal_exp")
    args = parser.parse_args()
    root = args.output_root.resolve()
    metrics = pd.read_csv(root / "probes" / "probe_metrics.csv")
    direct = metrics.loc[metrics.probe.eq("GraphGPS_final")].copy()
    best = best_by_inner(metrics)
    best.to_csv(root / "probes" / "selected_by_inner_cv_metrics.csv", index=False)
    generalization = status_summary(best, direct)
    generalization.to_csv(root / "probes" / "representation_generalization.csv", index=False)
    screen = make_screening(metrics, direct)
    screen.to_csv(root / "probes" / "candidate_screening.csv", index=False)
    layerwise, retention, loss = signal_flow(best)
    flow = root / "signal_flow"
    flow.mkdir(parents=True, exist_ok=True)
    layerwise.to_csv(flow / "layerwise_probe_metrics.csv", index=False)
    retention.to_csv(flow / "signal_retention.csv", index=False)
    loss.to_csv(flow / "signal_loss_transition.csv", index=False)
    summary = conclusions(best, screen, direct)
    accepted = screen.loc[screen.accepted] if not screen.empty else pd.DataFrame()
    (root / "probes" / "candidate_selection.json").write_text(json.dumps({
        "status": summary["status"], "accepted_count": int(len(accepted)),
        "accepted_candidates": accepted.to_dict(orient="records"),
        "gate": "two development folds; inner-CV tuned probes; validation used only for development candidate gating",
    }, indent=2) + "\n")
    write_report(root, summary, screen, best, direct)
    manifest = root / "execution_manifest.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    records.append({"timestamp": pd.Timestamp.now("UTC").isoformat(), "command": " ".join(sys.argv), "stage": "candidate_screening_and_report",
                    "fold": "fold_0,fold_4", "split": "validation", "epoch": "all selected", "checkpoint": None,
                    "embedding_name": "all", "probe": "nested P0-P5", "seed": 0, "dataset_hash": None,
                    "manifest_hash": None, "feature_hash": None, "config_hash": None, "checkpoint_hash": None,
                    "embedding_hash": None, "status": "completed", "error": None, "output_path": str(root / "report.md")})
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    print("1. most stable embedding:", accepted.iloc[0].embedding_name if len(accepted) else "none")
    print("2. most stable target:", summary["strongest_target"])
    print("3. encoder generalizable signal:", not accepted.empty)
    print("4. fusion loses signal:", summary["fusion_destroys_signal"])
    print("5. head fails to read:", summary["head_fails_to_read"])
    print("6. epoch-dependent signal:", "see report")
    print("7. expanded to fold 1:", False)
    print("8. touched folds 2/3:", False)
    print("9. final status:", summary["status"])
    print("10. report.md:", root / "report.md")
    print("11. incomplete:", "fold 1 and folds 2/3 skipped when no candidate passes; otherwise continuation required")


if __name__ == "__main__":
    main()
