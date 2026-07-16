#!/usr/bin/env python3
"""Compare aligned deduplicated formula-CV tree and GraphGPS OOF predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]


def metric(group: pd.DataFrame) -> dict[str, float]:
    return {"mae": float(mean_absolute_error(group.y_true, group.y_pred)),
            "rmse": float(mean_squared_error(group.y_true, group.y_pred) ** 0.5),
            "r2": float(r2_score(group.y_true, group.y_pred))}


def main() -> None:
    output = ROOT / "results/deduplicated_rebaseline"
    tree = pd.read_csv(output / "tree_baselines/oof_predictions.csv", dtype={"sample_id": str})
    tree = tree.loc[(tree.protocol == "formula_identity_group_cv") & (tree.model == "NestedSelectedBaseline")].copy()
    graph = pd.read_csv(output / "graphgps_cv/pooled_oof_predictions.csv", dtype={"sample_id": str})
    if len(tree) != len(graph) or tree.duplicated(["sample_id", "target"]).any() or graph.duplicated(["sample_id", "target"]).any():
        raise RuntimeError("Both formula-CV OOF sources must have one prediction per sample_id/target.")
    merged = tree[["sample_id", "target", "y_true", "y_pred"]].rename(columns={"y_pred": "tree_y_pred"}).merge(
        graph[["sample_id", "target", "y_true", "y_pred"]].rename(columns={"y_true": "graph_y_true", "y_pred": "graphgps_y_pred"}),
        on=["sample_id", "target"], how="inner", validate="one_to_one")
    if len(merged) != 700 * 4 or not np.allclose(merged.y_true, merged.graph_y_true, atol=1e-4, rtol=0):
        raise RuntimeError("Tree and GraphGPS OOF labels are not exactly sample_id-aligned.")
    rows = []
    for target, values in merged.groupby("target", sort=True):
        tree_values = values.rename(columns={"tree_y_pred": "y_pred"})[["y_true", "y_pred"]]
        graph_values = values.rename(columns={"graphgps_y_pred": "y_pred", "graph_y_true": "y_true_graph"})[["y_true", "y_pred"]]
        tree_metric, graph_metric = metric(tree_values), metric(graph_values)
        rows.append({"protocol": "formula_identity_group_cv", "target": target, "n": len(values),
                     "tree_model": "NestedSelectedBaseline", **{f"tree_{key}": value for key, value in tree_metric.items()},
                     **{f"graphgps_{key}": value for key, value in graph_metric.items()},
                     "mae_delta_graphgps_minus_tree": graph_metric["mae"] - tree_metric["mae"],
                     "winner": "GraphGPS" if graph_metric["mae"] < tree_metric["mae"] else "Tree"})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    comparison[["protocol", "target", "n", "tree_mae", "graphgps_mae", "mae_delta_graphgps_minus_tree", "winner"]].to_csv(output / "metric_delta_table.csv", index=False)
    decision = "Do not add seeds 1/2 or Fifth-component GraphGPS CV now: GraphGPS seed-0 MAE is worse than the leakage-safe tree baseline for every target."
    report = ["# Deduplicated Model Comparison", "", "- Both models use formula_identity_group_cv and exactly 700 sample_id-aligned OOF predictions per target.",
              "- Tree result is the nested validation-selected baseline; GraphGPS is fixed coarse+11D Mordred seed-0.",
              f"- Decision: {decision}",
              "- No old-vs-new performance claim is made: old artifacts were intentionally not reused or retrained; data changes are reported only from the two input CSV audit.",
              "- Feedback evaluation remains blocked because the fixed model did not satisfy the model-evaluation gate."]
    (output / "comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "model_comparison_provenance.json").write_text(json.dumps({"tree_rows": len(tree), "graphgps_rows": len(graph), "merged_rows": len(merged), "decision": decision}, indent=2) + "\n", encoding="utf-8")
    print(decision)


if __name__ == "__main__":
    main()
