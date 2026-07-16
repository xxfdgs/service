#!/usr/bin/env python3
"""Compare raw, audited replicate-median, and weighted tree training versions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import add_normalized_keys, discover_schema, metric_dict
from run_repeated_group_benchmark import make_pipeline
from stable_formulation import build_stable_feature_sets
from stage3_utils import append_execution


PROTOCOLS = ("fifth_component_group_cv", "formula_identity_group_cv")


def version_frame(version: str, schema) -> pd.DataFrame:
    """Load one immutable stage-three data version with explicit sample IDs."""
    paths = {
        "raw_records": schema.train_path,
        "replicate_median": ROOT / "results/generalization_stage2/data_audit/dataset_replicate_median.csv",
        "replicate_weighted": ROOT / "results/generalization_stage2/data_audit/dataset_replicate_weighted.csv",
    }
    frame = pd.read_csv(paths[version]).copy()
    frame["sample_id"] = frame[schema.id_column].astype(str)
    if frame.sample_id.duplicated().any():
        raise ValueError(f"{version} has duplicate sample IDs.")
    frame["raw_index"] = np.arange(len(frame), dtype=int)
    add_normalized_keys(frame, schema)
    return frame


def get_indexes(frame: pd.DataFrame, manifest: Path) -> dict[str, pd.Index]:
    """Map a version-specific manifest to rows through sample IDs only."""
    table = pd.read_csv(manifest, dtype={"sample_id": str})
    mapping = pd.Series(frame.index.to_numpy(), index=frame.sample_id)
    row_index = table.sample_id.map(mapping)
    if row_index.isna().any() or table.sample_id.duplicated().any():
        raise ValueError(f"Invalid version manifest: {manifest}")
    return {split: pd.Index(row_index[table.split == split].astype(int)) for split in ("train", "test")}


def estimators(seed: int, n_jobs: int) -> dict[str, object]:
    """Use the fixed tree settings established before external evaluation."""
    return {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.8,
                                            random_state=seed, n_jobs=n_jobs),
        "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.7,
                                                random_state=seed, n_jobs=n_jobs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generalization_stage3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    result_dir = output_dir / "replicate_comparison"
    result_dir.mkdir(parents=True, exist_ok=True)
    schema = discover_schema()
    frames = {version: version_frame(version, schema) for version in
              ("raw_records", "replicate_median", "replicate_weighted")}
    statistics = pd.DataFrame([{"data_version": version, "n_samples": len(frame),
                                 "n_true_replicate_rows": int((frame.get("replicate_group_class", pd.Series(dtype=str)) == "true_replicate").sum()),
                                 "has_sample_weight": any(column.startswith("sample_weight_") for column in frame.columns)}
                               for version, frame in frames.items()])
    statistics.to_csv(result_dir / "dataset_version_statistics.csv", index=False)
    metrics: list[dict[str, object]] = []
    for version, frame in frames.items():
        feature_sets, _, _ = build_stable_feature_sets(frame, schema)
        for protocol in PROTOCOLS:
            manifest_version = "raw_records" if version == "raw_records" else "replicate_median"
            for manifest in sorted((output_dir / "manifests" / protocol / manifest_version).glob("fold_*.csv")):
                indexes = get_indexes(frame, manifest)
                for target in schema.targets:
                    train_y = frame.loc[indexes["train"], target].astype(float)
                    test_y = frame.loc[indexes["test"], target].astype(float)
                    sample_weight = None
                    if version == "replicate_weighted":
                        weight_column = f"sample_weight_{target}"
                        if weight_column not in frame:
                            raise ValueError(f"Missing target-specific weight column: {weight_column}")
                        sample_weight = frame.loc[indexes["train"], weight_column].astype(float).to_numpy()
                    for model_name, estimator in estimators(arguments.seed, arguments.n_jobs).items():
                        pipeline = make_pipeline(feature_sets["F2_identity_ratio"], clone(estimator))
                        fit_kwargs = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
                        pipeline.fit(feature_sets["F2_identity_ratio"].loc[indexes["train"]], train_y, **fit_kwargs)
                        prediction = pipeline.predict(feature_sets["F2_identity_ratio"].loc[indexes["test"]])
                        metrics.append({"data_version": version, "protocol": protocol, "fold": manifest.stem,
                                        "target": target, "model": model_name, "n_train": len(indexes["train"]),
                                        "n_test": len(indexes["test"]), "used_sample_weight": sample_weight is not None,
                                        **metric_dict(test_y, prediction)})
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(result_dir / "tree_replicate_metrics.csv", index=False)
    paired_rows: list[dict[str, object]] = []
    for (protocol, fold, target, model), group in metric_frame.groupby(["protocol", "fold", "target", "model"]):
        raw = group.loc[group.data_version == "raw_records"]
        for version in ("replicate_median", "replicate_weighted"):
            candidate = group.loc[group.data_version == version]
            if raw.empty or candidate.empty:
                continue
            raw_row, candidate_row = raw.iloc[0], candidate.iloc[0]
            paired_rows.append({"protocol": protocol, "fold": fold, "target": target, "model": model,
                                "comparison": f"{version}_minus_raw_records",
                                "mae_difference": candidate_row.mae - raw_row.mae,
                                "relative_mae_improvement": (raw_row.mae - candidate_row.mae) / raw_row.mae,
                                "test_count_difference": int(candidate_row.n_test - raw_row.n_test)})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(result_dir / "paired_version_comparison.csv", index=False)
    median_gate = paired.loc[paired.comparison == "replicate_median_minus_raw_records"].groupby(["target", "model"]).agg(
        both_protocols_positive=("relative_mae_improvement", lambda values: len(values) == 10 and values.mean() > 0.05),
        improved_folds=("mae_difference", lambda values: int((values < 0).sum())),
        max_test_count_change=("test_count_difference", lambda values: int(np.abs(values).max())),
    ).reset_index()
    median_gate["eligible_for_graphgps_seeds_1_2"] = (
        median_gate.both_protocols_positive & (median_gate.improved_folds >= 6) & (median_gate.max_test_count_change == 0)
    )
    median_gate.to_csv(result_dir / "replicate_graphgps_gate.csv", index=False)
    report = ["# Replicate Treatment Effect", "",
              "- Only audited `true_replicate` groups were aggregated in the median/weighted source files.",
              "- Tree comparisons use version-matched manifests and F2_identity_ratio; weighted variants pass replicate_count only as training sample weight.",
              "- GraphGPS replicate training is deferred unless replicate_median passes the prespecified two-protocol, >5%, majority-fold, no-test-count-reduction gate."]
    (result_dir / "replicate_effect_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    append_execution(output_dir, command=[sys.executable, *sys.argv], protocol="both", fold="all", seed=arguments.seed,
                     data_version="raw_records|replicate_median|replicate_weighted", output=result_dir)
    print(f"Wrote {result_dir}")


if __name__ == "__main__":
    main()
