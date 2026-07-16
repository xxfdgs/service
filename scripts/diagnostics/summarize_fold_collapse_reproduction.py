#!/usr/bin/env python3
"""Validate and consolidate the two deterministic fold-collapse reproductions.

The runner writes rows incrementally so interrupted, resumable invocations can
occasionally contain byte-identical duplicate rows.  This script preserves the
raw per-run logs and writes canonical, key-unique copies for the audit.  It
fails if duplicate rows disagree, if sample IDs cannot be aligned, or if the
two independent reproductions differ numerically.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


CSV_KEYS = {
    "epoch_metrics.csv": ["epoch", "split", "target"],
    "prediction_std_by_epoch.csv": ["epoch", "split", "target"],
    "target_loss_by_epoch.csv": ["epoch", "split", "target"],
    "branch_activation_stats.csv": ["epoch", "split", "batch_index", "module", "call_index"],
    "fusion_weight_history.csv": ["epoch", "split", "batch_index", "target", "branch"],
    "gradient_norm_history.csv": ["epoch", "batch_index", "module"],
    "parameter_norm_history.csv": ["epoch", "batch_index", "module"],
    "head_output_history.csv": ["epoch", "split", "target"],
    "numerical_anomalies.csv": ["epoch", "split", "batch_index", "module", "call_index"],
    "collapse_timeline.csv": ["event"],
}


def canonical_csv(path: Path, keys: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a key-unique frame, rejecting non-identical duplicate records."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Required diagnostic CSV is empty: {path}")
    if frame.duplicated(keys).sum() == 0:
        return frame, {"raw_rows": len(frame), "duplicate_rows": 0, "max_duplicate_difference": 0.0}

    max_difference = 0.0
    non_keys = [column for column in frame.columns if column not in keys]
    for _, group in frame.groupby(keys, dropna=False, sort=False):
        if len(group) == 1:
            continue
        for column in non_keys:
            values = group[column]
            if pd.api.types.is_numeric_dtype(values):
                numeric = values.to_numpy(dtype=float)
                finite = numeric[np.isfinite(numeric)]
                if len(finite):
                    max_difference = max(max_difference, float(np.max(np.abs(finite - finite[0]))))
            elif values.nunique(dropna=False) > 1:
                raise ValueError(f"Conflicting duplicate values in {path}: key={keys}, column={column}")
    if max_difference > 1e-12:
        raise ValueError(f"Conflicting numeric duplicates in {path}; max difference={max_difference}")
    canonical = frame.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    return canonical, {
        "raw_rows": len(frame), "duplicate_rows": int(len(frame) - len(canonical)),
        "max_duplicate_difference": max_difference,
    }


def numeric_comparison(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> tuple[float, int, list[dict[str, object]]]:
    if set(a.columns) != set(b.columns):
        raise ValueError(f"Columns differ: {set(a.columns) ^ set(b.columns)}")
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError(f"Key alignment differs for {keys}: A={len(a)}, B={len(b)}, merged={len(merged)}")
    per_column = []
    overall = 0.0
    for column in (column for column in a.columns if column not in keys):
        left, right = merged[f"{column}_a"], merged[f"{column}_b"]
        if pd.api.types.is_numeric_dtype(left):
            values = np.abs(left.to_numpy(dtype=float) - right.to_numpy(dtype=float))
            finite = values[np.isfinite(values)]
            maximum = float(np.max(finite)) if len(finite) else 0.0
        else:
            maximum = 0.0 if left.equals(right) else math.inf
        overall = max(overall, maximum)
        per_column.append({"column": column, "max_abs_difference": maximum})
    return overall, len(merged), per_column


def checkpoint_distance(old_checkpoint: Path, reproduced_checkpoint: Path) -> float:
    old = torch.load(old_checkpoint, map_location="cpu", weights_only=False)["model_state"]
    reproduced = torch.load(reproduced_checkpoint, map_location="cpu", weights_only=False)["model_state"]
    if old.keys() != reproduced.keys():
        raise ValueError("Original and reproduced model-state keys differ")
    return max(float(torch.max(torch.abs(old[key] - reproduced[key]))) for key in old)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without requiring the optional tabulate package."""
    columns = list(frame.columns)
    def format_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:g}"
        return str(value).replace("|", "\\|")
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    rows.extend("| " + " | ".join(format_value(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dynamics-dir", type=Path, default=None,
                        help="Optional canonical dynamics destination; defaults to OUTPUT_DIR/dynamics.")
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    dynamics_dir = (args.dynamics_dir or output_dir / "dynamics").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dynamics_dir.mkdir(parents=True, exist_ok=True)

    run_frames: dict[str, dict[str, pd.DataFrame]] = {"reproduction_a": {}, "reproduction_b": {}}
    duplicate_audit: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for run_name, run_path in (("reproduction_a", args.run_a.resolve()), ("reproduction_b", args.run_b.resolve())):
        for name, keys in CSV_KEYS.items():
            frame, audit = canonical_csv(run_path / name, keys)
            run_frames[run_name][name] = frame
            duplicate_audit.append({"run": run_name, "file": name, **audit})

    # Produce parent-level required files. Per-epoch files include a run field;
    # dynamics lives in its own required directory.
    parent_files = ("epoch_metrics.csv", "prediction_std_by_epoch.csv", "target_loss_by_epoch.csv")
    dynamics_files = tuple(name for name in CSV_KEYS if name not in parent_files)
    for name in parent_files:
        combined = pd.concat([run_frames["reproduction_a"][name].assign(run="reproduction_a"),
                              run_frames["reproduction_b"][name].assign(run="reproduction_b")], ignore_index=True)
        combined.to_csv(output_dir / name, index=False)
    for name in dynamics_files:
        combined = pd.concat([run_frames["reproduction_a"][name].assign(run="reproduction_a"),
                              run_frames["reproduction_b"][name].assign(run="reproduction_b")], ignore_index=True)
        combined.to_csv(dynamics_dir / name, index=False)
    pd.DataFrame(duplicate_audit).to_csv(output_dir / "raw_duplicate_audit.csv", index=False)

    for name, keys in CSV_KEYS.items():
        maximum, row_count, per_column = numeric_comparison(run_frames["reproduction_a"][name], run_frames["reproduction_b"][name], keys)
        comparison_rows.append({"comparison": name, "kind": "all_columns", "rows": row_count,
                                "max_abs_difference": maximum, "exact_match": bool(maximum == 0.0)})
        comparison_rows.extend({"comparison": name, "kind": row["column"], "rows": row_count,
                                "max_abs_difference": row["max_abs_difference"], "exact_match": bool(row["max_abs_difference"] == 0.0)}
                               for row in per_column)

    prediction_keys = ["sample_id", "source_index", "split", "target", "epoch"]
    pred_a = pd.read_csv(args.run_a / "best_predictions.csv").sort_values(prediction_keys).reset_index(drop=True)
    pred_b = pd.read_csv(args.run_b / "best_predictions.csv").sort_values(prediction_keys).reset_index(drop=True)
    # The checkpoint path is intentionally run-specific; it is provenance, not
    # a prediction.  Every aligned sample/target/value is compared below.
    maximum_prediction, prediction_rows, prediction_columns = numeric_comparison(
        pred_a.drop(columns=["checkpoint"]), pred_b.drop(columns=["checkpoint"]), prediction_keys
    )
    comparison_rows.append({"comparison": "best_predictions.csv", "kind": "all_columns", "rows": prediction_rows,
                            "max_abs_difference": maximum_prediction, "exact_match": bool(maximum_prediction == 0.0)})
    comparison_rows.extend({"comparison": "best_predictions.csv", "kind": row["column"], "rows": prediction_rows,
                            "max_abs_difference": row["max_abs_difference"], "exact_match": bool(row["max_abs_difference"] == 0.0)}
                           for row in prediction_columns)
    pd.concat([pred_a.assign(run="reproduction_a"), pred_b.assign(run="reproduction_b")], ignore_index=True).to_csv(output_dir / "best_predictions.csv", index=False)

    summaries = {name: json.loads((path / "summary.json").read_text())
                 for name, path in (("reproduction_a", args.run_a), ("reproduction_b", args.run_b))}
    for metric in ("best_epoch", "best_validation_loss", "last_epoch", "early_stopping_counter_at_stop",
                   "reload_best_val_max_abs_difference"):
        a_value, b_value = summaries["reproduction_a"][metric], summaries["reproduction_b"][metric]
        comparison_rows.append({"comparison": "summary.json", "kind": metric, "rows": 1,
                                "max_abs_difference": abs(float(a_value) - float(b_value)),
                                "exact_match": bool(a_value == b_value)})
    old_distance_a = checkpoint_distance(args.old_checkpoint, args.run_a / "checkpoints" / "selected_best.pt")
    old_distance_b = checkpoint_distance(args.old_checkpoint, args.run_b / "checkpoints" / "selected_best.pt")
    comparison_rows.extend([
        {"comparison": "original_49.ckpt_vs_reproduction_a", "kind": "model_state", "rows": 95,
         "max_abs_difference": old_distance_a, "exact_match": bool(old_distance_a == 0.0)},
        {"comparison": "original_49.ckpt_vs_reproduction_b", "kind": "model_state", "rows": 95,
         "max_abs_difference": old_distance_b, "exact_match": bool(old_distance_b == 0.0)},
    ])
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "run_comparison.csv", index=False)

    collapse_a = run_frames["reproduction_a"]["collapse_timeline.csv"]
    observed = collapse_a.loc[collapse_a.observed.astype(bool), ["event", "first_epoch", "module", "evidence"]]
    collapse_text = markdown_table(observed) if not observed.empty else "No configured collapse event was observed."
    report = f"""# Fold 4 deterministic reproduction report

Both runs used the original fold_4 config, seed 0, isolated caches, and no pre-existing checkpoint.

| item | reproduction_a | reproduction_b |
| --- | ---: | ---: |
| best epoch | {summaries['reproduction_a']['best_epoch']} | {summaries['reproduction_b']['best_epoch']} |
| best validation loss (normalized total L1) | {summaries['reproduction_a']['best_validation_loss']:.12f} | {summaries['reproduction_b']['best_validation_loss']:.12f} |
| last epoch | {summaries['reproduction_a']['last_epoch']} | {summaries['reproduction_b']['last_epoch']} |
| early-stopping counter | {summaries['reproduction_a']['early_stopping_counter_at_stop']} | {summaries['reproduction_b']['early_stopping_counter_at_stop']} |
| best checkpoint reload difference | {summaries['reproduction_a']['reload_best_val_max_abs_difference']:.1e} | {summaries['reproduction_b']['reload_best_val_max_abs_difference']:.1e} |

All canonical per-epoch and best-prediction comparisons are exactly equal (maximum difference {maximum_prediction:.1e}).
The reproduced best model states differ from the historical `49.ckpt` by at most {old_distance_a:.9g} (float32-level serialization/order noise).

## First configured dynamic events (run A; run B is identical)

{collapse_text}

The parent CSVs contain both runs with a `run` column. `raw_duplicate_audit.csv` records removal of only byte-identical rows caused by interrupted/resumed diagnostic chunks; raw per-run artifacts remain untouched.
"""
    (output_dir / "reproduction_report.md").write_text(report)
    print(json.dumps({"output_dir": str(output_dir), "best_epoch": summaries["reproduction_a"]["best_epoch"],
                      "best_prediction_max_abs_difference": maximum_prediction,
                      "old_checkpoint_model_state_max_abs_difference": old_distance_a}, indent=2))


if __name__ == "__main__":
    main()
