#!/usr/bin/env python3
"""Audit and summarize the ten frozen input-only O12 log1p Norm runs.

Only continuous regression metrics are calculated here.  External feedback
data and threshold-derived criteria are deliberately outside this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGETS = ("Norm_before", "Norm_after")
EXPECTED_ROWS = 700
ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation(first: np.ndarray, second: np.ndarray, method: str) -> float:
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(pd.Series(first).corr(pd.Series(second), method=method))


def metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["y_true"].to_numpy(float)
    prediction = frame["y_pred"].to_numpy(float)
    return {
        "n": int(len(frame)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)),
        "pearson": correlation(truth, prediction, "pearson"),
        "spearman": correlation(truth, prediction, "spearman"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--raw-input", type=Path,
        default=ROOT / "datasets_lrx/raw/input/20260703_sum_utf8.csv",
        help="Canonical input CSV whose hash must match the configured source.")
    parser.add_argument("--run-prefix", default="O12Log_split")
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=10)
    args = parser.parse_args()
    runs_root = args.runs_root.resolve()
    raw_input = args.raw_input.resolve()
    if not raw_input.is_file():
        raise FileNotFoundError(f"Canonical input CSV is missing: {raw_input}")
    raw_input_hash = sha256(raw_input)

    inventory: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    input_sources: set[str] = set()
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive.")
    for seed in range(args.first_seed, args.first_seed + args.seed_count):
        run_dir = runs_root / f"{args.run_prefix}{seed}"
        settings_path = run_dir / "run_settings.json"
        summary_path = run_dir / "summary.json"
        config_path = run_dir / "effective_config.yaml"
        predictions_path = run_dir / "predictions.csv"
        checkpoint_path = run_dir / "checkpoints" / "selected_best.pt"
        required = (
            settings_path, summary_path, config_path, predictions_path,
            checkpoint_path)
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(f"Incomplete O12 log1p run: {run_dir}")

        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected_settings = {
            "target_set": "norm2",
            "loss_targets": list(TARGETS),
            "target_transform": "log1p",
            "training_loss": "mae",
            "model_type": "OneHotEmbedGPS",
            "outer_test_read_during_selection": False,
            "seed": seed,
        }
        mismatches = {
            key: (settings.get(key), expected)
            for key, expected in expected_settings.items()
            if settings.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                f"Unexpected training protocol in {run_dir}: {mismatches}")
        serialized_protocol = json.dumps(
            {"settings": settings, "config": config}, ensure_ascii=False).lower()
        if "new_validation" in serialized_protocol or "/feedback/" in serialized_protocol:
            raise RuntimeError(
                f"External feedback path appears in training metadata: {run_dir}")
        input_source = str(Path(config["read_csv"]).resolve())
        input_sources.add(input_source)
        if sha256(Path(input_source)) != raw_input_hash:
            raise RuntimeError(
                f"Configured source does not match canonical input data: "
                f"{input_source}")

        manifest_path = Path(settings["split_manifest"]).resolve()
        manifest = pd.read_csv(manifest_path)
        if len(manifest) != EXPECTED_ROWS:
            raise RuntimeError(f"Unexpected manifest size: {manifest_path}")
        manifest_counts = {
            split: int(count)
            for split, count in manifest["split"].value_counts().items()
        }
        if set(manifest_counts) != {"train", "val", "test"} or any(
                count <= 0 for count in manifest_counts.values()):
            raise RuntimeError(
                f"Unexpected split counts in {manifest_path}: {manifest_counts}")

        predictions = pd.read_csv(predictions_path)
        if predictions.duplicated(["sample_id", "split", "target"]).any():
            raise RuntimeError(f"Duplicate saved prediction rows: {run_dir}")
        expected_rows = EXPECTED_ROWS * len(TARGETS)
        if len(predictions) != expected_rows:
            raise RuntimeError(
                f"Expected {expected_rows} predictions in {run_dir}, "
                f"found {len(predictions)}")
        selected_epochs = predictions["epoch"].drop_duplicates().tolist()
        if selected_epochs != [summary["best_epoch"]]:
            raise RuntimeError(
                f"Prediction/checkpoint epoch mismatch in {run_dir}")

        inventory.append({
            "split_seed": seed,
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "input_source": input_source,
            "best_epoch": int(summary["best_epoch"]),
            "model_type": settings["model_type"],
            "target_transform": settings["target_transform"],
            "use_fifth_class_embedding": bool(
                settings.get("use_fifth_class_embedding", False)),
            "selection_metric": "continuous raw-scale validation MAE",
            "outer_test_read_during_selection": False,
        })
        for split in ("train", "val", "test"):
            count = manifest_counts[split]
            for target in TARGETS:
                part = predictions.loc[
                    predictions["split"].eq(split)
                    & predictions["target"].eq(target)]
                if len(part) != count:
                    raise RuntimeError(
                        f"Expected {count} {split}/{target} rows in "
                        f"{run_dir}, found {len(part)}")
                metric_rows.append({
                    "split_seed": seed,
                    "split": split,
                    "target": target,
                    **metric_values(part),
                })

    inventory_frame = pd.DataFrame(inventory)
    metrics = pd.DataFrame(metric_rows)
    target_average = (
        metrics.groupby(["split", "target"], as_index=False)
        .agg(
            completed_seeds=("split_seed", "nunique"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_pearson=("pearson", "mean"),
            mean_spearman=("spearman", "mean"),
        )
    )
    macro = (
        metrics.groupby(["split_seed", "split"], as_index=False)
        [["mae", "rmse", "r2", "pearson", "spearman"]]
        .mean()
        .groupby("split", as_index=False)
        .agg(
            completed_seeds=("split_seed", "nunique"),
            mean_macro_mae=("mae", "mean"),
            std_macro_mae=("mae", "std"),
            mean_macro_rmse=("rmse", "mean"),
            mean_macro_r2=("r2", "mean"),
            mean_macro_pearson=("pearson", "mean"),
            mean_macro_spearman=("spearman", "mean"),
        )
    )
    inventory_frame.to_csv(runs_root / "checkpoint_inventory.csv", index=False)
    metrics.to_csv(
        runs_root / "input_continuous_metrics_by_seed_target.csv", index=False)
    target_average.to_csv(
        runs_root / "input_continuous_metrics_target_average.csv", index=False)
    macro.to_csv(
        runs_root / "input_continuous_metrics_macro_average.csv", index=False)
    protocol = {
        "completed_checkpoints": int(len(inventory_frame)),
        "model_type": "OneHotEmbedGPS",
        "target_transform": "log1p",
        "training_loss": "MAE in log1p target space",
        "checkpoint_selection": "continuous raw-scale validation MAE",
        "input_sources": sorted(input_sources),
        "canonical_raw_input": str(raw_input),
        "canonical_raw_input_sha256": raw_input_hash,
        "external_feedback_read": False,
        "threshold_or_side_criterion_used": False,
        "outer_test_read_during_selection": False,
        "use_fifth_class_embedding_values": sorted(
            inventory_frame["use_fifth_class_embedding"].unique().tolist()),
        "run_prefix": args.run_prefix,
        "split_seeds": list(
            range(args.first_seed, args.first_seed + args.seed_count)),
    }
    (runs_root / "input_only_training_protocol_audit.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(target_average.to_string(index=False))
    print()
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
