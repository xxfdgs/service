#!/usr/bin/env python3
"""Run paired O12/O22 repeats and validation-only frozen ensembles.

For each seed, O12 and O22 are trained independently from scratch.  Their
checkpoint is selected by that run's validation MAE only.  The paired ensemble
then fits non-negative per-target weights, and optional affine Huber heads, on
that same validation prediction table only.  Test predictions are evaluated
only after those choices have been written to disk.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/diagnostics/run_fusion_head_experiment.py"
ENSEMBLE = ROOT / "scripts/diagnostics/build_validation_selected_ensemble.py"
CALIBRATION = ROOT / "scripts/diagnostics/build_validation_huber_calibration.py"
TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]
SPLITS = ["train", "val", "test"]


def call(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log_path}")


def train_command(run_dir: Path, candidate: str, seed: int, fusion_type: str,
                  resume: bool) -> list[str]:
    command = [sys.executable, str(RUNNER),
               "--config", str(ROOT / "results/new_dataset_benchmark_20260713/graphgps_standard.yaml"),
               "--run-dir", str(run_dir), "--fold", "fixed_input", "--group", "B",
               "--candidate", candidate, "--model-type", "OneHotEmbedGPS",
               "--fusion-type", fusion_type, "--head-type", "baseline",
               "--execution-max-epochs", "150", "--base-lr", ".001", "--weight-decay", ".00001",
               "--batch-size", "8", "--warmup-epochs", "50", "--seed", str(seed),
               "--gt-dropout", ".1", "--gt-attn-dropout", ".2",
               "--use-component-aux-features", "--use-mordred-features",
               "--mordred-feature-path", str(ROOT / "results/input_graphgps_optimization/features/mordred11_train_standardized.csv"),
               "--mordred-feature-dim", "11", "--include-test"]
    if resume:
        command.append("--resume")
    return command


def run_paired_training(seed_dir: Path, seed: int, parallel: bool) -> tuple[Path, Path]:
    o12, o22 = seed_dir / "O12", seed_dir / "O22"
    log_dir = seed_dir / "logs"
    specs = [(o12, f"R{seed}_O12", "concat_mlp"), (o22, f"R{seed}_O22", "gated_concat")]
    commands = []
    for run_dir, candidate, fusion in specs:
        if (run_dir / "summary.json").is_file():
            continue
        has_resume = (run_dir / "resume_state.pt").is_file()
        command = train_command(run_dir, candidate, seed, fusion, has_resume)
        if run_dir.exists() and not has_resume:
            command.append("--restart-incomplete")
        commands.append((run_dir, command))
    if parallel and len(commands) == 2:
        processes = []
        for run_dir, command in commands:
            log_path = log_dir / f"{run_dir.name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("w", encoding="utf-8")
            processes.append((run_dir, stream, subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                                                  stderr=subprocess.STDOUT)))
        failures = []
        for run_dir, stream, process in processes:
            code = process.wait()
            stream.close()
            if code:
                failures.append(f"{run_dir} (exit {code})")
        if failures:
            raise RuntimeError("Paired training failed: " + ", ".join(failures))
    else:
        for run_dir, command in commands:
            call(command, log_dir / f"{run_dir.name}.log")
    for run_dir, _, _ in specs:
        if not (run_dir / "summary.json").is_file() or not (run_dir / "predictions.csv").is_file():
            raise RuntimeError(f"Missing completed prediction artifacts: {run_dir}")
    return o12, o22


def correlations(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    if len(truth) < 2 or not np.std(truth) or not np.std(prediction):
        return np.nan, np.nan
    return float(pearsonr(truth, prediction).statistic), float(spearmanr(truth, prediction).statistic)


def metric_rows(prediction_path: Path, model: str, repeat: int) -> list[dict[str, object]]:
    table = pd.read_csv(prediction_path)
    rows = []
    for split in SPLITS:
        for target in TARGETS:
            part = table.loc[(table.split == split) & (table.target == target)]
            truth, prediction = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
            pearson, spearman = correlations(truth, prediction)
            rows.append({"repeat": repeat, "model": model, "split": split, "target": target,
                         "n": int(len(part)), "mae": float(mean_absolute_error(truth, prediction)),
                         "rmse": float(mean_squared_error(truth, prediction) ** .5),
                         "r2": float(r2_score(truth, prediction)), "pearson": pearson,
                         "spearman": spearman})
    return rows


def run_ensemble(seed_dir: Path, o12: Path, o22: Path) -> Path:
    output = seed_dir / "ensemble_huber"
    if not (output / "metrics.csv").is_file():
        uncalibrated = seed_dir / "ensemble_validation"
        if not (uncalibrated / "predictions.csv").is_file():
            call([sys.executable, str(ENSEMBLE), "--experiments-root", str(seed_dir),
                  "--output-dir", str(uncalibrated), "--runs", "O12", "O22"],
                 uncalibrated / "ensemble.log")
        call([sys.executable, str(CALIBRATION), "--predictions", str(uncalibrated / "predictions.csv"),
              "--output-dir", str(output)], output / "calibration.log")
    return output


def aggregate(root: Path, rows: list[dict[str, object]]) -> None:
    metrics = pd.DataFrame(rows).sort_values(["model", "repeat", "split", "target"])
    metrics.to_csv(root / "repeat_metrics.csv", index=False)
    per_repeat = metrics.groupby(["model", "repeat", "split"], as_index=False).agg(
        mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"),
        mean_pearson=("pearson", "mean"), mean_spearman=("spearman", "mean"))
    per_repeat.to_csv(root / "repeat_macro_metrics.csv", index=False)
    summary = per_repeat.groupby(["model", "split"], as_index=False).agg(
        repeats=("repeat", "nunique"), mean_mae=("mean_mae", "mean"), std_mae=("mean_mae", "std"),
        median_mae=("mean_mae", "median"), min_mae=("mean_mae", "min"), max_mae=("mean_mae", "max"),
        mean_r2=("mean_r2", "mean"), std_r2=("mean_r2", "std"),
        mean_pearson=("mean_pearson", "mean"), mean_spearman=("mean_spearman", "mean"))
    summary.to_csv(root / "repeat_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/input_graphgps_optimization/repeat10_o12_o22")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--parallel-base-models", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "protocol.json").write_text(json.dumps({
        "repeats": args.repeats, "seeds": list(range(args.seed_start, args.seed_start + args.repeats)),
        "base_models": {"O12": "concat_mlp", "O22": "gated_concat"},
        "selection": "best epoch by each run's validation MAE only",
        "ensemble": "per-target non-negative convex validation weights followed by validation Huber affine calibration",
        "test_selection": "not used for checkpoint, weight, or calibration selection",
    }, indent=2) + "\n", encoding="utf-8")
    all_rows = []
    for repeat, seed in enumerate(range(args.seed_start, args.seed_start + args.repeats), start=1):
        seed_dir = root / "repeats" / f"repeat_{repeat:02d}_seed_{seed}"
        o12, o22 = run_paired_training(seed_dir, seed, args.parallel_base_models)
        ensemble = run_ensemble(seed_dir, o12, o22)
        all_rows.extend(metric_rows(o12 / "predictions.csv", "O12", repeat))
        all_rows.extend(metric_rows(o22 / "predictions.csv", "O22", repeat))
        all_rows.extend(metric_rows(ensemble / "predictions.csv", "O12_O22_ensemble_huber", repeat))
        aggregate(root, all_rows)
        print(json.dumps({"repeat": repeat, "seed": seed, "status": "completed"}), flush=True)
    aggregate(root, all_rows)
    print(pd.read_csv(root / "repeat_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
