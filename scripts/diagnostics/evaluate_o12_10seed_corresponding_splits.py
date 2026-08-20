#!/usr/bin/env python3
"""Evaluate frozen O12 core4/norm2 checkpoints on their own val/test splits.

Each selected-best checkpoint is loaded once, evaluated once on the validation
loader and once on the test loader defined by its saved split manifest.  It
also averages all selected-best checkpoints on the full training-source CSV
and writes per-target true-vs-predicted scatter plots.  The latter is a
descriptive in-sample ensemble plot, not an independent generalisation score.
The script never trains, selects, calibrates, or changes any checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphgps  # noqa: F401,E402
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402


TARGET_GROUPS = {
    "core4": (
        ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"],
        100.0,
    ),
    "norm2": (["Norm_before", "Norm_after"], 1.0),
}
SPLIT_TO_LOADER_INDEX = {"val": 1, "test": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_correlation(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or not np.std(truth) or not np.std(prediction):
        return math.nan
    return float(function(truth, prediction).statistic)


def metric_values(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(truth)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)) if np.std(truth) else math.nan,
        "pearson": safe_correlation(pearsonr, truth, prediction),
        "spearman": safe_correlation(spearmanr, truth, prediction),
    }


def resolve_manifest(settings: dict, manifest_root: Path) -> Path:
    recorded = Path(settings["split_manifest"])
    if recorded.is_file():
        return recorded.resolve()
    local = manifest_root / recorded.name
    if local.is_file():
        return local.resolve()
    raise FileNotFoundError(f"Cannot find split manifest: {recorded}")


def relocate_saved_path(path_value: str) -> Path:
    """Resolve a moved ``results/...`` path recorded by a remote run."""
    configured = Path(path_value)
    if configured.is_file():
        return configured.resolve()
    if "results" in configured.parts:
        candidate = ROOT.joinpath(*configured.parts[configured.parts.index("results"):])
        if candidate.is_file():
            return candidate.resolve()
    return configured


def build_context(config_path: Path, manifest_path: Path, cache_dir: Path,
                  expected_targets: list[str]):
    set_cfg_gps(cfg)
    cfg.set_new_allowed(True)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(config_path.resolve()), opts=[]))
    if cfg.model.type != "OneHotEmbedGPS" or int(cfg.property_num) != len(expected_targets):
        raise RuntimeError(f"Unexpected checkpoint config: {config_path}")
    if not str(cfg.component_vocab_source).strip():
        raise RuntimeError(f"Missing component vocabulary source: {config_path}")
    input_path = relocate_saved_path(str(cfg.read_csv))
    vocab_path = relocate_saved_path(str(cfg.component_vocab_source))
    if not input_path.is_file() or not vocab_path.is_file():
        raise FileNotFoundError(f"Saved input or component vocabulary source is unavailable for {config_path}")
    cfg.read_csv = str(input_path)
    cfg.component_vocab_source = str(vocab_path)
    # Feature-off ablations deliberately store an empty Mordred path. Do not
    # reject those checkpoints; a loaded lookup is required only when their
    # saved model actually consumes the branch.
    if cfg.use_mordred_features:
        mordred_path = relocate_saved_path(str(cfg.mordred_feature_path))
        if not mordred_path.is_file():
            raise FileNotFoundError(f"Saved Mordred feature file is unavailable for {config_path}")
        cfg.mordred_feature_path = str(mordred_path)
    cfg.dataset.dir = str(cache_dir.resolve())
    cfg.dataset.cache_tag = f"frozen_split_eval_{config_path.parent.name}"
    cfg.dataset.cache_refresh = True
    cfg.dataset.diagnostic_split_path = str(manifest_path)
    cfg.dataset.diagnostic_id_column = "ID"
    cfg.dataset.diagnostic_manifest_id_column = "sample_id"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "cache_build.log").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            loaders = create_loader_5()
    return create_model_gps().to(torch.device(cfg.accelerator, cfg.gpu_serial)), loaders, torch.device(cfg.accelerator, cfg.gpu_serial), input_path


def predict_split(model: torch.nn.Module, loaders, device: torch.device, split: str,
                  targets: list[str], scale: float, expected_indices: set[int]) -> tuple[np.ndarray, np.ndarray]:
    loader_index = SPLIT_TO_LOADER_INDEX[split]
    rows: list[tuple[int, np.ndarray]] = []
    model.eval()
    with torch.no_grad():
        for batches in zip(*[group[loader_index] for group in loaders]):
            for suffix, batch in zip(("", "_2", "_3", "_4", "_5"), batches):
                batch.split = split + suffix
                batch.to(device)
            output, _ = model(*batches)
            values = output.detach().cpu().reshape(-1, len(targets)).numpy() * scale
            source = batches[0].sample_uid.detach().cpu().numpy().reshape(-1)
            rows.extend((int(index), value) for index, value in zip(source, values))
    rows.sort(key=lambda item: item[0])
    indices = np.asarray([index for index, _ in rows], dtype=int)
    if set(indices) != expected_indices or len(indices) != len(expected_indices):
        raise RuntimeError(f"{split} sample IDs do not match the saved manifest.")
    return indices, np.vstack([value for _, value in rows])


def predict_full_source(model: torch.nn.Module, loaders, device: torch.device,
                        targets: list[str], scale: float,
                        expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Predict every row from all three loaders and restore reporting units."""
    rows: list[tuple[int, np.ndarray]] = []
    model.eval()
    with torch.no_grad():
        for loader_index in range(3):
            for batches in zip(*[group[loader_index] for group in loaders]):
                for suffix, batch in zip(("", "_2", "_3", "_4", "_5"), batches):
                    batch.split = "full_source" + suffix
                    batch.to(device)
                output, _ = model(*batches)
                values = output.detach().cpu().reshape(-1, len(targets)).numpy() * scale
                source = batches[0].sample_uid.detach().cpu().numpy().reshape(-1)
                rows.extend((int(index), value) for index, value in zip(source, values))
    rows.sort(key=lambda item: item[0])
    indices = np.asarray([index for index, _ in rows], dtype=int)
    if not np.array_equal(indices, np.arange(expected_rows, dtype=int)):
        raise RuntimeError("Full-source predictions do not align one-to-one with input CSV rows.")
    return indices, np.vstack([value for _, value in rows])


def evaluate_run(model_root: Path, target_group: str, split_seed: int,
                 manifest_root: Path, output: Path) -> tuple[list[dict], list[dict], dict[str, object]]:
    targets, scale = TARGET_GROUPS[target_group]
    run_dir = model_root / target_group / f"O12_split{split_seed}"
    checkpoint_path = run_dir / "checkpoints" / "selected_best.pt"
    config_path = run_dir / "effective_config.yaml"
    settings_path = run_dir / "run_settings.json"
    if not all(path.is_file() for path in (checkpoint_path, config_path, settings_path)):
        raise FileNotFoundError(f"Incomplete saved model: {run_dir}")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if settings.get("loss_targets") != targets:
        raise RuntimeError(f"Target group mismatch in {run_dir}")
    manifest_path = resolve_manifest(settings, manifest_root)
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    expected_by_split = {
        split: manifest.loc[manifest.split.eq(split), "sample_id"].astype(str).tolist()
        for split in SPLIT_TO_LOADER_INDEX
    }
    if not all(expected_by_split.values()):
        raise RuntimeError(f"Manifest lacks val/test rows: {manifest_path}")

    cache_dir = output / "cache" / target_group / f"O12_split{split_seed}"
    model, loaders, device, input_path = build_context(config_path, manifest_path, cache_dir, targets)
    input_frame = pd.read_csv(input_path, dtype={"ID": str})
    if input_frame.ID.duplicated().any():
        raise RuntimeError(f"Input IDs are not unique: {input_path}")
    source_by_id = pd.Series(input_frame.index.to_numpy(), index=input_frame.ID.astype(str))
    expected_indices = {
        split: set(source_by_id.loc[ids].to_numpy(int)) for split, ids in expected_by_split.items()
    }

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    metric_rows, prediction_rows = [], []
    base = {
        "target_group": target_group,
        "split_seed": split_seed,
        "run": run_dir.name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "split_manifest": str(manifest_path),
    }
    for split in SPLIT_TO_LOADER_INDEX:
        indices, prediction = predict_split(model, loaders, device, split, targets, scale, expected_indices[split])
        truth = input_frame.loc[indices, targets].to_numpy(float)
        ids = input_frame.loc[indices, "ID"].astype(str).to_numpy()
        for target_index, target in enumerate(targets):
            metric_rows.append({**base, "split": split, "target": target,
                                **metric_values(truth[:, target_index], prediction[:, target_index])})
            prediction_rows.extend({**base, "split": split, "target": target,
                                    "sample_id": sample_id, "source_index": int(index),
                                    "y_true": float(true_value), "y_pred": float(predicted_value)}
                                   for sample_id, index, true_value, predicted_value in zip(
                                       ids, indices, truth[:, target_index], prediction[:, target_index]))
    full_indices, full_prediction = predict_full_source(
        model, loaders, device, targets, scale, len(input_frame))
    full_result = {
        "target_group": target_group,
        "split_seed": split_seed,
        "sample_ids": input_frame.loc[full_indices, "ID"].astype(str).to_numpy(),
        "source_indices": full_indices,
        "truth": input_frame.loc[full_indices, targets].to_numpy(float),
        "prediction": full_prediction,
        "targets": targets,
        "input_csv": str(input_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    return metric_rows, prediction_rows, full_result


def write_full_source_ensemble(target_group: str, results: list[dict[str, object]],
                               output: Path) -> None:
    """Average completed checkpoints on the input CSV and plot each target."""
    if not results:
        return
    first = results[0]
    targets = first["targets"]
    sample_ids = first["sample_ids"]
    source_indices = first["source_indices"]
    truth = first["truth"]
    for result in results[1:]:
        if (result["targets"] != targets
                or not np.array_equal(result["sample_ids"], sample_ids)
                or not np.array_equal(result["source_indices"], source_indices)
                or not np.allclose(result["truth"], truth)):
            raise RuntimeError(
                f"Cannot average {target_group} checkpoints with different input rows or labels.")

    predictions = np.stack([result["prediction"] for result in results], axis=0)
    prediction_mean = predictions.mean(axis=0)
    prediction_std = predictions.std(axis=0, ddof=0)
    ensemble_dir = output / "full_training_data_ensemble" / target_group
    plots_dir = ensemble_dir / "scatter_by_target"
    plots_dir.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame({"source_index": source_indices, "sample_id": sample_ids})
    metric_rows = []
    for target_index, target in enumerate(targets):
        table[f"y_true_{target}"] = truth[:, target_index]
        table[f"pred_{target}_mean"] = prediction_mean[:, target_index]
        table[f"pred_{target}_std_10models"] = prediction_std[:, target_index]
        summary = metric_values(truth[:, target_index], prediction_mean[:, target_index])
        metric_rows.append({"target_group": target_group, "target": target, **summary})

        lower = min(float(truth[:, target_index].min()), float(prediction_mean[:, target_index].min()))
        upper = max(float(truth[:, target_index].max()), float(prediction_mean[:, target_index].max()))
        padding = max((upper - lower) * .06, .1)
        limits = (lower - padding, upper + padding)
        figure, axis = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
        axis.scatter(truth[:, target_index], prediction_mean[:, target_index], s=28, alpha=.78,
                     color="#4c78a8", edgecolor="#222", linewidth=.3)
        axis.plot(limits, limits, "--", color="#d62728", linewidth=1.35, label="y = x")
        axis.set(xlabel="True value", ylabel="Mean prediction", xlim=limits, ylim=limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=.25)
        axis.legend(loc="upper left", fontsize=8)
        axis.set_title(
            f"O12 {target_group}, full training-source data: {target}\n"
            f"MAE = {summary['mae']:.3f}, R² = {summary['r2']:.3f}"
        )
        figure.savefig(plots_dir / f"{target}_true_vs_pred.png", dpi=180, bbox_inches="tight")
        figure.savefig(plots_dir / f"{target}_true_vs_pred.pdf", bbox_inches="tight")
        plt.close(figure)

    table.to_csv(ensemble_dir / "ensemble_mean_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(ensemble_dir / "metrics_ensemble.csv", index=False)
    (ensemble_dir / "provenance.json").write_text(json.dumps({
        "input_csv": first["input_csv"],
        "rows": int(len(table)),
        "target_group": target_group,
        "targets": targets,
        "ensemble": "unweighted arithmetic mean over completed selected-best checkpoints",
        "plot_interpretation": (
            "All rows from the training-source CSV are predicted by every checkpoint. "
            "This is an in-sample ensemble description, not an out-of-fold or external score."),
        "checkpoints": [{
            "split_seed": int(result["split_seed"]),
            "path": result["checkpoint"], "sha256": result["checkpoint_sha256"],
        } for result in results],
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path,
                        default=ROOT / "results/input_graphgps_optimization/O12-10-seeds-prediction-models")
    parser.add_argument("--manifest-root", type=Path,
                        default=ROOT / "results/input_graphgps_optimization/five_split_manifests")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/input_graphgps_optimization/O12-10-seeds-prediction-models/corresponding_split_single_inference")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 110)))
    parser.add_argument("--target-groups", choices=tuple(TARGET_GROUPS), nargs="+",
                        default=list(TARGET_GROUPS),
                        help="Completed O12 target groups to evaluate. Defaults to both core4 and norm2.")
    args = parser.parse_args()
    if sorted(args.seeds) != list(dict.fromkeys(args.seeds)) or any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be unique non-negative integers.")
    if len(args.target_groups) != len(set(args.target_groups)):
        raise ValueError("--target-groups must not contain duplicates.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_rows = [], []
    full_source_results: dict[str, list[dict[str, object]]] = {
        target_group: [] for target_group in args.target_groups
    }
    for target_group in args.target_groups:
        for split_seed in args.seeds:
            metrics, predictions, full_result = evaluate_run(
                args.model_root.resolve(), target_group, split_seed,
                args.manifest_root.resolve(), output)
            metric_rows.extend(metrics)
            prediction_rows.extend(predictions)
            full_source_results[target_group].append(full_result)
            print(f"completed {target_group} split{split_seed}", flush=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["target_group", "split", "split_seed", "target"])
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["target_group", "split", "split_seed", "target", "source_index"])
    metrics.to_csv(output / "metrics_by_checkpoint_target.csv", index=False)
    predictions.to_csv(output / "predictions_by_checkpoint.csv", index=False)
    target_summary = metrics.groupby(["target_group", "split", "target"], as_index=False).agg(
        checkpoints=("split_seed", "nunique"),
        mean_mae=("mae", "mean"), variance_mae=("mae", "var"), std_mae=("mae", "std"),
        mean_r2=("r2", "mean"), variance_r2=("r2", "var"), std_r2=("r2", "std"),
        mean_rmse=("rmse", "mean"), variance_rmse=("rmse", "var"),
    ).sort_values(["split", "target_group", "target"])
    target_summary.to_csv(output / "metrics_target_10seed_mean_variance.csv", index=False)
    macro = metrics.groupby(["target_group", "split", "split_seed"], as_index=False).agg(
        targets=("target", "count"), mean_mae=("mae", "mean"), mean_r2=("r2", "mean"), mean_rmse=("rmse", "mean"))
    macro.to_csv(output / "metrics_macro_by_checkpoint.csv", index=False)
    macro_summary = macro.groupby(["target_group", "split"], as_index=False).agg(
        checkpoints=("split_seed", "nunique"),
        mean_mae=("mean_mae", "mean"), variance_mae=("mean_mae", "var"), std_mae=("mean_mae", "std"),
        mean_r2=("mean_r2", "mean"), variance_r2=("mean_r2", "var"), std_r2=("mean_r2", "std"),
    ).sort_values(["split", "target_group"])
    macro_summary.to_csv(output / "metrics_macro_10seed_mean_variance.csv", index=False)
    for target_group, results in full_source_results.items():
        write_full_source_ensemble(target_group, results, output)
    (output / "provenance.json").write_text(json.dumps({
        "model_root": str(args.model_root.resolve()),
        "manifest_root": str(args.manifest_root.resolve()),
        "seeds": args.seeds,
        "target_groups": args.target_groups,
        "checkpoint_selection": "pre-existing selected_best.pt; no training or selection performed",
        "metric_variance": "sample variance across checkpoint split seeds (pandas ddof=1)",
        "full_training_source_ensemble": (
            "Each completed checkpoint predicts all source CSV rows; per-target plots and metrics "
            "use the unweighted mean. These are descriptive in-sample ensemble results."),
    }, indent=2) + "\n", encoding="utf-8")
    print("\nPer-target mean / variance:")
    print(target_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nMacro mean / variance:")
    print(macro_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
