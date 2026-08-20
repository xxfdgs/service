#!/usr/bin/env python3
"""Inference and diagnostics with the training-consistent five-component loader.

Examples
--------
python scripts/predict_with_split_plots.py --model O12
python scripts/predict_with_split_plots.py --model O22
python scripts/predict_with_split_plots.py --model O12-O22 --apply-huber-calibration

``O12-O22`` uses per-property convex weights fitted on the fixed validation
split only.  Huber calibration is optional because it changes the ensemble
from the raw validation-selected ensemble to the deployment ensemble.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphgps  # noqa: F401,E402
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from torch_geometric import seed_everything  # noqa: E402
from torch_geometric.graphgym.config import cfg as gcfg  # noqa: E402


TARGETS = [
    "EE_before", "EE_after", "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
]
MODEL_SPECS = {
    "O12": {
        "config": ROOT / "results/input_graphgps_optimization/experiments/"
                         "O12_input_onehot_aux_all_mordred_attn20_seed43/effective_config.yaml",
        "checkpoint": ROOT / "results/input_graphgps_optimization/experiments/"
                             "O12_input_onehot_aux_all_mordred_attn20_seed43/checkpoints/selected_best.pt",
    },
    "O22": {
        "config": ROOT / "results/input_graphgps_optimization/experiments/"
                         "O22_input_onehot_aux_all_mordred_attn20_gated_seed43/effective_config.yaml",
        "checkpoint": ROOT / "results/input_graphgps_optimization/experiments/"
                             "O22_input_onehot_aux_all_mordred_attn20_gated_seed43/checkpoints/selected_best.pt",
    },
}
DEFAULT_INPUT = ROOT / "results/new_dataset_benchmark_20260713/input_sanitized_utf8.csv"
DEFAULT_MANIFEST = ROOT / "results/new_dataset_benchmark_20260713/split_manifest.csv"
DEFAULT_WEIGHTS = ROOT / (
    "results/input_graphgps_optimization/calibration/"
    "O12_O22_validation_convex_ensemble/validation_weights.csv"
)
DEFAULT_CALIBRATION = ROOT / (
    "results/input_graphgps_optimization/calibration/"
    "O12_O22_validation_convex_ensemble_huber/coefficients.csv"
)


def setup_config(config_path: Path, input_csv: Path, manifest_path: Path,
                 cache_root: Path) -> None:
    """Load the exact saved architecture, then configure safe inference I/O."""
    cache_root.mkdir(parents=True, exist_ok=True)
    set_cfg_gps(gcfg)
    gcfg.set_new_allowed(True)
    gcfg.merge_from_file(str(config_path))
    gcfg.accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    gcfg.gpu_serial = 0
    gcfg.read_csv = str(input_csv.resolve())
    gcfg.train.mode = "double"  # Select LRX_five_multi, not the legacy predict loader.
    gcfg.data_rate = False
    gcfg.data_rate_type = False
    gcfg.dataset.dir = str(cache_root.resolve())
    gcfg.dataset.cache_per_run = True
    gcfg.dataset.cache_refresh = True
    gcfg.dataset.cache_tag = f"inference_{config_path.parent.name}"
    gcfg.dataset.diagnostic_split_path = str(manifest_path.resolve())
    gcfg.dataset.diagnostic_id_column = "ID"
    gcfg.dataset.diagnostic_manifest_id_column = "sample_id"
    for pe_name in ["posenc_LapPE", "posenc_SignNet", "posenc_RWSE",
                    "posenc_HKdiagSE", "posenc_ElstaticSE"]:
        pe_cfg = getattr(gcfg, pe_name, None)
        if pe_cfg and hasattr(pe_cfg, "kernel") and pe_cfg.kernel.times_func:
            pe_cfg.kernel.times = list(eval(pe_cfg.kernel.times_func))


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    device = torch.device(gcfg.accelerator, gcfg.gpu_serial) if gcfg.accelerator == "cuda" else torch.device("cpu")
    model = create_model_gps(dim_in=1, dim_out=gcfg.property_num).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def predict_one_model(name: str, config_path: Path, checkpoint_path: Path,
                      input_csv: Path, manifest_path: Path, cache_root: Path,
                      source_ids: np.ndarray) -> pd.DataFrame:
    setup_config(config_path, input_csv, manifest_path, cache_root / name)
    seed_everything(int(gcfg.seed))
    loaders, loaders_2, loaders_3, loaders_4, loaders_5 = create_loader_5()
    model = load_model(checkpoint_path)
    device = next(model.parameters()).device
    rows = []
    split_loaders = {"train": 0, "val": 1, "test": 2}
    with torch.no_grad():
        for split_name, loader_index in split_loaders.items():
            batches_by_component = [
                loaders[loader_index], loaders_2[loader_index], loaders_3[loader_index],
                loaders_4[loader_index], loaders_5[loader_index],
            ]
            for batches in tqdm(zip(*batches_by_component), total=len(loaders[loader_index]),
                                desc=f"{name}:{split_name}"):
                for suffix, batch in zip(("", "_2", "_3", "_4", "_5"), batches):
                    batch.split = split_name + suffix
                    batch.to(device)
                prediction, label = model(*batches)
                batch_size = batches[0].num_graphs
                prediction = prediction.detach().cpu().numpy().reshape(batch_size, len(TARGETS)) * 100.0
                label = label.detach().cpu().numpy().reshape(batch_size, len(TARGETS)) * 100.0
                if not hasattr(batches[0], "sample_uid"):
                    raise RuntimeError("Training-consistent loader did not provide sample_uid.")
                source_indices = batches[0].sample_uid.detach().cpu().numpy().reshape(-1).astype(int)
                for source_index, true_values, predicted_values in zip(source_indices, label, prediction):
                    rows.append({
                        "source_index": source_index,
                        "sample_id": str(source_ids[source_index]),
                        "split": split_name,
                        **{f"y_true_{target}": float(true_values[index])
                           for index, target in enumerate(TARGETS)},
                        **{f"y_pred_{target}": float(predicted_values[index])
                           for index, target in enumerate(TARGETS)},
                    })
    output = pd.DataFrame(rows).sort_values("source_index", kind="stable").reset_index(drop=True)
    if output.source_index.tolist() != list(range(len(source_ids))):
        raise RuntimeError(f"{name}: predictions do not cover the input CSV exactly once.")
    return output


def build_ensemble(o12: pd.DataFrame, o22: pd.DataFrame, weights_path: Path,
                   calibration_path: Path | None) -> pd.DataFrame:
    if not o12[["source_index", "sample_id", "split"]].equals(o22[["source_index", "sample_id", "split"]]):
        raise RuntimeError("O12 and O22 prediction rows are not aligned.")
    weights = pd.read_csv(weights_path).pivot(index="target", columns="experiment", values="weight")
    names = {
        "O12_input_onehot_aux_all_mordred_attn20_seed43": "O12",
        "O22_input_onehot_aux_all_mordred_attn20_gated_seed43": "O22",
    }
    output = o12.copy()
    coefficients = None if calibration_path is None else pd.read_csv(calibration_path).set_index("target")
    for target in TARGETS:
        prediction = sum(float(weights.loc[target, experiment]) * frame[f"y_pred_{target}"].to_numpy()
                         for experiment, frame in [(key, o12 if name == "O12" else o22)
                                                   for key, name in names.items()])
        if coefficients is not None:
            prediction = (prediction * float(coefficients.loc[target, "coefficient"])
                          + float(coefficients.loc[target, "intercept"]))
        output[f"y_pred_{target}"] = prediction
    return output


def compute_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test", "all_input"]:
        subset = predictions if split == "all_input" else predictions.loc[predictions.split.eq(split)]
        for target in TARGETS:
            truth = subset[f"y_true_{target}"].to_numpy(float)
            prediction = subset[f"y_pred_{target}"].to_numpy(float)
            rows.append({"split": split, "target": target, "n": len(subset),
                         "mae": mean_absolute_error(truth, prediction),
                         "r2": r2_score(truth, prediction)})
    return pd.DataFrame(rows)


def plot_per_split(predictions: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path,
                   title_prefix: str) -> None:
    colors = {"train": "#2e75b6", "val": "#f5a623", "test": "#b44742"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for axis, target in zip(axes.flat, TARGETS):
        truth = predictions[f"y_true_{target}"].to_numpy(float)
        prediction = predictions[f"y_pred_{target}"].to_numpy(float)
        lower, upper = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
        padding = max((upper - lower) * .04, 1.0)
        limits = (lower - padding, upper + padding)
        for split, color in colors.items():
            subset = predictions.loc[predictions.split.eq(split)]
            axis.scatter(subset[f"y_true_{target}"], subset[f"y_pred_{target}"], s=18,
                         alpha=.65, color=color, edgecolors="none", label=split)
        summary = metrics.loc[(metrics.split == "all_input") & (metrics.target == target)].iloc[0]
        axis.plot(limits, limits, "--", color="black", lw=1.2, label="y = x")
        axis.set(xlim=limits, ylim=limits, xlabel="True value", ylabel="Predicted value",
                 title=f"{target}\nMAE={summary.mae:.3f}, R²={summary.r2:.3f}")
        axis.grid(alpha=.22)
        axis.legend(fontsize=8)
    figure.suptitle(f"{title_prefix}: complete input dataset (n={len(predictions)})", fontsize=14)
    figure.savefig(output_dir / "full_input_true_vs_pred.png", dpi=220)
    figure.savefig(output_dir / "full_input_true_vs_pred.pdf")
    plt.close(figure)


def resolve_single_spec(args: argparse.Namespace) -> tuple[str, Path, Path]:
    if args.model in MODEL_SPECS:
        spec = MODEL_SPECS[args.model]
        return args.model, spec["config"], spec["checkpoint"]
    if args.config is None or args.checkpoint is None:
        raise ValueError("--model single requires both --config and --checkpoint.")
    return "single", args.config.resolve(), args.checkpoint.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["single", "O12", "O22", "O12-O22"], default="O12")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, help="Required only for --model single.")
    parser.add_argument("--checkpoint", type=Path, help="Required only for --model single.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--apply-huber-calibration", action="store_true")
    args = parser.parse_args()

    input_csv, manifest = args.input_csv.resolve(), args.manifest.resolve()
    source = pd.read_csv(input_csv)
    if source.ID.isna().any() or source.ID.astype(str).duplicated().any():
        raise ValueError("Input CSV must contain a complete, unique ID column.")
    required = {"sample_id", "split"}
    manifest_data = pd.read_csv(manifest)
    if required - set(manifest_data.columns) or len(manifest_data) != len(source):
        raise ValueError("Manifest must contain sample_id and split and cover the input CSV once.")
    output_dir = (args.output_dir or ROOT / "results/input_graphgps_optimization/input_inference" /
                  args.model.replace("-", "_")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "O12-O22":
        base = {}
        for name, spec in MODEL_SPECS.items():
            base[name] = predict_one_model(name, spec["config"], spec["checkpoint"], input_csv,
                                           manifest, output_dir / "cache", source.ID.astype(str).to_numpy())
            base[name].to_csv(output_dir / f"predictions_{name}.csv", index=False)
        predictions = build_ensemble(base["O12"], base["O22"], args.weights.resolve(),
                                     args.calibration.resolve() if args.apply_huber_calibration else None)
        title = "O12-O22 validation-weighted ensemble"
        if args.apply_huber_calibration:
            title += " + Huber calibration"
    else:
        name, config_path, checkpoint_path = resolve_single_spec(args)
        predictions = predict_one_model(name, config_path, checkpoint_path, input_csv, manifest,
                                        output_dir / "cache", source.ID.astype(str).to_numpy())
        title = name
    predictions.to_csv(output_dir / "predictions_by_split.csv", index=False)
    metrics = compute_metrics(predictions)
    metrics.to_csv(output_dir / "metrics_by_split.csv", index=False)
    metrics.loc[metrics.split.eq("all_input")].to_csv(output_dir / "metrics_full_input.csv", index=False)
    plot_per_split(predictions, metrics, output_dir, title)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Wrote complete-input predictions and plots to: {output_dir}")


if __name__ == "__main__":
    main()
