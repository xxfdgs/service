#!/usr/bin/env python3
"""Run the frozen O12/O22 validation-selected ensemble on labelled feedback.

The two GraphGPS checkpoints, convex ensemble weights, and affine Huber heads
are all pre-existing frozen artifacts.  Feedback labels are never used for
feature construction or prediction; they are read only after predictions are
formed to calculate the requested diagnostic metrics and scatter plots.
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
from rdkit import Chem
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


TARGETS = [
    "EE_before", "EE_after", "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
]
SMILES_COLUMNS = ["IL_SMILE", "HL_SMILE", "Chol_SMILE", "PEG_SMILE", "Fifth_SMILE"]
MORDRED_11 = [
    "SsNH3", "SMR_VSA9", "SlogP_VSA11", "SlogP_VSA10", "TopoPSA", "MW",
    "nRot", "nRing", "nAromAtom", "nHBDon", "nHBAcc",
]
COLORS = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b"]
DISPLAY_NAMES = ["EE before", "EE after", "Aerosolization", "mRNA recovery"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_smiles(value: object) -> str:
    if pd.isna(value) or str(value).strip() in {"", "nan", "None", "[Fr]"}:
        return ""
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        raise ValueError(f"Cannot parse feedback SMILES: {value!r}")
    return Chem.MolToSmiles(molecule, canonical=True)


def build_feedback_mordred_lookup(feedback: pd.DataFrame, metadata_path: Path,
                                  output_path: Path) -> dict[str, int]:
    """Compute feedback descriptors with the input-trained fixed scaler."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    means = np.asarray(metadata["means"], dtype=float)
    stds = np.asarray(metadata["stds"], dtype=float)
    if len(means) != len(MORDRED_11) or np.any(stds <= 0):
        raise RuntimeError("The O12 Mordred standardisation metadata is invalid.")
    keys = sorted({canonical_smiles(value) for column in SMILES_COLUMNS
                   for value in feedback[column] if canonical_smiles(value)})
    if not hasattr(np, "product"):
        np.product = np.prod  # Mordred 1.x compatibility under NumPy 2.x.
    from mordred import Calculator, descriptors  # pylint: disable=import-outside-toplevel

    calculator = Calculator(descriptors, ignore_3D=True)
    available = {str(descriptor): descriptor for descriptor in calculator.descriptors}
    missing = [name for name in MORDRED_11 if name not in available]
    if missing:
        raise RuntimeError(f"Installed Mordred misses descriptors: {missing}")
    selected = Calculator([available[name] for name in MORDRED_11], ignore_3D=True)
    rows = []
    for key in keys:
        values = selected(Chem.MolFromSmiles(key)).asdict()
        raw = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)
        raw = np.where(np.isfinite(raw), raw, means)
        standardized = (raw - means) / stds
        rows.append({"smiles": key, **{f"feature_{index}": value
                                        for index, value in enumerate(standardized)}})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["smiles", *[f"feature_{index}" for index in range(len(MORDRED_11))]]).to_csv(
        output_path, index=False)
    return {"unique_valid_feedback_smiles": len(keys), "feature_count": len(MORDRED_11)}


def make_feedback_manifest(feedback: pd.DataFrame, output_path: Path) -> None:
    if feedback.ID.isna().any() or feedback.ID.astype(str).duplicated().any():
        raise ValueError("Feedback ID must be complete and unique.")
    if len(feedback) < 3:
        raise ValueError("Feedback inference requires at least three rows.")
    split = np.full(len(feedback), "test", dtype=object)
    split[0], split[1] = "train", "val"  # Loader contract only; all rows are evaluated identically.
    pd.DataFrame({"ID": feedback.ID.astype(str), "split": split,
                  "split_order": np.arange(len(feedback), dtype=int)}).to_csv(output_path, index=False)


def prepare_config(config_path: Path, feedback_path: Path, manifest_path: Path,
                   cache_dir: Path, mordred_lookup: Path) -> None:
    set_cfg_gps(cfg)
    # effective_config.yaml intentionally records runtime provenance fields
    # (for example run_dir) in addition to the registered GraphGym schema.
    # They are harmless at inference, but must be admitted to recover the
    # exact saved architecture configuration.
    cfg.set_new_allowed(True)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(config_path.resolve()), opts=[]))
    cfg.read_csv = str(feedback_path.resolve())
    cfg.dataset.dir = str(cache_dir.resolve())
    cfg.dataset.cache_tag = f"feedback_inference_{config_path.parent.name}"
    cfg.dataset.cache_refresh = True
    cfg.dataset.diagnostic_split_path = str(manifest_path.resolve())
    cfg.dataset.diagnostic_id_column = "ID"
    cfg.dataset.diagnostic_manifest_id_column = "ID"
    cfg.mordred_feature_path = str(mordred_lookup.resolve())
    # component_vocab_source remains the original input CSV from effective_config.
    if not str(cfg.component_vocab_source).strip():
        raise RuntimeError("Inference config does not preserve the original component vocabulary source.")


def predict_checkpoint(config_path: Path, checkpoint_path: Path, feedback_path: Path,
                       manifest_path: Path, cache_dir: Path, mordred_lookup: Path,
                       feedback: pd.DataFrame) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepare_config(config_path, feedback_path, manifest_path, cache_dir, mordred_lookup)
    with (cache_dir / "cache_build.log").open("w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            loaders = create_loader_5()
    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model = create_model_gps().to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    rows = []
    with torch.no_grad():
        for loader_index in range(3):
            for batches in zip(*[group[loader_index] for group in loaders]):
                for suffix, batch in zip(("", "_2", "_3", "_4", "_5"), batches):
                    batch.split = "test" + suffix
                    batch.to(device)
                prediction, _ = model(*batches)
                values = prediction.detach().cpu().reshape(-1, len(TARGETS)).numpy() * 100.0
                source = batches[0].sample_uid.detach().cpu().numpy().reshape(-1)
                rows.extend((int(index), value) for index, value in zip(source, values))
    rows.sort(key=lambda item: item[0])
    indices = [index for index, _ in rows]
    if indices != list(range(len(feedback))):
        raise RuntimeError("Feedback prediction rows do not align one-to-one with the source CSV.")
    return np.vstack([value for _, value in rows])


def safe_correlation(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(function(truth, prediction).statistic) if len(truth) > 1 and np.std(truth) and np.std(prediction) else math.nan


def metrics_and_plots(predictions: np.ndarray, feedback: pd.DataFrame, output: Path) -> pd.DataFrame:
    metric_rows, prediction_rows = [], []
    for index, target in enumerate(TARGETS):
        truth, prediction = feedback[target].to_numpy(float), predictions[:, index]
        valid = np.isfinite(truth) & np.isfinite(prediction)
        truth, prediction = truth[valid], prediction[valid]
        metric_rows.append({"target": target, "n": int(len(truth)),
                            "mae": float(mean_absolute_error(truth, prediction)),
                            "rmse": float(mean_squared_error(truth, prediction) ** .5),
                            "r2": float(r2_score(truth, prediction)),
                            "pearson": safe_correlation(pearsonr, truth, prediction),
                            "spearman": safe_correlation(spearmanr, truth, prediction)})
        for sample_id, true_value, predicted_value in zip(feedback.loc[valid, "ID"], truth, prediction):
            prediction_rows.append({"sample_id": str(sample_id), "split": "feedback", "target": target,
                                    "y_true": float(true_value), "y_pred": float(predicted_value)})
    metrics = pd.DataFrame(metric_rows)
    pd.DataFrame(prediction_rows).to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    metrics[["mae", "rmse", "r2", "pearson", "spearman"]].mean().to_frame().T.to_csv(
        output / "metrics_summary.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 9.2))
    long_predictions = pd.DataFrame(prediction_rows)
    for axis, target, name, color in zip(axes.flat, TARGETS, DISPLAY_NAMES, COLORS):
        values = long_predictions.loc[long_predictions.target.eq(target)]
        truth, prediction = values.y_true.to_numpy(float), values.y_pred.to_numpy(float)
        lower, upper = float(min(truth.min(), prediction.min())), float(max(truth.max(), prediction.max()))
        padding = max((upper - lower) * .05, 1.0)
        limits = (lower - padding, upper + padding)
        summary = metrics.loc[metrics.target.eq(target)].iloc[0]
        axis.scatter(truth, prediction, s=35, alpha=.78, color=color, edgecolor="#222222", linewidth=.35)
        axis.plot(limits, limits, color="#d62728", linestyle="--", linewidth=1.5, label="y = x")
        axis.set(xlim=limits, ylim=limits, xlabel="True value", ylabel="Predicted value")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{name}\nMAE = {summary.mae:.3f}, R² = {summary.r2:.3f}")
        axis.grid(alpha=.25)
        axis.legend(loc="upper left", fontsize=8)
    figure.suptitle("Frozen O12/O22 ensemble: feedback predictions", fontsize=15, y=.98)
    figure.tight_layout(rect=(0, 0, 1, .96))
    figure.savefig(output / "feedback_true_vs_pred.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / "feedback_true_vs_pred.pdf", bbox_inches="tight")
    plt.close(figure)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-csv", type=Path, default=ROOT / "datasets_lrx/raw/feedback/20260703_validation.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/input_graphgps_optimization/feedback_inference/O12_O22_huber")
    parser.add_argument("--o12-config", type=Path, default=ROOT / "results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/effective_config.yaml")
    parser.add_argument("--o22-config", type=Path, default=ROOT / "results/input_graphgps_optimization/experiments/O22_input_onehot_aux_all_mordred_attn20_gated_seed43/effective_config.yaml")
    parser.add_argument("--o12-checkpoint", type=Path, default=ROOT / "results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/checkpoints/best_candidate_epoch_74.pt")
    parser.add_argument("--o22-checkpoint", type=Path, default=ROOT / "results/input_graphgps_optimization/experiments/O22_input_onehot_aux_all_mordred_attn20_gated_seed43/checkpoints/best_candidate_epoch_70.pt")
    parser.add_argument("--weights", type=Path, default=ROOT / "results/input_graphgps_optimization/calibration/O12_O22_validation_convex_ensemble/validation_weights.csv")
    parser.add_argument("--calibration", type=Path, default=ROOT / "results/input_graphgps_optimization/calibration/O12_O22_validation_convex_ensemble_huber/coefficients.csv")
    parser.add_argument("--mordred-metadata", type=Path, default=ROOT / "results/input_graphgps_optimization/features/mordred11_train_standardized.json")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    feedback_path = args.feedback_csv.resolve()
    feedback = pd.read_csv(feedback_path)
    required = {"ID", *SMILES_COLUMNS, *TARGETS}
    if missing := required - set(feedback.columns):
        raise ValueError(f"Feedback CSV misses required columns: {sorted(missing)}")
    manifest_path = output / "feedback_loader_manifest.csv"
    make_feedback_manifest(feedback, manifest_path)
    lookup_path = output / "mordred11_feedback_standardized.csv"
    descriptor_summary = build_feedback_mordred_lookup(feedback, args.mordred_metadata.resolve(), lookup_path)

    model_specs = [("O12", args.o12_config.resolve(), args.o12_checkpoint.resolve()),
                   ("O22", args.o22_config.resolve(), args.o22_checkpoint.resolve())]
    base = {}
    for name, config_path, checkpoint_path in model_specs:
        base[name] = predict_checkpoint(config_path, checkpoint_path, feedback_path, manifest_path,
                                        output / f"cache_{name}", lookup_path, feedback)
    weights = pd.read_csv(args.weights.resolve()).pivot(index="target", columns="experiment", values="weight")
    expected_columns = {
        "O12_input_onehot_aux_all_mordred_attn20_seed43": "O12",
        "O22_input_onehot_aux_all_mordred_attn20_gated_seed43": "O22",
    }
    ensemble = np.empty_like(base["O12"])
    for index, target in enumerate(TARGETS):
        row = weights.loc[target]
        ensemble[:, index] = sum(float(row[column]) * base[name][:, index]
                                 for column, name in expected_columns.items())
    coefficients = pd.read_csv(args.calibration.resolve()).set_index("target")
    calibrated = np.empty_like(ensemble)
    for index, target in enumerate(TARGETS):
        calibrated[:, index] = (ensemble[:, index] * float(coefficients.loc[target, "coefficient"])
                                + float(coefficients.loc[target, "intercept"]))
    np.savez_compressed(output / "base_and_ensemble_predictions.npz", O12=base["O12"], O22=base["O22"],
                        ensemble_before_calibration=ensemble, final=calibrated)
    metrics = metrics_and_plots(calibrated, feedback, output)
    (output / "provenance.json").write_text(json.dumps({
        "feedback_csv": str(feedback_path), "feedback_sha256": sha256(feedback_path),
        "feedback_rows": int(len(feedback)), "checkpoints": {name: {"path": str(checkpoint), "sha256": sha256(checkpoint)}
          for name, _, checkpoint in model_specs},
        "weights": str(args.weights.resolve()), "weights_sha256": sha256(args.weights.resolve()),
        "calibration": str(args.calibration.resolve()), "calibration_sha256": sha256(args.calibration.resolve()),
        "mordred_metadata": str(args.mordred_metadata.resolve()), "mordred_metadata_sha256": sha256(args.mordred_metadata.resolve()),
        "mordred_lookup": str(lookup_path), "mordred_lookup_sha256": sha256(lookup_path),
        "descriptor_summary": descriptor_summary,
        "weights_and_calibration_fit_on": "fixed input validation split only",
        "feedback_labels_used_only_for": "post-prediction metrics and scatter plots",
    }, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
