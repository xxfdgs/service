#!/usr/bin/env python3
"""Run a ten-checkpoint O12 ensemble on arbitrary formulation tables.

Each input table is evaluated by every selected-best checkpoint in ten run
directories.  The default layout is
``O12-10-seeds-prediction-models/<target-group>/O12_split100`` through
``split109``; ``--direct-run-root`` and ``--run-prefix`` support another
frozen ten-run root.  ``core4`` predictions are converted back to percentage
units; ``norm2`` predictions remain in their native scale.  Saved target
transforms are inverted after model inference.  Any target columns present in
an input file are explicitly replaced by zeros in the loader-only copy, so
labels can never affect a prediction.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphgps  # noqa: F401,E402
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402

from scripts.diagnostics.predict_o12_o22_feedback_ensemble import (  # noqa: E402
    MORDRED_11,
    SMILES_COLUMNS,
    build_feedback_mordred_lookup,
)


CORE_TARGETS = [
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
]
NORM_TARGETS = ["Norm_before", "Norm_after"]
TARGET_GROUPS = {
    "core4": (CORE_TARGETS, 100.0),
    "norm2": (NORM_TARGETS, 1.0),
}
REQUIRED_COLUMNS = {
    "ID", *SMILES_COLUMNS,
    "mol%_IL", "mol%_HL", "mol%_Chol", "mol%_PEG", "mol%_Fifth",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def read_table(path: Path, staging_dir: Path, excel_python: Path) -> pd.DataFrame:
    """Read CSV directly; use the supplied Python only if xlsx support is absent."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"ID": str})
    if path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"Unsupported input suffix: {path}")
    if importlib.util.find_spec("openpyxl") is not None:
        return pd.read_excel(path, dtype={"ID": str})
    if not excel_python.is_file():
        raise RuntimeError(
            f"Cannot read {path.name}: current Python lacks openpyxl and "
            f"--excel-python does not exist: {excel_python}")
    converted = staging_dir / f"{input_stem(path)}_xlsx_source.csv"
    converter = (
        "import pandas as pd, sys; "
        "pd.read_excel(sys.argv[1], dtype={'ID': str}).to_csv(sys.argv[2], index=False)"
    )
    subprocess.run(
        [str(excel_python), "-c", converter, str(path), str(converted)],
        check=True,
    )
    return pd.read_csv(converted, dtype={"ID": str})


def validate_and_stage(frame: pd.DataFrame, source: Path, staging_dir: Path) -> tuple[pd.DataFrame, Path]:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{source.name} is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["ID"] = frame["ID"].astype(str)
    if frame["ID"].isna().any() or frame["ID"].duplicated().any():
        raise ValueError(f"{source.name} requires non-null, unique ID values.")
    if len(frame) < 3:
        raise ValueError(f"{source.name} requires at least three rows for the loader contract.")

    # The saved loader constructs y even at inference.  These values are never
    # used by model.forward, and setting all six fields to zero rules out label
    # leakage for the labelled Excel file.
    loader_frame = frame.copy()
    for target in (*CORE_TARGETS, *NORM_TARGETS):
        loader_frame[target] = 0.0
    staging_path = staging_dir / f"{input_stem(source)}_model_input.csv"
    loader_frame.to_csv(staging_path, index=False)
    return frame, staging_path


def make_manifest(frame: pd.DataFrame, output: Path) -> None:
    split = np.full(len(frame), "test", dtype=object)
    split[0], split[1] = "train", "val"  # Required only by the loader API.
    pd.DataFrame({
        "ID": frame["ID"].astype(str),
        "split": split,
        "split_order": np.arange(len(frame), dtype=int),
    }).to_csv(output, index=False)


def relocate_saved_path(path_value: str) -> str:
    """Resolve paths saved before this checkout moved from ``blology`` to ``biology``."""
    configured = Path(path_value)
    if configured.is_file():
        return str(configured.resolve())
    parts = configured.parts
    if "results" in parts:
        candidate = ROOT.joinpath(*parts[parts.index("results"):])
        if candidate.is_file():
            return str(candidate.resolve())
    return path_value


def build_context(config_path: Path, model_input: Path, manifest: Path,
                  cache_dir: Path, mordred_lookup: Path | None, targets: list[str],
                  component_vocab_source: Path | None = None):
    set_cfg_gps(cfg)
    cfg.set_new_allowed(True)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(config_path.resolve()), opts=[]))
    if cfg.model.type != "OneHotEmbedGPS" or int(cfg.property_num) != len(targets):
        raise RuntimeError(
            f"Expected a {len(targets)}-target OneHotEmbedGPS config, got {config_path}")
    if not str(cfg.component_vocab_source).strip():
        raise RuntimeError(f"Saved config has no component vocabulary source: {config_path}")
    # The saved model configuration contains the original 700-row vocabulary
    # source.  Preserve it even if an experiment was moved to a checkout whose
    # absolute path differs from the one embedded in the checkpoint config.
    cfg.component_vocab_source = str(component_vocab_source.resolve()) if component_vocab_source else relocate_saved_path(
        str(cfg.component_vocab_source))
    if not Path(str(cfg.component_vocab_source)).is_file():
        raise FileNotFoundError(
            f"Cannot locate checkpoint component vocabulary CSV: {cfg.component_vocab_source}"
        )
    cfg.read_csv = str(model_input.resolve())
    cfg.dataset.dir = str(cache_dir.resolve())
    cfg.dataset.cache_tag = f"o12_10seed_predict_{config_path.parent.name}"
    cfg.dataset.cache_refresh = True
    cfg.dataset.diagnostic_split_path = str(manifest.resolve())
    cfg.dataset.diagnostic_id_column = "ID"
    cfg.dataset.diagnostic_manifest_id_column = "ID"
    if cfg.use_mordred_features:
        if mordred_lookup is None or not mordred_lookup.is_file():
            raise FileNotFoundError(
                f"Checkpoint requires a Mordred lookup but none was supplied: {config_path}")
        cfg.mordred_feature_path = str(mordred_lookup.resolve())
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "cache_build.log").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            loaders = create_loader_5()
    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    return create_model_gps().to(device), loaders, device


def predict_checkpoint(model: torch.nn.Module, loaders, device: torch.device,
                       expected_rows: int, targets: list[str], scale: float,
                       target_transform: str) -> np.ndarray:
    model.eval()
    rows: list[tuple[int, np.ndarray]] = []
    with torch.no_grad():
        for loader_index in range(3):
            for batches in zip(*[group[loader_index] for group in loaders]):
                for suffix, batch in zip(("", "_2", "_3", "_4", "_5"), batches):
                    batch.split = "predict" + suffix
                    batch.to(device)
                output, _ = model(*batches)
                values = output.detach().cpu().reshape(-1, len(targets)).numpy()
                if target_transform == "log1p":
                    values = np.maximum(np.expm1(values), 0.0)
                elif target_transform != "identity":
                    raise ValueError(f"Unsupported target transform: {target_transform}")
                values *= scale
                source = batches[0].sample_uid.detach().cpu().numpy().reshape(-1)
                rows.extend((int(index), value) for index, value in zip(source, values))
    rows.sort(key=lambda item: item[0])
    if [index for index, _ in rows] != list(range(expected_rows)):
        raise RuntimeError("Predictions do not align one-to-one with source rows.")
    prediction = np.vstack([value for _, value in rows])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Non-finite O12 prediction encountered.")
    return prediction


def checkpoint_specs(model_root: Path, target_group: str, targets: list[str],
                     run_prefix: str, direct_run_root: bool, first_seed: int,
                     seed_count: int,
                     ) -> list[tuple[int, Path, Path, dict]]:
    specs = []
    runs_root = model_root if direct_run_root else model_root / target_group
    for seed in range(first_seed, first_seed + seed_count):
        run_dir = runs_root / f"{run_prefix}{seed}"
        checkpoint = run_dir / "checkpoints" / "selected_best.pt"
        config = run_dir / "effective_config.yaml"
        settings_path = run_dir / "run_settings.json"
        if not all(path.is_file() for path in (checkpoint, config, settings_path)):
            raise FileNotFoundError(f"Incomplete O12 checkpoint run: {run_dir}")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if settings.get("loss_targets") != targets:
            raise RuntimeError(f"Checkpoint is not the requested O12 {target_group} model: {run_dir}")
        if settings.get("model_type") != "OneHotEmbedGPS":
            raise RuntimeError(f"Checkpoint is not a GraphGPS OneHotEmbedGPS model: {run_dir}")
        if settings.get("target_transform", "identity") not in {"identity", "log1p"}:
            raise RuntimeError(f"Unsupported saved target transform in {run_dir}")
        if settings.get("outer_test_read_during_selection") is not False:
            raise RuntimeError(f"Checkpoint selection may have read the outer test split: {run_dir}")
        specs.append((seed, checkpoint, config, settings))
    return specs


def run_dataset(source: Path, specs: list[tuple[int, Path, Path, dict]],
                output_root: Path, mordred_metadata: Path, excel_python: Path,
                target_group: str, targets: list[str], scale: float,
                component_vocab_source: Path | None = None) -> dict[str, object]:
    output = output_root / input_stem(source)
    staging = output / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    frame = read_table(source, staging, excel_python)
    original, model_input = validate_and_stage(frame, source, staging)
    manifest = output / "loader_manifest.csv"
    make_manifest(original, manifest)
    mordred_usage = {bool(settings.get("use_mordred_features", False))
                     for _, _, _, settings in specs}
    if len(mordred_usage) != 1:
        raise RuntimeError("An ensemble cannot mix checkpoints with and without Mordred11 features.")
    use_mordred_features = mordred_usage.pop()
    mordred_lookup = output / "mordred11_standardized.csv" if use_mordred_features else None
    descriptor_summary = (
        build_feedback_mordred_lookup(original, mordred_metadata, mordred_lookup)
        if use_mordred_features else {
            "enabled": False,
            "policy": "Checkpoint has use_mordred_features=false; no Mordred lookup was built or loaded.",
        }
    )

    model, loaders, device = build_context(
        specs[0][2], model_input, manifest, output / f"cache_{target_group}", mordred_lookup, targets,
        component_vocab_source)
    all_predictions = []
    long_rows = []
    transforms = {settings.get("target_transform", "identity")
                  for _, _, _, settings in specs}
    if len(transforms) != 1:
        raise RuntimeError(f"Ensemble mixes target transforms: {sorted(transforms)}")
    for seed, checkpoint_path, _, settings in specs:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        prediction = predict_checkpoint(
            model, loaders, device, len(original), targets, scale,
            settings.get("target_transform", "identity"))
        all_predictions.append(prediction)
        for target_index, target in enumerate(targets):
            long_rows.extend({
                "ID": sample_id,
                "target_group": target_group,
                "split_seed": seed,
                "target": target,
                "prediction": float(value),
            } for sample_id, value in zip(original["ID"], prediction[:, target_index]))

    stacked = np.stack(all_predictions, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=0)
    ensemble = original.copy()
    for target_index, target in enumerate(targets):
        ensemble[f"pred_{target}_mean"] = mean[:, target_index]
        ensemble[f"pred_{target}_std_10models"] = std[:, target_index]
    output.mkdir(parents=True, exist_ok=True)
    ensemble_path = output / f"ensemble_mean_predictions_{target_group}.csv"
    long_path = output / f"predictions_by_model_long_{target_group}.csv"
    summary_path = output / f"ensemble_prediction_summary_{target_group}.csv"
    ensemble.to_csv(ensemble_path, index=False)
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    pd.DataFrame({
        "target_group": target_group,
        "target": targets,
        "mean_prediction": mean.mean(axis=0),
        "std_prediction_across_samples": mean.std(axis=0, ddof=0),
        "mean_model_std": std.mean(axis=0),
        "max_model_std": std.max(axis=0),
    }).to_csv(summary_path, index=False)
    (output / f"provenance_{target_group}.json").write_text(json.dumps({
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "source_rows": int(len(original)),
        "target_group": target_group,
        "targets": targets,
        "ensemble": "unweighted arithmetic mean over ten O12 selected-best checkpoints",
        "uncertainty": "population standard deviation over the ten checkpoint predictions",
        "target_transform": next(iter(transforms)),
        "target_transform_inverse": "expm1_clamped_at_zero" if next(iter(transforms)) == "log1p" else "identity",
        "labels_used_for_model_input": False,
        "mordred_features": MORDRED_11 if use_mordred_features else [],
        "mordred_lookup": (str(mordred_lookup.resolve()) if mordred_lookup else None),
        "mordred_lookup_sha256": sha256(mordred_lookup) if mordred_lookup else None,
        "mordred_summary": descriptor_summary,
        "checkpoints": [{
            "split_seed": seed,
            "path": str(checkpoint.resolve()),
            "sha256": sha256(checkpoint),
        } for seed, checkpoint, _, _ in specs],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "dataset": source.name,
        "rows": len(original),
        "models": len(specs),
        "target_group": target_group,
        "output": str(ensemble_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path,
                        default=ROOT / "results/input_graphgps_optimization/O12-10-seeds-prediction-models")
    parser.add_argument("--input-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/input_graphgps_optimization/O12-10-seeds-prediction-models/predict_ensemble_10seed")
    parser.add_argument("--mordred-metadata", type=Path,
                        default=ROOT / "results/input_graphgps_optimization/features/mordred11_train_standardized.json")
    parser.add_argument("--component-vocab-source", type=Path,
                        help="Optional original training CSV used to build the frozen component vocabularies. "
                             "Use this only when a relocated checkpoint config contains a stale absolute path.")
    parser.add_argument("--target-group", choices=tuple(TARGET_GROUPS), default="core4",
                        help="Checkpoint subdirectory and matching target group to infer.")
    parser.add_argument("--run-prefix", default="O12_split",
                        help="Run-directory prefix immediately before split seeds 100..109.")
    parser.add_argument("--direct-run-root", action="store_true",
                        help="Find run directories directly below --model-root, without a target-group subdirectory.")
    parser.add_argument("--first-seed", type=int, default=100,
                        help="First split seed in the consecutive ensemble.")
    parser.add_argument("--seed-count", type=int, default=10,
                        help="Number of consecutive split seeds; final O12 reports expect ten.")
    parser.add_argument("--excel-python", type=Path,
                        default=Path("/home/puzexuan/anaconda3/bin/python"))
    args = parser.parse_args()
    model_root = args.model_root.resolve()
    output_root = args.output_root.resolve()
    metadata = args.mordred_metadata.resolve()
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing O12 Mordred metadata: {metadata}")
    component_vocab_source = (args.component_vocab_source.resolve()
                              if args.component_vocab_source is not None else None)
    if component_vocab_source is not None and not component_vocab_source.is_file():
        raise FileNotFoundError(
            f"Component-vocabulary source does not exist: {component_vocab_source}")
    targets, scale = TARGET_GROUPS[args.target_group]
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive.")
    specs = checkpoint_specs(
        model_root, args.target_group, targets, args.run_prefix,
        args.direct_run_root, args.first_seed, args.seed_count)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in args.input_files:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Input file does not exist: {source}")
        rows.append(run_dataset(source, specs, output_root, metadata, args.excel_python,
                                args.target_group, targets, scale, component_vocab_source))
    pd.DataFrame(rows).to_csv(output_root / f"run_summary_{args.target_group}.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
