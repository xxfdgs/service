#!/usr/bin/env python3
"""Infer frozen O13-E Fifth-OOD checkpoints on labelled new_validation.

For each checkpoint seed, the 11 Mordred and twelve Fifth-only mechanism
descriptors are transformed with that seed's pre-existing train-only scaler.
Labels are set to zero before loader construction and are never read by this
script.  Scoring and scatter plots are intentionally a separate step.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphgps  # noqa: F401,E402
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from graphgps.lrx_add.fifth_mechanistic_descriptors import (  # noqa: E402
    MECHANISTIC_DESCRIPTOR_NAMES,
    descriptor_vector,
)
from graphgps.lrx_add.fifth_semantic_features import semantic_features  # noqa: E402
from scripts.diagnostics.build_o13g_structured_features import raw as o13g_raw  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402

from scripts.diagnostics.predict_o12_10seed_ensemble import (  # noqa: E402
    CORE_TARGETS,
    NORM_TARGETS,
    TARGET_GROUPS,
    input_stem,
    make_manifest,
    sha256,
    validate_and_stage,
)
from scripts.diagnostics.predict_o12_o22_feedback_ensemble import (  # noqa: E402
    SMILES_COLUMNS,
    build_feedback_mordred_lookup,
)


ALL_TARGETS = [*CORE_TARGETS, *NORM_TARGETS]


def target_key(target_group: str, single_target: str | None) -> str:
    """Filename-safe label separating single- and multi-task predictions."""
    if single_target is None:
        return target_group
    return f"single_{single_target.lower()}"


def feature_lookup_for_seed(frame: pd.DataFrame, seed_root: Path, output: Path) -> tuple[Path, Path, dict]:
    """Generate inference-only standardized lookups using frozen seed scalers."""
    g_metadata = seed_root / "feature_audit.json"
    if g_metadata.is_file():
        meta = json.loads(g_metadata.read_text(encoding="utf-8"))
        if "aa_vocab" in meta:
            table = o13g_raw(frame)
            aa, terminal = meta["aa_vocab"], meta["terminal_vocab"]
            mean, std = float(meta["tail_mean"]), float(meta["tail_population_std"])
            output_table = pd.DataFrame({"smiles": table.canonical_smiles,
                "aa_id": table.parsed_AA.map(aa).fillna(0).astype(int),
                "terminal_id": table.terminal_state.map(terminal).fillna(0).astype(int),
                "tail_length_normalized": ((table.tail_length - mean) / std).fillna(0.),
                "tail_length_present_mask": table.tail_present})
            output_table = output_table.loc[output_table.smiles.ne("[Fr]")]
            fifth_path = output / "fifth_structured_standardized.csv"; output_table.to_csv(fifth_path, index=False)
            return None, fifth_path, {"structured_scaler_metadata": str(g_metadata.resolve()),
                                      "structured_lookup_sha256": sha256(fifth_path)}
    m_metadata = seed_root / "mordred11_all_components_train_only.json"
    f_metadata = (seed_root / "fifth_mechanistic_train_only.json")
    semantic_metadata = seed_root / "fifth_semantic_train_only.json"
    if not f_metadata.is_file() and semantic_metadata.is_file():
        f_metadata = semantic_metadata
    if not m_metadata.is_file() or not f_metadata.is_file():
        raise FileNotFoundError(f"Missing strict scaler metadata under {seed_root}")
    mordred_path = output / "mordred11_standardized.csv"
    m_summary = build_feedback_mordred_lookup(frame, m_metadata, mordred_path)
    fifth_metadata = json.loads(f_metadata.read_text(encoding="utf-8"))
    if "feature_layout" in fifth_metadata:
        names = fifth_metadata["feature_layout"]
        means = np.asarray(fifth_metadata["numeric_mean"], dtype=float)
        stds = np.asarray(fifth_metadata["numeric_effective_std"], dtype=float)
        vocabs = fifth_metadata["categorical_vocabularies"]
        rows = []
        for value in frame["Fifth_SMILE"]:
            if pd.isna(value) or str(value).strip() in {"", "nan", "[Fr]"}: continue
            molecule = Chem.MolFromSmiles(str(value))
            if molecule is None: continue
            key = Chem.MolToSmiles(molecule, canonical=True); result = semantic_features(key)
            numeric = result.numeric_vector().astype(float)
            vector = list((numeric - means) / stds)
            for column in ("family_type", "UC_amino_acid_type"):
                value = result.family_type if column == "family_type" else result.uc_amino_acid_type
                vocab = vocabs[column]; one = np.zeros(len(vocab)); one[vocab.get(value, 0)] = 1.; vector.extend(one)
            rows.append({"smiles": key, **{f"feature_{i}": x for i, x in enumerate(vector)}})
        fifth_path = output / "fifth_semantic_standardized.csv"
        pd.DataFrame(rows).drop_duplicates("smiles").to_csv(fifth_path, index=False)
        return mordred_path, fifth_path, {"mordred_summary": m_summary, "fifth_lookup_sha256": sha256(fifth_path)}
    names = fifth_metadata.get("descriptor_names")
    means = np.asarray(fifth_metadata.get("means"), dtype=float)
    stds = np.asarray(fifth_metadata.get("effective_stds"), dtype=float)
    if names != list(MECHANISTIC_DESCRIPTOR_NAMES) or len(means) != len(names) or np.any(stds <= 0):
        raise RuntimeError(f"Invalid O13-E fifth descriptor scaler: {f_metadata}")
    rows = []
    keys = set()
    for value in frame["Fifth_SMILE"]:
        if pd.isna(value) or str(value).strip() in {"", "nan", "[Fr]"}:
            continue
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            continue
        keys.add(Chem.MolToSmiles(molecule, canonical=True))
    for key in sorted(keys):
        values = descriptor_vector(key).astype(float)
        standardized = (values - means) / stds
        rows.append({"smiles": key, **{f"feature_{index}": value
                                       for index, value in enumerate(standardized)}})
    fifth_path = output / "fifth_mechanistic_standardized.csv"
    pd.DataFrame(rows, columns=["smiles", *[f"feature_{i}" for i in range(len(names))]]).to_csv(
        fifth_path, index=False)
    return mordred_path, fifth_path, {
        "mordred_scaler_metadata": str(m_metadata.resolve()), "mordred_scaler_sha256": sha256(m_metadata),
        "fifth_scaler_metadata": str(f_metadata.resolve()), "fifth_scaler_sha256": sha256(f_metadata),
        "mordred_lookup_sha256": sha256(mordred_path), "fifth_lookup_sha256": sha256(fifth_path),
        "mordred_summary": m_summary, "unique_present_fifth_smiles": len(rows),
    }


def predict_checkpoint(config_path: Path, checkpoint_path: Path, model_input: Path, manifest: Path,
                       cache: Path, mordred_lookup: Path, fifth_lookup: Path,
                       targets: list[str], scale: float, target_transform: str) -> tuple[np.ndarray, np.ndarray | None]:
    set_cfg_gps(cfg)
    cfg.set_new_allowed(True)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(config_path.resolve()), opts=[]))
    if cfg.model.type != "OneHotEmbedGPS" or not (cfg.use_fifth_mechanistic_descriptors or cfg.use_fifth_semantic_features or cfg.use_fifth_structured_features):
        raise RuntimeError(f"Not an O13-E OneHotEmbedGPS config: {config_path}")
    if int(cfg.property_num) != len(targets) or (
            cfg.use_fifth_mechanistic_descriptors
            and int(cfg.fifth_mechanistic_descriptor_dim) != len(MECHANISTIC_DESCRIPTOR_NAMES)):
        raise RuntimeError(f"O13-E target/descriptor dimensions do not match: {config_path}")
    cfg.read_csv = str(model_input.resolve())
    cfg.dataset.dir = str(cache.resolve())
    cfg.dataset.cache_tag = f"o13e_new_validation_{config_path.parent.name}"
    cfg.dataset.cache_refresh = True
    cfg.dataset.diagnostic_split_path = str(manifest.resolve())
    cfg.dataset.diagnostic_id_column = "ID"
    cfg.dataset.diagnostic_manifest_id_column = "ID"
    if cfg.use_mordred_features:
        cfg.mordred_feature_path = str(mordred_lookup.resolve())
    if cfg.use_fifth_mechanistic_descriptors:
        cfg.fifth_mechanistic_descriptor_path = str(fifth_lookup.resolve())
    else:
        cfg.fifth_semantic_feature_path = str(fifth_lookup.resolve())
    if cfg.use_fifth_structured_features:
        cfg.fifth_structured_feature_path = str(fifth_lookup.resolve())
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "cache_build.log").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            loaders = create_loader_5()
    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model = create_model_gps().to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    rows, probability_rows = [], []
    with torch.no_grad():
        for loader_index in range(3):
            for batches in zip(*[group[loader_index] for group in loaders]):
                for suffix, batch in zip(("", "_2", "_3", "_4", "_5"), batches):
                    batch.split = "predict" + suffix
                    batch.to(device)
                model_output = model(*batches)
                if not isinstance(model_output, tuple) or len(model_output) not in {2, 3}:
                    raise RuntimeError('Unexpected model forward output during O14/O13 inference.')
                output = model_output[0]
                high_logit = model_output[2] if len(model_output) == 3 else None
                values = output.detach().cpu().reshape(-1, len(targets)).numpy()
                if target_transform == "log1p":
                    values = np.maximum(np.expm1(values), 0.0)
                elif target_transform != "identity":
                    raise ValueError(f"Unsupported target transform: {target_transform}")
                values *= scale
                indices = batches[0].sample_uid.detach().cpu().numpy().reshape(-1)
                rows.extend((int(index), value) for index, value in zip(indices, values))
                if high_logit is not None:
                    probabilities = torch.sigmoid(high_logit).detach().cpu().numpy().reshape(-1)
                    probability_rows.extend((int(index), float(value))
                                            for index, value in zip(indices, probabilities))
    rows.sort(key=lambda item: item[0])
    if [index for index, _ in rows] != list(range(len(pd.read_csv(model_input)))):
        raise RuntimeError("O13-E predictions do not align one-to-one with new_validation rows")
    probability = None
    if probability_rows:
        probability_rows.sort(key=lambda item: item[0])
        if [index for index, _ in probability_rows] != list(range(len(pd.read_csv(model_input)))):
            raise RuntimeError('O14 classifier probabilities do not align one-to-one with new_validation rows')
        probability = np.asarray([value for _, value in probability_rows], dtype=float)
    return np.vstack([value for _, value in rows]), probability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=ROOT / "results/input_graphgps_optimization/o13e_mean_pooling_fifth_descriptors/o13e_strict_train_only_scaling")
    parser.add_argument("--preprocessing-root", type=Path, default=ROOT / "results/input_graphgps_optimization/o13e_mean_pooling_fifth_descriptors/preprocessing")
    parser.add_argument("--input-csv", type=Path, default=ROOT / "datasets_lrx/raw/feedback/new_validation.csv")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/input_graphgps_optimization/o13e_mean_pooling_fifth_descriptors/new_validation_ensemble")
    parser.add_argument("--target-group", choices=tuple(TARGET_GROUPS), required=True,
                        help="Target group for a multi-task run, or the group containing --single-target.")
    parser.add_argument("--single-target", choices=ALL_TARGETS, default=None,
                        help="Load O13G single-task runs from single_task/<target>/.")
    parser.add_argument("--o14a-ablation", choices=("A0", "A1", "A2", "A3"), default=None,
                        help="Load an O14-A single-target layout instead of the historical O13 layout.")
    parser.add_argument("--o14a-domain", choices=("full", "double"), default=None,
                        help="O14-A training domain. Required together with --o14a-ablation.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 110)))
    args = parser.parse_args()
    if (args.o14a_ablation is None) != (args.o14a_domain is None):
        parser.error('--o14a-ablation and --o14a-domain must be supplied together.')
    if args.o14a_ablation is not None and args.single_target is None:
        parser.error('O14-A inference requires --single-target Norm_before or Norm_after.')
    source, output_root = args.input_csv.resolve(), args.output_root.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"new_validation CSV is missing: {source}")
    frame = pd.read_csv(source, dtype={"ID": str})
    output = output_root / input_stem(source)
    staging = output / "staging"; staging.mkdir(parents=True, exist_ok=True)
    original, model_input = validate_and_stage(frame, source, staging)
    manifest = output / "loader_manifest.csv"; make_manifest(original, manifest)
    if args.single_target is None:
        targets, scale = TARGET_GROUPS[args.target_group]
    else:
        targets = [args.single_target]
        scale = 100.0 if args.single_target in CORE_TARGETS else 1.0
    key = target_key(args.target_group, args.single_target)
    arrays, probability_arrays, long_rows, provenance = [], [], [], []
    for seed in args.seeds:
        if args.o14a_ablation is not None:
            slug = args.single_target.lower()
            target_title = 'NormBefore' if args.single_target == 'Norm_before' else 'NormAfter'
            domain_title = args.o14a_domain.title()
            run = (args.model_root.resolve() / args.o14a_ablation / args.o14a_domain / slug /
                   f'O14{args.o14a_ablation}{domain_title}_FifthOOD_{target_title}_seed{seed}')
            feature_root = (args.preprocessing_root.resolve() / args.o14a_ablation /
                            args.o14a_domain / slug / f'seed{seed}')
        elif args.single_target is not None:
            slug = args.single_target.lower()
            run = (args.model_root.resolve() / "single_task" / slug /
                   f"O13G_{slug}_split{seed}")
            feature_root = args.preprocessing_root.resolve() / f"seed{seed}"
        else:
            group_root = args.model_root.resolve() / args.target_group
            run = group_root / f"O12_split{seed}"
            if not run.exists():
                run = group_root / f"O13G_norm2_split{seed}"
            feature_root = args.preprocessing_root.resolve() / f"seed{seed}"
        checkpoint, config, settings_path = (run / "checkpoints/selected_best.pt", run / "effective_config.yaml", run / "run_settings.json")
        if not all(path.is_file() for path in (checkpoint, config, settings_path)):
            raise FileNotFoundError(f"Incomplete O13-E checkpoint: {run}")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if (settings.get("loss_targets") != targets
                or settings.get("single_target") != args.single_target
                or settings.get("outer_test_read_during_selection") is not False):
            raise RuntimeError(f"Checkpoint violates frozen O13-E inference contract: {run}")
        seed_dir = output / "seed_specific_features" / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        mordred, fifth, feature_provenance = feature_lookup_for_seed(
            original, feature_root, seed_dir)
        prediction, high_probability = predict_checkpoint(
            config, checkpoint, model_input, manifest, seed_dir / "cache", mordred, fifth,
            targets, scale, settings.get("target_transform", "identity"))
        arrays.append(prediction)
        probability_arrays.append(high_probability)
        provenance.append({"split_seed": seed, "checkpoint": str(checkpoint.resolve()),
                           "checkpoint_sha256": sha256(checkpoint), **feature_provenance})
        for target_index, target in enumerate(targets):
            long_rows.extend({"ID": sample_id, "target_group": args.target_group,
                              "single_target": args.single_target, "split_seed": seed,
                              "target": target, "prediction": float(value),
                              "prob_gt1_from_classifier": (
                                  float(high_probability[row_index]) if high_probability is not None else np.nan)}
                             for row_index, (sample_id, value) in enumerate(
                                 zip(original.ID, prediction[:, target_index])))
    stacked = np.stack(arrays)
    mean, std = stacked.mean(axis=0), stacked.std(axis=0, ddof=0)
    ensemble = original.copy()
    for index, target in enumerate(targets):
        ensemble[f"pred_{target}_mean"] = mean[:, index]
        ensemble[f"pred_{target}_std_10models"] = std[:, index]
        if all(value is not None for value in probability_arrays):
            probabilities = np.stack(probability_arrays)
            ensemble[f"prob_{target}_gt1_mean"] = probabilities.mean(axis=0)
            ensemble[f"prob_{target}_gt1_std_10models"] = probabilities.std(axis=0, ddof=0)
    output.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(output / f"ensemble_mean_predictions_{key}.csv", index=False)
    pd.DataFrame(long_rows).to_csv(output / f"predictions_by_model_long_{key}.csv", index=False)
    pd.DataFrame({"target_group": args.target_group, "target": targets,
                  "mean_prediction": mean.mean(axis=0), "std_prediction_across_samples": mean.std(axis=0, ddof=0),
                  "mean_model_std": std.mean(axis=0), "max_model_std": std.max(axis=0)}).to_csv(
        output / f"ensemble_prediction_summary_{key}.csv", index=False)
    (output / f"provenance_{key}.json").write_text(json.dumps({
        "source": str(source), "source_sha256": sha256(source), "rows": len(original),
        "target_group": args.target_group, "single_target": args.single_target, "targets": targets,
        "labels_used_for_model_input": False,
        "per_seed_preprocessing": "Each checkpoint uses its own frozen train-only Mordred11 and Fifth descriptor scaler; new_validation labels never enter feature calculation.",
        "checkpoints_and_features": provenance,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(args.seeds)}-checkpoint ensemble ({key}): {output}")


if __name__ == "__main__":
    main()
