
"""Infer frozen O13-E/O13G/O14-A checkpoints on labelled new_validation.

For each checkpoint seed, the checkpoint-specific frozen preprocessing is
reused. Labels are set to zero before loader construction and are never read by
the model or feature-building path. After all checkpoint predictions have been
computed, the original labels are used only for post-hoc ensemble metrics and
true-vs-predicted scatter plots.
"""

from __future__ import annotations

import argparse
import contextlib
import json
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
from sklearn.metrics import mean_absolute_error, r2_score


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
    validate_and_stage,
)
from scripts.diagnostics.predict_o12_o22_feedback_ensemble import (  # noqa: E402
    build_feedback_mordred_lookup,
)


ALL_TARGETS = [*CORE_TARGETS, *NORM_TARGETS]


def target_key(target_group: str, single_target: str | None) -> str:
    """Filename-safe label separating single- and multi-task predictions."""
    if single_target is None:
        return target_group
    return f"single_{single_target.lower()}"


def feature_lookup_for_seed(
    frame: pd.DataFrame,
    seed_root: Path,
    output: Path,
) -> tuple[Path | None, Path, dict]:
    """Generate inference-only standardized lookups using frozen seed scalers."""
    g_metadata = seed_root / "feature_audit.json"
    if g_metadata.is_file():
        meta = json.loads(g_metadata.read_text(encoding="utf-8"))
        if "aa_vocab" in meta:
            table = o13g_raw(frame)
            aa, terminal = meta["aa_vocab"], meta["terminal_vocab"]
            mean, std = float(meta["tail_mean"]), float(meta["tail_population_std"])
            output_table = pd.DataFrame({
                "smiles": table.canonical_smiles,
                "aa_id": table.parsed_AA.map(aa).fillna(0).astype(int),
                "terminal_id": table.terminal_state.map(terminal).fillna(0).astype(int),
                "tail_length_normalized": ((table.tail_length - mean) / std).fillna(0.0),
                "tail_length_present_mask": table.tail_present,
            })
            output_table = output_table.loc[output_table.smiles.ne("[Fr]")]
            fifth_path = output / "fifth_structured_standardized.csv"
            output_table.to_csv(fifth_path, index=False)
            return None, fifth_path, {
                "structured_scaler_metadata": str(g_metadata.resolve()),
            }

    m_metadata = seed_root / "mordred11_all_components_train_only.json"
    f_metadata = seed_root / "fifth_mechanistic_train_only.json"
    semantic_metadata = seed_root / "fifth_semantic_train_only.json"
    if not f_metadata.is_file() and semantic_metadata.is_file():
        f_metadata = semantic_metadata
    if not m_metadata.is_file() or not f_metadata.is_file():
        raise FileNotFoundError(f"Missing strict scaler metadata under {seed_root}")

    mordred_path = output / "mordred11_standardized.csv"
    m_summary = build_feedback_mordred_lookup(frame, m_metadata, mordred_path)
    fifth_metadata = json.loads(f_metadata.read_text(encoding="utf-8"))

    if "feature_layout" in fifth_metadata:
        means = np.asarray(fifth_metadata["numeric_mean"], dtype=float)
        stds = np.asarray(fifth_metadata["numeric_effective_std"], dtype=float)
        vocabs = fifth_metadata["categorical_vocabularies"]
        rows = []
        for value in frame["Fifth_SMILE"]:
            if pd.isna(value) or str(value).strip() in {"", "nan", "[Fr]"}:
                continue
            molecule = Chem.MolFromSmiles(str(value))
            if molecule is None:
                continue
            key = Chem.MolToSmiles(molecule, canonical=True)
            result = semantic_features(key)
            numeric = result.numeric_vector().astype(float)
            vector = list((numeric - means) / stds)
            for column in ("family_type", "UC_amino_acid_type"):
                category = (
                    result.family_type
                    if column == "family_type"
                    else result.uc_amino_acid_type
                )
                vocab = vocabs[column]
                one = np.zeros(len(vocab))
                one[vocab.get(category, 0)] = 1.0
                vector.extend(one)
            rows.append({
                "smiles": key,
                **{f"feature_{i}": x for i, x in enumerate(vector)},
            })
        fifth_path = output / "fifth_semantic_standardized.csv"
        pd.DataFrame(rows).drop_duplicates("smiles").to_csv(fifth_path, index=False)
        return mordred_path, fifth_path, {
            "mordred_summary": m_summary,
        }

    names = fifth_metadata.get("descriptor_names")
    means = np.asarray(fifth_metadata.get("means"), dtype=float)
    stds = np.asarray(fifth_metadata.get("effective_stds"), dtype=float)
    if (
        names != list(MECHANISTIC_DESCRIPTOR_NAMES)
        or len(means) != len(names)
        or np.any(stds <= 0)
    ):
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
        rows.append({
            "smiles": key,
            **{f"feature_{index}": value for index, value in enumerate(standardized)},
        })
    fifth_path = output / "fifth_mechanistic_standardized.csv"
    pd.DataFrame(
        rows,
        columns=["smiles", *[f"feature_{i}" for i in range(len(names))]],
    ).to_csv(fifth_path, index=False)
    return mordred_path, fifth_path, {
        "mordred_scaler_metadata": str(m_metadata.resolve()),
        "fifth_scaler_metadata": str(f_metadata.resolve()),
        "mordred_summary": m_summary,
        "unique_present_fifth_smiles": len(rows),
    }


def predict_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    model_input: Path,
    manifest: Path,
    cache: Path,
    mordred_lookup: Path | None,
    fifth_lookup: Path,
    targets: list[str],
    scale: float,
    target_transform: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    set_cfg_gps(cfg)
    cfg.set_new_allowed(True)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(config_path.resolve()), opts=[]))
    if int(cfg.property_num) != len(targets) or (
        cfg.use_fifth_mechanistic_descriptors
        and int(cfg.fifth_mechanistic_descriptor_dim) != len(MECHANISTIC_DESCRIPTOR_NAMES)
    ):
        raise RuntimeError(f"O13-E target/descriptor dimensions do not match: {config_path}")

    cfg.read_csv = str(model_input.resolve())
    cfg.dataset.dir = str(cache.resolve())
    cfg.dataset.cache_tag = f"o13e_new_validation_{config_path.parent.name}"
    cfg.dataset.cache_refresh = True
    cfg.dataset.diagnostic_split_path = str(manifest.resolve())
    cfg.dataset.diagnostic_id_column = "ID"
    cfg.dataset.diagnostic_manifest_id_column = "ID"
    if cfg.use_mordred_features:
        if mordred_lookup is None:
            raise RuntimeError("Checkpoint requires Mordred features but no inference lookup was generated.")
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
                    raise RuntimeError("Unexpected model forward output during O14/O13 inference.")
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
                    probability_rows.extend(
                        (int(index), float(value))
                        for index, value in zip(indices, probabilities)
                    )

    rows.sort(key=lambda item: item[0])
    if [index for index, _ in rows] != list(range(len(pd.read_csv(model_input)))):
        raise RuntimeError("O13-E predictions do not align one-to-one with new_validation rows")

    probability = None
    if probability_rows:
        probability_rows.sort(key=lambda item: item[0])
        if [index for index, _ in probability_rows] != list(range(len(pd.read_csv(model_input)))):
            raise RuntimeError("O14 classifier probabilities do not align one-to-one with new_validation rows")
        probability = np.asarray([value for _, value in probability_rows], dtype=float)
    return np.vstack([value for _, value in rows]), probability


def _safe_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Return R2 when defined; otherwise NaN."""
    if len(truth) < 2 or np.allclose(truth, truth[0]):
        return float("nan")
    return float(r2_score(truth, prediction))


def ensemble_metrics_by_class(ensemble: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Compute post-hoc ensemble MAE/R2 overall and by Fifth_class."""
    classes = (
        ensemble["Fifth_class"].fillna("other").astype(str).str.strip().str.lower()
        if "Fifth_class" in ensemble.columns
        else pd.Series("other", index=ensemble.index)
    )
    rows = []
    for target in targets:
        pred_col = f"pred_{target}_mean"
        if target not in ensemble.columns or pred_col not in ensemble.columns:
            continue
        for subset_name, mask in (
            ("all", pd.Series(True, index=ensemble.index)),
            ("single", classes.eq("single")),
            ("double", classes.eq("double")),
        ):
            selected = mask & ensemble[target].notna() & ensemble[pred_col].notna()
            if not selected.any():
                continue
            truth = ensemble.loc[selected, target].to_numpy(dtype=float)
            prediction = ensemble.loc[selected, pred_col].to_numpy(dtype=float)
            rows.append({
                "target": target,
                "subset": subset_name,
                "n": int(len(truth)),
                "mae": float(mean_absolute_error(truth, prediction)),
                "r2": _safe_r2(truth, prediction),
            })
    return pd.DataFrame(rows)


def plot_ensemble_scatter(
    ensemble: pd.DataFrame,
    targets: list[str],
    output: Path,
    key: str,
    model_label: str,
) -> list[Path]:
    """Plot true-vs-predicted ensemble scatter using the established format.

    Single and double samples share each subplot but use different marker
    shapes.  Each subplot contains y=x, x=1 and y=1 helper lines.  The labels
    are accessed only here, after all model inference has completed.
    """
    plottable = [
        target
        for target in targets
        if target in ensemble.columns and f"pred_{target}_mean" in ensemble.columns
    ]
    if not plottable:
        print("No labelled targets are available; skipping scatter plots.", flush=True)
        return []

    classes = (
        ensemble["Fifth_class"].fillna("other").astype(str).str.strip().str.lower()
        if "Fifth_class" in ensemble.columns
        else pd.Series("other", index=ensemble.index)
    )

    if len(plottable) == 1:
        figure, axis = plt.subplots(figsize=(6.0, 5.5), constrained_layout=True)
        axes = [axis]
    elif len(plottable) == 2:
        figure, axes_array = plt.subplots(1, 2, figsize=(11.8, 5.3), constrained_layout=True)
        axes = list(np.ravel(axes_array))
    else:
        ncols = 2
        nrows = int(np.ceil(len(plottable) / ncols))
        figure, axes_array = plt.subplots(
            nrows,
            ncols,
            figsize=(11.8, 5.3 * nrows),
            constrained_layout=True,
        )
        axes = list(np.ravel(np.asarray(axes_array, dtype=object)))

    marker_spec = (("single", "o"), ("double", "s"))
    for axis, target in zip(axes, plottable):
        pred_col = f"pred_{target}_mean"
        valid = ensemble[target].notna() & ensemble[pred_col].notna()
        truth = ensemble.loc[valid, target].to_numpy(dtype=float)
        prediction = ensemble.loc[valid, pred_col].to_numpy(dtype=float)
        if len(truth) == 0:
            axis.set_visible(False)
            continue

        lower = min(float(np.min(truth)), float(np.min(prediction)), 1.0)
        upper = max(float(np.max(truth)), float(np.max(prediction)), 1.0)
        padding = max((upper - lower) * 0.06, 0.1)
        limits = (lower - padding, upper + padding)

        plotted = pd.Series(False, index=ensemble.index)
        for label, marker in marker_spec:
            selected = valid & classes.eq(label)
            if selected.any():
                axis.scatter(
                    ensemble.loc[selected, target],
                    ensemble.loc[selected, pred_col],
                    s=42,
                    alpha=0.84,
                    marker=marker,
                    linewidth=0.45,
                    label=label,
                )
                plotted |= selected
        other = valid & ~plotted
        if other.any():
            axis.scatter(
                ensemble.loc[other, target],
                ensemble.loc[other, pred_col],
                s=46,
                alpha=0.84,
                marker="X",
                linewidth=0.45,
                label="other/missing",
            )

        axis.plot(limits, limits, "--", linewidth=1.25, label="y = x")
        axis.axvline(1.0, linestyle=":", linewidth=1.15, label="x = 1")
        axis.axhline(1.0, linestyle="-.", linewidth=1.15, label="y = 1")
        axis.set(
            xlabel="True value",
            ylabel="Predicted value",
            xlim=limits,
            ylim=limits,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)

        mae = float(mean_absolute_error(truth, prediction))
        r2 = _safe_r2(truth, prediction)
        axis.set_title(f"{target}\nMAE = {mae:.3f}, R² = {r2:.3f}")
        axis.legend(loc="best", fontsize=8)

    for axis in axes[len(plottable):]:
        axis.set_visible(False)

    figure.suptitle(model_label)
    scatter_dir = output / "scatter_plots"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    png_path = scatter_dir / f"scatter_{key}.png"
    pdf_path = scatter_dir / f"scatter_{key}.pdf"
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return [png_path, pdf_path]


def automatic_model_label(args: argparse.Namespace) -> str:
    if args.model_label:
        return args.model_label
    if args.o14a_ablation is not None:
        return f"O14-{args.o14a_ablation} {args.o14a_domain.title()} ensemble"
    if args.single_target is not None:
        return f"O13G {args.single_target} ensemble"
    return "O13-E/O13G ensemble"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "results/input_graphgps_optimization/o13e_mean_pooling_fifth_descriptors/o13e_strict_train_only_scaling",
    )
    parser.add_argument(
        "--preprocessing-root",
        type=Path,
        default=ROOT / "results/input_graphgps_optimization/o13e_mean_pooling_fifth_descriptors/preprocessing",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT / "datasets_lrx/raw/feedback/new_validation.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/input_graphgps_optimization/o13e_mean_pooling_fifth_descriptors/new_validation_ensemble",
    )
    parser.add_argument(
        "--target-group",
        choices=tuple(TARGET_GROUPS),
        required=True,
        help="Target group for a multi-task run, or the group containing --single-target.",
    )
    parser.add_argument(
        "--single-target",
        choices=ALL_TARGETS,
        default=None,
        help="Load O13G single-task runs from single_task/<target>/.",
    )
    parser.add_argument(
        "--o14a-ablation",
        choices=("A0", "A1", "A2", "A3"),
        default=None,
        help="Load an O14-A single-target layout instead of the historical O13 layout.",
    )
    parser.add_argument(
        "--o14a-domain",
        choices=("full", "double"),
        default=None,
        help="O14-A training domain. Required together with --o14a-ablation.",
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help="Optional title used above the scatter plot; otherwise inferred from the run type.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 110)))
    args = parser.parse_args()

    if (args.o14a_ablation is None) != (args.o14a_domain is None):
        parser.error("--o14a-ablation and --o14a-domain must be supplied together.")
    if args.o14a_ablation is not None and args.single_target is None:
        parser.error("O14-A inference requires --single-target Norm_before or Norm_after.")

    source, output_root = args.input_csv.resolve(), args.output_root.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"new_validation CSV is missing: {source}")

    frame = pd.read_csv(source, dtype={"ID": str})
    output = output_root / input_stem(source)
    staging = output / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    original, model_input = validate_and_stage(frame, source, staging)
    manifest = output / "loader_manifest.csv"
    make_manifest(original, manifest)

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
            target_title = "NormBefore" if args.single_target == "Norm_before" else "NormAfter"
            domain_title = args.o14a_domain.title()
            run = (
                args.model_root.resolve()
                / args.o14a_ablation
                / args.o14a_domain
                / slug
                / f"O14{args.o14a_ablation}{domain_title}_FifthOOD_{target_title}_seed{seed}"
            )
            feature_root = (
                args.preprocessing_root.resolve()
                / args.o14a_ablation
                / args.o14a_domain
                / slug
                / f"seed{seed}"
            )
        elif args.single_target is not None:
            slug = args.single_target.lower()
            run = (
                args.model_root.resolve()
                / slug
            )
            for child in run.iterdir():
                if child.is_dir() and child.name.endswith(f"split{seed}"):
                    run = child
                    break
            feature_root = args.preprocessing_root.resolve() / f"seed{seed}"
        else:
            group_root = args.model_root.resolve() / args.target_group
            run = group_root / f"O12_split{seed}"
            if not run.exists():
                run = group_root / f"O13G_norm2_split{seed}"
            feature_root = args.preprocessing_root.resolve() / f"seed{seed}"

        checkpoint = run / "checkpoints/selected_best.pt"
        config = run / "effective_config.yaml"
        settings_path = run / "run_settings.json"
        if not all(path.is_file() for path in (checkpoint, config, settings_path)):
            raise FileNotFoundError(f"Incomplete O13-E checkpoint: {run}")

        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if (
            settings.get("loss_targets") != targets
            or settings.get("single_target") != args.single_target
            or settings.get("outer_test_read_during_selection") is not False
        ):
            raise RuntimeError(f"Checkpoint violates frozen O13-E inference contract: {run}")

        seed_dir = output / "seed_specific_features" / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        mordred, fifth, feature_provenance = feature_lookup_for_seed(
            original,
            feature_root,
            seed_dir,
        )
        prediction, high_probability = predict_checkpoint(
            config,
            checkpoint,
            model_input,
            manifest,
            seed_dir / "cache",
            mordred,
            fifth,
            targets,
            scale,
            settings.get("target_transform", "identity"),
        )
        arrays.append(prediction)
        probability_arrays.append(high_probability)
        provenance.append({
            "split_seed": seed,
            "checkpoint": str(checkpoint.resolve()),
            **feature_provenance,
        })

        for target_index, target in enumerate(targets):
            long_rows.extend(
                {
                    "ID": sample_id,
                    "target_group": args.target_group,
                    "single_target": args.single_target,
                    "split_seed": seed,
                    "target": target,
                    "prediction": float(value),
                    "prob_gt1_from_classifier": (
                        float(high_probability[row_index])
                        if high_probability is not None
                        else np.nan
                    ),
                }
                for row_index, (sample_id, value) in enumerate(
                    zip(original.ID, prediction[:, target_index])
                )
            )

    stacked = np.stack(arrays)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=0)
    ensemble = original.copy()
    for index, target in enumerate(targets):
        ensemble[f"pred_{target}_mean"] = mean[:, index]
        ensemble[f"pred_{target}_std_10models"] = std[:, index]
        if all(value is not None for value in probability_arrays):
            probabilities = np.stack(probability_arrays)
            ensemble[f"prob_{target}_gt1_mean"] = probabilities.mean(axis=0)
            ensemble[f"prob_{target}_gt1_std_10models"] = probabilities.std(axis=0, ddof=0)

    output.mkdir(parents=True, exist_ok=True)
    ensemble_path = output / f"ensemble_mean_predictions_{key}.csv"
    ensemble.to_csv(ensemble_path, index=False)
    pd.DataFrame(long_rows).to_csv(
        output / f"predictions_by_model_long_{key}.csv",
        index=False,
    )
    pd.DataFrame({
        "target_group": args.target_group,
        "target": targets,
        "mean_prediction": mean.mean(axis=0),
        "std_prediction_across_samples": mean.std(axis=0, ddof=0),
        "mean_model_std": std.mean(axis=0),
        "max_model_std": std.max(axis=0),
    }).to_csv(output / f"ensemble_prediction_summary_{key}.csv", index=False)

    metrics = ensemble_metrics_by_class(ensemble, targets)
    metrics_path = output / f"ensemble_label_metrics_{key}.csv"
    metrics.to_csv(metrics_path, index=False)

    scatter_paths = plot_ensemble_scatter(
        ensemble,
        targets,
        output,
        key,
        automatic_model_label(args),
    )

    (output / f"provenance_{key}.json").write_text(
        json.dumps({
            "source": str(source),
            "rows": len(original),
            "target_group": args.target_group,
            "single_target": args.single_target,
            "targets": targets,
            "labels_used_for_model_input": False,
            "labels_used_posthoc_for_scatter_and_metrics": True,
            "per_seed_preprocessing": (
                "Each checkpoint uses its own frozen train-only preprocessing; "
                "new_validation labels never enter feature calculation or model inference."
            ),
            "ensemble_label_metrics": str(metrics_path.resolve()),
            "scatter_plots": [str(path.resolve()) for path in scatter_paths],
            "checkpoints_and_features": provenance,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(args.seeds)}-checkpoint ensemble ({key}): {output}")
    if not metrics.empty:
        print("\nPost-hoc labelled ensemble metrics:")
        print(metrics.to_string(index=False))
    if scatter_paths:
        print("\nScatter plots:")
        for path in scatter_paths:
            print(path)


if __name__ == "__main__":
    main()