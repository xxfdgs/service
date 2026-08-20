#!/usr/bin/env python3
"""
Stage-5 P0/P1/P2 ensemble inference on labelled new_validation.

Purpose
-------
For each Stage-5 variant:
    P0_random
    P1_PT_D
    P2_PT_DF

load selected-best checkpoints for split seeds 100/101/102, run inference on
new_validation, average the three predictions, calculate MAE/R2, and generate:

    scatter_P0_random.png
    scatter_P1_PT_D.png
    scatter_P2_PT_DF.png
    scatter_P0_P1_P2_comparison.png

Each scatter contains:
    - x = true Norm_before
    - y = 3-checkpoint ensemble mean prediction
    - y=x reference line
    - MAE
    - R^2

Leakage policy
--------------
The true labels from new_validation are NOT passed to the model.

A temporary loader-only CSV is made in which all six property labels are set
to zero.  Labels are joined back only AFTER inference for scoring/plotting.

Why the temporary CSV is triplicated
------------------------------------
The current csv_pyg_five_multi diagnostic loader requires non-empty
train/val/test partitions.  For inference only, each external sample is copied
three times into loader-only train/val/test partitions.  No optimization is
performed; only the TEST copy is forwarded through the model.

The checkpoint's original training component-vocabulary source is retained,
so external categorical vocabularies are not rebuilt from new_validation.

Mordred descriptor policy
-------------------------
The training lookup is not reused directly for external rows.  For every
distinct frozen Mordred scaler required by the selected checkpoints, this
script calculates descriptors for every valid external component SMILES and
standardizes them with that scaler.  This avoids treating a real, previously
unseen molecule as a missing descriptor (the loader's all-zero fallback).

This preserves the feature scale used by an existing checkpoint; it does not
retroactively make a historical checkpoint's scaler seed-specific.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROPERTY_COLUMNS = [
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
    "Norm_before",
    "Norm_after",
]

TARGET_INDEX = {
    "Norm_before": 4,
    "Norm_after": 5,
}

DEFAULT_MODELS = ["P0_random", "P1_PT_D", "P2_PT_DF"]
DEFAULT_SPLITS = [100, 101, 102]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_training_vocab_source(path_value: str, repo: Path) -> Path:
    """Resolve a provenance path after a completed run is copied to another checkout.

    Training artifacts retain their original absolute component-vocabulary
    location.  External inference must use the same CSV, but a copied result
    directory can legitimately refer to a different checkout root (for
    example ``/public/.../biology_prediction``).  Rebase only a missing path
    by its repository-relative suffix; never substitute an arbitrary file.
    """
    source = Path(path_value).expanduser()
    if source.is_file():
        return source.resolve()

    repo_markers = ("biology_prediction/", "biology_prediction\\")
    text = str(source)
    for marker in repo_markers:
        if marker not in text:
            continue
        relative = text.split(marker, 1)[1]
        candidate = repo / relative
        if candidate.is_file():
            print(
                "Rebased missing component vocabulary source from "
                f"{source} to {candidate}",
                flush=True,
            )
            return candidate.resolve()
    raise FileNotFoundError(
        "Frozen training component vocabulary source is missing and could not "
        f"be safely rebased within this checkout: {source}"
    )


def read_csv_robust(path: Path) -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype={"ID": str})
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError("Unable to decode CSV:\n" + "\n".join(errors))



def validate_external_labels(frame: pd.DataFrame,single_task:str) -> None:
    required = {
        "ID",
        "IL_SMILE",
        "HL_SMILE",
        "Chol_SMILE",
        "PEG_SMILE",
        "Fifth_SMILE",
        "mol%_IL",
        "mol%_HL",
        "mol%_Chol",
        "mol%_PEG",
        "mol%_Fifth",
        "Fifth_class",
        #"Norm_before",
        *PROPERTY_COLUMNS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "new_validation is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if frame["ID"].isna().any() or frame["ID"].duplicated().any():
        raise ValueError("new_validation ID must be non-null and unique.")
    y = pd.to_numeric(frame[single_task], errors="coerce")
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError(f"new_validation {single_task} contains non-finite labels.")


def build_loader_only_external(
    labels: pd.DataFrame,
    output_csv: Path,
    manifest_csv: Path,
) -> None:
    """Make 3 zero-label copies solely to satisfy the diagnostic loader."""
    copies = []
    manifest_rows = []

    for split in ("train", "val", "test"):
        part = labels.copy()
        original_ids = part["ID"].astype(str).tolist()
        part["ID"] = [
            f"__stage5_external_{split}__{index:04d}__{sample_id}"
            for index, sample_id in enumerate(original_ids)
        ]
        part["_external_original_id"] = original_ids
        part["_external_split"] = split

        # Prevent any true response from entering graph construction/model input.
        for column in PROPERTY_COLUMNS:
            part[column] = 0.0

        copies.append(part)

        manifest_rows.extend(
            {
                "sample_id": synthetic_id,
                "split": split,
                "split_order": index,
            }
            for index, synthetic_id in enumerate(part["ID"].astype(str))
        )

    loader_frame = pd.concat(copies, ignore_index=True)
    loader_frame.to_csv(output_csv, index=False)

    manifest = pd.DataFrame(manifest_rows)
    if len(manifest) != 3 * len(labels):
        raise RuntimeError("External manifest row-count invariant failed.")
    manifest.to_csv(manifest_csv, index=False)


def run_worker(
    script_path: Path,
    run_dir: Path,
    loader_csv: Path,
    manifest_csv: Path,
    worker_dir: Path,
    mordred_feature_path: Path | None,
    single_target: str,
) -> Path:
    worker_dir.mkdir(parents=True, exist_ok=True)
    output_csv = worker_dir / "predictions.csv"

    command = [
        sys.executable,
        "-u",
        str(script_path),
        "--worker",
        "--run-dir",
        str(run_dir),
        "--loader-csv",
        str(loader_csv),
        "--external-manifest",
        str(manifest_csv),
        "--worker-output",
        str(output_csv),
        "--worker-cache",
        str(worker_dir / "cache"),
        "--single-target",
        single_target,
    ]
    if mordred_feature_path is not None:
        command.extend([
            "--mordred-feature-path",
            str(mordred_feature_path),
        ])
    print("[worker]", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return output_csv


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, r2_score

    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)) if len(y_true) else math.nan,
        "r2": (
            float(r2_score(y_true, y_pred))
            if len(y_true) > 1 and np.std(y_true) > 0
            else math.nan
        ),
    }


def external_mordred_lookup(
    labels: pd.DataFrame,
    training_lookup: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, object]]:
    """Calculate external descriptors using a checkpoint's frozen scaler.

    ``mordred_feature_vector`` only performs a lookup and silently uses zeros
    for an absent key.  Existing Stage-5 checkpoints were trained with a
    standardized lookup, so the correct inference-compatible repair is to
    compute the raw descriptors for external SMILES and apply the *same*
    saved means/stds.  The shared O12 helper already implements that exact
    descriptor schema and canonical-SMILES convention.
    """
    training_lookup = training_lookup.resolve()
    if not training_lookup.is_file():
        raise FileNotFoundError(
            f"Checkpoint Mordred lookup is missing: {training_lookup}"
        )

    metadata_path = training_lookup.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Checkpoint Mordred scaler metadata is required to construct "
            f"external descriptors: {metadata_path}"
        )

    repo = Path(__file__).resolve().parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scripts.diagnostics.predict_o12_o22_feedback_ensemble import (
        build_feedback_mordred_lookup,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mordred11_external_standardized.csv"
    summary = build_feedback_mordred_lookup(
        labels,
        metadata_path,
        output_path,
    )

    lookup = pd.read_csv(output_path)
    feature_columns = [
        column for column in lookup.columns if column.startswith("feature_")
    ]
    if len(feature_columns) != 11 or lookup.empty:
        raise RuntimeError(
            f"External Mordred lookup is malformed: {output_path}"
        )
    if lookup["smiles"].duplicated().any():
        raise RuntimeError(
            f"External Mordred lookup has duplicate canonical SMILES: {output_path}"
        )
    if not np.isfinite(lookup[feature_columns].to_numpy(dtype=float)).all():
        raise RuntimeError(
            f"External Mordred lookup contains non-finite features: {output_path}"
        )

    provenance = {
        "training_lookup": str(training_lookup),
        "training_lookup_sha256": file_sha256(training_lookup),
        "scaler_metadata": str(metadata_path),
        "scaler_metadata_sha256": file_sha256(metadata_path),
        "external_lookup": str(output_path),
        "external_lookup_sha256": file_sha256(output_path),
        "external_lookup_rows": int(len(lookup)),
        "descriptor_summary": summary,
        "policy": (
            "Computed every valid external component SMILES and standardized "
            "it with the checkpoint's existing frozen Mordred scaler."
        ),
    }
    return output_path, provenance


def normalized_fifth_class(series: pd.Series) -> pd.Series:
    values = series.fillna("__unknown__").astype(str).str.strip().str.lower()
    mapping = {
        "single": "single",
        "double": "double",
    }
    return values.map(mapping).fillna(values)


def metrics_by_subset(
    frame: pd.DataFrame,
    prediction_column: str,
    model: str,
    ensemble_size: int,
) -> list[dict[str, float | int | str]]:
    if "Fifth_class" not in frame.columns:
        raise ValueError(
            "new_validation must contain Fifth_class to report single/double metrics."
        )

    classes = normalized_fifth_class(frame["Fifth_class"])
    y_all = frame["y_true"].to_numpy(dtype=float)
    p_all = frame[prediction_column].to_numpy(dtype=float)

    rows = []
    for subset, mask in (
        ("all", np.ones(len(frame), dtype=bool)),
        ("single", classes.eq("single").to_numpy()),
        ("double", classes.eq("double").to_numpy()),
    ):
        y = y_all[mask]
        p = p_all[mask]
        rows.append({
            "model": model,
            "subset": subset,
            "ensemble_size": int(ensemble_size),
            **calculate_metrics(y, p),
            "prediction_mean": float(np.mean(p)) if len(p) else math.nan,
            "prediction_std": float(np.std(p, ddof=0)) if len(p) else math.nan,
            "target_mean": float(np.mean(y)) if len(y) else math.nan,
            "target_std": float(np.std(y, ddof=0)) if len(y) else math.nan,
        })
    return rows


def make_plots(
    all_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    models: list[str],
    output_dir: Path,
    target: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "Fifth_class" not in all_predictions.columns:
        raise ValueError("Fifth_class is required for single/double plotting.")

    classes = normalized_fifth_class(all_predictions["Fifth_class"])
    masks = {
        "single": classes.eq("single").to_numpy(),
        "double": classes.eq("double").to_numpy(),
    }

    # Use identical x/y limits for every model, so visual comparison is fair.
    values = [all_predictions["y_true"].to_numpy(dtype=float)]
    for model in models:
        values.append(
            all_predictions[f"{model}_ensemble_mean"].to_numpy(dtype=float)
        )

    merged = np.concatenate(values)
    finite = merged[np.isfinite(merged)]

    lo = float(min(0.0, finite.min()))
    hi = float(finite.max())
    span = max(hi - lo, 1e-6)
    pad = 0.06 * span
    axis_lo, axis_hi = lo - pad, hi + pad

    metric_map = {
        (row["model"], row["subset"]): row
        for _, row in metrics.iterrows()
    }

    def metric_text(model: str) -> str:
        lines = []
        for subset, label in (
            ("all", "All"),
            ("single", "Single"),
            ("double", "Double"),
        ):
            m = metric_map[(model, subset)]
            lines.append(
                f"{label}: MAE={m['mae']:.4f}, $R^2$={m['r2']:.4f}"
            )
        return "\n".join(lines)

    def add_reference_lines(ax) -> None:
        """Add y=x and threshold lines x=1 / y=1."""
        ax.plot(
            [axis_lo, axis_hi],
            [axis_lo, axis_hi],
            linestyle="--",
            linewidth=1.3,
            label="y = x",
        )

        ax.axvline(
            x=1.0,
            linestyle=":",
            linewidth=1.2,
            label="x = 1",
        )
        ax.axhline(
            y=1.0,
            linestyle=":",
            linewidth=1.2,
            label="y = 1",
        )

    def add_metric_box(ax, text: str, fontsize: int = 10) -> None:
        """Place metric text consistently in the upper-right corner."""
        ax.text(
            0.97,
            0.97,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=fontsize,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.88,
            },
        )

    # ------------------------------------------------------------------
    # Individual model plots
    # ------------------------------------------------------------------
    for model in models:
        y = all_predictions["y_true"].to_numpy(dtype=float)
        p = all_predictions[
            f"{model}_ensemble_mean"
        ].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(6.8, 6.2))

        if masks["single"].any():
            ax.scatter(
                y[masks["single"]],
                p[masks["single"]],
                s=52,
                alpha=0.82,
                marker="o",
                label=f"single (n={int(masks['single'].sum())})",
            )

        if masks["double"].any():
            ax.scatter(
                y[masks["double"]],
                p[masks["double"]],
                s=58,
                alpha=0.82,
                marker="^",
                label=f"double (n={int(masks['double'].sum())})",
            )

        unknown = ~(masks["single"] | masks["double"])
        if unknown.any():
            ax.scatter(
                y[unknown],
                p[unknown],
                s=48,
                alpha=0.70,
                marker="x",
                label=f"other/unknown (n={int(unknown.sum())})",
            )

        # y=x, x=1, y=1
        add_reference_lines(ax)

        ax.set_xlim(axis_lo, axis_hi)
        ax.set_ylim(axis_lo, axis_hi)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(f"True {target}")
        ax.set_ylabel(
            f"Predicted {target} "
            f"(model ensemble mean)"
        )
        ax.set_title(f"{model} — new_validation")
        ax.grid(alpha=0.22)

        ax.legend(loc="upper left")

        # Metrics -> upper right
        add_metric_box(
            ax,
            metric_text(model),
        )

        fig.tight_layout()

        fig.savefig(
            output_dir / f"scatter_{model}.png",
            dpi=220,
        )
        fig.savefig(
            output_dir / f"scatter_{model}.pdf",
        )
        plt.close(fig)

        # --------------------------------------------------------------
        # Class-separated panels
        # --------------------------------------------------------------
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11.6, 5.5),
            squeeze=False,
        )
        axes = axes[0]

        for ax, subset in zip(
            axes,
            ("single", "double"),
        ):
            mask = masks[subset]

            ax.scatter(
                y[mask],
                p[mask],
                s=52,
                alpha=0.82,
            )

            # y=x, x=1, y=1
            add_reference_lines(ax)

            ax.set_xlim(axis_lo, axis_hi)
            ax.set_ylim(axis_lo, axis_hi)
            ax.set_aspect("equal", adjustable="box")

            ax.set_xlabel(f"True {target}")
            ax.set_title(f"{model} — {subset}")
            ax.grid(alpha=0.22)

            m = metric_map[(model, subset)]

            # Metrics -> upper right
            add_metric_box(
                ax,
                (
                    f"n = {int(m['n'])}\n"
                    f"MAE = {m['mae']:.4f}\n"
                    f"$R^2$ = {m['r2']:.4f}"
                ),
            )

        axes[0].set_ylabel(
            f"Predicted {target} (ensemble mean)"
        )

        fig.tight_layout()

        fig.savefig(
            output_dir / f"scatter_{model}_single_double.png",
            dpi=220,
            bbox_inches="tight",
        )
        fig.savefig(
            output_dir / f"scatter_{model}_single_double.pdf",
            bbox_inches="tight",
        )

        plt.close(fig)

    # ------------------------------------------------------------------
    # Multi-model comparison
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(6.2 * len(models), 5.8),
        squeeze=False,
    )
    axes = axes[0]

    for ax, model in zip(axes, models):
        y = all_predictions[
            "y_true"
        ].to_numpy(dtype=float)

        p = all_predictions[
            f"{model}_ensemble_mean"
        ].to_numpy(dtype=float)

        if masks["single"].any():
            ax.scatter(
                y[masks["single"]],
                p[masks["single"]],
                s=44,
                alpha=0.82,
                marker="o",
                label="single",
            )

        if masks["double"].any():
            ax.scatter(
                y[masks["double"]],
                p[masks["double"]],
                s=50,
                alpha=0.82,
                marker="^",
                label="double",
            )

        # y=x, x=1, y=1
        add_reference_lines(ax)

        ax.set_xlim(axis_lo, axis_hi)
        ax.set_ylim(axis_lo, axis_hi)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(model)
        ax.set_xlabel(f"True {target}")
        ax.grid(alpha=0.22)

        # Metrics -> upper right
        add_metric_box(
            ax,
            metric_text(model),
            fontsize=9,
        )

    axes[0].set_ylabel(
        f"Predicted {target} (ensemble mean)"
    )

    # Keep legend in upper-left so it does not overlap metric box.
    axes[-1].legend(loc="upper left")

    fig.suptitle(
        f"Stage-5 checkpoint ensembles on new_validation — {target}",
        y=1.01,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "scatter_P0_P1_P2_comparison.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "scatter_P0_P1_P2_comparison.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)



def main_controller(args) -> None:
    stage5_root = args.stage5_root.resolve()
    new_validation = args.new_validation.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = read_csv_robust(new_validation)
    validate_external_labels(labels,args.single_target)

    work_dir = output_dir / "_loader_work"
    if work_dir.exists() and not args.keep_work:
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    loader_csv = work_dir / "new_validation_ZERO_LABELS_triplicated.csv"
    manifest_csv = work_dir / "new_validation_triplicated_manifest.csv"
    build_loader_only_external(labels, loader_csv, manifest_csv)

    script_path = Path(__file__).resolve()
    mordred_lookups: dict[str, tuple[Path, dict[str, object]]] = {}
    mordred_provenance: list[dict[str, object]] = []

    long_rows = []
    target = args.single_target

    wide = pd.DataFrame({
        "ID": labels["ID"].astype(str),
        "y_true": pd.to_numeric(
        labels[target],
        errors="raise",
        ).astype(float),
    })
    if "Fifth_class" in labels.columns:
        wide["Fifth_class"] = normalized_fifth_class(labels["Fifth_class"])
    if "Fifth" in labels.columns:
        wide["Fifth"] = labels["Fifth"].astype(str)

    for model in args.models:
        model_predictions = []

        for split_seed in args.splits:
            run_dir = stage5_root / model / f"split{split_seed}"
            required = [
                run_dir / "checkpoints" / "selected_best.pt",
                run_dir / "effective_config.yaml",
                run_dir / "run_settings.json",
            ]
            missing = [path for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{model} split{split_seed} is incomplete: {missing}"
                )

            settings = json.loads(
                (run_dir / "run_settings.json").read_text(encoding="utf-8")
            )
            mordred_path = None
            if bool(settings.get("use_mordred_features", False)):
                training_lookup = Path(
                    str(settings.get("mordred_feature_path", ""))
                ).resolve()
                lookup_key = str(training_lookup)
                if lookup_key not in mordred_lookups:
                    # Keep generated descriptors alongside final predictions;
                    # ``_loader_work`` is intentionally deleted after a normal
                    # run, but inference provenance must remain reproducible.
                    lookup_dir = output_dir / "mordred_features" / (
                        file_sha256(training_lookup)[:16]
                    )
                    mordred_lookups[lookup_key] = external_mordred_lookup(
                        labels,
                        training_lookup,
                        lookup_dir,
                    )
                mordred_path, lookup_provenance = mordred_lookups[lookup_key]
                mordred_provenance.append({
                    "model": model,
                    "split_seed": int(split_seed),
                    **lookup_provenance,
                })

            worker_dir = work_dir / model / f"split{split_seed}"
            prediction_path = run_worker(
                script_path,
                run_dir,
                loader_csv,
                manifest_csv,
                worker_dir,
                mordred_path,
                args.single_target,
            )
            pred = pd.read_csv(prediction_path, dtype={"ID": str})

            expected_ids = labels["ID"].astype(str).tolist()
            if pred["ID"].astype(str).tolist() != expected_ids:
                raise RuntimeError(
                    f"{model} split{split_seed}: external prediction ID/order mismatch."
                )
            model_predictions.append(pred["y_pred"].to_numpy(dtype=float))

            long_rows.extend(
                {
                    "model": model,
                    "split_seed": int(split_seed),
                    "ID": sample_id,
                    "y_pred": float(value),
                }
                for sample_id, value in zip(expected_ids, pred["y_pred"])
            )

        matrix = np.vstack(model_predictions)
        if matrix.shape != (len(args.splits), len(labels)):
            raise RuntimeError(f"{model}: unexpected ensemble matrix shape {matrix.shape}")

        wide[f"{model}_ensemble_mean"] = matrix.mean(axis=0)
        wide[f"{model}_ensemble_std"] = matrix.std(axis=0, ddof=0)

        for row_index, split_seed in enumerate(args.splits):
            wide[f"{model}_split{split_seed}"] = matrix[row_index]

    long_frame = pd.DataFrame(long_rows)
    long_frame.to_csv(output_dir / "predictions_by_checkpoint_long.csv", index=False)
    wide.to_csv(output_dir / "ensemble_predictions_with_labels.csv", index=False)

    metric_rows = []
    for model in args.models:
        metric_rows.extend(
            metrics_by_subset(
                wide,
                f"{model}_ensemble_mean",
                model=model,
                ensemble_size=len(args.splits),
            )
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "ensemble_metrics.csv", index=False)

    make_plots(wide, metrics, args.models, output_dir,args.single_target)

    provenance = {
        "stage5_root": str(stage5_root),
        "new_validation": str(new_validation),
        "models": args.models,
        "split_seeds": args.splits,
        "target": args.single_target,
        "ensemble": "arithmetic mean of selected_best checkpoint predictions",
        "label_leakage_policy": (
            "All six property labels are zeroed in the loader-only "
            "external CSV; "
            f"true {args.single_target} is joined only after inference."
        ),
        "loader_workaround": (
            "Each external row is triplicated into non-empty train/val/test loader "
            "partitions; only TEST-copy predictions are retained. No training occurs."
        ),
        "mordred_external_feature_policy": (
            "For checkpoints using Mordred11, every valid external component "
            "SMILES is calculated and standardized with that checkpoint's existing "
            "frozen scaler; the training lookup itself is not used for external rows."
        ),
        "mordred_external_feature_provenance": mordred_provenance,
    }
    (output_dir / "inference_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("STAGE-5 NEW_VALIDATION ENSEMBLE")
    print("=" * 88)
    print(
        metrics[
            [
                "model",
                "subset",
                "ensemble_size",
                "n",
                "mae",
                "r2",
                "prediction_mean",
                "prediction_std",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Outputs: {output_dir}")
    print("  ensemble_predictions_with_labels.csv")
    print("  predictions_by_checkpoint_long.csv")
    print("  ensemble_metrics.csv")
    print("  scatter_P0_random.png/.pdf")
    print("  scatter_P1_PT_D.png/.pdf")
    print("  scatter_P2_PT_DF.png/.pdf")
    print("  scatter_<model>_single_double.png/.pdf")
    print("  scatter_P0_P1_P2_comparison.png/.pdf")

    if not args.keep_work:
        shutil.rmtree(work_dir)


def inverse_prediction(
    prediction,
    target_transform: str,
    target_scaler: dict | None,
):
    import torch

    value = prediction
    if target_scaler is not None and target_scaler.get("type") != "identity":
        mean = torch.as_tensor(
            target_scaler["mean"],
            dtype=value.dtype,
            device=value.device,
        )
        std = torch.as_tensor(
            target_scaler["std"],
            dtype=value.dtype,
            device=value.device,
        )
        if mean.numel() != 1:
            raise ValueError(
                "Stage-5 external worker expects a one-output checkpoint."
            )
        value = value * std[0] + mean[0]

    if target_transform == "identity":
        return value
    if target_transform == "log1p":
        return torch.expm1(value).clamp_min(0)
    raise ValueError(f"Unsupported target transform: {target_transform}")


def main_worker(args) -> None:
    # Imports are intentionally local: every worker gets a fresh GraphGym cfg.
    import torch

    ROOT = Path(__file__).resolve()
    # The script is expected under <repo>/scripts/pretrain/stage5/.
    repo = ROOT.parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import graphgps  # noqa: F401
    from graphgps.config.config_gps import set_cfg_gps
    from graphgps.create_model_gps import create_model_gps
    from graphgps.determinism import configure_determinism
    from loader_5 import create_loader_5
    from torch_geometric.graphgym.config import cfg, load_cfg

    run_dir = args.run_dir.resolve()
    checkpoint_path = run_dir / "checkpoints" / "selected_best.pt"
    config_path = run_dir / "effective_config.yaml"
    settings_path = run_dir / "run_settings.json"

    for path in (checkpoint_path, config_path, settings_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    settings = json.loads(settings_path.read_text(encoding="utf-8"))

    target = args.single_target

    if settings.get("single_target") != target:
        raise ValueError(
        f"Expected single_target={target}, "
        f"got {settings.get('single_target')!r}"
    )

    if int(settings.get("property_num", -1)) != 1:
        raise ValueError(
            f"Expected property_num=1, got {settings.get('property_num')!r}"
        )

    set_cfg_gps(cfg)

    # `effective_config.yaml` is a runtime snapshot written after the training
    # runner has added fields such as `run_dir` / `out_dir` (and potentially
    # other data-dependent keys).  A fresh GraphGym cfg does not necessarily
    # pre-register every one of those keys, so vanilla YACS merge would fail
    # with e.g.:
    #     KeyError: Non-existent config key: run_dir
    #
    # Temporarily allow new keys while reconstructing the exact training-time
    # effective configuration, then lock the schema again before inference.
    if not hasattr(cfg, "set_new_allowed"):
        raise AttributeError(
            "GraphGym cfg does not expose YACS set_new_allowed(); "
            "cannot safely reload the saved effective_config.yaml."
        )
    cfg.set_new_allowed(True)
    try:
        load_cfg(
            cfg,
            SimpleNamespace(cfg_file=str(config_path), opts=[]),
        )
    finally:
        cfg.set_new_allowed(False)

    loader_csv = args.loader_csv.resolve()
    external_manifest = args.external_manifest.resolve()
    worker_cache = args.worker_cache.resolve()

    cfg.read_csv = str(loader_csv)
    training_vocab_source = resolve_training_vocab_source(
        settings["component_vocab_source"], repo
    )
    cfg.component_vocab_source = str(training_vocab_source)


    target = args.single_target
    target_index = TARGET_INDEX[target]

    cfg.property_num = 1
    cfg.property_serial = target_index
    cfg.single_task_target_index = target_index

    cfg.dataset.diagnostic_split_path = str(external_manifest)
    cfg.dataset.diagnostic_id_column = "ID"
    cfg.dataset.diagnostic_manifest_id_column = "sample_id"
    cfg.dataset.dir = str(worker_cache)
    cfg.dataset.cache_tag = "stage5-new-validation-external-inference"
    cfg.dataset.cache_refresh = True

    if bool(settings.get("use_mordred_features", False)):
        if args.mordred_feature_path is None:
            raise ValueError(
                "Mordred-enabled Stage-5 checkpoint requires an externally "
                "constructed Mordred feature lookup."
            )
        mordred_feature_path = args.mordred_feature_path.resolve()
        if not mordred_feature_path.is_file():
            raise FileNotFoundError(mordred_feature_path)
        cfg.use_mordred_features = True
        cfg.mordred_feature_path = str(mordred_feature_path)
        cfg.mordred_feature_dim = int(settings["mordred_feature_dim"])
    elif args.mordred_feature_path is not None:
        raise ValueError(
            "Received an external Mordred lookup for a checkpoint that does not "
            "use Mordred features."
        )

    worker_cache.mkdir(parents=True, exist_ok=True)
    cfg.run_dir = str(worker_cache.parent)
    cfg.out_dir = str(worker_cache.parent)

    configure_determinism(int(cfg.seed), bool(cfg.train.deterministic))

    # Loader processing also materializes input-derived vocab sizes.
    loaders = create_loader_5()

    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model = create_model_gps().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state" not in checkpoint:
        raise KeyError(f"{checkpoint_path} has no model_state.")

    # Guard against accidentally loading a different task/architecture.
    if checkpoint.get("single_target") not in (None, target):
        raise ValueError(
            f"Checkpoint single_target="
            f"{checkpoint.get('single_target')!r}; "
            f"expected {target}"
        )
    if checkpoint.get("property_num") not in (None, 1):
        raise ValueError(
            f"Checkpoint property_num={checkpoint.get('property_num')!r}"
        )

    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    target_transform = checkpoint.get(
        "target_transform",
        settings.get("target_transform", "identity"),
    )
    target_scaler = checkpoint.get(
        "target_scaler",
        settings.get(
            "target_scaler",
            {"type": "identity", "mean": [0.0], "std": [1.0]},
        ),
    )

    def prepare_batches(items):
        for suffix, batch in zip(("", "_2", "_3", "_4", "_5"), items):
            batch.split = "test" + suffix
            batch.to(device)
        return items

    predictions = []
    with torch.no_grad():
        test_batches = zip(*[group[2] for group in loaders])
        for batches in test_batches:
            batches = prepare_batches(list(batches))
            output = model(*batches)
            if not isinstance(output, tuple) or len(output) not in (2, 3):
                raise TypeError(
                    "Unexpected model forward contract during external inference."
                )
            prediction = output[0]
            prediction = inverse_prediction(
                prediction,
                target_transform,
                target_scaler,
            )
            predictions.extend(
                prediction.detach().cpu().reshape(-1).numpy().astype(float).tolist()
            )

    loader_source = pd.read_csv(loader_csv, dtype={"ID": str})
    test_source = loader_source.loc[
        loader_source["_external_split"].eq("test")
    ].copy()

    if len(predictions) != len(test_source):
        raise RuntimeError(
            f"Prediction count {len(predictions)} != test-copy rows {len(test_source)}"
        )

    result = pd.DataFrame({
        "ID": test_source["_external_original_id"].astype(str).tolist(),
        "y_pred": predictions,
    })
    result.to_csv(args.worker_output.resolve(), index=False)

    print(
        f"[worker complete] {run_dir.parent.name}/{run_dir.name}: "
        f"{len(result)} external predictions",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--stage5-root",
        type=Path,
        default=Path(
            "results/fifth_pretraining/stage5_downstream_transfer"
        ),
    )
    parser.add_argument(
        "--new-validation",
        type=Path,
        default=Path("datasets_lrx/raw/feedback/new_validation.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/fifth_pretraining/stage5_downstream_transfer/"
            "new_validation_3seed_ensemble"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        default=DEFAULT_SPLITS,
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Retain temporary zero-label loader CSVs and per-checkpoint caches.",
    )

    # Internal worker mode.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--loader-csv", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--external-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-cache", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--mordred-feature-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
    "--single-target",
    choices=("Norm_before", "Norm_after"),
    default="Norm_before",
    help="Single Norm target predicted by the checkpoints.",
)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        required = [
            args.run_dir,
            args.loader_csv,
            args.external_manifest,
            args.worker_output,
            args.worker_cache,
        ]
        if any(value is None for value in required):
            raise ValueError("Internal worker arguments are incomplete.")
        main_worker(args)
    else:
        main_controller(args)


if __name__ == "__main__":
    main()
