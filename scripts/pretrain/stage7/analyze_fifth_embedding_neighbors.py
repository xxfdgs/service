#!/usr/bin/env python3
"""
Stage 7 — diagnose Fifth-component embedding geometry and nearest neighbors.

Goal
----
Determine whether the pretrained / downstream Comp5GraphEncoder embeddings
place new_validation Fifth structures near training Fifth identities with
compatible Norm_before behavior.

The analysis is STRUCTURE-BRANCH-ONLY:
    Fifth_SMILE
        -> exact graph_feature.smiles2graph pipeline
        -> exact local Comp5GraphEncoder
        -> configured graph pooling
        -> graph embedding

It intentionally does NOT use:
    components 1-4,
    component ratios,
    Fifth_class embedding,
    Mordred features,
    component auxiliary features,
    downstream fusion MLP/head.

Therefore any association between embedding neighbors and Norm_before is a
diagnostic of the Fifth structural representation, not a substitute for the
full property model.

Training-reference labels
-------------------------
The 700-row training set may contain repeated Fifth identities under different
formulations. Because the Comp5 embedding depends only on Fifth structure,
reference rows are first aggregated to one MODEL-VISIBLE Fifth connectivity.
For every training Fifth identity the script records:
    Norm_before mean / median / std / min / max
    fraction with Norm_before > 1
    row count
This avoids allowing repeated formulations of one Fifth structure to dominate
nearest-neighbor ranks.

Default representations
-----------------------
Stage 4:
    Stage4_PT_D
    Stage4_PT_DF

Downstream selected-best Comp5 encoders, split100..109:
    P0_random
    P1_PT_D
    P2_PT_DF

Optional Stage-6 representation families can be added with repeated:
    --extra-family LABEL=PATH_TEMPLATE
where PATH_TEMPLATE may contain "{split}", e.g.
    --extra-family P1_PT_D_diffLR1e4='results/.../P1_PT_D_diffLR1e4/split{split}/checkpoints/selected_best.pt'

Outputs
-------
inventory/
    training_fifth_identity_reference.csv
    new_validation_queries.csv
    structure_inventory.csv

embeddings/
    <representation>.csv

nearest_neighbors_cosine.csv
query_embedding_diagnostics.csv
representation_summary.csv
family_summary.csv
top1_neighbor_consensus.csv
anchor_double_neighbors.csv

pca/
    Stage4_PT_D_pca.png
    Stage4_PT_DF_pca.png

worker/
    exact_graph_cache.pt
    graph_cache_metadata.json

Important interpretation
------------------------
A nearest-neighbor Norm_before mismatch does NOT prove the full model is wrong:
the full model also sees formulation/context features. This analysis asks the
narrower mechanistic question:
    "Does the Fifth structural embedding itself organize molecules in a way
     that is compatible with downstream Norm_before?"
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


PROPERTY = "Norm_before"
THRESHOLD = 1.0
DEFAULT_SPLITS = list(range(100, 110))
DEFAULT_ANCHORS = ["13-F1", "13-F3", "13-F10", "13-F5"]


# =============================================================================
# Generic controller utilities
# =============================================================================

def read_csv_robust(path: Path) -> pd.DataFrame:
    failures = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype={"ID": str})
        except UnicodeDecodeError as exc:
            failures.append(f"{encoding}: {exc}")
    raise UnicodeError(
        f"Unable to decode {path}:\n" + "\n".join(failures)
    )


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def canonical_connectivity(smiles: object) -> str | None:
    """Canonical non-isomeric SMILES: current model-visible stereo-free identity."""
    from rdkit import Chem

    text = clean_text(smiles)
    if text in {"", "[Fr]", "0", "0.0"}:
        return None

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"RDKit failed on Fifth_SMILE={text!r}")

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )


def first_nonempty(series: pd.Series) -> str:
    for value in series:
        text = clean_text(value)
        if text:
            return text
    return ""


def joined_unique(series: pd.Series, max_items: int | None = None) -> str:
    values = []
    seen = set()
    for value in series:
        text = clean_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        values.append(text)
    if max_items is not None and len(values) > max_items:
        return "|".join(values[:max_items]) + f"|...(+{len(values)-max_items})"
    return "|".join(values)


def normalize_class(value: object) -> str:
    text = clean_text(value).lower()
    if text in {"single", "double"}:
        return text
    return text if text else "unknown"


def build_inventories(
    train: pd.DataFrame,
    external: pd.DataFrame,
    inventory_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_train = {"ID", "Fifth_SMILE", PROPERTY}
    required_external = {"ID", "Fifth_SMILE", PROPERTY, "Fifth_class"}

    missing = required_train.difference(train.columns)
    if missing:
        raise ValueError(f"Training CSV missing columns: {sorted(missing)}")
    missing = required_external.difference(external.columns)
    if missing:
        raise ValueError(f"new_validation missing columns: {sorted(missing)}")

    if train["ID"].isna().any() or train["ID"].duplicated().any():
        raise ValueError("Training ID must be complete and unique.")
    if external["ID"].isna().any() or external["ID"].duplicated().any():
        raise ValueError("new_validation ID must be complete and unique.")

    train = train.copy()
    external = external.copy()

    train["canonical_connectivity"] = train["Fifth_SMILE"].map(
        canonical_connectivity
    )
    external["canonical_connectivity"] = external["Fifth_SMILE"].map(
        canonical_connectivity
    )

    train[PROPERTY] = pd.to_numeric(train[PROPERTY], errors="coerce")
    external[PROPERTY] = pd.to_numeric(external[PROPERTY], errors="coerce")

    # Only non-empty Fifth structures can enter Comp5 structural diagnosis.
    train_struct = train.loc[
        train["canonical_connectivity"].notna()
        & np.isfinite(train[PROPERTY].to_numpy(dtype=float))
    ].copy()

    query = external.loc[
        external["canonical_connectivity"].notna()
        & np.isfinite(external[PROPERTY].to_numpy(dtype=float))
    ].copy()

    if train_struct.empty:
        raise ValueError("No usable structural Fifth references in training CSV.")
    if query.empty:
        raise ValueError("No usable Fifth structures in new_validation.")

    fifth_name_col_train = "Fifth" if "Fifth" in train_struct.columns else None
    fifth_name_col_query = "Fifth" if "Fifth" in query.columns else None
    class_col_train = "Fifth_class" if "Fifth_class" in train_struct.columns else None

    ref_rows = []
    for index, (connectivity, group) in enumerate(
        train_struct.groupby("canonical_connectivity", sort=True),
        start=1,
    ):
        y = group[PROPERTY].to_numpy(dtype=float)
        classes = (
            joined_unique(group[class_col_train].map(normalize_class))
            if class_col_train is not None
            else ""
        )

        ref_rows.append({
            "reference_id": f"TRF_{index:04d}",
            "canonical_connectivity": connectivity,
            "representative_smiles": first_nonempty(group["Fifth_SMILE"]),
            "Fifth": (
                first_nonempty(group[fifth_name_col_train])
                if fifth_name_col_train is not None else ""
            ),
            "training_classes": classes,
            "training_row_count": int(len(group)),
            "training_ids": joined_unique(group["ID"].astype(str), max_items=20),
            "norm_before_mean": float(np.mean(y)),
            "norm_before_median": float(np.median(y)),
            "norm_before_std": float(np.std(y, ddof=0)),
            "norm_before_min": float(np.min(y)),
            "norm_before_max": float(np.max(y)),
            "norm_before_high_fraction_gt1": float(np.mean(y > THRESHOLD)),
        })

    references = pd.DataFrame(ref_rows)

    query_rows = []
    for _, row in query.iterrows():
        query_rows.append({
            "query_id": str(row["ID"]),
            "canonical_connectivity": row["canonical_connectivity"],
            "representative_smiles": clean_text(row["Fifth_SMILE"]),
            "Fifth": (
                clean_text(row[fifth_name_col_query])
                if fifth_name_col_query is not None else ""
            ),
            "Fifth_class": normalize_class(row["Fifth_class"]),
            "true_norm_before": float(row[PROPERTY]),
            "true_high_gt1": bool(float(row[PROPERTY]) > THRESHOLD),
        })
    queries = pd.DataFrame(query_rows)

    # One structure ID per model-visible connectivity across train+external.
    structure_by_conn: dict[str, dict[str, Any]] = {}
    for _, row in references.iterrows():
        structure_by_conn.setdefault(
            row["canonical_connectivity"],
            {
                "canonical_connectivity": row["canonical_connectivity"],
                "representative_smiles": row["representative_smiles"],
                "seen_in_training": True,
                "seen_in_new_validation": False,
            },
        )
        structure_by_conn[row["canonical_connectivity"]]["seen_in_training"] = True

    for _, row in queries.iterrows():
        if row["canonical_connectivity"] not in structure_by_conn:
            structure_by_conn[row["canonical_connectivity"]] = {
                "canonical_connectivity": row["canonical_connectivity"],
                "representative_smiles": row["representative_smiles"],
                "seen_in_training": False,
                "seen_in_new_validation": True,
            }
        else:
            structure_by_conn[row["canonical_connectivity"]][
                "seen_in_new_validation"
            ] = True

    structures = pd.DataFrame(
        [
            {"structure_id": f"STR_{i:04d}", **record}
            for i, record in enumerate(
                sorted(
                    structure_by_conn.values(),
                    key=lambda x: x["canonical_connectivity"],
                ),
                start=1,
            )
        ]
    )

    conn_to_sid = structures.set_index("canonical_connectivity")[
        "structure_id"
    ].to_dict()

    references["structure_id"] = references["canonical_connectivity"].map(
        conn_to_sid
    )
    queries["structure_id"] = queries["canonical_connectivity"].map(
        conn_to_sid
    )
    queries["exact_training_connectivity"] = queries[
        "canonical_connectivity"
    ].isin(set(references["canonical_connectivity"]))

    inventory_dir.mkdir(parents=True, exist_ok=True)
    references.to_csv(
        inventory_dir / "training_fifth_identity_reference.csv",
        index=False,
    )
    queries.to_csv(
        inventory_dir / "new_validation_queries.csv",
        index=False,
    )
    structures.to_csv(
        inventory_dir / "structure_inventory.csv",
        index=False,
    )

    print(
        "[inventory] "
        f"training rows usable={len(train_struct)}, "
        f"training Fifth identities={len(references)}, "
        f"new_validation queries={len(queries)}, "
        f"unique structures total={len(structures)}"
    )
    print(
        "[inventory] exact model-visible train/external overlap: "
        f"{int(queries['exact_training_connectivity'].sum())}/{len(queries)}"
    )

    return references, queries, structures


# =============================================================================
# Representation specifications
# =============================================================================

def safe_rep_name(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def make_representation_specs(args) -> list[dict[str, Any]]:
    stage4_root = args.stage4_root.resolve()
    stage5_root = args.stage5_root.resolve()

    specs = [
        {
            "representation": "Stage4_PT_D",
            "family": "Stage4_PT_D",
            "split_seed": None,
            "checkpoint": (
                stage4_root
                / "PT_D"
                / "checkpoints"
                / "best_comp5_encoder_state_dict.pt"
            ),
        },
        {
            "representation": "Stage4_PT_DF",
            "family": "Stage4_PT_DF",
            "split_seed": None,
            "checkpoint": (
                stage4_root
                / "PT_DF"
                / "checkpoints"
                / "best_comp5_encoder_state_dict.pt"
            ),
        },
    ]

    for family in args.stage5_models:
        for split_seed in args.splits:
            specs.append({
                "representation": f"{family}_split{split_seed}",
                "family": family,
                "split_seed": int(split_seed),
                "checkpoint": (
                    stage5_root
                    / family
                    / f"split{split_seed}"
                    / "checkpoints"
                    / "selected_best.pt"
                ),
            })

    for item in args.extra_family:
        if "=" not in item:
            raise ValueError(
                "--extra-family must be LABEL=PATH_TEMPLATE; "
                f"got {item!r}"
            )
        family, template = item.split("=", 1)
        family = family.strip()
        template = template.strip()
        if not family or not template:
            raise ValueError(f"Invalid --extra-family {item!r}")

        for split_seed in args.splits:
            checkpoint = Path(
                template.format(split=split_seed)
            )
            specs.append({
                "representation": f"{family}_split{split_seed}",
                "family": family,
                "split_seed": int(split_seed),
                "checkpoint": checkpoint,
            })

    seen = set()
    unique = []
    for spec in specs:
        name = spec["representation"]
        if name in seen:
            raise ValueError(f"Duplicate representation name: {name}")
        seen.add(name)
        checkpoint = Path(spec["checkpoint"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"{name}: missing checkpoint {checkpoint}"
            )
        spec = dict(spec)
        spec["checkpoint"] = checkpoint
        unique.append(spec)

    return unique


# =============================================================================
# Worker invocation
# =============================================================================

def run_subprocess(command: list[str]) -> None:
    print("[worker]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def prepare_graph_cache(
    script_path: Path,
    config: Path,
    structures_csv: Path,
    cache_path: Path,
    metadata_path: Path,
) -> None:
    run_subprocess([
        sys.executable,
        "-u",
        str(script_path),
        "--worker-mode",
        "prepare",
        "--reference-config",
        str(config),
        "--structures-csv",
        str(structures_csv),
        "--graph-cache",
        str(cache_path),
        "--graph-cache-metadata",
        str(metadata_path),
    ])


def extract_representation(
    script_path: Path,
    config: Path,
    graph_cache: Path,
    graph_metadata: Path,
    checkpoint: Path,
    output_csv: Path,
    batch_size: int,
) -> None:
    run_subprocess([
        sys.executable,
        "-u",
        str(script_path),
        "--worker-mode",
        "embed",
        "--reference-config",
        str(config),
        "--graph-cache",
        str(graph_cache),
        "--graph-cache-metadata",
        str(graph_metadata),
        "--checkpoint",
        str(checkpoint),
        "--worker-output",
        str(output_csv),
        "--embedding-batch-size",
        str(batch_size),
    ])


# =============================================================================
# Nearest-neighbor analysis
# =============================================================================

def embedding_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("emb_")]
    if not columns:
        raise ValueError("Embedding file has no emb_* columns.")
    return columns


def cosine_distance_matrix(
    query: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    qnorm = np.linalg.norm(query, axis=1, keepdims=True)
    rnorm = np.linalg.norm(reference, axis=1, keepdims=True)

    if np.any(qnorm <= 0) or np.any(rnorm <= 0):
        raise ValueError("Zero-norm embedding encountered.")

    q = query / qnorm
    r = reference / rnorm
    return 1.0 - q @ r.T


def euclidean_distance_matrix(
    query: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    q2 = np.sum(query * query, axis=1, keepdims=True)
    r2 = np.sum(reference * reference, axis=1, keepdims=True).T
    d2 = np.maximum(q2 + r2 - 2.0 * query @ reference.T, 0.0)
    return np.sqrt(d2)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    from scipy.stats import spearmanr

    a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return math.nan
    return float(spearmanr(a[mask], b[mask]).statistic)


def safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    from sklearn.metrics import roc_auc_score

    y = y_true.astype(bool).to_numpy()
    s = pd.to_numeric(score, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return math.nan
    return float(roc_auc_score(y.astype(int), s))


def analyze_one_representation(
    spec: dict[str, Any],
    embedding_path: Path,
    references: pd.DataFrame,
    queries: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    emb = pd.read_csv(embedding_path)
    if emb["structure_id"].duplicated().any():
        raise ValueError(f"{embedding_path}: duplicate structure_id")

    cols = embedding_columns(emb)
    emb_by_id = emb.set_index("structure_id")

    missing_ref = set(references["structure_id"]) - set(emb_by_id.index)
    missing_query = set(queries["structure_id"]) - set(emb_by_id.index)
    if missing_ref or missing_query:
        raise ValueError(
            f"{spec['representation']}: embedding IDs missing "
            f"references={sorted(missing_ref)[:10]}, "
            f"queries={sorted(missing_query)[:10]}"
        )

    ref_matrix = emb_by_id.loc[
        references["structure_id"], cols
    ].to_numpy(dtype=float)
    query_matrix = emb_by_id.loc[
        queries["structure_id"], cols
    ].to_numpy(dtype=float)

    cosine = cosine_distance_matrix(query_matrix, ref_matrix)
    euclidean = euclidean_distance_matrix(query_matrix, ref_matrix)

    k_eff = min(k, len(references))
    neighbor_rows = []
    diagnostic_rows = []

    for qi, query in queries.reset_index(drop=True).iterrows():
        order = np.argsort(cosine[qi], kind="stable")[:k_eff]

        neighbors = references.iloc[order].reset_index(drop=True)
        cos_values = cosine[qi, order]
        eu_values = euclidean[qi, order]

        for rank, (ref_idx, neighbor) in enumerate(
            zip(order, neighbors.to_dict("records")),
            start=1,
        ):
            neighbor_rows.append({
                "representation": spec["representation"],
                "family": spec["family"],
                "split_seed": spec["split_seed"],
                "query_id": query["query_id"],
                "query_fifth": query["Fifth"],
                "query_class": query["Fifth_class"],
                "query_true_norm_before": query["true_norm_before"],
                "query_true_high_gt1": query["true_high_gt1"],
                "query_exact_training_connectivity": query[
                    "exact_training_connectivity"
                ],
                "rank": int(rank),
                "cosine_distance": float(cosine[qi, ref_idx]),
                "euclidean_distance": float(euclidean[qi, ref_idx]),
                "neighbor_reference_id": neighbor["reference_id"],
                "neighbor_fifth": neighbor["Fifth"],
                "neighbor_training_classes": neighbor["training_classes"],
                "neighbor_training_row_count": neighbor["training_row_count"],
                "neighbor_norm_before_mean": neighbor["norm_before_mean"],
                "neighbor_norm_before_median": neighbor["norm_before_median"],
                "neighbor_norm_before_std": neighbor["norm_before_std"],
                "neighbor_norm_before_min": neighbor["norm_before_min"],
                "neighbor_norm_before_max": neighbor["norm_before_max"],
                "neighbor_high_fraction_gt1": neighbor[
                    "norm_before_high_fraction_gt1"
                ],
                "neighbor_connectivity": neighbor["canonical_connectivity"],
            })

        norm_means = neighbors["norm_before_mean"].to_numpy(dtype=float)
        high_fractions = neighbors[
            "norm_before_high_fraction_gt1"
        ].to_numpy(dtype=float)

        # Inverse-distance score is diagnostic only. Add a small floor so exact
        # structural matches dominate without numerical overflow.
        weights = 1.0 / np.maximum(cos_values, 1e-6)
        weights = weights / weights.sum()

        diagnostic_rows.append({
            "representation": spec["representation"],
            "family": spec["family"],
            "split_seed": spec["split_seed"],
            "query_id": query["query_id"],
            "query_fifth": query["Fifth"],
            "query_class": query["Fifth_class"],
            "true_norm_before": query["true_norm_before"],
            "true_high_gt1": query["true_high_gt1"],
            "exact_training_connectivity": query[
                "exact_training_connectivity"
            ],
            "top1_cosine_distance": float(cos_values[0]),
            "top1_euclidean_distance": float(eu_values[0]),
            "top1_neighbor_reference_id": neighbors.iloc[0][
                "reference_id"
            ],
            "top1_neighbor_fifth": neighbors.iloc[0]["Fifth"],
            "top1_neighbor_training_classes": neighbors.iloc[0][
                "training_classes"
            ],
            "top1_neighbor_norm_mean": float(norm_means[0]),
            "top1_neighbor_high_fraction_gt1": float(high_fractions[0]),
            f"knn{k_eff}_norm_mean": float(np.mean(norm_means)),
            f"knn{k_eff}_norm_median": float(np.median(norm_means)),
            f"knn{k_eff}_norm_inverse_cosine_weighted": float(
                np.sum(weights * norm_means)
            ),
            f"knn{k_eff}_high_fraction_mean": float(
                np.mean(high_fractions)
            ),
            f"knn{k_eff}_neighbors_mean_gt1_count": int(
                np.sum(norm_means > THRESHOLD)
            ),
            f"knn{k_eff}_same_class_count": int(
                sum(
                    query["Fifth_class"]
                    in str(value).lower().split("|")
                    for value in neighbors["training_classes"]
                )
            ),
        })

    neighbors_df = pd.DataFrame(neighbor_rows)
    diagnostics_df = pd.DataFrame(diagnostic_rows)

    double = diagnostics_df.loc[
        diagnostics_df["query_class"].eq("double")
    ].copy()
    single = diagnostics_df.loc[
        diagnostics_df["query_class"].eq("single")
    ].copy()

    score_col = f"knn{k_eff}_norm_mean"
    weighted_col = f"knn{k_eff}_norm_inverse_cosine_weighted"

    summary = {
        "representation": spec["representation"],
        "family": spec["family"],
        "split_seed": spec["split_seed"],
        "embedding_dim": len(cols),
        "k": int(k_eff),
        "n_queries": int(len(diagnostics_df)),
        "n_single": int(len(single)),
        "n_double": int(len(double)),
        "mean_top1_cosine_distance_all": float(
            diagnostics_df["top1_cosine_distance"].mean()
        ),
        "mean_top1_cosine_distance_single": (
            float(single["top1_cosine_distance"].mean())
            if len(single) else math.nan
        ),
        "mean_top1_cosine_distance_double": (
            float(double["top1_cosine_distance"].mean())
            if len(double) else math.nan
        ),
        "double_spearman_true_vs_top1_neighbor_norm": safe_spearman(
            double["true_norm_before"],
            double["top1_neighbor_norm_mean"],
        ) if len(double) else math.nan,
        f"double_spearman_true_vs_knn{k_eff}_norm_mean": safe_spearman(
            double["true_norm_before"],
            double[score_col],
        ) if len(double) else math.nan,
        f"double_spearman_true_vs_knn{k_eff}_weighted_norm": safe_spearman(
            double["true_norm_before"],
            double[weighted_col],
        ) if len(double) else math.nan,
        f"double_auc_high_gt1_from_knn{k_eff}_norm_mean": safe_auc(
            double["true_high_gt1"],
            double[score_col],
        ) if len(double) else math.nan,
        "double_top1_neighbor_norm_abs_error_mean": (
            float(
                np.mean(
                    np.abs(
                        double["true_norm_before"].to_numpy(dtype=float)
                        - double[
                            "top1_neighbor_norm_mean"
                        ].to_numpy(dtype=float)
                    )
                )
            )
            if len(double) else math.nan
        ),
        f"double_knn{k_eff}_norm_abs_error_mean": (
            float(
                np.mean(
                    np.abs(
                        double["true_norm_before"].to_numpy(dtype=float)
                        - double[score_col].to_numpy(dtype=float)
                    )
                )
            )
            if len(double) else math.nan
        ),
    }

    return neighbors_df, diagnostics_df, summary


def family_summary(representation_summary: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        column
        for column in representation_summary.columns
        if column not in {
            "representation",
            "family",
            "split_seed",
        }
        and pd.api.types.is_numeric_dtype(representation_summary[column])
    ]

    rows = []
    for family, group in representation_summary.groupby("family", sort=False):
        row: dict[str, Any] = {
            "family": family,
            "representation_count": int(len(group)),
            "split_seed_count": int(group["split_seed"].notna().sum()),
        }
        for column in numeric_cols:
            values = pd.to_numeric(group[column], errors="coerce")
            finite = values[np.isfinite(values)]
            row[f"{column}_mean"] = (
                float(finite.mean()) if len(finite) else math.nan
            )
            row[f"{column}_std"] = (
                float(finite.std(ddof=1))
                if len(finite) > 1 else math.nan
            )
            row[f"{column}_median"] = (
                float(finite.median()) if len(finite) else math.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def build_top1_consensus(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    # Consensus is meaningful for families represented by multiple downstream
    # split-specific encoders.
    for (family, query_id), group in diagnostics.groupby(
        ["family", "query_id"],
        sort=False,
    ):
        total = len(group)
        counts = (
            group.groupby(
                [
                    "top1_neighbor_reference_id",
                    "top1_neighbor_fifth",
                ],
                dropna=False,
            )
            .agg(
                top1_count=("representation", "size"),
                mean_top1_cosine_distance=(
                    "top1_cosine_distance",
                    "mean",
                ),
                mean_top1_neighbor_norm=(
                    "top1_neighbor_norm_mean",
                    "mean",
                ),
            )
            .reset_index()
            .sort_values(
                ["top1_count", "mean_top1_cosine_distance"],
                ascending=[False, True],
            )
        )

        query_meta = group.iloc[0]
        for rank, (_, row) in enumerate(counts.iterrows(), start=1):
            rows.append({
                "family": family,
                "query_id": query_id,
                "query_fifth": query_meta["query_fifth"],
                "query_class": query_meta["query_class"],
                "true_norm_before": query_meta["true_norm_before"],
                "consensus_rank": int(rank),
                "top1_neighbor_reference_id": row[
                    "top1_neighbor_reference_id"
                ],
                "top1_neighbor_fifth": row["top1_neighbor_fifth"],
                "top1_count": int(row["top1_count"]),
                "representation_count": int(total),
                "top1_fraction": float(row["top1_count"] / total),
                "mean_top1_cosine_distance": float(
                    row["mean_top1_cosine_distance"]
                ),
                "mean_top1_neighbor_norm": float(
                    row["mean_top1_neighbor_norm"]
                ),
            })

    return pd.DataFrame(rows)


# =============================================================================
# PCA diagnostic plots
# =============================================================================

def make_pca_plot(
    representation: str,
    embedding_path: Path,
    references: pd.DataFrame,
    queries: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    emb = pd.read_csv(embedding_path).set_index("structure_id")
    cols = [column for column in emb.columns if column.startswith("emb_")]

    train_matrix = emb.loc[
        references["structure_id"], cols
    ].to_numpy(dtype=float)
    query_matrix = emb.loc[
        queries["structure_id"], cols
    ].to_numpy(dtype=float)

    if len(references) < 3:
        return

    pca = PCA(n_components=2)
    train_xy = pca.fit_transform(train_matrix)
    query_xy = pca.transform(query_matrix)

    fig, ax = plt.subplots(figsize=(8.2, 6.7))

    scatter = ax.scatter(
        train_xy[:, 0],
        train_xy[:, 1],
        c=references["norm_before_mean"].to_numpy(dtype=float),
        s=36,
        alpha=0.72,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Training Fifth mean Norm_before")

    single_mask = queries["Fifth_class"].eq("single").to_numpy()
    double_mask = queries["Fifth_class"].eq("double").to_numpy()

    if single_mask.any():
        ax.scatter(
            query_xy[single_mask, 0],
            query_xy[single_mask, 1],
            marker="o",
            s=80,
            facecolors="none",
            edgecolors="black",
            label="new_validation single",
        )

    if double_mask.any():
        ax.scatter(
            query_xy[double_mask, 0],
            query_xy[double_mask, 1],
            marker="X",
            s=110,
            label="new_validation double",
        )
        for xy, query_id in zip(
            query_xy[double_mask],
            queries.loc[double_mask, "query_id"],
        ):
            ax.annotate(
                str(query_id),
                (xy[0], xy[1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% train variance)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% train variance)")
    ax.set_title(f"{representation}: Fifth structural embedding PCA")
    ax.grid(alpha=0.20)
    ax.legend(loc="best")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{safe_rep_name(representation)}_pca.png", dpi=220)
    fig.savefig(output_dir / f"{safe_rep_name(representation)}_pca.pdf")
    plt.close(fig)


# =============================================================================
# Controller
# =============================================================================

def main_controller(args) -> None:
    output_dir = args.output_dir.resolve()
    inventory_dir = output_dir / "inventory"
    embedding_dir = output_dir / "embeddings"
    worker_dir = output_dir / "worker"
    pca_dir = output_dir / "pca"

    if output_dir.exists() and args.refresh:
        shutil.rmtree(output_dir)

    for path in (inventory_dir, embedding_dir, worker_dir, pca_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_csv = args.training_csv.resolve()
    new_validation = args.new_validation.resolve()
    reference_config = args.reference_config.resolve()

    for path in (train_csv, new_validation, reference_config):
        if not path.is_file():
            raise FileNotFoundError(path)

    train = read_csv_robust(train_csv)
    external = read_csv_robust(new_validation)

    references, queries, structures = build_inventories(
        train,
        external,
        inventory_dir,
    )

    script_path = Path(__file__).resolve()
    graph_cache = worker_dir / "exact_graph_cache.pt"
    graph_metadata = worker_dir / "graph_cache_metadata.json"
    structures_csv = inventory_dir / "structure_inventory.csv"

    if not graph_cache.is_file() or not graph_metadata.is_file():
        prepare_graph_cache(
            script_path,
            reference_config,
            structures_csv,
            graph_cache,
            graph_metadata,
        )

    specs = make_representation_specs(args)
    print(f"[representations] total={len(specs)}")

    neighbor_frames = []
    diagnostic_frames = []
    summary_rows = []
    embedding_paths: dict[str, Path] = {}

    for index, spec in enumerate(specs, start=1):
        rep = spec["representation"]
        output_path = embedding_dir / f"{safe_rep_name(rep)}.csv"
        embedding_paths[rep] = output_path

        print(
            f"\n[{index}/{len(specs)}] extracting {rep} "
            f"from {spec['checkpoint']}"
        )

        if not output_path.is_file():
            extract_representation(
                script_path,
                reference_config,
                graph_cache,
                graph_metadata,
                spec["checkpoint"],
                output_path,
                args.embedding_batch_size,
            )

        neighbors, diagnostics, summary = analyze_one_representation(
            spec,
            output_path,
            references,
            queries,
            args.k,
        )
        neighbor_frames.append(neighbors)
        diagnostic_frames.append(diagnostics)
        summary_rows.append(summary)

    all_neighbors = pd.concat(neighbor_frames, ignore_index=True)
    all_diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    family_df = family_summary(summary_df)
    consensus_df = build_top1_consensus(all_diagnostics)

    all_neighbors.to_csv(
        output_dir / "nearest_neighbors_cosine.csv",
        index=False,
    )
    all_diagnostics.to_csv(
        output_dir / "query_embedding_diagnostics.csv",
        index=False,
    )
    summary_df.to_csv(
        output_dir / "representation_summary.csv",
        index=False,
    )
    family_df.to_csv(
        output_dir / "family_summary.csv",
        index=False,
    )
    consensus_df.to_csv(
        output_dir / "top1_neighbor_consensus.csv",
        index=False,
    )

    anchors = set(args.anchors)
    anchor_df = all_neighbors.loc[
        all_neighbors["query_id"].astype(str).isin(anchors)
        & all_neighbors["query_class"].eq("double")
    ].copy()
    anchor_df.to_csv(
        output_dir / "anchor_double_neighbors.csv",
        index=False,
    )

    # PCA only for the two clean pretrained structural spaces by default.
    for rep in ("Stage4_PT_D", "Stage4_PT_DF"):
        if rep in embedding_paths:
            make_pca_plot(
                rep,
                embedding_paths[rep],
                references,
                queries,
                pca_dir,
            )

    print()
    print("=" * 116)
    print("STAGE 7 — FIFTH EMBEDDING / NEAREST-NEIGHBOR DIAGNOSIS")
    print("=" * 116)

    display = [
        "representation",
        "family",
        "split_seed",
        "mean_top1_cosine_distance_double",
        "double_spearman_true_vs_top1_neighbor_norm",
    ]
    knn_cols = [
        c for c in summary_df.columns
        if c.startswith("double_spearman_true_vs_knn")
        or c.startswith("double_auc_high_gt1_from_knn")
        or c.startswith("double_knn") and c.endswith("norm_abs_error_mean")
    ]
    display.extend(knn_cols)

    print(summary_df[display].to_string(index=False))

    print()
    print("Family-level aggregate:")
    family_display = [
        column
        for column in family_df.columns
        if column == "family"
        or column == "representation_count"
        or (
            "double_spearman_true_vs_knn" in column
            and column.endswith("_mean")
        )
        or (
            "double_auc_high_gt1_from_knn" in column
            and column.endswith("_mean")
        )
        or (
            "mean_top1_cosine_distance_double" in column
            and column.endswith("_mean")
        )
    ]
    print(family_df[family_display].to_string(index=False))

    print()
    print("Anchor double queries:")
    anchor_diag = all_diagnostics.loc[
        all_diagnostics["query_id"].astype(str).isin(anchors)
    ]
    cols = [
        "representation",
        "query_id",
        "true_norm_before",
        "top1_cosine_distance",
        "top1_neighbor_fifth",
        "top1_neighbor_norm_mean",
    ]
    print(anchor_diag[cols].to_string(index=False))

    print()
    print("Outputs:")
    for name in (
        "nearest_neighbors_cosine.csv",
        "query_embedding_diagnostics.csv",
        "representation_summary.csv",
        "family_summary.csv",
        "top1_neighbor_consensus.csv",
        "anchor_double_neighbors.csv",
    ):
        print(f"  {output_dir / name}")
    print(f"  {pca_dir}")


# =============================================================================
# Worker: exact graph cache and embedding extraction
# =============================================================================

def load_worker_cfg(config_path: Path):
    import graphgps  # noqa: F401
    from graphgps.config.config_gps import set_cfg_gps
    from torch_geometric.graphgym.config import cfg, load_cfg

    set_cfg_gps(cfg)

    if not hasattr(cfg, "set_new_allowed"):
        raise AttributeError("YACS cfg lacks set_new_allowed().")

    cfg.set_new_allowed(True)
    try:
        load_cfg(
            cfg,
            SimpleNamespace(
                cfg_file=str(config_path.resolve()),
                opts=[],
            ),
        )
    finally:
        cfg.set_new_allowed(False)

    return cfg


PE_NAMES = (
    "LapPE",
    "EquivStableLapPE",
    "SignNet",
    "RWSE",
    "HKdiagSE",
    "ElstaticSE",
)


def enabled_pe_types(cfg) -> list[str]:
    result = []
    for name in PE_NAMES:
        cfg_name = f"posenc_{name}"
        if (
            hasattr(cfg, cfg_name)
            and bool(getattr(getattr(cfg, cfg_name), "enable", False))
        ):
            result.append(name)
    return result


def materialize_posenc_kernel_times(cfg, pe_types: list[str]) -> dict[str, list]:
    resolved = {}
    safe_globals = {"__builtins__": {}}
    safe_locals = {"range": range, "list": list, "tuple": tuple}

    for pe_name in pe_types:
        pe_cfg = getattr(cfg, f"posenc_{pe_name}")
        if not hasattr(pe_cfg, "kernel"):
            continue

        kernel = pe_cfg.kernel
        times_func = str(getattr(kernel, "times_func", "")).strip()
        current_times = list(getattr(kernel, "times", []))

        if times_func:
            try:
                times = list(eval(times_func, safe_globals, safe_locals))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not materialize posenc_{pe_name}.kernel.times "
                    f"from {times_func!r}"
                ) from exc
            if not times:
                raise RuntimeError(
                    f"posenc_{pe_name}.kernel.times_func produced empty list."
                )
            kernel.times = times
            current_times = list(kernel.times)

        if not current_times:
            raise RuntimeError(
                f"Enabled PE {pe_name} has no usable kernel times."
            )
        resolved[pe_name] = current_times

    return resolved


def apply_posenc_if_needed(data, cfg, pe_types: list[str]):
    if not pe_types:
        return data

    from graphgps.transform.posenc_stats import compute_posenc_stats

    signature = inspect.signature(compute_posenc_stats)
    parameters = list(signature.parameters)

    if len(parameters) >= 4:
        result = compute_posenc_stats(data, pe_types, True, cfg)
    elif len(parameters) == 3:
        result = compute_posenc_stats(data, pe_types, True)
    else:
        raise RuntimeError(
            f"Unsupported compute_posenc_stats signature: {signature}"
        )
    return data if result is None else result


def resolve_encoder_class(repo_root: Path):
    import importlib
    import inspect as pyinspect
    import torch_geometric.graphgym.register as register
    import graphgps  # noqa: F401

    if "OneHotEmbedGPS" not in register.network_dict:
        raise RuntimeError("OneHotEmbedGPS not registered after import graphgps.")

    network_cls = register.network_dict["OneHotEmbedGPS"]
    module = importlib.import_module(network_cls.__module__)
    if not hasattr(module, "Comp5GraphEncoder"):
        raise AttributeError(
            f"{network_cls.__module__} has no Comp5GraphEncoder"
        )

    encoder_cls = module.Comp5GraphEncoder
    source = Path(pyinspect.getfile(encoder_cls)).resolve()
    if repo_root not in source.parents:
        raise RuntimeError(
            f"Resolved Comp5GraphEncoder is outside repository: {source}"
        )
    return encoder_cls


def prepare_graph_worker(args) -> None:
    import torch
    from rdkit import Chem
    from torch_geometric.data import Data
    from graph_feature import smiles2graph

    cfg = load_worker_cfg(args.reference_config.resolve())
    pe_types = enabled_pe_types(cfg)
    resolved_times = materialize_posenc_kernel_times(cfg, pe_types)

    structures = pd.read_csv(args.structures_csv.resolve())
    required = {"structure_id", "representative_smiles"}
    missing = required.difference(structures.columns)
    if missing:
        raise ValueError(
            f"structures CSV missing columns: {sorted(missing)}"
        )

    coarse_enable = bool(getattr(cfg, "coarse_grain_enable", False))
    coarse_min = int(getattr(cfg, "coarse_grain_min_chain_length", 0))

    data_list = []
    for row_index, row in structures.iterrows():
        sid = str(row["structure_id"])
        smiles = clean_text(row["representative_smiles"])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"{sid}: RDKit failed for {smiles!r}")

        graph = smiles2graph(mol, coarse_enable, coarse_min)

        if len(graph["edge_feat"]) != graph["edge_index"].shape[1]:
            raise ValueError(f"{sid}: edge feature/index mismatch")
        if len(graph["node_feat"]) != graph["num_nodes"]:
            raise ValueError(f"{sid}: node feature/count mismatch")

        data = Data(
            x=torch.from_numpy(
                np.asarray(graph["node_feat"])
            ).to(torch.int64),
            edge_index=torch.from_numpy(
                np.asarray(graph["edge_index"])
            ).to(torch.int64),
            edge_attr=torch.from_numpy(
                np.asarray(graph["edge_feat"]).flatten()
            ).to(torch.long),
            structure_index=torch.tensor([row_index], dtype=torch.long),
        )
        data = apply_posenc_if_needed(data, cfg, pe_types)
        data_list.append(data)

    if not data_list:
        raise ValueError("No graph data built.")

    dim_in = int(data_list[0].x.shape[-1])
    for i, data in enumerate(data_list):
        if data.x.ndim != 2 or int(data.x.shape[-1]) != dim_in:
            raise ValueError(
                f"graph {i} raw x mismatch: {tuple(data.x.shape)}, "
                f"expected second dim={dim_in}"
            )

    args.graph_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "data_list": data_list,
            "structure_ids": structures["structure_id"].astype(str).tolist(),
        },
        args.graph_cache.resolve(),
    )

    metadata = {
        "structure_count": len(data_list),
        "dim_in": dim_in,
        "graph_pooling": str(cfg.model.graph_pooling),
        "pe_types": pe_types,
        "resolved_pe_kernel_times": resolved_times,
        "coarse_grain_enable": coarse_enable,
        "coarse_grain_min_chain_length": coarse_min,
        "reference_config": str(args.reference_config.resolve()),
    }
    args.graph_cache_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.graph_cache_metadata.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "[prepare complete] "
        f"structures={len(data_list)}, dim_in={dim_in}, "
        f"pooling={cfg.model.graph_pooling}, PE={pe_types}"
    )


def extract_encoder_state(
    checkpoint_path: Path,
    target_state: dict[str, Any],
) -> dict[str, Any]:
    import torch

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # Stage-4 raw transfer artifact.
    if (
        isinstance(payload, dict)
        and payload
        and all(torch.is_tensor(value) for value in payload.values())
        and set(payload.keys()) == set(target_state.keys())
    ):
        state = payload
    elif isinstance(payload, dict) and "encoder_state_dict" in payload:
        state = payload["encoder_state_dict"]
    elif isinstance(payload, dict) and "model_state" in payload:
        full = payload["model_state"]
        state = {}
        for key, target in target_state.items():
            exact = [
                candidate
                for candidate in (
                    key,
                    f"comp5_encoder.{key}",
                    f"model.comp5_encoder.{key}",
                    f"model.model.comp5_encoder.{key}",
                )
                if candidate in full
            ]
            if len(exact) != 1:
                suffix = f"comp5_encoder.{key}"
                exact = [
                    candidate for candidate in full
                    if candidate.endswith(suffix)
                ]
            if len(exact) != 1:
                raise KeyError(
                    f"Could not uniquely extract Comp5 key {key!r} "
                    f"from {checkpoint_path}; candidates={exact[:20]}"
                )
            state[key] = full[exact[0]]
    else:
        raise TypeError(
            f"Unsupported checkpoint format: {checkpoint_path}"
        )

    source_keys = set(state)
    target_keys = set(target_state)
    if source_keys != target_keys:
        raise RuntimeError(
            f"Encoder key mismatch for {checkpoint_path}: "
            f"missing={sorted(target_keys-source_keys)}, "
            f"unexpected={sorted(source_keys-target_keys)}"
        )

    for key in sorted(target_keys):
        if tuple(state[key].shape) != tuple(target_state[key].shape):
            raise RuntimeError(
                f"{checkpoint_path}: shape mismatch {key}: "
                f"{tuple(state[key].shape)} != {tuple(target_state[key].shape)}"
            )

    return state


def embed_worker(args) -> None:
    import torch
    import torch_geometric.graphgym.register as register
    from torch_geometric.loader import DataLoader

    cfg = load_worker_cfg(args.reference_config.resolve())

    repo_root = Path(__file__).resolve().parents[3]
    EncoderClass = resolve_encoder_class(repo_root)

    cache = torch.load(
        args.graph_cache.resolve(),
        map_location="cpu",
        weights_only=False,
    )
    data_list = cache["data_list"]
    structure_ids = [str(x) for x in cache["structure_ids"]]

    metadata = json.loads(
        args.graph_cache_metadata.resolve().read_text(encoding="utf-8")
    )
    dim_in = int(metadata["dim_in"])

    if len(data_list) != len(structure_ids):
        raise RuntimeError("Graph-cache structure count mismatch.")

    encoder = EncoderClass(dim_in)
    state = extract_encoder_state(
        args.checkpoint.resolve(),
        encoder.state_dict(),
    )
    encoder.load_state_dict(state, strict=True)

    pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

    if str(cfg.model.graph_pooling) != str(metadata["graph_pooling"]):
        raise RuntimeError(
            "Graph pooling config differs from graph-cache metadata: "
            f"{cfg.model.graph_pooling!r} vs {metadata['graph_pooling']!r}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    encoder = encoder.to(device)
    encoder.eval()

    loader = DataLoader(
        data_list,
        batch_size=int(args.embedding_batch_size),
        shuffle=False,
        num_workers=0,
    )

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            encoded = encoder(batch)
            pooled = pooling_fun(encoded.x, encoded.batch)
            embeddings.append(
                pooled.detach().cpu().numpy().astype(np.float32)
            )

    matrix = np.concatenate(embeddings, axis=0)
    if matrix.shape[0] != len(structure_ids):
        raise RuntimeError(
            f"Embedding rows {matrix.shape[0]} != structures {len(structure_ids)}"
        )

    output = pd.DataFrame({
        "structure_id": structure_ids,
    })
    for column_index in range(matrix.shape[1]):
        output[f"emb_{column_index:03d}"] = matrix[:, column_index]

    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.worker_output.resolve(), index=False)

    print(
        "[embed complete] "
        f"checkpoint={args.checkpoint.resolve()}, "
        f"structures={matrix.shape[0]}, dim={matrix.shape[1]}, device={device}"
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--training-csv",
        type=Path,
        default=Path(
            "results/input_graphgps_optimization/"
            "o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_"
            "freshcache_baseline/staging/20260812-sum-700_utf8.csv"
        ),
    )
    parser.add_argument(
        "--new-validation",
        type=Path,
        default=Path("datasets_lrx/raw/feedback/new_validation.csv"),
    )
    parser.add_argument(
        "--stage4-root",
        type=Path,
        default=Path(
            "results/fifth_pretraining/stage4_graphgps_pretraining"
        ),
    )
    parser.add_argument(
        "--stage5-root",
        type=Path,
        default=Path(
            "results/fifth_pretraining/stage5_downstream_transfer"
        ),
    )
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=Path(
            "results/fifth_pretraining/stage5_downstream_transfer/"
            "P1_PT_D/split100/effective_config.yaml"
        ),
        help=(
            "Encoder-compatible effective config used solely to reconstruct "
            "the exact Comp5 graph/encoder interface. Stage-4 transfer strict "
            "state matching independently verifies compatibility."
        ),
    )
    parser.add_argument(
        "--stage5-models",
        nargs="+",
        default=["P0_random", "P1_PT_D", "P2_PT_DF"],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        default=DEFAULT_SPLITS,
    )
    parser.add_argument(
        "--extra-family",
        action="append",
        default=[],
        help=(
            "Optional LABEL=PATH_TEMPLATE for another split-specific family. "
            "PATH_TEMPLATE may contain {split}. Repeat this option as needed."
        ),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--anchors",
        nargs="+",
        default=DEFAULT_ANCHORS,
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/fifth_pretraining/stage7_embedding_diagnosis"
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Delete the Stage-7 output directory and recompute everything.",
    )

    # Internal workers.
    parser.add_argument(
        "--worker-mode",
        choices=["prepare", "embed"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--structures-csv",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--graph-cache",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--graph-cache-metadata",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-output",
        type=Path,
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.k <= 0:
        raise ValueError("--k must be positive.")
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be positive.")

    # Make project root importable for worker mode.
    script = Path(__file__).resolve()
    repo_root = script.parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if args.worker_mode == "prepare":
        required = (
            args.reference_config,
            args.structures_csv,
            args.graph_cache,
            args.graph_cache_metadata,
        )
        if any(value is None for value in required):
            raise ValueError("prepare worker arguments are incomplete.")
        prepare_graph_worker(args)
        return

    if args.worker_mode == "embed":
        required = (
            args.reference_config,
            args.graph_cache,
            args.graph_cache_metadata,
            args.checkpoint,
            args.worker_output,
        )
        if any(value is None for value in required):
            raise ValueError("embed worker arguments are incomplete.")
        embed_worker(args)
        return

    main_controller(args)


if __name__ == "__main__":
    main()
