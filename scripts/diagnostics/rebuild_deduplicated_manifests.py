#!/usr/bin/env python3
"""Create explicit, hash-bound nested Group CV manifests for the new dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from audit_deduplicated_dataset import TARGETS, append_execution, sha256_file, sha256_text  # noqa: E402


PROTOCOLS = {
    "fifth_component_group_cv": "fifth_component_key",
    "formula_identity_group_cv": "formula_identity_key",
}


def json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def canonical_manifest_hash(manifest: pd.DataFrame) -> str:
    """Hash manifest contents excluding the self-referential hash field."""
    columns = [column for column in manifest.columns if column != "manifest_sha256"]
    payload = manifest[columns].sort_values(["original_row_index", "split"], kind="stable")
    return sha256_text(payload.to_csv(index=False, lineterminator="\n"))


def require_audited_dataset(output_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    source = json.loads((output_dir / "data_source.json").read_text(encoding="utf-8"))
    if source.get("audit_status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise RuntimeError(f"Manifest build blocked by audit status {source.get('audit_status')!r}.")
    raw_path = Path(source["dataset_path"])
    if sha256_file(raw_path) != source["dataset_sha256"]:
        raise RuntimeError("Selected raw CSV has changed since audit; rerun audit first.")
    dataset_path = output_dir / "data_audit" / "dataset_with_sample_id.csv"
    frame = pd.read_csv(dataset_path, dtype={"sample_id": str})
    required = {"sample_id", "original_row_index", *PROTOCOLS.values(), *TARGETS}
    missing = required - set(frame.columns)
    if missing or frame.sample_id.duplicated().any():
        raise ValueError(f"Audited dataset cannot form manifests: missing={sorted(missing)}, duplicate IDs={frame.sample_id.duplicated().any()}")
    expected_hash = sha256_text("\n".join(sorted(frame.sample_id)))
    if expected_hash != source["sample_id_hash"]:
        raise RuntimeError("sample_id hash differs from the audit output.")
    return frame, source


def build_fold_manifest(frame: pd.DataFrame, source: dict[str, object], protocol: str, group_column: str,
                        fold: int, train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> pd.DataFrame:
    split = np.full(len(frame), "", dtype=object)
    split[train] = "train"
    split[validation] = "val"
    split[test] = "test"
    manifest = pd.DataFrame({
        "sample_id": frame.sample_id.astype(str), "split": split, "protocol": protocol,
        "outer_fold": fold, "group_id": frame[group_column].astype(str),
        "original_row_index": frame.original_row_index.astype(int),
        "dataset_sha256": source["dataset_sha256"],
    })
    if (manifest.split == "").any() or manifest.sample_id.duplicated().any():
        raise AssertionError(f"Incomplete or duplicate assignment in {protocol}/fold_{fold}.")
    group_splits = manifest.groupby("group_id").split.nunique()
    if group_splits.gt(1).any():
        leaked = group_splits.loc[group_splits.gt(1)].index.tolist()[:5]
        raise AssertionError(f"Group leakage in {protocol}/fold_{fold}: {leaked}")
    manifest["manifest_sha256"] = canonical_manifest_hash(manifest)
    return manifest.sort_values("original_row_index", kind="stable").reset_index(drop=True)


def fold_distribution(frame: pd.DataFrame, manifest: pd.DataFrame, protocol: str, fold: int) -> list[dict[str, object]]:
    lookup = frame.set_index("sample_id")
    if not lookup.index.is_unique:
        raise AssertionError("Audited sample IDs are not unique.")
    rows: list[dict[str, object]] = []
    for split, subset in manifest.groupby("split"):
        values = lookup.loc[subset.sample_id, TARGETS]
        for target in TARGETS:
            series = pd.to_numeric(values[target], errors="coerce")
            rows.append({"protocol": protocol, "outer_fold": fold, "split": split, "target": target,
                         "n_samples": int(len(subset)), "n_groups": int(subset.group_id.nunique()),
                         "mean": float(series.mean()), "std": float(series.std(ddof=1)),
                         "q05": float(series.quantile(0.05)), "median": float(series.median()), "q95": float(series.quantile(0.95))})
    return rows


def build_group_kfold(frame: pd.DataFrame, source: dict[str, object], protocol: str, group_column: str,
                      output_dir: Path, seed: int, n_splits: int) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    groups = frame[group_column].astype(str).to_numpy()
    distinct_groups = np.unique(groups)
    if len(distinct_groups) < n_splits:
        return build_repeated_shuffle(frame, source, protocol, group_column, output_dir, seed)
    fold_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    outer_test_count = np.zeros(len(frame), dtype=int)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (outer_train_val, outer_test) in enumerate(splitter.split(frame, groups=groups)):
        inner_groups = groups[outer_train_val]
        inner_splitter = GroupShuffleSplit(n_splits=1, test_size=1 / n_splits, random_state=seed + fold)
        train_relative, val_relative = next(inner_splitter.split(outer_train_val, groups=inner_groups))
        outer_train = outer_train_val[train_relative]
        outer_val = outer_train_val[val_relative]
        manifest = build_fold_manifest(frame, source, protocol, group_column, fold, outer_train, outer_val, outer_test)
        path = output_dir / protocol / f"fold_{fold}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(path, index=False)
        file_hash = sha256_file(path)
        outer_test_count[outer_test] += 1
        fold_rows.append({"protocol": protocol, "mode": "group_kfold_5", "outer_fold": fold, "path": str(path),
                          "file_sha256": file_hash, "manifest_sha256": manifest.manifest_sha256.iloc[0],
                          "n_total": len(manifest), "n_train": int((manifest.split == "train").sum()),
                          "n_val": int((manifest.split == "val").sum()), "n_test": int((manifest.split == "test").sum()),
                          "n_groups_total": int(manifest.group_id.nunique()), "n_groups_train": int(manifest.loc[manifest.split == "train", "group_id"].nunique()),
                          "n_groups_val": int(manifest.loc[manifest.split == "val", "group_id"].nunique()), "n_groups_test": int(manifest.loc[manifest.split == "test", "group_id"].nunique()),
                          "sample_id_unique": bool(not manifest.sample_id.duplicated().any()), "group_leakage": bool(manifest.groupby("group_id").split.nunique().gt(1).any()),
                          "outer_test_coverage_exact_once": None})
        distribution_rows.extend(fold_distribution(frame, manifest, protocol, fold))
    exact_once = bool(np.all(outer_test_count == 1))
    for record in fold_rows:
        record["outer_test_coverage_exact_once"] = exact_once
    if not exact_once:
        raise AssertionError(f"{protocol} outer test folds do not cover every sample exactly once.")
    return fold_rows, distribution_rows, {"protocol": protocol, "mode": "group_kfold_5", "n_groups": len(distinct_groups), "n_folds": n_splits}


def build_repeated_shuffle(frame: pd.DataFrame, source: dict[str, object], protocol: str, group_column: str,
                           output_dir: Path, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Fallback only when five group folds are mathematically unavailable."""
    groups = frame[group_column].astype(str).to_numpy()
    outer = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=seed)
    fold_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    for fold, (outer_train_val, outer_test) in enumerate(outer.split(frame, groups=groups)):
        inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 100 + fold)
        train_relative, val_relative = next(inner.split(outer_train_val, groups=groups[outer_train_val]))
        manifest = build_fold_manifest(frame, source, protocol, group_column, fold, outer_train_val[train_relative], outer_train_val[val_relative], outer_test)
        path = output_dir / protocol / f"fold_{fold}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(path, index=False)
        fold_rows.append({"protocol": protocol, "mode": "group_shuffle_repeat_10", "outer_fold": fold, "path": str(path),
                          "file_sha256": sha256_file(path), "manifest_sha256": manifest.manifest_sha256.iloc[0],
                          "n_total": len(manifest), "n_train": int((manifest.split == "train").sum()), "n_val": int((manifest.split == "val").sum()),
                          "n_test": int((manifest.split == "test").sum()), "n_groups_total": int(manifest.group_id.nunique()),
                          "n_groups_train": int(manifest.loc[manifest.split == "train", "group_id"].nunique()), "n_groups_val": int(manifest.loc[manifest.split == "val", "group_id"].nunique()),
                          "n_groups_test": int(manifest.loc[manifest.split == "test", "group_id"].nunique()), "sample_id_unique": True,
                          "group_leakage": bool(manifest.groupby("group_id").split.nunique().gt(1).any()), "outer_test_coverage_exact_once": False})
        distribution_rows.extend(fold_distribution(frame, manifest, protocol, fold))
    return fold_rows, distribution_rows, {"protocol": protocol, "mode": "group_shuffle_repeat_10", "n_groups": int(np.unique(groups).size), "n_folds": 10}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "deduplicated_rebaseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Validate one protocol in a throwaway manifest directory.")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    frame, source = require_audited_dataset(output_dir)
    manifest_root = output_dir / ("manifests_smoke" if arguments.smoke else "manifests")
    protocols = {"fifth_component_group_cv": PROTOCOLS["fifth_component_group_cv"]} if arguments.smoke else PROTOCOLS
    integrity_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    settings: list[dict[str, object]] = []
    for protocol, group_column in protocols.items():
        fold_rows, rows, protocol_settings = build_group_kfold(frame, source, protocol, group_column, manifest_root, arguments.seed, 5)
        integrity_rows.extend(fold_rows)
        distribution_rows.extend(rows)
        settings.append(protocol_settings)
    integrity = pd.DataFrame(integrity_rows)
    integrity.to_csv(manifest_root / "manifest_integrity.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(manifest_root / "fold_distribution_summary.csv", index=False)
    json_dump({"dataset_sha256": source["dataset_sha256"], "sample_id_hash": source["sample_id_hash"], "seed": arguments.seed,
               "protocols": settings, "manifest_hash_definition": "SHA256 of canonical manifest content excluding manifest_sha256 field"}, manifest_root / "manifest_settings.json")
    lines = ["# Deduplicated Group CV Manifest Report", "", f"- Dataset SHA256: `{source['dataset_sha256']}`", f"- sample_id hash: `{source['sample_id_hash']}`"]
    for setting in settings:
        lines.append(f"- `{setting['protocol']}`: {setting['mode']}; {setting['n_groups']} groups; {setting['n_folds']} manifests.")
    lines.extend(["- Train, validation, and test group identities are pairwise disjoint within every manifest.",
                  "- `manifest_sha256` hashes canonical contents excluding itself; `file_sha256` is recorded in manifest_integrity.csv."])
    (manifest_root / "manifest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_execution(output_dir, {"timestamp": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
        "dataset_path": str(output_dir / "data_audit" / "dataset_with_sample_id.csv"), "dataset_sha256": source["dataset_sha256"],
        "protocol": "smoke_fifth" if arguments.smoke else "both", "fold": "all", "seed": arguments.seed,
        "manifest_sha256": sha256_file(manifest_root / "manifest_integrity.csv"), "feature_hash": None, "config_hash": None,
        "checkpoint": None, "status": "PASS", "error": None, "output_path": str(manifest_root)})
    print(f"Wrote {manifest_root}")


if __name__ == "__main__":
    main()
