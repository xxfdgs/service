#!/usr/bin/env python3
"""Verify required checkpoint and prediction artifacts before advancing GraphGPS seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import TARGET_COLUMNS
from stage3_utils import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generalization_stage3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--data-version", default="raw_records")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    graph_dir = output_dir / "graphgps_raw_cv"
    rows: list[dict[str, object]] = []
    for protocol in ("fifth_component_group_cv", "formula_identity_group_cv"):
        for fold_index in range(5):
            fold = f"fold_{fold_index}"
            manifest = output_dir / "manifests" / protocol / arguments.data_version / f"{fold}.csv"
            config = graph_dir / "configs" / f"{protocol}_{arguments.data_version}_{fold}_seed_{arguments.seed}.yaml"
            run_dir = graph_dir / "training" / config.stem / str(arguments.seed)
            checkpoints = sorted((run_dir / "ckpt").glob("*.ckpt"))
            checkpoint_ok = len(checkpoints) == 1
            metadata_ok = False
            if checkpoint_ok:
                payload = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
                metadata_ok = bool(payload.get("stage3_checkpoint_metadata")) and payload.get("seed") == arguments.seed \
                    and payload.get("manifest_hash") == sha256_file(manifest)
            for split in ("val", "test"):
                path = graph_dir / "seed_predictions" / protocol / f"{fold}_seed_{arguments.seed}_{split}.csv"
                manifest_frame = pd.read_csv(manifest, dtype={"sample_id": str})
                expected = manifest_frame.loc[manifest_frame.split == split, "sample_id"].astype(str)
                if path.is_file():
                    prediction = pd.read_csv(path, dtype={"sample_id": str})
                    unique = not prediction.duplicated(["sample_id", "target"]).any()
                    ids_equal = set(prediction.sample_id) == set(expected)
                    target_equal = set(prediction.target) == set(TARGET_COLUMNS)
                    n_expected = len(expected) * len(TARGET_COLUMNS)
                    prediction_ok = unique and ids_equal and target_equal and len(prediction) == n_expected
                else:
                    unique = ids_equal = target_equal = prediction_ok = False
                    n_expected = len(expected) * len(TARGET_COLUMNS)
                rows.append({"protocol": protocol, "fold": fold, "seed": arguments.seed, "split": split,
                             "checkpoint_ok": checkpoint_ok, "metadata_ok": metadata_ok,
                             "prediction_path": str(path), "prediction_ok": prediction_ok,
                             "sample_target_unique": unique, "sample_id_set_equal": ids_equal,
                             "target_set_equal": target_equal, "n_expected": n_expected,
                             "n_actual": len(prediction) if path.is_file() else 0,
                             "status": "pass" if checkpoint_ok and metadata_ok and prediction_ok else "pending_or_fail"})
    report = pd.DataFrame(rows)
    report.to_csv(graph_dir / f"seed_{arguments.seed}_completion_audit.csv", index=False)
    print(f"Seed {arguments.seed} completion pass: {bool((report.status == 'pass').all())}")


if __name__ == "__main__":
    main()
