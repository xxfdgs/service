#!/usr/bin/env python3
"""Consolidate non-M3 shard execution metadata into the root audit manifest.

Shard runners keep their exact command and data-hash metadata next to their
outputs.  This collector makes the root manifest complete without opening any
M3 artifact, which the user explicitly deferred for later review.
"""

from __future__ import annotations

import json
from pathlib import Path

from prepare_hybrid_embedding_tree_experiment import ROOT, append_execution, json_dump


OUTPUT = ROOT / "results/hybrid_embedding_tree_exp"


def is_deferred_m3(path: Path) -> bool:
    return any(part.endswith("_m3") for part in path.parts)


def main() -> None:
    root_path = OUTPUT / "execution_manifest.json"
    existing = json.loads(root_path.read_text(encoding="utf-8")) if root_path.exists() else []
    # Earlier collection used an inconsistent outer-fold key and could append
    # duplicate historical shard records.  Keep one auditable record per old
    # shard and mark it invalidated rather than pretending it is a rerun.
    repaired = []
    old_shards = set()
    for row in existing:
        if row.get("status") != "COMPLETED_NON_M3_SHARD":
            repaired.append(row)
            continue
        key = (str(row.get("stage")), str(row.get("outer_fold")), str(row.get("output_path")))
        if key in old_shards:
            continue
        old_shards.add(key)
        row = dict(row)
        row["status"] = "INVALIDATED_PREPROCESSOR_CACHE_BUG"
        row["error"] = "Superseded before selection: inner preprocessing cache was corrected and the shard is being rerun."
        repaired.append(row)
    if repaired != existing:
        json_dump(repaired, root_path)
    existing = repaired
    seen = {
        (str(row.get("stage")), str(row.get("outer_fold")), str(row.get("output_path")))
        for row in existing
        if row.get("status") != "INVALIDATED_PREPROCESSOR_CACHE_BUG"
    }
    linear_marker = OUTPUT / "stage1/fixed_preprocessor_linear_complete.marker"
    if not linear_marker.is_file():
        correction = OUTPUT / "audit/preprocessor_rerun_note.md"
        correction_key = ("preprocessor_correction", "fold_0,fold_4", str(correction.resolve()))
        if correction.is_file() and correction_key not in seen:
            append_execution(
                OUTPUT,
                command=["code_audit", "rerun non-M3 shards after inner-preprocessing cache correction"],
                stage="preprocessor_correction",
                target="all",
                outer_fold="fold_0,fold_4",
                feature_family="A0-A7,B1-B11",
                embedding_name="not_read_for_M3",
                model="M0,M1,M2,M4,M5,M6",
                status="PREVIOUS_NON_M3_SHARDS_INVALIDATED_RERUN_REQUIRED",
                output_path=str(correction.resolve()),
            )
        print("EXECUTION_MANIFEST_WAITING_FOR_FIXED_LINEAR_RERUN")
        return
    records = []
    for stage in ("stage1", "stage2"):
        shard_root = OUTPUT / stage / "shards"
        for path in sorted(shard_root.glob("*/fold_*_execution.json")):
            if is_deferred_m3(path):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            output = str(path.resolve())
            key = (str(payload.get("stage")), f"fold_{payload.get('outer_fold')}", output)
            if key in seen:
                continue
            records.append((payload, output))
            seen.add(key)
    for payload, output in records:
        append_execution(
            OUTPUT,
            command=payload.get("command"),
            stage=payload.get("stage"),
            target="all",
            outer_fold=f"fold_{payload.get('outer_fold')}",
            inner_fold="GroupKFold(5)",
            feature_family=",".join(payload.get("families", [])),
            embedding_name="descriptor_branch_raw,fused_embedding,graph_branch_raw",
            model=",".join(payload.get("models", [])),
            dataset_hash=payload.get("dataset_hash"),
            manifest_hash=payload.get("manifest_hash"),
            hyperparameters={"shard_execution_timestamp": payload.get("timestamp"), "preprocessor_generation": "fixed"},
            status="COMPLETED_FIXED_PREPROCESSOR_RERUN",
            output_path=output,
        )
    note = OUTPUT / "stage1/m3_deferred_protocol_note.md"
    note_key = ("m3_deferment", "none", str(note.resolve()))
    if note.is_file() and note_key not in seen:
        append_execution(
            OUTPUT,
            command=["user_instruction", "defer M3 from current decision path while retaining its independent run"],
            stage="m3_deferment",
            target="all",
            outer_fold="none",
            feature_family="all",
            embedding_name="not_read",
            model="M3",
            status="USER_DEFERRED_NOT_READ",
            output_path=str(note.resolve()),
        )
    print("EXECUTION_MANIFEST_CONSOLIDATED", len(records), "non-M3 shards")


if __name__ == "__main__":
    main()
