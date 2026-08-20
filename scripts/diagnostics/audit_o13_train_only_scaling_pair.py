#!/usr/bin/env python3
"""Fail-closed audit for the one-seed strict-scaling O12/O13-C pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


DYNAMIC = {
    "cfg_dest", "out_dir", "run_dir", "model.architecture_name",
    "dataset.dir", "dataset.cache_tag", "dataset.diagnostic_split_path",
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, dict):
        return {prefix: value}
    output: dict[str, object] = {}
    for key, child in value.items():
        output.update(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o12-effective", type=Path, required=True)
    parser.add_argument("--o13c-effective", type=Path, required=True)
    parser.add_argument("--strict-lookup", type=Path, required=True)
    parser.add_argument("--scaler-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    o12, o13c = load(args.o12_effective), load(args.o13c_effective)
    left, right = flatten(o12), flatten(o13c)
    differences = [{"path": path, "o12": left.get(path), "o13c": right.get(path)}
                   for path in sorted(set(left) | set(right))
                   if path not in DYNAMIC and left.get(path) != right.get(path)]
    expected = [{"path": "model.graph_pooling", "o12": "add", "o13c": "mean"}]
    if differences != expected:
        raise RuntimeError(f"Strict-scaling pair differs beyond pooling: {json.dumps(differences, indent=2)}")
    lookup = args.strict_lookup.resolve()
    lookup_hash = sha256(lookup)
    for label, cfg in (("O12", o12), ("O13-C", o13c)):
        configured = Path(str(cfg.get("mordred_feature_path", ""))).resolve()
        if configured != lookup or sha256(configured) != lookup_hash:
            raise RuntimeError(f"{label} does not use the exact seed-specific strict lookup.")
        if bool(cfg.get("mordred_fifth_only", False)):
            raise RuntimeError(f"{label} violates full-O12 Mordred fusion semantics.")
    metadata = json.loads(args.scaler_metadata.read_text(encoding="utf-8"))
    if metadata.get("leakage_check", {}).get("status") != "PASS":
        raise RuntimeError("Scaler metadata did not pass train/val/test leakage checks.")
    report = {
        "status": "pass", "o12_effective": str(args.o12_effective.resolve()),
        "o13c_effective": str(args.o13c_effective.resolve()), "strict_lookup": str(lookup),
        "strict_lookup_sha256": lookup_hash, "scaler_metadata": str(args.scaler_metadata.resolve()),
        "scaler_leakage_check": metadata["leakage_check"], "semantic_config_differences": differences,
        "statement": "The two Part-A refits share the same train-only descriptor lookup; graph pooling is the only effective model difference.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
