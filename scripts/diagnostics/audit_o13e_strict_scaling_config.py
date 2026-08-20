#!/usr/bin/env python3
"""Fail closed: O13-E differs from strict-scaled O13-C only by fifth descriptors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


DYNAMIC = {
    "cfg_dest", "out_dir", "run_dir", "model.architecture_name", "dataset.dir",
    "dataset.cache_tag", "dataset.diagnostic_split_path",
    "fifth_mechanistic_descriptor_path", "fifth_mechanistic_descriptor_dim",
    "use_fifth_mechanistic_descriptors",
}


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, dict):
        return {prefix: value}
    out: dict[str, object] = {}
    for key, child in value.items():
        out.update(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
    return out


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o13c-effective", type=Path, required=True)
    parser.add_argument("--o13e-effective", type=Path, required=True)
    parser.add_argument("--strict-mordred", type=Path, required=True)
    parser.add_argument("--strict-fifth", type=Path, required=True)
    parser.add_argument("--mordred-scaler-metadata", type=Path, required=True)
    parser.add_argument("--fifth-scaler-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    c, e = load(args.o13c_effective), load(args.o13e_effective)
    left, right = flatten(c), flatten(e)
    differences = [{"path": path, "o13c": left.get(path), "o13e": right.get(path)}
                   for path in sorted(set(left) | set(right))
                   if path not in DYNAMIC and left.get(path) != right.get(path)]
    if differences:
        raise RuntimeError(f"O13-E differs from O13-C beyond descriptors: {json.dumps(differences, indent=2)}")
    if c.get("model", {}).get("graph_pooling") != "mean" or e.get("model", {}).get("graph_pooling") != "mean":
        raise RuntimeError("O13-C/E contract requires mean pooling in both models")
    for cfg, label in ((c, "O13-C"), (e, "O13-E")):
        forbidden = {"model.fifth_only_fusion": cfg["model"].get("fifth_only_fusion"),
                     "use_fifth_class_embedding": cfg.get("use_fifth_class_embedding"),
                     "model.ratio_polynomial_features": cfg["model"].get("ratio_polynomial_features"),
                     "mordred_fifth_only": cfg.get("mordred_fifth_only")}
        if any(value for value in forbidden.values()):
            raise RuntimeError(f"{label} violates O13-E single-variable contract: {forbidden}")
    if c.get("use_fifth_mechanistic_descriptors") or not e.get("use_fifth_mechanistic_descriptors"):
        raise RuntimeError("Descriptor branch must be disabled in O13-C and enabled in O13-E")
    mordred_path, fifth_path = args.strict_mordred.resolve(), args.strict_fifth.resolve()
    for cfg, label in ((c, "O13-C"), (e, "O13-E")):
        path = Path(str(cfg.get("mordred_feature_path", ""))).resolve()
        if path != mordred_path or sha256(path) != sha256(mordred_path):
            raise RuntimeError(f"{label} does not use the seed-specific strict Mordred lookup")
    if Path(str(e.get("fifth_mechanistic_descriptor_path", ""))).resolve() != fifth_path:
        raise RuntimeError("O13-E does not use the specified seed-specific fifth lookup")
    if int(e.get("fifth_mechanistic_descriptor_dim", 0)) != 12:
        raise RuntimeError("O13-E must use exactly the 12 declared mechanism descriptors")
    mordred_metadata = json.loads(args.mordred_scaler_metadata.read_text(encoding="utf-8"))
    fifth_metadata = json.loads(args.fifth_scaler_metadata.read_text(encoding="utf-8"))
    if (mordred_metadata.get("leakage_check", {}).get("status") != "PASS"
            or fifth_metadata.get("leakage_check", {}).get("status") != "PASS"):
        raise RuntimeError("A strict descriptor scaler leakage audit did not pass")
    report = {
        "status": "pass", "o13c_effective": str(args.o13c_effective.resolve()),
        "o13e_effective": str(args.o13e_effective.resolve()),
        "strict_mordred": str(mordred_path), "strict_mordred_sha256": sha256(mordred_path),
        "strict_fifth": str(fifth_path), "strict_fifth_sha256": sha256(fifth_path),
        "mordred_leakage_check": mordred_metadata["leakage_check"],
        "fifth_leakage_check": fifth_metadata["leakage_check"],
        "semantic_config_differences": [
            {"path": "use_fifth_mechanistic_descriptors", "o13c": False, "o13e": True},
            {"path": "fifth_mechanistic_descriptor_dim", "o13c": 0, "o13e": 12},
            {"path": "fifth_mechanistic_descriptor_path", "o13c": "", "o13e": str(fifth_path)},
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
