#!/usr/bin/env python3
"""Fail-closed configuration audit for the O13-C add-to-mean pooling ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/diagnostics/run_fusion_head_experiment.py"
NETWORK = ROOT / "graphgps/network/onehot_embed_gps.py"

# These vary per frozen seed/protocol run but cannot affect model semantics.
DYNAMIC_PATHS = {
    "cfg_dest", "out_dir", "run_dir", "read_csv", "component_vocab_source",
    "mordred_feature_path", "dataset.dir", "dataset.cache_tag",
    "dataset.diagnostic_split_path", "model.architecture_name",
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        output: dict[str, object] = {}
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten(child, name))
        return output
    return {prefix: value}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get(config: dict, path: str) -> object:
    value: object = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Config misses {path}")
        value = value[key]
    return value


def source_checks() -> dict[str, object]:
    runner = RUNNER.read_text(encoding="utf-8")
    network = NETWORK.read_text(encoding="utf-8")
    required_runner = [
        "parser.add_argument('--graph-pooling'",
        "cfg.model.graph_pooling = args.graph_pooling",
    ]
    required_network = [
        "self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]",
        "emb5 = self.pooling_fun(data5_encoded.x, data5_encoded.batch)",
    ]
    missing = [text for text in required_runner if text not in runner]
    missing += [text for text in required_network if text not in network]
    if missing:
        raise RuntimeError(
            "O13-C pooling override cannot be verified in the runner/network: "
            f"missing {missing}")
    return {
        "runner": str(RUNNER),
        "network": str(NETWORK),
        "verified_forward": (
            "OneHotEmbedGPS resolves cfg.model.graph_pooling through the registered "
            "pooling dictionary and applies it to the component-5 GraphGPS node states. "
            "Components 1–4 are O12 categorical embeddings and are not graph-pooled."
        ),
    }


def baseline_checks(config: dict) -> None:
    if get(config, "model.type") != "OneHotEmbedGPS":
        raise RuntimeError("O13-C requires the frozen O12 OneHotEmbedGPS baseline.")
    if get(config, "model.graph_pooling") != "add":
        raise RuntimeError("Frozen O12 baseline must use add graph pooling.")
    expected_false = {
        "model.fifth_only_fusion": False,
        "model.ratio_polynomial_features": False,
        "use_fifth_class_embedding": False,
        "use_fifth_identity_embedding": False,
        "use_fifth_ratio_modulation": False,
        "mordred_fifth_only": False,
    }
    for path, expected in expected_false.items():
        if get(config, path) != expected:
            raise RuntimeError(f"Frozen O12 baseline violates O13-C contract: {path} != {expected}")


def compare(baseline: dict, candidate: dict) -> list[dict[str, object]]:
    left, right = flatten(baseline), flatten(candidate)
    differences = []
    for path in sorted(set(left) | set(right)):
        if path in DYNAMIC_PATHS or left.get(path) == right.get(path):
            continue
        differences.append({"path": path, "o12": left.get(path), "o13c": right.get(path)})
    return differences


def file_hash(config: dict, path: str, fallback: Path | None = None) -> dict[str, str]:
    configured_path = Path(str(get(config, path))).resolve()
    file_path = configured_path
    used_fallback = False
    if not file_path.is_file() and fallback is not None:
        file_path = fallback.resolve()
        used_fallback = True
    if not file_path.is_file():
        raise FileNotFoundError(f"Configured file is missing: {path}={configured_path}")
    return {"configured_path": str(configured_path), "hashed_path": str(file_path),
            "used_locked_fallback": used_fallback, "sha256": sha256(file_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-effective", type=Path, required=True)
    parser.add_argument("--candidate-effective", type=Path)
    parser.add_argument("--locked-input-csv", type=Path,
                        help="Existing locked input used when a historical O12 path was cleaned.")
    parser.add_argument("--locked-mordred", type=Path,
                        help="Existing locked Mordred lookup used when a historical O12 path was cleaned.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline_effective)
    baseline_checks(baseline)
    report: dict[str, object] = {
        "experiment": "O13-C = O12 + graph_pooling add→mean",
        "baseline_effective": str(args.baseline_effective.resolve()),
        "source_checks": source_checks(),
        "baseline_input": file_hash(baseline, "read_csv", args.locked_input_csv),
        "baseline_mordred": file_hash(baseline, "mordred_feature_path", args.locked_mordred),
        "allowed_semantic_difference": {
            "model.graph_pooling": {"o12": "add", "o13c": "mean"}
        },
    }
    if args.candidate_effective is None:
        report["phase"] = "preflight"
        report["status"] = "pass"
        report["note"] = (
            "The training command must pass --graph-pooling mean. Post-training, each "
            "saved effective_config.yaml is compared fail-closed to this baseline.")
    else:
        candidate = load(args.candidate_effective)
        candidate_input = file_hash(candidate, "read_csv", args.locked_input_csv)
        candidate_mordred = file_hash(candidate, "mordred_feature_path", args.locked_mordred)
        if candidate_input["sha256"] != report["baseline_input"]["sha256"]:
            raise RuntimeError("O13-C input CSV content differs from frozen O12 baseline.")
        if candidate_mordred["sha256"] != report["baseline_mordred"]["sha256"]:
            raise RuntimeError("O13-C Mordred lookup content differs from frozen O12 baseline.")
        if get(candidate, "model.graph_pooling") != "mean":
            raise RuntimeError("O13-C candidate does not use mean graph pooling.")
        for path in ("model.fifth_only_fusion", "model.ratio_polynomial_features",
                     "use_fifth_class_embedding", "use_fifth_identity_embedding",
                     "use_fifth_ratio_modulation", "mordred_fifth_only"):
            if get(candidate, path) != get(baseline, path):
                raise RuntimeError(f"Forbidden O13-C change: {path}")
        differences = compare(baseline, candidate)
        if differences != [{"path": "model.graph_pooling", "o12": "add", "o13c": "mean"}]:
            raise RuntimeError(
                "O13-C effective-config audit failed; differences outside the single pooling "
                f"ablation: {json.dumps(differences, indent=2)}")
        report.update({
            "phase": "post_training",
            "candidate_effective": str(args.candidate_effective.resolve()),
            "candidate_input": candidate_input,
            "candidate_mordred": candidate_mordred,
            "semantic_config_differences": differences,
            "status": "pass",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
