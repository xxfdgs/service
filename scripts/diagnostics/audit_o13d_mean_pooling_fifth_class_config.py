#!/usr/bin/env python3
"""Fail-closed configuration audit for O13-D: mean pooling + Fifth_class."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/diagnostics/run_fusion_head_experiment.py"
NETWORK = ROOT / "graphgps/network/onehot_embed_gps.py"
LOADER = ROOT / "graphgps/lrx_add/csv_pyg_five_multi.py"

DYNAMIC_PATHS = {
    "cfg_dest", "out_dir", "run_dir", "read_csv", "component_vocab_source",
    "mordred_feature_path", "dataset.dir", "dataset.cache_tag",
    "dataset.diagnostic_split_path", "model.architecture_name",
}
EXPECTED_DIFFS = [
    {"path": "model.graph_pooling", "o12": "add", "o13d": "mean"},
    {"path": "use_fifth_class_embedding", "o12": False, "o13d": True},
]


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return result
    return {prefix: value}


def nested(config: dict, path: str) -> object:
    value: object = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Config misses {path}")
        value = value[key]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(config: dict, field: str, fallback: Path | None) -> dict[str, object]:
    configured = Path(str(nested(config, field))).resolve()
    path, fallback_used = configured, False
    if not path.is_file() and fallback is not None:
        path, fallback_used = fallback.resolve(), True
    if not path.is_file():
        raise FileNotFoundError(f"Missing {field}: {configured}")
    return {"configured_path": str(configured), "hashed_path": str(path),
            "used_locked_fallback": fallback_used, "sha256": sha256(path)}


def source_checks() -> dict[str, object]:
    runner, network, loader = (path.read_text(encoding="utf-8") for path in (RUNNER, NETWORK, LOADER))
    required = {
        "runner": ["parser.add_argument('--graph-pooling'", "cfg.model.graph_pooling = args.graph_pooling",
                   "parser.add_argument('--use-fifth-class-embedding'"],
        "network": ["self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]",
                    "emb5 = self.pooling_fun(data5_encoded.x, data5_encoded.batch)",
                    "emb5 = emb5 + self.fifth_class_embedding(class_ids)"],
        "loader": ["def build_input_fifth_class_vocab", "def input_fifth_class_id",
                   "fifth_class_id = input_fifth_class_id"],
    }
    text_by_name = {"runner": runner, "network": network, "loader": loader}
    missing = [f"{name}:{snippet}" for name, snippets in required.items()
               for snippet in snippets if snippet not in text_by_name[name]]
    if missing:
        raise RuntimeError(f"Cannot verify O13-D implementation: {missing}")
    return {
        "runner": str(RUNNER), "network": str(NETWORK), "loader": str(LOADER),
        "verified_forward": (
            "Mean pooling is resolved through cfg.model.graph_pooling and applied to "
            "the component-5 GraphGPS node states; Fifth_class embedding is added only "
            "to that component-5 graph representation. Components 1–4 remain O12 "
            "categorical embeddings, not graph-pooled representations."
        ),
    }


def baseline_checks(config: dict) -> None:
    expected = {
        "model.type": "OneHotEmbedGPS", "model.graph_pooling": "add",
        "model.fifth_only_fusion": False, "model.ratio_polynomial_features": False,
        "use_fifth_class_embedding": False, "use_fifth_identity_embedding": False,
        "use_fifth_ratio_modulation": False, "mordred_fifth_only": False,
    }
    for path, value in expected.items():
        if nested(config, path) != value:
            raise RuntimeError(f"Frozen O12 violates O13-D contract: {path} != {value!r}")


def differences(o12: dict, o13d: dict) -> list[dict[str, object]]:
    left, right = flatten(o12), flatten(o13d)
    return [{"path": path, "o12": left.get(path), "o13d": right.get(path)}
            for path in sorted(set(left) | set(right))
            if path not in DYNAMIC_PATHS and left.get(path) != right.get(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-effective", type=Path, required=True)
    parser.add_argument("--candidate-effective", type=Path)
    parser.add_argument("--locked-input-csv", type=Path, required=True)
    parser.add_argument("--locked-mordred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = load(args.baseline_effective)
    baseline_checks(baseline)
    baseline_input = checked_file(baseline, "read_csv", args.locked_input_csv)
    baseline_mordred = checked_file(baseline, "mordred_feature_path", args.locked_mordred)
    report: dict[str, object] = {
        "experiment": "O13-D = O12 + graph_pooling add→mean + Fifth_class embedding",
        "baseline_effective": str(args.baseline_effective.resolve()), "source_checks": source_checks(),
        "baseline_input": baseline_input, "baseline_mordred": baseline_mordred,
        "allowed_semantic_differences": EXPECTED_DIFFS,
    }
    if args.candidate_effective is None:
        report.update({"phase": "preflight", "status": "pass"})
    else:
        candidate = load(args.candidate_effective)
        candidate_input = checked_file(candidate, "read_csv", args.locked_input_csv)
        candidate_mordred = checked_file(candidate, "mordred_feature_path", args.locked_mordred)
        if candidate_input["sha256"] != baseline_input["sha256"]:
            raise RuntimeError("O13-D input content differs from the frozen O12 baseline.")
        if candidate_mordred["sha256"] != baseline_mordred["sha256"]:
            raise RuntimeError("O13-D Mordred content differs from the frozen O12 baseline.")
        actual = differences(baseline, candidate)
        if actual != EXPECTED_DIFFS:
            raise RuntimeError(
                "O13-D effective configuration has changes outside the two allowed "
                f"ablation flags: {json.dumps(actual, indent=2)}")
        if int(nested(candidate, "fifth_class_vocab_size")) < 3:
            raise RuntimeError("O13-D Fifth_class vocabulary cannot represent unknown/double/single.")
        report.update({"phase": "post_training", "status": "pass",
                       "candidate_effective": str(args.candidate_effective.resolve()),
                       "candidate_input": candidate_input, "candidate_mordred": candidate_mordred,
                       "semantic_config_differences": actual})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
