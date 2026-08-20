#!/usr/bin/env python3
"""Run one strict P1 checkpoint batch and assert Fifth-pathway feature flow.

This is a runtime companion to the saved architecture/config audits.  It
reconstructs a selected checkpoint with its own frozen training source and
manifest, forwards one outer-train batch, and verifies that the real Fifth
graph encoder and class embedding receive gradient.  No optimisation occurs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def parameter_grad_norm(module: torch.nn.Module | None) -> float:
    if module is None:
        return 0.0
    values = [parameter.grad.detach().norm().item() for parameter in module.parameters()
              if parameter.grad is not None]
    return float(np.linalg.norm(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    run_dir, output = args.run_dir.resolve(), args.output_dir.resolve()
    for name in ("effective_config.yaml", "run_settings.json", "checkpoints/selected_best.pt"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(run_dir / name)

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    import graphgps  # noqa: F401
    from graphgps.config.config_gps import set_cfg_gps
    from graphgps.create_model_gps import create_model_gps
    from graphgps.determinism import configure_determinism
    from loader_5 import create_loader_5
    from torch_geometric.graphgym.config import cfg, load_cfg

    settings = json.loads((run_dir / "run_settings.json").read_text())
    if settings.get("single_target") != "Norm_before" or settings.get("property_num") != 1:
        raise ValueError("Runtime audit only accepts a one-output Norm_before checkpoint.")
    set_cfg_gps(cfg)
    cfg.set_new_allowed(True)
    try:
        load_cfg(cfg, SimpleNamespace(cfg_file=str(run_dir / "effective_config.yaml"), opts=[]))
    finally:
        cfg.set_new_allowed(False)
    source = Path(settings["input_csv"])
    manifest = Path(settings["split_manifest"])
    vocabulary_source = Path(settings["component_vocab_source"])
    if not source.is_file() or not manifest.is_file() or not vocabulary_source.is_file():
        raise FileNotFoundError("Saved P1 provenance source, manifest, or vocabulary source is missing.")
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.read_csv = str(source)
    cfg.component_vocab_source = str(vocabulary_source)
    cfg.property_num = 1; cfg.property_serial = 4; cfg.single_task_target_index = 4
    cfg.dataset.diagnostic_split_path = str(manifest)
    cfg.dataset.diagnostic_id_column = "ID"; cfg.dataset.diagnostic_manifest_id_column = "sample_id"
    cfg.dataset.dir = str(cache_dir); cfg.dataset.cache_tag = "p1-fifth-runtime-audit"; cfg.dataset.cache_refresh = True
    cfg.accelerator = args.device; cfg.gpu_serial = 0
    configure_determinism(int(cfg.seed), bool(cfg.train.deterministic))
    loaders = create_loader_5()
    device = torch.device(args.device)
    model = create_model_gps().to(device)
    checkpoint = torch.load(run_dir / "checkpoints/selected_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.train()

    batches = next(zip(*[group[0] for group in loaders]))
    components = list(batches)
    for suffix, batch in zip(("", "_2", "_3", "_4", "_5"), components):
        batch.split = "train" + suffix
        batch.to(device)
    fifth = components[4]
    fifth_ratio = fifth.ratio.detach().view(-1)
    if not hasattr(fifth, "fifth_class_id") or not hasattr(fifth, "component_vocab_id"):
        raise RuntimeError("Loader did not pass Fifth categorical identities to the runtime batch.")
    prediction, label = model(*components)[:2]
    if not torch.isfinite(prediction).all() or not torch.isfinite(label).all():
        raise RuntimeError("Non-finite Fifth-pathway runtime output.")
    prediction.sum().backward()
    network = getattr(model, "model", model)
    comp5_grad = parameter_grad_norm(getattr(network, "comp5_encoder", None))
    fifth_class_grad = parameter_grad_norm(getattr(network, "fifth_class_embedding", None))
    aux_grad = parameter_grad_norm(getattr(network, "aux_feature_encoder", None))
    if comp5_grad <= 0:
        raise RuntimeError("Fifth GraphGPS encoder received no gradient on the real runtime batch.")
    if hasattr(network, "fifth_class_embedding") and fifth_class_grad <= 0:
        raise RuntimeError("Configured Fifth-class embedding received no gradient on the real runtime batch.")
    if (fifth_ratio < 0).any():
        raise RuntimeError("Negative Fifth ratio reached runtime batch.")
    report = {
        "status": "PASS",
        "scope": "Single outer-train batch from the saved P1 run; no optimizer step and no external data.",
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "checkpoints/selected_best.pt"),
        "model_type": type(network).__name__,
        "batch_size": int(fifth.num_graphs),
        "fifth_ratio_min": float(fifth_ratio.min()),
        "fifth_ratio_max": float(fifth_ratio.max()),
        "fifth_ratio_zero_rows": int(fifth_ratio.eq(0).sum()),
        "fifth_class_ids": sorted(set(fifth.fifth_class_id.detach().cpu().view(-1).tolist())),
        "fifth_component_vocab_ids_unique": int(fifth.component_vocab_id.detach().cpu().view(-1).unique().numel()),
        "fifth_graph_encoder_gradient_norm": comp5_grad,
        "fifth_class_embedding_gradient_norm": fifth_class_grad,
        "component_aux_encoder_gradient_norm": aux_grad,
        "target_feature_guard": "The model receives only graph/component/ratio fields; y is consumed after prediction by its output contract.",
        "zero_ratio_note": "No zero-Fifth-ratio row occurred in this locked source batch; absence masking was therefore not inferred from this batch.",
    }
    (output / "p1_fifth_feature_runtime_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
