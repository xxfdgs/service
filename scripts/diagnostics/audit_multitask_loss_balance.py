#!/usr/bin/env python3
"""Audit per-target L1 and gradient contributions on a frozen train batch."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphgps  # noqa: F401,E402
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from graphgps.determinism import configure_determinism  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402

TARGETS = ["EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency"]


def prepare_batches(items, device: torch.device):
    for batch, suffix in zip(items, ("", "_2", "_3", "_4", "_5")):
        batch.split = "train" + suffix
        batch.to(device)
    return items


def norm(module: torch.nn.Module) -> float:
    chunks = [parameter.grad.detach().reshape(-1).float() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(chunks))) if chunks else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    (output_dir / "cache").mkdir(parents=True)

    set_cfg_gps(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(args.config.resolve()), opts=[]))
    cfg.run_dir = str(output_dir)
    cfg.out_dir = str(output_dir)
    cfg.dataset.dir = str(output_dir / "cache")
    cfg.dataset.cache_tag = f"loss-balance-{args.fold}"
    cfg.dataset.cache_refresh = True
    configure_determinism(int(cfg.seed), bool(cfg.train.deterministic))
    with (output_dir / "cache_build.log").open("w") as cache_log:
        with contextlib.redirect_stdout(cache_log), contextlib.redirect_stderr(cache_log):
            loader_groups = create_loader_5()
    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model = create_model_gps().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model_state"], strict=True)
    core = model.model
    modules = {
        "graph_encoder": core.gnn, "ratio_encoder": core.ratio_encoder,
        "main_head": core.FC_layers[2], "direct_head": core.FC_layers_2mlp[0],
        "middle_head": core.FC_layers_midle_mlp[0], "branch_weight_mlp": core.branch_weight_mlp,
    }
    batches = prepare_batches(list(next(zip(*[group[0] for group in loader_groups]))), device)
    model.train()
    prediction, labels = model(*batches)
    losses = [torch.mean(torch.abs(prediction[index::4] - labels[index::4])) for index in range(4)]
    total = torch.stack(losses).sum()
    rows = []
    for index, (target, loss) in enumerate(zip(TARGETS, losses)):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        for name, module in modules.items():
            rows.append({"fold": args.fold, "target": target, "kind": "target_loss_gradient", "module": name,
                         "loss_normalized": float(loss.detach()), "loss_fraction": float(loss.detach() / total.detach()),
                         "n_valid": int(torch.isfinite(labels[index::4]).sum()), "gradient_norm": norm(module)})
    model.zero_grad(set_to_none=True)
    total.backward()
    for name, module in modules.items():
        rows.append({"fold": args.fold, "target": "total", "kind": "summed_loss_gradient", "module": name,
                     "loss_normalized": float(total.detach()), "loss_fraction": 1.0,
                     "n_valid": int(torch.isfinite(labels).sum()), "gradient_norm": norm(module)})
    pd.DataFrame(rows).to_csv(output_dir / "loss_balance_audit.csv", index=False)
    (output_dir / "settings.json").write_text(json.dumps({"fold": args.fold, "checkpoint": str(args.checkpoint.resolve()),
                                                            "split": "train", "batch_index": 0, "outer_test_used": False}, indent=2) + "\n")
    print(pd.DataFrame(rows).to_csv(index=False))


if __name__ == "__main__":
    main()
