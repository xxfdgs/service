#!/usr/bin/env python3
"""Train train-only linear probes on frozen GraphGPS branch representations."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def prepare_batches(items, split: str, device: torch.device):
    for batch, suffix in zip(items, ("", "_2", "_3", "_4", "_5")):
        batch.split = split + suffix
        batch.to(device)
    return items


class FrozenRepresentationExtractor:
    """Capture frozen branch inputs from one normal model forward pass."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.graph_parts: list[torch.Tensor] = []
        self.fused: torch.Tensor | None = None
        self.handles = [
            model.gnn.register_forward_hook(self._graph_hook),
            model.FC_layers[0].register_forward_pre_hook(self._fusion_hook),
        ]

    def _graph_hook(self, _module, _inputs, output) -> None:
        self.graph_parts.append(self.model.pooling_fun(output.x, output.batch).detach().cpu())

    def _fusion_hook(self, _module, inputs) -> None:
        self.fused = inputs[0].detach().cpu()

    def start(self) -> None:
        self.graph_parts = []
        self.fused = None

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.graph_parts) != 5 or self.fused is None:
            raise RuntimeError(f"Expected five graph branches and a fused input, got {len(self.graph_parts)} / {self.fused is not None}")
        return torch.cat(self.graph_parts, dim=1).numpy(), self.fused.numpy()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def collect_split(model, core, loaders, split: str, device: torch.device):
    extractor = FrozenRepresentationExtractor(core)
    results = {"graph": [], "descriptor": [], "formula": [], "fused": [], "label": [], "source_index": []}
    try:
        model.eval()
        with torch.no_grad():
            for batches in zip(*[group[{"train": 0, "val": 1, "test": 2}[split]] for group in loaders]):
                batches = prepare_batches(list(batches), split, device)
                descriptor = torch.cat([batch.mordred_feat.view(batch.num_graphs, -1).float() for batch in batches], dim=1)
                formula = torch.cat([core._ratio_features(batch.ratio).float() for batch in batches], dim=1)
                extractor.start()
                _prediction, label = model(*batches)
                graph, fused = extractor.result()
                results["graph"].append(graph)
                results["descriptor"].append(descriptor.detach().cpu().numpy())
                results["formula"].append(formula.detach().cpu().numpy())
                results["fused"].append(fused)
                results["label"].append(label.detach().cpu().reshape(-1, 4).numpy())
                results["source_index"].append(batches[0].sample_uid.detach().cpu().numpy().reshape(-1))
    finally:
        extractor.close()
    return {key: np.vstack(value) if key != "source_index" else np.concatenate(value) for key, value in results.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite probe directory: {output_dir}")
    (output_dir / "cache").mkdir(parents=True)

    set_cfg_gps(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(args.config.resolve()), opts=[]))
    cfg.run_dir = str(output_dir)
    cfg.out_dir = str(output_dir)
    cfg.dataset.dir = str(output_dir / "cache")
    cfg.dataset.cache_tag = f"frozen-branch-probe-{args.fold}"
    cfg.dataset.cache_refresh = True
    configure_determinism(int(cfg.seed), bool(cfg.train.deterministic))
    with (output_dir / "cache_build.log").open("w") as cache_log:
        with contextlib.redirect_stdout(cache_log), contextlib.redirect_stderr(cache_log):
            loaders = create_loader_5()
    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model = create_model_gps().to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    train, validation = (collect_split(model, model.model, loaders, split, device) for split in ("train", "val"))

    manifest = pd.read_csv(cfg.train.manifest_path, dtype={"sample_id": str})
    lookup = dict(zip(manifest.original_row_index.astype(int), manifest.sample_id.astype(str)))
    metric_rows, prediction_rows = [], []
    for branch in ("graph", "descriptor", "formula", "fused"):
        for index, target in enumerate(TARGETS):
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            estimator.fit(train[branch], train["label"][:, index])
            prediction = estimator.predict(validation[branch])
            truth = validation["label"][:, index]
            metric_rows.append({"fold": args.fold, "branch": branch, "target": target, "split": "validation",
                                "n_train": len(train["label"]), "n_validation": len(truth), "ridge_alpha": 1.0,
                                "mae": float(mean_absolute_error(truth, prediction) * 100),
                                "r2": float(r2_score(truth, prediction)),
                                "spearman": float(spearmanr(truth, prediction).statistic),
                                "prediction_std": float(prediction.std(ddof=1) * 100),
                                "target_std": float(truth.std(ddof=1) * 100)})
            for source_index, y_true, y_pred in zip(validation["source_index"], truth, prediction):
                prediction_rows.append({"fold": args.fold, "branch": branch, "target": target,
                                        "sample_id": lookup[int(source_index)], "source_index": int(source_index),
                                        "split": "validation", "y_true": float(y_true * 100), "y_pred": float(y_pred * 100)})
    pd.DataFrame(metric_rows).to_csv(output_dir / "branch_probe_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "branch_probe_predictions.csv", index=False)
    (output_dir / "probe_settings.json").write_text(json.dumps({
        "fold": args.fold, "checkpoint": str(args.checkpoint.resolve()), "alpha": 1.0,
        "fit_split": "train", "evaluation_split": "validation", "outer_test_used": False,
    }, indent=2) + "\n")
    print(pd.DataFrame(metric_rows).to_csv(index=False))


if __name__ == "__main__":
    main()
