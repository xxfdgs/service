#!/usr/bin/env python3
"""Train one five-component multi-task baseline on an explicit input split.

Supported baselines share the same formulation-level prediction head.  GCN,
GIN, MPNN (edge-aware GINE message passing), and Transformer encode every one
of the five molecular graphs with a shared molecular encoder.  MLP instead
uses the five RDKit/Morgan 136-dimensional component vectors.  In all cases
the five molar fractions are appended immediately before the shared head.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, GINEConv, GINConv, TransformerConv, global_mean_pool

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphgps.lrx_add.csv_pyg_five_multi import molecular_aux_features


TARGET_GROUPS = {
    "core4": (["EE_before", "EE_after", "Aerosolization_Efficiency",
               "mRNA_Recovery_Efficiency"], 100.0),
    "norm2": (["Norm_before", "Norm_after"], 1.0),
}
SMILES_COLUMNS = ["IL_SMILE", "HL_SMILE", "Chol_SMILE", "PEG_SMILE", "Fifth_SMILE"]
RATIO_COLUMNS = ["mol%_IL", "mol%_HL", "mol%_Chol", "mol%_PEG", "mol%_Fifth"]
MODELS = ("GCN", "GIN", "MPNN", "Transformer", "MLP")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def safe_molecule(value: object) -> Chem.Mol:
    if pd.isna(value) or str(value).strip() in {"", "nan", "None", "[Fr]"}:
        return Chem.MolFromSmiles("[Fr]")
    molecule = Chem.MolFromSmiles(str(value))
    return molecule if molecule is not None else Chem.MolFromSmiles("[Fr]")


def atom_features(atom: Chem.Atom) -> list[float]:
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetDegree() / 6.0,
        atom.GetFormalCharge() / 5.0,
        float(atom.GetIsAromatic()),
        atom.GetMass() / 250.0,
        float(atom.IsInRing()),
    ]


def bond_features(bond: Chem.Bond) -> list[float]:
    kind = bond.GetBondType()
    return [float(kind == Chem.BondType.SINGLE), float(kind == Chem.BondType.DOUBLE),
            float(kind == Chem.BondType.TRIPLE), float(kind == Chem.BondType.AROMATIC)]


def graph_from_molecule(molecule: Chem.Mol, label: np.ndarray, sample_index: int, ratio: float,
                        tabular: np.ndarray | None = None) -> Data:
    x = torch.tensor([atom_features(atom) for atom in molecule.GetAtoms()], dtype=torch.float)
    edges, attributes = [], []
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_features(bond)
        edges.extend(((left, right), (right, left)))
        attributes.extend((feature, feature))
    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                  if edges else torch.empty((2, 0), dtype=torch.long))
    edge_attr = (torch.tensor(attributes, dtype=torch.float)
                 if attributes else torch.empty((0, 4), dtype=torch.float))
    fields = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr,
              "y": torch.tensor(label, dtype=torch.float).view(1, -1),
              "ratio": torch.tensor([ratio], dtype=torch.float),
              # Do not end this field with ``index``: PyG would increment it
              # while batching as if it were a graph edge index.
              "source_row": torch.tensor([sample_index], dtype=torch.long)}
    if tabular is not None:
        fields["tabular"] = torch.tensor(tabular, dtype=torch.float).view(1, -1)
    return Data(**fields)


class FormulationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, row_indices: list[int], targets: list[str], scale: float):
        self.items: list[tuple[Data, Data, Data, Data, Data]] = []
        for source_index in row_indices:
            row = frame.iloc[source_index]
            label = row[targets].to_numpy(dtype=np.float32) / scale
            molecules = [safe_molecule(row[column]) for column in SMILES_COLUMNS]
            auxiliary = np.concatenate([molecular_aux_features(molecule) for molecule in molecules]).astype(np.float32)
            # Match ``csv_pyg_five_multi``: an absent component-5 ratio is
            # represented as zero rather than allowing NaN to enter a model.
            ratios = np.nan_to_num(row[RATIO_COLUMNS].to_numpy(dtype=np.float32) / 100.0,
                                   nan=0.0, posinf=0.0, neginf=0.0)
            tabular = np.concatenate((auxiliary, ratios)).astype(np.float32)
            graphs = tuple(
                graph_from_molecule(molecule, label, source_index, float(ratios[component]),
                                    tabular if component == 0 else None)
                for component, molecule in enumerate(molecules)
            )
            self.items.append(graphs)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Data, Data, Data, Data, Data]:
        return self.items[index]


def collate_formulations(items: list[tuple[Data, Data, Data, Data, Data]]) -> tuple[Batch, Batch, Batch, Batch, Batch]:
    return tuple(Batch.from_data_list([item[component] for item in items])
                 for component in range(5))  # type: ignore[return-value]


class MolecularEncoder(nn.Module):
    def __init__(self, model_name: str, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.model_name = model_name
        self.node_input = nn.Linear(6, hidden_dim)
        self.edge_input = nn.Linear(4, hidden_dim)
        self.dropout = float(dropout)
        convolutions = []
        for _ in range(layers):
            if model_name == "GCN":
                convolutions.append(GCNConv(hidden_dim, hidden_dim))
            elif model_name == "GIN":
                convolutions.append(GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
            elif model_name == "MPNN":
                convolutions.append(GINEConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
                    edge_dim=hidden_dim))
            elif model_name == "Transformer":
                convolutions.append(TransformerConv(hidden_dim, hidden_dim, heads=1,
                                                     concat=False, edge_dim=hidden_dim,
                                                     dropout=dropout))
            else:
                raise ValueError(f"Unsupported graph baseline: {model_name}")
        self.convolutions = nn.ModuleList(convolutions)
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(layers))

    def forward(self, batch: Batch) -> torch.Tensor:
        value = self.node_input(batch.x.float())
        edge = self.edge_input(batch.edge_attr.float())
        for convolution, norm in zip(self.convolutions, self.norms):
            if self.model_name == "GCN":
                updated = convolution(value, batch.edge_index)
            elif self.model_name == "GIN":
                updated = convolution(value, batch.edge_index)
            else:
                updated = convolution(value, batch.edge_index, edge)
            value = norm(functional.relu(updated))
            value = functional.dropout(value, p=self.dropout, training=self.training)
        return global_mean_pool(value, batch.batch)


class FiveComponentBaseline(nn.Module):
    def __init__(self, model_name: str, output_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.model_name = model_name
        if model_name == "MLP":
            fusion_dim = 5 * 136 + 5
            self.encoder = None
        else:
            self.encoder = MolecularEncoder(model_name, hidden_dim, layers, dropout)
            fusion_dim = 5 * hidden_dim + 5
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim),
        )

    def forward(self, batches: tuple[Batch, Batch, Batch, Batch, Batch]) -> torch.Tensor:
        if self.encoder is None:
            features = batches[0].tabular.float()
        else:
            embeddings = [self.encoder(batch) for batch in batches]
            ratios = torch.cat([batch.ratio.view(batch.num_graphs, 1).float() for batch in batches], dim=1)
            features = torch.cat([*embeddings, ratios], dim=1)
        return self.head(features)


def split_indices(frame: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, list[int]]:
    if "ID" not in frame or "sample_id" not in manifest or "split" not in manifest:
        raise ValueError("Input frame requires ID; manifest requires sample_id and split.")
    ids = frame.ID.astype(str)
    if ids.duplicated().any() or manifest.sample_id.astype(str).duplicated().any():
        raise ValueError("Input IDs and manifest sample_id values must be unique.")
    source = {sample_id: index for index, sample_id in enumerate(ids)}
    membership = {str(row.sample_id): row.split for row in manifest.itertuples(index=False)}
    if set(source) != set(membership):
        raise ValueError("Manifest must cover exactly the input IDs.")
    result = {split: [source[str(row.sample_id)] for row in manifest.itertuples(index=False)
                      if row.split == split] for split in ("train", "val", "test")}
    if not all(result.values()):
        raise ValueError("Every manifest partition must be non-empty.")
    return result


def correlation(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or not np.std(truth) or not np.std(prediction):
        return math.nan
    return float(function(truth, prediction).statistic)


def metric_rows(prediction: np.ndarray, truth: np.ndarray, targets: list[str], scale: float,
                split: str, best_epoch: int, checkpoint: str) -> list[dict[str, object]]:
    rows = []
    for index, target in enumerate(targets):
        y, p = truth[:, index] * scale, prediction[:, index] * scale
        rows.append({"split": split, "target": target, "n": int(len(y)),
                     "mae": float(mean_absolute_error(y, p)),
                     "rmse": float(mean_squared_error(y, p) ** .5),
                     "r2": float(r2_score(y, p)) if np.std(y) else math.nan,
                     "pearson": correlation(pearsonr, y, p),
                     "spearman": correlation(spearmanr, y, p),
                     "best_epoch": int(best_epoch), "checkpoint": checkpoint})
    return rows


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions, labels, indices = [], [], []
    with torch.no_grad():
        for batches in loader:
            batches = tuple(batch.to(device) for batch in batches)
            predictions.append(model(batches).detach().cpu().numpy())
            labels.append(batches[0].y.view(-1, batches[0].y.shape[-1]).detach().cpu().numpy())
            indices.append(batches[0].source_row.view(-1).detach().cpu().numpy())
    return np.vstack(predictions), np.vstack(labels), np.concatenate(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--target-group", choices=tuple(TARGET_GROUPS), required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=50)
    parser.add_argument("--early-stop-min-delta", type=float, default=.001)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.layers < 1 or arguments.batch_size < 1:
        raise ValueError("epochs, layers, and batch-size must be positive.")
    run_dir = arguments.run_dir.resolve()
    if run_dir.exists() and not arguments.resume:
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_seed(arguments.seed)
    frame = pd.read_csv(arguments.input_csv.resolve())
    manifest = pd.read_csv(arguments.split_manifest.resolve(), dtype={"sample_id": str})
    targets, scale = TARGET_GROUPS[arguments.target_group]
    missing = set(["ID", *SMILES_COLUMNS, *RATIO_COLUMNS, *targets]).difference(frame.columns)
    if missing:
        raise ValueError(f"Input CSV lacks required columns: {sorted(missing)}")
    indices = split_indices(frame, manifest)
    datasets = {split: FormulationDataset(frame, rows, targets, scale) for split, rows in indices.items()}
    loaders = {split: DataLoader(dataset, batch_size=arguments.batch_size,
                                 shuffle=split == "train", num_workers=0,
                                 collate_fn=collate_formulations)
               for split, dataset in datasets.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FiveComponentBaseline(arguments.model, len(targets), arguments.hidden_dim,
                                  arguments.layers, arguments.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.lr, weight_decay=arguments.weight_decay)
    def lr_multiplier(epoch: int) -> float:
        if epoch < arguments.warmup_epochs:
            return max(1e-6, float(epoch + 1) / max(1, arguments.warmup_epochs))
        progress = (epoch - arguments.warmup_epochs) / max(1, arguments.epochs - arguments.warmup_epochs)
        return max(1e-3, .5 * (1. + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    state_path = run_dir / "resume_state.pt"
    start_epoch, best_epoch, best_loss, best_state, early_counter = 0, -1, math.inf, None, 0
    if arguments.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_epoch, best_loss = state["next_epoch"], state["best_epoch"], state["best_loss"]
        best_state, early_counter = state["best_state"], state["early_counter"]
    history_path = run_dir / "history.csv"
    if not arguments.resume:
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_loss", "lr", "best_epoch", "best_val_loss", "early_counter"]).writeheader()
        (run_dir / "run_settings.json").write_text(json.dumps({
            "model": arguments.model, "target_group": arguments.target_group, "targets": targets,
            "target_scale": scale, "input_csv": str(arguments.input_csv.resolve()),
            "input_sha256": file_sha256(arguments.input_csv.resolve()),
            "split_manifest": str(arguments.split_manifest.resolve()),
            "manifest_sha256": file_sha256(arguments.split_manifest.resolve()),
            "split_sizes": {key: len(value) for key, value in indices.items()},
            "seed": arguments.seed, "epochs": arguments.epochs, "batch_size": arguments.batch_size,
            "hidden_dim": arguments.hidden_dim, "layers": arguments.layers, "dropout": arguments.dropout,
            "lr": arguments.lr, "weight_decay": arguments.weight_decay,
            "graph_input": "five RDKit molecular graphs + five molar fractions" if arguments.model != "MLP" else None,
            "mlp_input": "five 136D RDKit/Morgan component vectors + five molar fractions" if arguments.model == "MLP" else None,
        }, indent=2) + "\n", encoding="utf-8")
    for epoch in range(start_epoch, arguments.epochs):
        model.train()
        train_losses = []
        for batches in loaders["train"]:
            batches = tuple(batch.to(device) for batch in batches)
            optimizer.zero_grad(set_to_none=True)
            output = model(batches)
            label = batches[0].y.view(-1, len(targets))
            loss = torch.abs(output - label).mean(dim=0).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_prediction, val_truth, _ = predict(model, loaders["val"], device)
        val_loss = float(np.abs(val_prediction - val_truth).mean(axis=0).sum())
        train_loss = float(np.mean(train_losses))
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, best_state, early_counter = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            early_counter += 1
        lr = float(optimizer.param_groups[0]["lr"])
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_loss", "lr", "best_epoch", "best_val_loss", "early_counter"]).writerow(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr,
                 "best_epoch": best_epoch, "best_val_loss": best_loss, "early_counter": early_counter})
        scheduler.step()
        torch.save({"next_epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "best_epoch": best_epoch, "best_loss": best_loss,
                    "best_state": best_state, "early_counter": early_counter}, state_path)
        print(json.dumps({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                          "best_epoch": best_epoch, "best_val_loss": best_loss,
                          "early_counter": early_counter}), flush=True)
        if early_counter >= arguments.early_stop_patience:
            break
    if best_state is None:
        raise RuntimeError("No training epoch completed.")
    checkpoint_path = run_dir / "selected_best.pt"
    torch.save({"model_state": best_state, "best_epoch": best_epoch, "best_validation_loss": best_loss,
                "model": arguments.model, "target_group": arguments.target_group, "targets": targets}, checkpoint_path)
    model.load_state_dict(best_state)
    metric, prediction_rows = [], []
    for split in ("train", "val", "test"):
        output, truth, source_indices = predict(model, loaders[split], device)
        metric.extend(metric_rows(output, truth, targets, scale, split, best_epoch, str(checkpoint_path)))
        for row, source_index in enumerate(source_indices):
            for target_index, target in enumerate(targets):
                prediction_rows.append({"sample_id": str(frame.iloc[int(source_index)].ID),
                                        "source_index": int(source_index), "split": split, "target": target,
                                        "y_true": float(truth[row, target_index] * scale),
                                        "y_pred": float(output[row, target_index] * scale),
                                        "best_epoch": best_epoch, "checkpoint": str(checkpoint_path)})
    pd.DataFrame(metric).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps({
        "model": arguments.model, "target_group": arguments.target_group, "targets": targets,
        "best_epoch": best_epoch, "best_validation_loss_normalized": best_loss,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "selected_checkpoint": str(checkpoint_path), "completed": True,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
