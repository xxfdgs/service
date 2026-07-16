"""
OneHotEmbedGPSModel: Dual-pathway polymer property predictor.

Components 1-4 (IL, HL, Chol, PEG) — low-diversity, near-fixed molecules:
    one-hot encode by molecule identity → learnable dense embeddings.

Component 5 (Fifth) — high-diversity, variable molecules:
    full GraphGPS encoder (GINE+Transformer, no augmented descriptors).

Fusion: concatenate 5 embeddings + 5 ratios → MLP → multi-property predictions.
"""

import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (
    new_layer_config,
    BatchNorm1dNode,
)
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gps_layer import GPSLayer
from graphgps.component_vocab import (
    get_vocab_id,
    build_component_embeddings,
)


# ---------------------------------------------------------------------------
# Component-5 GraphGPS encoder (reuses FeatureEncoder + GPSLayers from GPSModel)
# ---------------------------------------------------------------------------

class Comp5GraphEncoder(nn.Module):
    """Graph encoder for the fifth component — same as GPSModel minus the head."""

    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in

        # Node encoder
        if cfg.dataset.node_encoder:
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1,
                                     has_act=False, has_bias=False, cfg=cfg))
            self.dim_in = cfg.gnn.dim_inner

        # Edge encoder
        if cfg.dataset.edge_encoder:
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1,
                                     has_act=False, has_bias=False, cfg=cfg))

        # Pre-MP layers
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == self.dim_in, \
            "The inner and hidden dims must match."

        # GPS layers
        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except Exception:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")

        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(GPSLayer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                act=cfg.gnn.act,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                log_attn_weights=cfg.train.mode == 'log-attn-weights',
            ))
        self.layers = nn.Sequential(*layers)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


# ---------------------------------------------------------------------------
# Main dual-pathway model
# ---------------------------------------------------------------------------

@register_network('OneHotEmbedGPS')
class OneHotEmbedGPSModel(nn.Module):
    """
    Dual-pathway model for 5-component polymer property prediction.

    Pathway A (components 1-4):
        Identify molecule by atom count → one-hot embedding → ratio modulation.

    Pathway B (component 5):
        Full GraphGPS encoder → graph pooling → graph embedding.

    Fusion:
        Concat(emb1..emb5, ratios) → 3-layer MLP → predictions.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.hidden_dim = cfg.gt.dim_hidden
        self.num_components = 5
        self.out_dim = cfg.property_num

        # --- Pathway A: one-hot embeddings for components 1-4 ---
        self.comp_embeddings = build_component_embeddings(
            self.hidden_dim, cfg.component_vocab_sizes)

        # Ratio-aware modulation for each component 1-4:
        # ratio_feat = [r, sqrt(r), log1p(50r), 1(r>0)] → gamma, beta
        self.ratio_modulators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            )
            for _ in range(4)  # only for comps 1-4
        ])

        # --- Pathway B: GraphGPS encoder for component 5 ---
        self.comp5_encoder = Comp5GraphEncoder(dim_in)
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

        # --- Fusion MLP ---
        # 5 embeddings + 5 ratios = 5*hidden_dim + 5
        self.mordred_feature_dim = cfg.mordred_feature_dim if cfg.use_mordred_features else 0
        mordred_multiplier = 1 if cfg.mordred_fifth_only else self.num_components
        fusion_in_dim = (self.num_components * self.hidden_dim + self.num_components
                         + mordred_multiplier * self.mordred_feature_dim)
        dropout = float(getattr(cfg.gt, 'dropout', 0.0))
        act_fn = nn.SiLU

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, 256),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            act_fn(),
            nn.Linear(64, self.out_dim),
        )

    @staticmethod
    def _build_ratio_feature(ratio):
        """ratio: [B, 1] → ratio_feat: [B, 4]"""
        eps = 1e-8
        return torch.cat([
            ratio,
            torch.sqrt(ratio.clamp_min(0.0) + eps),
            torch.log1p(50.0 * ratio.clamp_min(0.0)),
            (ratio > 0).float(),
        ], dim=1)

    def _encode_components_1_to_4(self, batch, comp_idx):
        """Encode one of the first 4 components via one-hot embedding + ratio modulation.

        Args:
            batch: PyG DataBatch for this component.
            comp_idx: 0-based component index (0=IL, 1=HL, 2=Chol, 3=PEG).

        Returns:
            embedding: [B, hidden_dim]
        """
        B = batch.num_graphs

        # New datasets carry input-derived canonical-SMILES IDs.  Retain the
        # atom-count fallback solely for old processed caches.
        if hasattr(batch, 'component_vocab_id'):
            vocab_ids = batch.component_vocab_id.view(-1).long()
        else:
            num_nodes_per_graph = batch.ptr[1:] - batch.ptr[:-1]
            vocab_ids = get_vocab_id(comp_idx + 1, num_nodes_per_graph)
            vocab_ids[vocab_ids < 0] = 0

        # One-hot embedding lookup
        emb = self.comp_embeddings[f'comp{comp_idx + 1}'](vocab_ids)  # [B, hidden_dim]

        # Ratio modulation
        ratio = batch.ratio.view(B, 1).float()
        ratio_feat = self._build_ratio_feature(ratio)                # [B, 4]
        gamma, beta = self.ratio_modulators[comp_idx](ratio_feat).chunk(2, dim=-1)
        emb = emb * (1.0 + torch.tanh(gamma)) + beta

        return emb, ratio

    def _apply_index(self, batch):
        pred = batch.graph_feature
        out_dim = self.out_dim

        if out_dim == 1:
            label = batch.y.view(-1).unsqueeze(1)
        elif out_dim == 2:
            y0 = getattr(batch, 'y').view(-1)
            y1 = getattr(batch, 'y1').view(-1)
            label = torch.stack([y0, y1], dim=1)
        elif out_dim == 4:
            y0 = getattr(batch, 'y').view(-1)
            y1 = getattr(batch, 'y1').view(-1)
            y2 = getattr(batch, 'y2').view(-1)
            y3 = getattr(batch, 'y3').view(-1)
            label = torch.stack([y0, y1, y2, y3], dim=1)
        elif out_dim == 6:
            y0 = getattr(batch, 'y').view(-1)
            y1 = getattr(batch, 'y1').view(-1)
            y2 = getattr(batch, 'y2').view(-1)
            y3 = getattr(batch, 'y3').view(-1)
            y4 = getattr(batch, 'y4').view(-1)
            y5 = getattr(batch, 'y5').view(-1)
            label = torch.stack([y0, y1, y2, y3, y4, y5], dim=1)
        else:
            label = batch.y.view(-1, 1)

        return pred, label

    def forward(self, data1, data2, data3, data4, data5):
        # --- Pathway A: one-hot embed components 1-4 ---
        emb1, ratio1 = self._encode_components_1_to_4(data1, 0)
        emb2, ratio2 = self._encode_components_1_to_4(data2, 1)
        emb3, ratio3 = self._encode_components_1_to_4(data3, 2)
        emb4, ratio4 = self._encode_components_1_to_4(data4, 3)

        # --- Pathway B: GraphGPS for component 5 ---
        data5_encoded = self.comp5_encoder(data5)
        emb5 = self.pooling_fun(data5_encoded.x, data5_encoded.batch)  # [B, hidden_dim]
        ratio5 = data5.ratio.view(-1, 1).float()

        # --- Fusion ---
        all_ratios = torch.cat([ratio1, ratio2, ratio3, ratio4, ratio5], dim=1)
        all_embs = torch.cat([emb1, emb2, emb3, emb4, emb5], dim=1)
        fusion_parts = [all_embs, all_ratios]
        if cfg.use_mordred_features:
            mordred_batches = [data5] if cfg.mordred_fifth_only else [data1, data2, data3, data4, data5]
            mordred_input = torch.cat([data.mordred_feat.view(data.num_graphs, -1).float() for data in mordred_batches], dim=1)
            fusion_parts.append(mordred_input)
        combined = torch.cat(fusion_parts, dim=1)
        pred = self.fusion(combined)  # [B, out_dim]

        data1.graph_feature = pred
        pred, label = self._apply_index(data1)
        pred = pred.contiguous().view(-1)
        label = label.contiguous().view(-1)
        return pred, label
