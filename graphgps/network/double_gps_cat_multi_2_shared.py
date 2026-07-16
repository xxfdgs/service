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


class FeatureEncoder_double(torch.nn.Module):
    """
    Encoding node and edge features.
    Kept for compatibility with the original file.
    """

    def __init__(self, dim_in):
        super(FeatureEncoder_double, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_inner,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    )
                )
            self.dim_in = cfg.gnn.dim_inner

        if cfg.dataset.edge_encoder:
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_edge,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    )
                )

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == self.dim_in, (
            'The inner and hidden dims must match.'
        )

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except Exception as exc:
            raise ValueError(f'Unexpected layer type: {cfg.gt.layer_type}') from exc

        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(
                GPSLayer(
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
                )
            )
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class Double_gps(torch.nn.Module):
    """
    Shared graph encoder for all 5 components.
    """

    def __init__(self, dim_in, dim_out):
        super(Double_gps, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_inner,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    )
                )
            self.dim_in = cfg.gnn.dim_inner

        if cfg.dataset.edge_encoder:
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_edge,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    )
                )

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == self.dim_in, (
            'The inner and hidden dims must match.'
        )

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except Exception as exc:
            raise ValueError(f'Unexpected layer type: {cfg.gt.layer_type}') from exc

        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(
                GPSLayer(
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
                )
            )
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('GPSDoubleModel_multi2_cat_shared')
class GPSModel(torch.nn.Module):
    """
    5-component GPS model for 2-property regression.

    Main design changes relative to the original version:
    1) Remove the slot-specific treatment of component 5.
    2) Use a shared ratio encoder for all components.
    3) Replace ratio multiplicative suppression with residual FiLM modulation.
    4) Let the model learn token importance from structure + ratio jointly.
    5) Use permutation-friendly token mixing (no component-specific expert branch).
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gnn = Double_gps(dim_in, dim_out)
        self.hidden_dim = cfg.gt.dim_hidden
        self.num_components = 5
        self.out_dim = 2

        act_layer = register.act_dict[cfg.gnn.act]
        dropout = float(getattr(cfg.gt, 'dropout', 0.0))

        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
        self.token_norm = nn.LayerNorm(self.hidden_dim)
        self.post_mix_norm = nn.LayerNorm(self.hidden_dim)

        # Shared ratio encoder for all components.
        # Input features: [r, sqrt(r), log1p(50r), 1(r>0)]
        self.ratio_encoder = nn.Sequential(
            nn.Linear(4, 64, bias=True),
            act_layer(),
            nn.Linear(64, 2 * self.hidden_dim, bias=True),
        )

        # Mixer over 5 component tokens. No slot-specific parameters are introduced.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=cfg.gt.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.token_mixer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Learn token importance directly from mixed token representations.
        self.token_importance = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, bias=True),
            act_layer(),
            nn.Linear(self.hidden_dim // 2, 1, bias=True),
        )

        # Three shared, permutation-friendly prediction branches.
        summary_dim = self.hidden_dim * 3 + self.num_components
        self.attn_branch = nn.Sequential(
            nn.Linear(summary_dim, 256, bias=True),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(256, self.out_dim, bias=True),
        )
        self.mean_branch = nn.Sequential(
            nn.Linear(summary_dim, 256, bias=True),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(256, self.out_dim, bias=True),
        )
        self.max_branch = nn.Sequential(
            nn.Linear(summary_dim, 256, bias=True),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(256, self.out_dim, bias=True),
        )

        # Branch fusion weights are learned from structure + ratio, per property.
        self.branch_gate = nn.Sequential(
            nn.Linear(summary_dim, 128, bias=True),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(128, self.out_dim * 3, bias=True),
        )

    @staticmethod
    def _build_ratio_feature(ratio: torch.Tensor) -> torch.Tensor:
        """
        ratio: [B, 1]
        return: [B, 4]
        """
        eps = 1e-8
        return torch.cat(
            [
                ratio,
                torch.sqrt(torch.clamp(ratio, min=0.0) + eps),
                torch.log1p(50.0 * torch.clamp(ratio, min=0.0)),
                (ratio > 0).float(),
            ],
            dim=1,
        )

    def _apply_index(self, batch):
        pred = batch.graph_feature
        y_names = ['y', 'y1']

        ys = []
        for name in y_names:
            y = getattr(batch, name)
            y = y.view(-1)
            ys.append(y)

        label = torch.stack(ys, dim=1)
        return pred, label

    def _encode_component(self, rep):
        """
        Encode one component into one token using shared graph encoder + shared ratio encoder.
        """
        graph_emb = self.pooling_fun(rep.x, rep.batch)
        ratio = rep.ratio.unsqueeze(1)
        ratio_feat = self._build_ratio_feature(ratio)
        gamma, beta = self.ratio_encoder(ratio_feat).chunk(2, dim=-1)
        token = graph_emb * (1.0 + gamma) + beta
        token = self.token_norm(token)
        return token, ratio

    def forward(self, data1, data2, data3, data4, data5):
        datas = [data1, data2, data3, data4, data5]
        reps = [self.gnn(data) for data in datas]

        tokens = []
        ratios = []
        for rep in reps:
            token, ratio = self._encode_component(rep)
            tokens.append(token)
            ratios.append(ratio)

        # [B, 5, H], [B, 5]
        tokens = torch.stack(tokens, dim=1)
        ratio_cat = torch.cat(ratios, dim=1)

        # Cross-component interaction learning.
        tokens = self.token_mixer(tokens)
        tokens = self.post_mix_norm(tokens)

        # Learned token importance, entirely shared across all slots.
        token_scores = self.token_importance(tokens)          # [B, 5, 1]
        token_alpha = torch.softmax(token_scores, dim=1)     # [B, 5, 1]

        attn_pool = (token_alpha * tokens).sum(dim=1)        # [B, H]
        mean_pool = tokens.mean(dim=1)                       # [B, H]
        max_pool = tokens.max(dim=1).values                  # [B, H]

        # Shared summary used by each branch.
        summary_feat = torch.cat([attn_pool, mean_pool, max_pool, ratio_cat], dim=1)

        pred_attn = self.attn_branch(summary_feat)           # [B, 2]
        pred_mean = self.mean_branch(summary_feat)           # [B, 2]
        pred_max = self.max_branch(summary_feat)             # [B, 2]

        gate_logits = self.branch_gate(summary_feat)         # [B, 2*3]
        gate_logits = gate_logits.view(-1, self.out_dim, 3) # [B, 2, 3]
        gate_weights = torch.softmax(gate_logits, dim=-1)

        pred_stack = torch.stack([pred_attn, pred_mean, pred_max], dim=-1)  # [B, 2, 3]
        pred = (pred_stack * gate_weights).sum(dim=-1)                       # [B, 2]

        data1.graph_feature = pred
        pred, label = self._apply_index(data1)

        pred = pred.contiguous().view(-1)
        label = label.contiguous().view(-1)
        return pred, label
