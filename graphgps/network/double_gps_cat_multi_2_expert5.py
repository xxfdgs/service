import math
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
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
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

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == self.dim_in, \
            "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except Exception:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")

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
    Shared graph encoder for 5 components.
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

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == self.dim_in, \
            "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except Exception:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")

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


@register_network('GPSDoubleModel_multi2_cat_expert5')
class GPSModel(torch.nn.Module):
    """
    改进版：
    1. 5个组分共享同一GNN编码器。
    2. ratio不再直接把低占比组分乘小，而是采用残差式调制（FiLM-style）。
    3. 显式加入5组分间交互建模（Transformer mixer）。
    4. 为第5组分增加专门expert分支，增强“低占比但高影响”建模能力。
    5. 两个性质共享主干，但输出时按性质分别做分支加权融合。
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gnn = Double_gps(dim_in, dim_out)
        self.hidden_dim = cfg.gt.dim_hidden
        self.num_components = 5
        self.num_targets = 2
        self.ratio_feat_dim = 4  # [r, sqrt(r), log1p(50r), presence]

        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
        self.activation = nn.SiLU
        self.dropout = getattr(cfg.gt, 'dropout', 0.1)

        # 每个组分一个ratio encoder，避免“低比例=信息被乘没”
        self.ratio_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.ratio_feat_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 2 * self.hidden_dim),
            )
            for _ in range(self.num_components)
        ])

        # 给5个组分加入可学习的位置/组分标识
        self.component_embedding = nn.Parameter(
            torch.zeros(1, self.num_components, self.hidden_dim)
        )
        nn.init.trunc_normal_(self.component_embedding, std=0.02)

        # 5个组分之间显式交互
        mixer_nhead = self._safe_nhead(self.hidden_dim, cfg.gt.n_heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=mixer_nhead,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.component_mixer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 主干分支
        main_in_dim = self.hidden_dim * self.num_components + self.num_components
        self.main_backbone = nn.Sequential(
            nn.Linear(main_in_dim, 512),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(self.dropout),
        )
        self.main_head = nn.Linear(256, self.num_targets)

        # 直接分支：保留一个浅层shortcut，防止过拟合时主干过深
        self.direct_head = nn.Sequential(
            nn.Linear(main_in_dim, 256),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_targets),
        )

        # 第5组分专门分支：强化低占比高敏感信息
        comp5_in_dim = self.hidden_dim * 2 + self.num_components + self.ratio_feat_dim
        self.comp5_expert = nn.Sequential(
            nn.Linear(comp5_in_dim, 256),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_targets),
        )

        # 每个性质单独学习3个分支的融合权重：[main, direct, comp5]
        self.branch_gate = nn.Sequential(
            nn.Linear(main_in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, self.num_targets * 3),
        )

    @staticmethod
    def _safe_nhead(hidden_dim, target_nhead):
        if hidden_dim % target_nhead == 0:
            return target_nhead
        for h in [8, 4, 2, 1]:
            if hidden_dim % h == 0:
                return h
        return 1

    @staticmethod
    def build_ratio_feature(ratio):
        """
        ratio: [B, 1]
        为低占比区间提供更高分辨率。
        """
        eps = 1e-8
        return torch.cat([
            ratio,
            torch.sqrt(ratio.clamp_min(0.0) + eps),
            torch.log1p(50.0 * ratio.clamp_min(0.0)),
            (ratio > 0).float(),
        ], dim=1)

    def _apply_index(self, batch):
        pred = batch.graph_feature
        y_names = ["y", "y1"]

        ys = []
        for name in y_names:
            y = getattr(batch, name)
            y = y.view(-1)
            ys.append(y)
        label = torch.stack(ys, dim=1)
        return pred, label

    def forward(self, data1, data2, data3, data4, data5):
        datas = [data1, data2, data3, data4, data5]
        reps = [self.gnn(data) for data in datas]

        graph_embs = []
        ratio_list = []
        ratio_feat_list = []

        # Step 1: 每个组分编码 + 比例残差调制
        for i, rep in enumerate(reps):
            graph_emb = self.pooling_fun(rep.x, rep.batch)          # [B, H]
            ratio = rep.ratio.view(-1, 1).float()                   # [B, 1]
            ratio_feat = self.build_ratio_feature(ratio)            # [B, 4]

            gamma_beta = self.ratio_encoders[i](ratio_feat)         # [B, 2H]
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

            # 残差式调制：避免低ratio时信息被直接乘没
            graph_emb = graph_emb * (1.0 + torch.tanh(gamma)) + beta

            graph_embs.append(graph_emb)
            ratio_list.append(ratio)
            ratio_feat_list.append(ratio_feat)

        ratio_cat = torch.cat(ratio_list, dim=1)                   # [B, 5]

        # Step 2: 5组分交互建模
        tokens = torch.stack(graph_embs, dim=1)                    # [B, 5, H]
        tokens = tokens + self.component_embedding
        tokens = self.component_mixer(tokens)                      # [B, 5, H]

        # Step 3: 主干与shortcut分支
        mix_flat = tokens.reshape(tokens.size(0), -1)              # [B, 5H]
        global_feat = torch.cat([mix_flat, ratio_cat], dim=1)      # [B, 5H+5]

        h_main = self.main_backbone(global_feat)                   # [B, 256]
        pred_main = self.main_head(h_main)                         # [B, 2]
        pred_direct = self.direct_head(global_feat)                # [B, 2]

        # Step 4: 第5组分expert分支
        comp5_token = tokens[:, 4, :]                              # [B, H]
        other_context = tokens[:, :4, :].mean(dim=1)              # [B, H]
        comp5_ratio_feat = ratio_feat_list[4]                      # [B, 4]
        comp5_feat = torch.cat(
            [comp5_token, other_context, ratio_cat, comp5_ratio_feat], dim=1
        )                                                          # [B, 2H+5+4]
        pred_comp5 = self.comp5_expert(comp5_feat)                 # [B, 2]

        # Step 5: 对每个性质分别学习三分支融合权重
        gate_logits = self.branch_gate(global_feat)                # [B, 2*3]
        gate_logits = gate_logits.view(-1, self.num_targets, 3)    # [B, 2, 3]
        gate = torch.softmax(gate_logits, dim=-1)

        pred_stack = torch.stack([pred_main, pred_direct, pred_comp5], dim=-1)  # [B, 2, 3]
        pred = (pred_stack * gate).sum(dim=-1)                     # [B, 2]

        data1.graph_feature = pred
        pred, label = self._apply_index(data1)

        pred = pred.contiguous().view(-1)
        label = label.contiguous().view(-1)
        return pred, label
