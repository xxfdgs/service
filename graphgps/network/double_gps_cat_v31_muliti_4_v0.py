import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gps_layer import GPSLayer
import torch.nn as nn


def _redesign_dropout(value):
    """Return the explicitly configured redesign dropout probability."""
    if value in (None, 'current'):
        return 0.0
    return float(value)


def _redesign_norm(dim, value):
    """Keep the default redesign normalization as a no-op for clean ablations."""
    if value in (None, 'current', 'none'):
        return nn.Identity()
    if value == 'layernorm':
        return nn.LayerNorm(dim)
    raise ValueError(f'Unsupported branch normalization: {value}')


class RedesignFusion(nn.Module):
    """Fuse graph, descriptor, and formula representations for new variants.

    This module deliberately exists only outside the historical baseline path,
    so loading an existing checkpoint remains strict and parameter-identical.
    """

    def __init__(self, graph_dim, descriptor_dim, formula_dim, fusion_dim,
                 fusion_type, branch_normalization='current', dropout='current'):
        super().__init__()
        self.fusion_type = fusion_type
        self.fusion_dim = fusion_dim
        self.graph_project = nn.Sequential(
            nn.Linear(graph_dim, fusion_dim),
            _redesign_norm(fusion_dim, branch_normalization),
            nn.Dropout(_redesign_dropout(dropout)),
        )
        self.descriptor_project = nn.Sequential(
            nn.Linear(descriptor_dim, fusion_dim),
            _redesign_norm(fusion_dim, branch_normalization),
            nn.Dropout(_redesign_dropout(dropout)),
        )
        self.formula_project = nn.Sequential(
            nn.Linear(formula_dim, fusion_dim),
            _redesign_norm(fusion_dim, branch_normalization),
            nn.Dropout(_redesign_dropout(dropout)),
        )
        raw_dim = graph_dim + descriptor_dim + formula_dim

        if fusion_type == 'softmax_sum':
            self.gate = nn.Linear(raw_dim, 3)
            self.output_dim = fusion_dim
        elif fusion_type == 'concat':
            self.output_dim = fusion_dim * 3
        elif fusion_type == 'concat_mlp':
            self.concat_mlp = nn.Sequential(
                nn.Linear(fusion_dim * 3, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.SiLU(),
                nn.Dropout(_redesign_dropout(dropout)),
                nn.Linear(fusion_dim, fusion_dim),
            )
            self.output_dim = fusion_dim
        elif fusion_type == 'residual':
            self.residual_mlp = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.SiLU(),
                nn.Dropout(_redesign_dropout(dropout)),
                nn.Linear(fusion_dim, fusion_dim),
            )
            nn.init.zeros_(self.residual_mlp[-1].weight)
            nn.init.zeros_(self.residual_mlp[-1].bias)
            self.output_dim = fusion_dim
        elif fusion_type == 'gated_concat':
            self.gate = nn.Sequential(
                nn.Linear(fusion_dim * 3, fusion_dim),
                nn.Sigmoid(),
            )
            self.concat_project = nn.Linear(fusion_dim * 3, fusion_dim)
            self.output_dim = fusion_dim
        else:
            raise ValueError(f'Unsupported fusion type: {fusion_type}')

    def forward(self, graph_input, descriptor_input, formula_input):
        graph = self.graph_project(graph_input)
        descriptor = self.descriptor_project(descriptor_input)
        formula = self.formula_project(formula_input)
        concatenated = torch.cat([graph, descriptor, formula], dim=1)
        diagnostics = {
            'graph_branch': graph.detach(),
            'descriptor_branch': descriptor.detach(),
            'formula_branch': formula.detach(),
        }

        if self.fusion_type == 'softmax_sum':
            weights = torch.softmax(
                self.gate(torch.cat([graph_input, descriptor_input, formula_input], dim=1)),
                dim=1,
            )
            fused = (weights[:, 0:1] * graph + weights[:, 1:2] * descriptor
                     + weights[:, 2:3] * formula)
            diagnostics['fusion_weights'] = weights.detach()
        elif self.fusion_type == 'concat':
            fused = concatenated
        elif self.fusion_type == 'concat_mlp':
            fused = self.concat_mlp(concatenated)
        elif self.fusion_type == 'residual':
            fused = graph + self.residual_mlp(torch.cat([descriptor, formula], dim=1))
        else:  # gated_concat
            gate = self.gate(concatenated)
            fused = gate * self.concat_project(concatenated) + (1.0 - gate) * graph
            diagnostics['feature_gate'] = gate.detach()

        diagnostics['fused'] = fused.detach()
        return fused, diagnostics


class RedesignHead(nn.Module):
    """Prediction-head variants used by the fusion/head redesign experiment."""

    def __init__(self, dim_in, dim_out, head_type, hidden_dim, dropout='current'):
        super().__init__()
        self.head_type = head_type
        hidden_dim = int(hidden_dim)
        p = _redesign_dropout(dropout)
        if head_type == 'linear':
            self.head = nn.Linear(dim_in, dim_out)
        elif head_type in ('two_layer', 'baseline'):
            self.head = nn.Sequential(
                nn.Linear(dim_in, hidden_dim), nn.SiLU(), nn.Dropout(p),
                nn.Linear(hidden_dim, dim_out),
            )
        elif head_type in ('residual', 'residual_head'):
            self.base = nn.Linear(dim_in, dim_out)
            self.delta = nn.Sequential(
                nn.Linear(dim_in, hidden_dim), nn.SiLU(), nn.Dropout(p),
                nn.Linear(hidden_dim, dim_out),
            )
            nn.init.zeros_(self.delta[-1].weight)
            nn.init.zeros_(self.delta[-1].bias)
        elif head_type == 'target_specific':
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim_in, hidden_dim), nn.SiLU(), nn.Dropout(p),
                    nn.Linear(hidden_dim, 1),
                ) for _ in range(dim_out)
            ])
        else:
            raise ValueError(f'Unsupported head type: {head_type}')

    def forward(self, fused):
        diagnostics = {'head_input': fused.detach()}
        if self.head_type in ('linear', 'two_layer', 'baseline'):
            pred = self.head(fused)
        elif self.head_type in ('residual', 'residual_head'):
            pred = self.base(fused) + self.delta(fused)
        else:
            pred = torch.cat([head(fused) for head in self.heads], dim=1)
        diagnostics['head_output'] = pred.detach()
        return pred, diagnostics


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
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

        ####
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == self.dim_in, \
            "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
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
        self.layers = torch.nn.Sequential(*layers)
        ####

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class Double_gps(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """

    def __init__(self, dim_in, dim_out):
        super(Double_gps, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

        ####
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == self.dim_in, \
            "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
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
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('GPSDoubleModel_multi4_cat_v0')
class GPSModel(torch.nn.Module):
    """
    GPS model for 5-component formulation property prediction.

    核心修改：
    1. 不再使用 prefix_emb * graph_emb，避免低含量组分的结构信息被比例压低。
    2. 每个组分使用：图结构嵌入 + 比例多尺度嵌入 + 组分身份嵌入。
    3. ratio = 0 时，该组分嵌入置零，表示该组分不存在。
    4. 第1-4组分作为主体基体环境。
    5. 第5组分作为低含量强效调控组分，额外增加 additive_delta 修正分支。
    6. 输出层后不再接 activation，避免预测范围被压缩。
    7. 多性质预测时，每个性质使用独立的三分支融合权重。
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.gnn = Double_gps(dim_in, dim_out)

        # ---------------------------------------------------------
        # 1. 基础维度设置
        # ---------------------------------------------------------
        self.hidden_dim = cfg.gt.dim_hidden

        # 优先使用 cfg.property_num；如果没有，则使用外部传入 dim_out
        self.dim_out = int(getattr(cfg, "property_num", dim_out))

        if self.dim_out not in [1, 2, 3, 4]:
            raise ValueError(
                f"Unexpected property number: {self.dim_out}. "
                f"Please check cfg.property_num or dim_out."
            )

        # 每个比例扩展为4个特征：
        # [r, sqrt(r), log(1 + 100r) / log(101), is_present]
        self.ratio_feat_dim = 4

        # ---------------------------------------------------------
        # 2. 比例编码器
        # ---------------------------------------------------------
        self.ratio_encoder = nn.Sequential(
            nn.Linear(self.ratio_feat_dim, self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.ReLU()
        )

        # ---------------------------------------------------------
        # 3. 组分身份嵌入
        # ---------------------------------------------------------
        # 5个组分分别对应 id = 0, 1, 2, 3, 4
        # 这样模型可以区分同一分子出现在不同组分位置时的不同意义。
        self.component_type_emb = nn.Embedding(5, self.hidden_dim)

        if cfg.use_component_aux_features:
            self.aux_feature_encoder = nn.Sequential(
                nn.Linear(cfg.component_aux_dim, self.hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            )

        # 用于稳定 graph_emb + ratio_emb + type_emb 的融合
        self.component_norm = nn.LayerNorm(self.hidden_dim)

        # ---------------------------------------------------------
        # 4. 主预测分支
        # ---------------------------------------------------------
        # 输入包括：
        # 5个组分嵌入：5 * hidden_dim
        # 5个组分比例特征：5 * ratio_feat_dim
        self.mordred_feature_dim = cfg.mordred_feature_dim if cfg.use_mordred_features else 0
        fusion_input_dim = (self.hidden_dim * 5 + self.ratio_feat_dim * 5
                            + self.mordred_feature_dim * 5)

        self.FC_layers = nn.ModuleList([
            nn.Linear(fusion_input_dim, 256, bias=True),
            nn.Linear(256, 256, bias=True),
            nn.Linear(256, self.dim_out, bias=True)
        ])

        # 直接预测分支
        self.FC_layers_2mlp = nn.ModuleList([
            nn.Linear(fusion_input_dim, self.dim_out, bias=True)
        ])

        # 中间层预测分支
        self.FC_layers_midle_mlp = nn.ModuleList([
            nn.Linear(256, self.dim_out, bias=True)
        ])

        self.activation = register.act_dict[cfg.gnn.act]
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

        # ---------------------------------------------------------
        # 5. 性质特异性分支融合权重
        # ---------------------------------------------------------
        # 原代码只输出3个权重，并对所有性质共用。
        # 这里输出 dim_out * 3 个权重，使每个性质都有自己的三分支融合权重。
        self.branch_weight_mlp = nn.Sequential(
            nn.Linear(self.ratio_feat_dim * 5, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, self.dim_out * 3, bias=True)
        )

        # ---------------------------------------------------------
        # 6. 第5组分低含量强效修正分支
        # ---------------------------------------------------------
        # 前4组分形成主体基体环境 matrix_context。
        self.matrix_context_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.ReLU()
        )

        # 第5组分修正项：
        # 输入 = 主体环境 + 第5组分嵌入 + 第5组分比例特征
        self.additive_delta_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.ratio_feat_dim, 128, bias=True),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, self.dim_out, bias=True)
        )

        # -----------------------------------------------------------------
        # 7. Optional fusion/head redesign interface.
        #
        # The conditional is intentionally after every historical module and
        # does not instantiate any new parameter on the legacy path.  Thus a
        # legacy checkpoint sees the same state_dict keys and forward path.
        # -----------------------------------------------------------------
        self.fusion_type = str(getattr(cfg.model, 'fusion_type', 'softmax_sum'))
        self.head_type = str(getattr(cfg.model, 'head_type', 'baseline'))
        self.architecture_name = str(
            getattr(cfg.model, 'architecture_name', 'legacy_baseline'))
        self.validate_redesign_inputs = bool(
            getattr(cfg.model, 'validate_redesign_inputs', False))
        self.legacy_baseline = (
            self.fusion_type == 'softmax_sum'
            and self.head_type == 'baseline'
            and self.architecture_name == 'legacy_baseline'
        )
        # Group A must alter only the prediction head.  The historical model
        # gates *prediction branches*, rather than embeddings, so retaining
        # that exact fusion requires keeping the direct/middle branches and
        # branch-weight MLP active and replacing only pred_main.
        self.head_ablation_with_legacy_fusion = (
            not self.legacy_baseline
            and self.fusion_type == 'softmax_sum'
            and self.architecture_name.startswith('A_')
        )
        self.last_diagnostics = {}

        if self.head_ablation_with_legacy_fusion:
            head_hidden_dim = getattr(cfg.model, 'head_hidden_dim', None)
            head_hidden_dim = self.hidden_dim if head_hidden_dim is None else int(head_hidden_dim)
            self.redesign_head = RedesignHead(
                dim_in=fusion_input_dim,
                dim_out=self.dim_out,
                head_type=self.head_type,
                hidden_dim=head_hidden_dim,
                dropout=getattr(cfg.model, 'head_dropout', 'current'),
            )
        elif not self.legacy_baseline:
            descriptor_dim = max(1, self.mordred_feature_dim * 5)
            fusion_dim = getattr(cfg.model, 'fusion_hidden_dim', None)
            fusion_dim = self.hidden_dim if fusion_dim is None else int(fusion_dim)
            head_hidden_dim = getattr(cfg.model, 'head_hidden_dim', None)
            head_hidden_dim = fusion_dim if head_hidden_dim is None else int(head_hidden_dim)
            self.redesign_fusion = RedesignFusion(
                graph_dim=self.hidden_dim * 5,
                descriptor_dim=descriptor_dim,
                formula_dim=self.ratio_feat_dim * 5,
                fusion_dim=fusion_dim,
                fusion_type=self.fusion_type,
                branch_normalization=getattr(
                    cfg.model, 'branch_normalization', 'current'),
                dropout=getattr(cfg.model, 'fusion_dropout', 'current'),
            )
            self.redesign_head = RedesignHead(
                dim_in=self.redesign_fusion.output_dim,
                dim_out=self.dim_out,
                head_type=self.head_type,
                hidden_dim=head_hidden_dim,
                dropout=getattr(cfg.model, 'head_dropout', 'current'),
            )

    def _ratio_features(self, ratio):
        """
        将单个组分比例 r 扩展为多尺度比例特征。

        输入:
            ratio: [B] 或 [B, 1]

        输出:
            ratio_feat: [B, 4]

        第1列: r
        第2列: sqrt(r)
        第3列: log(1 + 100r) / log(101)
        第4列: is_present，r > 0 时为1，否则为0
        """
        r = ratio.view(-1, 1).float()

        # 防止异常比例进入模型
        r = torch.clamp(r, min=0.0, max=1.0)

        # ratio=0 时，sqrt_r 严格为0
        sqrt_r = torch.sqrt(r)

        log_base = torch.log(
            torch.tensor(101.0, device=r.device, dtype=r.dtype)
        )
        log_r = torch.log1p(100.0 * r) / log_base

        is_present = (r > 0).float()

        ratio_feat = torch.cat([r, sqrt_r, log_r, is_present], dim=1)
        return ratio_feat

    def _apply_index(self, batch):
        """
        从 batch 中读取多性质标签。

        cfg.property_num = 4 时，标签字段为 y, y1, y2, y3
        cfg.property_num = 3 时，标签字段为 y, y1, y2
        cfg.property_num = 2 时，标签字段为 y, y1
        cfg.property_num = 1 时，标签字段为 y
        """
        pred = batch.graph_feature

        if self.dim_out == 4:
            y_names = ["y", "y1", "y2", "y3"]
        elif self.dim_out == 3:
            y_names = ["y", "y1", "y2"]
        elif self.dim_out == 2:
            y_names = ["y", "y1"]
        elif self.dim_out == 1:
            y_names = ["y"]
        else:
            raise ValueError(f"Unexpected dim_out: {self.dim_out}")

        ys = []
        for name in y_names:
            if not hasattr(batch, name):
                raise AttributeError(
                    f"Batch does not have label field `{name}`. "
                    f"Current property number is {self.dim_out}."
                )
            y = getattr(batch, name)
            y = y.view(-1)
            ys.append(y)

        label = torch.stack(ys, dim=1)
        return pred, label

    def _encode_one_component(self, rep, component_index):
        """
        对单个组分进行编码。

        输入:
            rep: self.gnn(data_i) 的输出
            component_index: 0-4

        输出:
            comp_emb: [B, hidden_dim]
            ratio_feat: [B, 4]

        关键逻辑:
            ratio = 0 时，is_present = 0。
            此时 comp_emb 会被置零，表示该组分不存在。
        """
        # 图结构嵌入
        graph_emb = self.pooling_fun(rep.x, rep.batch)
        # [B, hidden_dim]

        # 比例多尺度特征
        ratio_feat = self._ratio_features(rep.ratio)
        # [B, 4]

        ratio_emb = self.ratio_encoder(ratio_feat)
        # [B, hidden_dim]

        # 组分身份嵌入
        B = graph_emb.size(0)
        comp_id = torch.full(
            (B,),
            component_index,
            dtype=torch.long,
            device=graph_emb.device
        )
        type_emb = self.component_type_emb(comp_id)
        # [B, hidden_dim]

        if cfg.use_component_aux_features:
            aux_feat = rep.aux_feat.view(B, -1).float()
            if aux_feat.size(1) != cfg.component_aux_dim:
                raise ValueError(
                    f"Expected {cfg.component_aux_dim} auxiliary features, "
                    f"got {aux_feat.size(1)}."
                )
            aux_emb = self.aux_feature_encoder(aux_feat)
        else:
            aux_emb = torch.zeros_like(graph_emb)
        # [B, hidden_dim]

        # 不再使用 prefix_emb * graph_emb
        # 改为图嵌入、比例、组分身份和分子级辅助特征的加和融合
        comp_emb = self.component_norm(graph_emb + ratio_emb + type_emb + aux_emb)

        # ratio = 0 时，该组分嵌入置零
        is_present = ratio_feat[:, 3:4]
        # [B, 1]

        comp_emb = comp_emb * is_present
        # [B, hidden_dim]

        return comp_emb, ratio_feat

    def _validate_redesign_inputs(self, graph_input, descriptor_input,
                                  formula_input, batch_size):
        """Fail early on malformed experiment inputs without changing legacy."""
        expected_graph_dim = self.hidden_dim * 5
        expected_formula_dim = self.ratio_feat_dim * 5
        if graph_input.shape != (batch_size, expected_graph_dim):
            raise ValueError(
                f'Graph input shape {tuple(graph_input.shape)} does not match '
                f'({batch_size}, {expected_graph_dim}).')
        if formula_input.shape != (batch_size, expected_formula_dim):
            raise ValueError(
                f'Formula input shape {tuple(formula_input.shape)} does not match '
                f'({batch_size}, {expected_formula_dim}).')
        if descriptor_input.size(0) != batch_size:
            raise ValueError('Descriptor batch size differs from graph batch size.')
        for name, tensor in {
            'graph': graph_input,
            'descriptor': descriptor_input,
            'formula': formula_input,
        }.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f'Non-finite values detected in {name} input.')

    def forward(self, data1, data2, data3, data4, data5):
        """
        输入:
            data1-data5: 5个组分的 PyG batch data

        每个 data 中需要包含:
            x, edge_index, edge_attr, batch, ratio

        标签字段默认存放在 data1 中:
            y, y1, y2, y3 等
        """
        datas = [data1, data2, data3, data4, data5]

        # ---------------------------------------------------------
        # 1. 五个组分分别经过共享 GPS 编码器
        # ---------------------------------------------------------
        reps = [self.gnn(data) for data in datas]

        component_embs = []
        ratio_feats = []

        for i, rep in enumerate(reps):
            comp_emb, ratio_feat = self._encode_one_component(rep, i)
            component_embs.append(comp_emb)
            ratio_feats.append(ratio_feat)

        # ---------------------------------------------------------
        # 2. 拼接5组分信息
        # ---------------------------------------------------------
        graph_emb_input = torch.cat(component_embs, dim=1)
        # [B, 5 * hidden_dim]

        ratio_input = torch.cat(ratio_feats, dim=1)
        # [B, 5 * 4]

        fusion_parts = [graph_emb_input, ratio_input]
        if cfg.use_mordred_features:
            mordred_input = torch.cat([
                rep.mordred_feat.view(graph_emb_input.size(0), -1).float()
                for rep in reps
            ], dim=1)
            fusion_parts.append(mordred_input)
        fusion_input = torch.cat(fusion_parts, dim=1)
        # [B, 5 * hidden_dim + 20]

        # Kept outside the historical fusion vector only as the descriptor
        # branch input for redesign variants.  It does not alter legacy data.
        if not cfg.use_mordred_features:
            mordred_input = torch.zeros(
                graph_emb_input.size(0), 1,
                dtype=graph_emb_input.dtype,
                device=graph_emb_input.device,
            )

        # ---------------------------------------------------------
        # 3. 历史三分支预测，或新的融合/预测头接口
        # ---------------------------------------------------------
        if self.legacy_baseline:
            # Do not refactor this block: it is the historical checkpoint
            # forward path used by the strict baseline-equivalence test.
            h1 = self.FC_layers[0](fusion_input)
            h1 = self.activation()(h1)

            h2 = self.FC_layers[1](h1)
            h2 = self.activation()(h2)

            # 输出层后不接 activation
            pred_main = self.FC_layers[2](h2)
            # [B, dim_out]

            # 输出层后不接 activation
            pred_direct = self.FC_layers_2mlp[0](fusion_input)
            # [B, dim_out]

            pred_middle = self.FC_layers_midle_mlp[0](h1)
            # [B, dim_out]

            branch_logits = self.branch_weight_mlp(ratio_input)
            # [B, dim_out * 3]

            branch_logits = branch_logits.view(-1, self.dim_out, 3)
            # [B, dim_out, 3]

            # This opt-in path is solely a diagnostic control for isolating
            # softmax-gating collapse.  The default remains the historical
            # learnable softmax used by all existing configs and checkpoints.
            if cfg.diagnostic_uniform_fusion:
                branch_weights = torch.full_like(branch_logits, 1.0 / 3.0)
            else:
                branch_weights = torch.softmax(branch_logits, dim=-1)
            # [B, dim_out, 3]

            pred_stack = torch.stack(
                [pred_main, pred_direct, pred_middle],
                dim=-1
            )
            # [B, dim_out, 3]

            pred = torch.sum(pred_stack * branch_weights, dim=-1)
            # [B, dim_out]
            self.last_diagnostics = {
                'graph_input': graph_emb_input.detach(),
                'descriptor_input': mordred_input.detach(),
                'formula_input': ratio_input.detach(),
                'legacy_branch_predictions': pred_stack.detach(),
                'legacy_branch_weights': branch_weights.detach(),
                'fused': pred.detach(),
            }
        elif self.head_ablation_with_legacy_fusion:
            # Group A control: direct and middle predictions, the ratio-driven
            # branch gate, and its softmax fusion are copied from legacy.  The
            # only changed computation is pred_main via redesign_head.
            if self.validate_redesign_inputs:
                if not hasattr(data1, 'sample_uid'):
                    raise AttributeError('Redesign experiments require sample_uid for alignment.')
                if data1.sample_uid.view(-1).numel() != graph_emb_input.size(0):
                    raise ValueError('sample_uid count differs from graph batch size.')
                self._validate_redesign_inputs(
                    graph_emb_input, mordred_input, ratio_input,
                    graph_emb_input.size(0))
            h1 = self.FC_layers[0](fusion_input)
            h1 = self.activation()(h1)
            h2 = self.FC_layers[1](h1)
            h2 = self.activation()(h2)
            pred_main, head_diagnostics = self.redesign_head(fusion_input)
            pred_direct = self.FC_layers_2mlp[0](fusion_input)
            pred_middle = self.FC_layers_midle_mlp[0](h1)
            branch_logits = self.branch_weight_mlp(ratio_input).view(
                -1, self.dim_out, 3)
            if cfg.diagnostic_uniform_fusion:
                branch_weights = torch.full_like(branch_logits, 1.0 / 3.0)
            else:
                branch_weights = torch.softmax(branch_logits, dim=-1)
            pred_stack = torch.stack([pred_main, pred_direct, pred_middle], dim=-1)
            pred = torch.sum(pred_stack * branch_weights, dim=-1)
            self.last_diagnostics = {
                'graph_input': graph_emb_input.detach(),
                'descriptor_input': mordred_input.detach(),
                'formula_input': ratio_input.detach(),
                'legacy_branch_predictions': pred_stack.detach(),
                'legacy_branch_weights': branch_weights.detach(),
                **head_diagnostics,
                'fused': pred.detach(),
            }
        else:
            if self.validate_redesign_inputs:
                if not hasattr(data1, 'sample_uid'):
                    raise AttributeError('Redesign experiments require sample_uid for alignment.')
                if data1.sample_uid.view(-1).numel() != graph_emb_input.size(0):
                    raise ValueError('sample_uid count differs from graph batch size.')
                self._validate_redesign_inputs(
                    graph_emb_input, mordred_input, ratio_input,
                    graph_emb_input.size(0))
            fused, fusion_diagnostics = self.redesign_fusion(
                graph_emb_input, mordred_input, ratio_input)
            pred, head_diagnostics = self.redesign_head(fused)
            self.last_diagnostics = {
                'graph_input': graph_emb_input.detach(),
                'descriptor_input': mordred_input.detach(),
                'formula_input': ratio_input.detach(),
                **fusion_diagnostics,
                **head_diagnostics,
            }

        # ---------------------------------------------------------
        # 6. 第5组分低含量强效修正
        # ---------------------------------------------------------
        # 前4个组分作为主体基体环境
        matrix_input = torch.cat(component_embs[:4], dim=1)
        # [B, 4 * hidden_dim]

        matrix_context = self.matrix_context_mlp(matrix_input)
        # [B, hidden_dim]

        additive_emb = component_embs[4]
        additive_ratio_feat = ratio_feats[4]

        additive_input = torch.cat(
            [matrix_context, additive_emb, additive_ratio_feat],
            dim=1
        )
        # [B, 2 * hidden_dim + 4]

        additive_delta = self.additive_delta_head(additive_input)
        # [B, dim_out]

        # 第5组分 ratio=0 时，additive_present=0，修正分支不生效。
        # 第5组分 ratio>0 时，即使比例很低，也允许非线性修正。
        additive_present = additive_ratio_feat[:, 3:4]
        # [B, 1]

        pred = pred + cfg.fifth_component_delta_weight * additive_present * additive_delta
        # [B, dim_out]

        # ---------------------------------------------------------
        # 7. 返回 pred 和 label
        # ---------------------------------------------------------
        data1.graph_feature = pred
        pred, label = self._apply_index(data1)

        pred = pred.contiguous().view(-1)
        label = label.contiguous().view(-1)

        return pred, label
