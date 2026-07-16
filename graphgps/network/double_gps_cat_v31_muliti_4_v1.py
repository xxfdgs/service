import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gps_layer import GPSLayer
import torch.nn as nn


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


@register_network('GPSDoubleModel_multi4_cat_v1')

class GPSModel(torch.nn.Module):
    """
    GPS model for 5-component formulation property prediction.

    当前版本说明：
    1. 采用方案A：删除原来的三分支融合输出层，改为轻量 MLP 预测头。
    2. 采用方案B：保留第5组分低含量修正分支，但降低自由度。
    3. 第5组分修正项由比例强度 additive_strength 控制，而不是只由 is_present 控制。
    4. 输出层后不接 activation，适合连续性质回归。
    5. 为兼容现有 train_five_multi.py 和 logger.py，最终返回一维 pred 和 label。
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        # ---------------------------------------------------------
        # 1. GPS 图编码器
        # ---------------------------------------------------------
        self.gnn = Double_gps(dim_in, dim_out)

        # ---------------------------------------------------------
        # 2. 基础维度设置
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
        # 3. 比例编码器
        # ---------------------------------------------------------
        self.ratio_encoder = nn.Sequential(
            nn.Linear(self.ratio_feat_dim, self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.ReLU()
        )

        # ---------------------------------------------------------
        # 4. 组分身份嵌入
        # ---------------------------------------------------------
        # 5个组分分别对应 id = 0, 1, 2, 3, 4。
        # 如果5个组分的位置有固定化学意义，可以保留该设计。
        self.component_type_emb = nn.Embedding(5, self.hidden_dim)

        # 稳定 graph_emb + ratio_emb + type_emb 的融合
        self.component_norm = nn.LayerNorm(self.hidden_dim)

        # ---------------------------------------------------------
        # 5. 轻量化主预测头：方案A
        # ---------------------------------------------------------
        # 输入包括：
        # 5个组分嵌入：5 * hidden_dim
        # 5个组分比例特征：5 * ratio_feat_dim = 20
        fusion_input_dim = self.hidden_dim * 5 + self.ratio_feat_dim * 5

        self.pred_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128, bias=True),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, self.dim_out, bias=True)
        )

        # ---------------------------------------------------------
        # 6. 第5组分低含量弱修正分支：方案B
        # ---------------------------------------------------------
        # 前4组分形成主体基体环境 matrix_context。
        self.matrix_context_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.ReLU()
        )

        # 第5组分修正项：
        # 输入 = 主体环境 + 第5组分嵌入 + 第5组分比例特征
        self.additive_delta_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.ratio_feat_dim, 64, bias=True),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, self.dim_out, bias=True)
        )

        # 是否启用第5组分修正分支
        # 默认启用；如果想关闭，可在 cfg 中设置 use_additive_delta=False
        self.use_additive_delta = bool(getattr(cfg, "use_additive_delta", True))

        # 图级池化函数
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

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
        # label: [B, dim_out]

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

        # 比例嵌入
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

        # 组分融合：图结构 + 比例 + 组分身份
        comp_emb = self.component_norm(graph_emb + ratio_emb + type_emb)
        # [B, hidden_dim]

        # ratio = 0 时，该组分嵌入置零
        is_present = ratio_feat[:, 3:4]
        comp_emb = comp_emb * is_present

        return comp_emb, ratio_feat

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
        # 2. 拼接5组分图嵌入和比例信息
        # ---------------------------------------------------------
        graph_emb_input = torch.cat(component_embs, dim=1)
        # [B, 5 * hidden_dim]

        ratio_input = torch.cat(ratio_feats, dim=1)
        # [B, 5 * 4]

        fusion_input = torch.cat([graph_emb_input, ratio_input], dim=1)
        # [B, 5 * hidden_dim + 20]

        # ---------------------------------------------------------
        # 3. 轻量主预测头：方案A
        # ---------------------------------------------------------
        pred = self.pred_head(fusion_input)
        # [B, dim_out]

        # ---------------------------------------------------------
        # 4. 第5组分低含量弱修正：方案B
        # ---------------------------------------------------------
        if self.use_additive_delta:
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

            # 使用比例强度控制修正幅度，而不是只用 is_present 开关。
            # additive_ratio_feat[:, 2:3] 对应 log(1 + 100r) / log(101)
            additive_strength = additive_ratio_feat[:, 2:3]
            # [B, 1]

            pred = pred + additive_strength * additive_delta
            # [B, dim_out]

        # ---------------------------------------------------------
        # 5. 返回 pred 和 label
        # ---------------------------------------------------------
        data1.graph_feature = pred
        pred, label = self._apply_index(data1)

        # ---------------------------------------------------------
        # 6. 关键修正：兼容现有 logger.py
        # ---------------------------------------------------------
        # 你的 logger.py 中存在类似逻辑：
        # true_2d = true_np.reshape(batch_size, PROPERTY_NUM)
        #
        # 因此模型这里需要返回一维形式：
        # pred:  [B * dim_out]
        # label: [B * dim_out]
        #
        # 如果这里返回 [B, dim_out]，logger 在 epoch 统计时容易出现 reshape 错误。
        pred = pred.contiguous().view(-1)
        label = label.contiguous().view(-1)

        return pred, label