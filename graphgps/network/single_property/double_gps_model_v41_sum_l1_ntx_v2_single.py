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


@register_network('GPSDoubleModel_l1_ntx_v2_single_sum')
class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former.
    version 1 (modified to 5 components input)
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gnn = Double_gps(dim_in, dim_out)

        # 修改1：全连接层输入维度 (5个组分 × hidden_dim)
        input_dim = cfg.gt.dim_hidden
        list_FC_layers = [
            nn.Linear(input_dim, 256, bias=True),
            nn.Linear(256, 512, bias=True),
            nn.Linear(512, dim_out, bias=True)
        ]
        # 中间预测层保持不变（仅维度依赖256中间层）
        list_FC_layers_middle_2mlp = [nn.Linear(256, 1, bias=True)]
        # 修改2：GPS直接预测层输入维度适配5组分
        list_FC_layers_2mlp = [nn.Linear(input_dim, 1, bias=True)]

        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.FC_layers_2mlp = nn.ModuleList(list_FC_layers_2mlp)
        # self.FC_layers_midle_mlp = nn.ModuleList(list_FC_layers_middle_2mlp)?
        self.activation = register.act_dict[cfg.gnn.act]

        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
        self.prefix_mlp = nn.Sequential(*[
            nn.Linear(1, cfg.train.batch_size, bias=False),
            nn.ReLU(),
            nn.Linear(cfg.train.batch_size, cfg.gt.dim_hidden, bias=False),
            nn.ReLU()
        ])


        self.tg_mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    # 修改4：forward输入参数扩展为5个data
    def forward(self, data1, data2, data3, data4, data5):
        # 步骤1：批量处理5个组分的GNN编码
        datas = [data1, data2, data3, data4, data5]
        reps = [self.gnn(data) for data in datas]  # 每个组分通过同一GNN编码

        # 步骤2：对每个组分做池化和前缀MLP变换
        graph_embs = []
        # prefix_embs = []
        tg_percents = []
        for rep in reps:
            graph_emb = self.pooling_fun(rep.x, rep.batch)
            prefix_emb = self.prefix_mlp(rep.ratio.unsqueeze(1))
            graph_embs.append(prefix_emb * graph_emb)
            tg_percents.append(rep.ratio.unsqueeze(1))

        # 步骤3：拼接5个组分的tg_percent并计算权重
        tg_percent_cat = torch.cat(tg_percents, dim=1)  # [batch_size, 5]
        logits = self.tg_mlp(tg_percent_cat)
        weights = torch.softmax(logits, dim=1)  # [batch_size, 2]
        tg_percent_end = weights[:, 0].unsqueeze(1)  # [batch_size, 1]
        tg_percent_start = weights[:, 1].unsqueeze(1)
        # tg_percent_middle = weights[:, 2].unsqueeze(1)


        # 步骤5：拼接5个组分的特征
        # graph_emb_input = torch.cat(graph_embs, dim=1)  # [batch_size, 5*hidden_dim]
        graph_emb_input = torch.sum(torch.stack(graph_embs, dim=1), dim=1)
        # 步骤6：全连接层前向传播
        graph_emb = graph_emb_input
        graph_emb_feature_middle = None
        graph_emb_feature = None
        for l in range(3):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation()(graph_emb)
            if l == 1:
                graph_emb_feature = graph_emb
            # if l == 0:
            #     graph_emb_feature_middle = graph_emb

        # 步骤7：GPS直接预测和中间层预测
        graph_emb_2mlp = self.FC_layers_2mlp[0](graph_emb_input)
        graph_emb_2mlp = self.activation()(graph_emb_2mlp)
        # graph_emb_middle_2mlp = self.FC_layers_midle_mlp[0](graph_emb_feature_middle)
        # graph_emb_middle_2mlp = self.activation()(graph_emb_middle_2mlp)

        # 步骤8：最终预测（加权求和：5个组分预测 + 中间预测）
        data1.graph_feature = graph_emb
        pred, label = self._apply_index(data1)

        # 加权逻辑：pred（主分支）× 各组分权重 + 直接预测×中间权重 + 中间层预测×中间权重
        pred = (pred * tg_percent_end) + (graph_emb_2mlp * tg_percent_start)


        return pred, label, graph_emb_input, graph_emb_feature_middle, graph_emb_feature