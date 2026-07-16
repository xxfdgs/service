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

# @register_network('GPSDoubleModel')
class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        # self.gnn = Double_gps(dim_in, dim_out)
        self.gnn1 = Double_gps(dim_in, dim_out)
        self.gnn2 = Double_gps(dim_in, dim_out)
        # self.fc = nn.Linear((cfg.gt.dim_hidden)*2, 1)
        # self.fc = nn.Linear((cfg.gt.dim_hidden) * 2, 1)
        # self.gnn_layers = nn.ModuleList([Double_gps(dim_in, dim_out), Double_gps(dim_in, dim_out)])
        ###
        list_FC_layers = [nn.Linear((((cfg.gt.dim_hidden) *2) +1), 256, bias=True), nn.Linear(256, 512, bias=True),
                          nn.Linear(512, dim_out, bias=True)]
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.activation = register.act_dict[cfg.gnn.act]

        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, data1, data2):
        # Get representations for each molecule
        rep1 = self.gnn1(data1)  ### 同一个网络
        rep2 = self.gnn2(data2)  ### 同一个网络
        # rep1 = self.gnn1(data1) ### 两个单独的网络
        # rep2 = self.gnn2(data2) ### 两个单独的网络
        # rep1 = self.gnn_layers[0](data1)  # 使用 ModuleList 中的 gnn1
        # rep2 = self.gnn_layers[1](data2)

        graph_emb_1 = self.pooling_fun(rep1.x, rep1.batch)
        graph_emb_2 = self.pooling_fun(rep2.x, rep2.batch)
        # Combine the representations
        graph_emb = torch.cat((graph_emb_1, graph_emb_2), dim=1)
        ### add mnw  ratio
        # add_mnw = (rep1.mnw).unsqueeze(1)
        add_ratio = (rep2.ratio).unsqueeze(1)
        # graph_emb = torch.cat((graph_emb, add_mnw), dim=1)
        graph_emb = torch.cat((graph_emb, add_ratio), dim=1)


        ###
        # MLP
        # graph_emb = self.fc(combined)
        for l in range(3):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation()(graph_emb)

        # Final prediction
        data1.graph_feature = graph_emb
        data2.graph_feature = graph_emb
        pred, label = self._apply_index(data1)
        return pred, label



