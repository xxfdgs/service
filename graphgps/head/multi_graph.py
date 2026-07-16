"""
@Name:  multi_graph.py
@Auth:  rongxing
@Date:  2024/4/24-下午8:32
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""

import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head



@register_head('multi_graph') #feat_graph
class FineGraphHead(nn.Module):
    """
    feat prediction head for graph prediction feature in contrast loss.

    Args:
        dim_in (int): Input dimension.
        dim_out1 (int): Output dimension for the first property.
        dim_out2 (int): Output dimension for the second property.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=3):
        super().__init__()
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
        list_FC_layers = [nn.Linear(64, 128, bias=True), nn.Linear(128, 256, bias=True),
                          nn.Linear(256, 512, bias=True)]
        list_FC_layers.append(
            nn.Linear(512, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

        # 2023.04.02

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation()(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label

    # def __init__(self, dim_in, dim_out, L=3):
    #     super().__init__()
    #     self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
    #
    #     # 第一个性质的多层感知器
    #     self.FC_layers1 = nn.ModuleList([
    #         nn.Linear(64, 128, bias=True),
    #         nn.Linear(128, 256, bias=True),
    #         nn.Linear(256, 512, bias=True),
    #         nn.Linear(512, dim_out, bias=True)
    #     ])
    #
    #     # # 第二个性质的多层感知器
    #     # self.FC_layers2 = nn.ModuleList([
    #     #     nn.Linear(64, 128, bias=True),
    #     #     nn.Linear(128, 256, bias=True),
    #     #     nn.Linear(256, 512, bias=True),
    #     #     nn.Linear(512, dim_out, bias=True)
    #     # ])
    #
    #     self.L = L
    #     self.activation = register.act_dict[cfg.gnn.act]
    #
    # def _apply_index(self, batch):
    #     return batch.graph_feature, batch.y
    #
    # def forward(self, batch):
    #     graph_emb = self.pooling_fun(batch.x, batch.batch)
    #
    #     # 对第一个性质的多层感知器进行前向传播
    #     for l in range(self.L):
    #         graph_emb = self.FC_layers1[l](graph_emb)
    #         graph_emb = self.activation()(graph_emb)
    #     pred1 = self.FC_layers1[self.L](graph_emb)
    #
    #     # # 对第二个性质的多层感知器进行前向传播
    #     # graph_emb = self.pooling_fun(batch.x, batch.batch)
    #     # for l in range(self.L):
    #     #     graph_emb = self.FC_layers2[l](graph_emb)
    #     #     graph_emb = self.activation()(graph_emb)
    #     # pred2 = self.FC_layers2[self.L](graph_emb)
    #
    #     # batch.graph_feature = [pred1,pred2]
    #     # pred, label = self._apply_index(batch)
    #     # return pred[0],pred[1], label
    #
    #     batch.graph_feature = pred1
    #     pred, label = self._apply_index(batch)
    #     return pred, label