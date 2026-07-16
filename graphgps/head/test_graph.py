"""
@Name:  test_graph.py
@Auth:  rongxing
@Date:  2023/2/22-上午9:28
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""

import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head


@register_head('test_graph') #test_graph
class FineGraphHead(nn.Module):
    """
    feat prediction head for graph prediction feature in contrast loss.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
        list_FC_layers = [nn.Linear(128, 128, bias=True),nn.Linear(128, 64, bias=True),]
        list_FC_layers.append(
            nn.Linear(64, 1, bias=True))
        # list_FC_layers = [
        #     nn.Linear(dim_in // 4 ** l, dim_in // 4 ** (l + 1), bias=True)
        #     for l in range(L)]
        # list_FC_layers.append(
        #     nn.Linear(dim_in // 4 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation()(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label