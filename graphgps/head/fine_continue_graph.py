"""
@Name:  fine_continue_graph.py
@Auth:  rongxing
@Date:  2023/4/18-下午4:14
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME}
basis on feature_graph.py and alter it to suitable for output  which
be uesed in Tm prediction
64 128
128 256
256 512
+
512 1
"""


import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head


@register_head('fine_continue_graph') #feat_graph
class FineGraphHead(nn.Module):
    """
    feat prediction head for graph prediction feature in contrast loss.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
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


    #2023.04.02
    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation()(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label

