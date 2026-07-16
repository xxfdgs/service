"""
@Name:  feature_graph.py
@Auth:  rongxing
@Date:  2023/2/20-上午10:42
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME}
basis on san_graph.py and alter it to suitable for output feature which
be uesed in contrast loss
"""
import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head


@register_head('feat_graph') #feat_graph
class FeatGraphHead(nn.Module):
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
        # original version
        # list_FC_layers = [
        #     nn.Linear(dim_in * 2 ** l, dim_in * 2 ** (l + 1), bias=True)
        #     for l in range(L)]
        # list_FC_layers.append(
        #     nn.Linear(dim_in * 2 ** L, dim_in * 2 ** (L+1), bias=True))
# 待修改 添加激活函数来实现MLP
# 添加隐层(参考molclr)
        #2023.03.27
        list_FC_layers = [
            nn.Linear(dim_in * 2 ** l, dim_in * 2 ** (l + 1), bias=True)
            for l in range(L)]
        # list_FC_layers.append(nn.Linear(dim_in * 2 ** L, dim_out, bias=True))
        list_FC_layers.append(nn.Linear(dim_in * 2 ** (L), 512, bias=True))
        # if cfg.dataset.data_mask == False:
        #     list_FC_layers.append(nn.Linear(512, dim_in * 2 ** (L-1), bias=True))
        #     list_FC_layers.append(nn.Linear(dim_in * 2 ** (L-1), dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y
        #graph_feature是FC前的输出
        #测试 batch.graph_feature.shape ,batch.y.shape

    #2023.03.27
    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        for l in range((self.L)+1):
            graph_emb = self.FC_layers[l](graph_emb)
            if l != ((self.L)+1):
                graph_emb = self.activation()(graph_emb)
        batch.graph_feature = graph_emb   #输出特征并赋予batch新的特征
        pred, label = self._apply_index(batch)
        return pred, label


    # #original version
    # def forward(self, batch):
    #     graph_emb = self.pooling_fun(batch.x, batch.batch)
    #     for l in range(self.L):
    #         # print('l',l)
    #         graph_emb = self.FC_layers[l](graph_emb)
    #         graph_emb = self.activation()(graph_emb)
    #     graph_emb = self.FC_layers[self.L](graph_emb)
    #     batch.graph_feature = graph_emb
    #     pred, label = self._apply_index(batch)
    #     return pred, label
