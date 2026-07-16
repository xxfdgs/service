"""
@Name:  feature_y_graph.py
@Auth:  rongxing
@Date:  2023/3/31-上午10:44
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME}
basis on feature_graph.py and alter it to suitable for output feature which
be uesed in contrast loss and value(predicted Tm)

"""

import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head


@register_head('feature_y_graph') #feat_graph
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
        #2023.04.01
        # list_FC_layers = [
        #     nn.Linear(dim_in * 2 ** l, dim_in * 2 ** (l + 1), bias=True)
        #     for l in range(L)]
        # # list_FC_layers.append(nn.Linear(dim_in * 2 ** L, dim_out, bias=True))
        # list_FC_layers.append(nn.Linear(dim_in * 2 ** (L), 512, bias=True))
        # list_FC_layers.append(nn.Linear(512, dim_in * 2 ** (L - 1), bias=True))
        # list_FC_layers.append(nn.Linear(dim_in * 2 ** (L - 1), dim_out, bias=True))

        #2023.04.02
        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y
        #graph_feature是FC前的输出
    def _apply_index_y(self, batch):
        return batch.graph_feature, batch.graph_feature_y, batch.y
    # def _apply_feature(self,batch):
    #     return batch.graph_feature_y
    # def _apply_y(self,batch):
    #     return batch.y
    #2023.04.01
    # def forward(self, batch):
    #     graph_emb = self.pooling_fun(batch.x, batch.batch)
    #     for l in range((self.L)+1):
    #         graph_emb = self.FC_layers[l](graph_emb)
    #         if l != ((self.L)+1):
    #             graph_emb = self.activation()(graph_emb)
    #     batch.graph_feature = graph_emb
    #     graph_emb_y = self.FC_layers[((self.L) + 1)](graph_emb)
    #     # graph_emb = self.activation()(graph_emb)
    #     graph_emb_y = self.FC_layers[((self.L) + 2)](graph_emb_y)
    #     batch.graph_feature_y = graph_emb_y
    #     pred,pred_y, label = self._apply_index_y(batch)
    #     return pred,pred_y, label

        #molclr
        # h = self.pool(h, data.batch)
        # h = self.feat_lin(h)
        # out = self.out_lin(h)
    # 2023.04.02
    #original version add feature,feature_y
    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        batch.graph_feature = graph_emb
        for l in range(self.L):
            # print('l',l)
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation()(graph_emb)
        graph_emb_y = self.FC_layers[self.L](graph_emb)
        batch.graph_feature_y = graph_emb_y
        pred,pred_y, label = self._apply_index_y(batch)
        return pred,pred_y, label

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
