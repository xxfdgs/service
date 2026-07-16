"""
@Name:  Regression_aware_Metric_Learning.py
@Auth:  rongxing
@Date:  2025/3/16-下午1:43
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg


class BatchwiseRegressionAwareLoss(nn.Module):
    def __init__(self):
        super(BatchwiseRegressionAwareLoss, self).__init__()

    def forward(self, features, labels):
        # features: (batch_size, feature_dim)  -> (32, 512)
        # labels: (batch_size,)  -> (32,)

        batch_size = features.size(0)

        # ------- 计算 Pairwise 特征距离 -------
        # Expand features for broadcasting
        f1 = features.unsqueeze(1)  # shape: (32, 1, 512)
        f2 = features.unsqueeze(0)  # shape: (1, 32, 512)

        # Pairwise L2 distance
        feature_dist = torch.norm(f1 - f2, p=2, dim=2)  # shape: (32, 32)

        # ------- 计算 Pairwise 标签距离 -------
        l1 = labels.unsqueeze(1)  # (32, 1)
        l2 = labels.unsqueeze(0)  # (1, 32)

        label_dist = torch.abs(l1 - l2)  # shape: (32, 32)

        # ------- 计算 Loss -------
        loss_matrix = (feature_dist - label_dist) ** 2

        # 通常去掉对角线（自己和自己比没有意义）
        mask = torch.eye(batch_size, device=features.device).bool()
        loss_matrix = loss_matrix.masked_fill(mask, 0.0)

        # 取平均
        loss = loss_matrix.sum() / (batch_size * (batch_size - 1))

        return loss

def RAML_losses(pred_feature, true):
    if cfg.model.loss_fun == 'NTX' or cfg.model.loss_fun == 'l1_ntx':
        loss_fn = BatchwiseRegressionAwareLoss()
        loss = loss_fn(pred_feature, true)
        return loss