"""
@Name:  weight_Regression_aware_Metric_Learning.py
@Auth:  rongxing
@Date:  2025/3/16-下午2:50
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg

class BatchwiseRegressionAwareLoss(nn.Module):
    def __init__(self, temp_threshold=300, max_weight=50.0):
        """
        temp_threshold: 高温的阈值 (比如设为 200)
        max_weight: 高温区的最大权重
        """
        super(BatchwiseRegressionAwareLoss, self).__init__()
        self.temp_threshold = temp_threshold
        self.max_weight = max_weight

    def forward(self, features, labels):
        # features: (batch_size, feature_dim)  -> (32, 512)
        # labels: (batch_size,)  -> (32,)

        batch_size = features.size(0)

        # ------- 计算 Pairwise 特征距离 -------
        f1 = features.unsqueeze(1)  # shape: (32, 1, 512)
        f2 = features.unsqueeze(0)  # shape: (1, 32, 512)

        # Pairwise L2 distance
        feature_dist = torch.norm(f1 - f2, p=2, dim=2)  # shape: (32, 32)

        # ------- 计算 Pairwise 标签距离 -------
        l1 = labels.unsqueeze(1)  # (32, 1)
        l2 = labels.unsqueeze(0)  # (1, 32)

        label_dist = torch.abs(l1 - l2)  # shape: (32, 32)

        # ------- 加权设计 -------
        # Step 1: 生成权重，越高温权重越大
        if self.temp_threshold is not None:
            # 简单线性权重示例
            weights1 = torch.where(labels > self.temp_threshold, self.max_weight, 1.0)
        else:
            # 归一化权重
            weights1 = (labels - labels.min()) / (labels.max() - labels.min() + 1e-6)
            weights1 = 1.0 + weights1 * (self.max_weight - 1.0)  # 最小权重为1，最大为max_weight

        # 生成 pairwise 权重矩阵
        weights_i = weights1.unsqueeze(1)  # (32, 1)
        weights_j = weights1.unsqueeze(0)  # (1, 32)
        weight_matrix = (weights_i + weights_j) / 2.0  # 对称性保持

        # ------- 计算 Loss -------
        loss_matrix = ((feature_dist - label_dist) ** 2) * weight_matrix  # 加权

        # 去掉对角线
        mask = torch.eye(batch_size, device=features.device).bool()
        loss_matrix = loss_matrix.masked_fill(mask, 0.0)

        # 平均
        loss = loss_matrix.sum() / (batch_size * (batch_size - 1))

        return loss

def WRAML_losses(pred_feature, true):
    if cfg.model.loss_fun == 'NTX' or cfg.model.loss_fun == 'l1_ntx':
        loss_fn = BatchwiseRegressionAwareLoss()
        loss = loss_fn(pred_feature, true)
        return loss