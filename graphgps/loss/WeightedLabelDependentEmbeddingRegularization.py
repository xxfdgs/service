"""
@Name:  WeightedLabelDependentEmbeddingRegularization.py
@Auth:  rongxing
@Date:  2025/3/16-下午2:00
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg

class WeightedLabelDependentEmbeddingRegularization(nn.Module):
    def __init__(self, alpha=10.0, temp_threshold=250.0, high_temp_weight=2.0):
        """
        参数：
        - alpha: 正则化项的权重系数。
        - temp_threshold: 高温区的温度阈值，超过此值视为高温。
        - high_temp_weight: 高温区样本的损失加权系数。
        """
        super(WeightedLabelDependentEmbeddingRegularization, self).__init__()
        self.alpha = alpha
        self.temp_threshold = temp_threshold
        self.high_temp_weight = high_temp_weight

    def forward(self, features, labels):
        """
        参数：
        - features: 特征张量，形状为 (batch_size, feature_dim)。
        - labels: 标签张量，形状为 (batch_size,)。
        返回：
        - 加权后的 LDER 损失值。
        """
        batch_size = features.size(0)

        # 1. 归一化特征以计算余弦相似度
        norm_features = F.normalize(features, p=2, dim=1)  # (batch_size, feature_dim)

        # 2. 计算特征余弦相似度矩阵
        sim_matrix = torch.matmul(norm_features, norm_features.T)  # (batch_size, batch_size)

        # 3. 计算标签差异矩阵
        label_diff = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))  # (batch_size, batch_size)

        # 4. 将标签差异归一化到 [0, 1] 区间
        label_diff_norm = label_diff / (label_diff.max() + 1e-8)

        # 5. 计算基础的 LDER 损失矩阵
        lder_loss_matrix = (sim_matrix - (1 - label_diff_norm)) ** 2

        # 6. 创建高温区掩码
        high_temp_mask = (labels > self.temp_threshold).float()  # (batch_size,)
        high_temp_weight_matrix = high_temp_mask.unsqueeze(1) * high_temp_mask.unsqueeze(0)  # (batch_size, batch_size)

        # 7. 对高温区样本对的损失进行加权
        weighted_lder_loss_matrix = lder_loss_matrix * (1 + self.high_temp_weight * high_temp_weight_matrix)

        # 8. 屏蔽对角线元素（自身与自身的比较）
        mask = torch.eye(batch_size, device=features.device).bool()
        weighted_lder_loss_matrix = weighted_lder_loss_matrix.masked_fill(mask, 0.0)

        # 9. 计算平均损失
        lder_loss = weighted_lder_loss_matrix.sum() / (batch_size * (batch_size - 1))

        return self.alpha * lder_loss

def WLDER_losses(pred_feature, true):
    if cfg.model.loss_fun == 'NTX' or cfg.model.loss_fun == 'l1_ntx':
        loss_fn = WeightedLabelDependentEmbeddingRegularization()
        loss = loss_fn(pred_feature, true)
        return loss