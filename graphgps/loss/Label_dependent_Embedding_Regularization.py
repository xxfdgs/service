"""
@Name:  Label_dependent_Embedding_Regularization.py
@Auth:  rongxing
@Date:  2025/3/16-下午1:56
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg

class LabelDependentEmbeddingRegularization(nn.Module):
    def __init__(self, margin=1.0, alpha=10.0):
        super(LabelDependentEmbeddingRegularization, self).__init__()
        self.margin = margin  # 控制最大距离
        self.alpha = alpha    # 正则项权重

    def forward(self, features, labels):
        """
        features: (batch_size, feature_dim)
        labels: (batch_size,)
        """

        batch_size = features.size(0)

        # ---- 1. 归一化特征（可选，稳定余弦相似度）
        norm_features = F.normalize(features, p=2, dim=1)  # (32, 512)

        # ---- 2. 计算 pairwise 特征余弦相似度 (32,32)
        sim_matrix = torch.matmul(norm_features, norm_features.T)  # Cosine similarity

        # ---- 3. 计算 pairwise 标签差异
        label_diff = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))  # (32,32)

        # ---- 4. 设计目标函数:
        # 希望标签差异小 → sim 高，标签差异大 → sim 低
        # 使用 margin，避免过拟合

        # 将标签差映射到 0~1 范围
        label_diff_norm = label_diff / (label_diff.max() + 1e-8)

        # 目标： 1 - label_diff_norm ≈ sim
        lder_loss = (sim_matrix - (1 - label_diff_norm)) ** 2

        # ---- 5. mask 自己和自己 (对角线)
        mask = torch.eye(batch_size, device=features.device).bool()
        lder_loss = lder_loss.masked_fill(mask, 0.0)

        # ---- 6. 取平均
        lder_loss = lder_loss.sum() / (batch_size * (batch_size - 1))

        return self.alpha * lder_loss

def LDER_losses(pred_feature, true):
    if cfg.model.loss_fun == 'NTX' or cfg.model.loss_fun == 'l1_ntx':
        loss_fn = LabelDependentEmbeddingRegularization()
        loss = loss_fn(pred_feature, true)
        return loss