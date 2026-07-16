"""
@Name:  loss_region.py
@Auth:  rongxing
@Date:  2025/3/3-下午4:13
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""
import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg


class TemperatureContrastiveLoss(torch.nn.Module):
    def __init__(self, tau=0.1, lambda_inter=1.0):
        super().__init__()
        self.tau = tau
        self.lambda_inter = lambda_inter


    def forward(self, features, input_labels):
        """
        features: (batch_size, feature_dim) - 分子特征
        labels: (batch_size,) - 预设的高温/中温/低温标签
        """
        # 计算余弦相似度矩阵
        # sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0),
        #                                  dim=2)  # (batch_size, batch_size)
        #
        # labels = input_labels
        #
        # # 组内损失 (L_intra)
        # intra_loss = 0.0
        # # 组内损失 (L_intra)
        # intra_loss = 0.0
        # intra_losses = []
        # for temp in torch.unique(labels):  # 遍历不同温度区
        #     mask = labels == temp  # 选出属于该温度区的样本
        #     if mask.sum() > 1:  # 至少需要两个样本
        #         sim_pos = sim_matrix[mask][:, mask]  # 取组内相似度
        #         # 计算 InfoNCE 损失
        #         #### 计算均值，确保维度匹配
        #         ### old
        #         # loss_per_class = torch.log(
        #         #     torch.exp(sim_pos / self.tau).sum(dim=1) / torch.exp(sim_pos / self.tau).sum())
        #         ## 0327
        #         loss_per_class = -torch.log(
        #             torch.exp(torch.diag(sim_pos) / self.tau) /
        #             torch.exp(sim_pos / self.tau).sum(dim=1)
        #         )
        #         intra_loss -= loss_per_class.mean()
        #     ### 存入列表后统一求和
        #     #     loss_per_class = torch.log(
        #     #         torch.exp(sim_pos / self.tau).sum(dim=1) / torch.exp(sim_pos / self.tau).sum())
        #     #     intra_losses.append(loss_per_class)
        #     #
        #     # if intra_losses:
        #     #     intra_loss -= torch.cat(intra_losses).sum()  # 统一求和
        #
        # # 组间损失 (L_inter)
        # inter_loss = 0.0
        # temp_list = torch.unique(labels)  # 统计 batch 内的所有类别
        # for i in range(len(temp_list)):
        #     for j in range(i + 1, len(temp_list)):
        #         mask_i = labels == temp_list[i]
        #         mask_j = labels == temp_list[j]
        #         if mask_i.sum() > 0 and mask_j.sum() > 0:
        #             sim_neg = sim_matrix[mask_i][:, mask_j]  # 计算组间相似度
        #             ### direct loss
        #             # inter_loss += torch.exp(sim_neg / self.tau).mean()
        #             ### loss = loss * distance
        #             inter_loss += (torch.exp(sim_neg / self.tau).mean()) * torch.abs(temp_list[i]-temp_list[j])
        #
        # loss = intra_loss + self.lambda_inter * inter_loss
        # return loss

        lambda_temp=0.05
        tau = 0.1
        soft_margin = 20
        # batch_size = features.shape[0]

        # # 计算特征的余弦相似度矩阵 (batch_size, batch_size)
        # features = F.normalize(features, p=2, dim=1)  # 归一化
        # similarity_matrix = torch.matmul(features, features.T)  # 计算余弦相似度
        #
        # # 计算温度标签之间的差值矩阵 ΔT
        # temperature_labels = input_labels.unsqueeze(1)  # (batch_size, 1)
        # delta_T = torch.abs(temperature_labels - temperature_labels.T)  # (batch_size, batch_size)
        #
        # # 计算温度相关权重 w_ij = exp(-λ * ΔT)
        # weight_matrix = torch.exp(-lambda_temp * delta_T)
        #
        # # 计算 positive mask (同温度样本对)
        # positive_mask = (delta_T == 0).float()  # (batch_size, batch_size)，同温度样本为 1，其余为 0
        #
        # # 计算 numerator 和 denominator
        # exp_sim = torch.exp(similarity_matrix / tau)  # 计算 exp(S / τ)
        #
        # numerator = torch.sum(exp_sim * positive_mask, dim=1)  # 仅保留正样本对
        # denominator = torch.sum(exp_sim * weight_matrix, dim=1)  # 负样本加权求和
        #
        # # 计算 InfoNCE 损失
        # loss = -torch.mean(torch.log(numerator / denominator))

        ##### region version 1
        # **计算特征余弦相似度矩阵** (batch_size, batch_size)
        features = F.normalize(features, p=2, dim=1)  # 归一化
        similarity_matrix = torch.matmul(features, features.T)  # (batch_size, batch_size)

        # **计算温度标签之间的差值矩阵 ΔT**
        temperature_labels = input_labels.unsqueeze(1)  # (batch_size, 1)
        delta_T = torch.abs(temperature_labels - temperature_labels.T)  # (batch_size, batch_size)

        # **计算温度相关权重** (Soft negative sampling)
        weight_matrix = torch.exp(-lambda_temp * delta_T)  # 温差越大，权重越小

        # **正样本 mask** (完全相同的温度)
        #  delta_T ==0, or < cut_temp
        positive_mask = (delta_T == 0).float()  # (batch_size, batch_size)，同温度样本为 1，其余为 0

        # **引入弱正样本** (温差 <= soft_margin 也算部分正样本)
        weak_positive_mask = ((delta_T > 0) & (delta_T <= soft_margin)).float()

        # 计算 exp(S / τ)
        exp_sim = torch.exp(similarity_matrix / tau)

        # 计算 numerator (包括正样本和弱正样本)
        numerator = torch.sum(exp_sim * (positive_mask + 0.5 * weak_positive_mask), dim=1)

        # 计算 denominator (所有负样本加权求和)
        denominator = torch.sum(exp_sim * weight_matrix, dim=1)

        # 计算损失
        loss = -torch.mean(torch.log(numerator / denominator))

        return loss

def region_losses(pred_feature, true,region_te):
    if cfg.model.loss_fun == 'NTX' or cfg.model.loss_fun == 'l1_ntx':
        # NTX_loss = TemperatureContrastiveLoss(torch.device(cfg.accelerator, cfg.gpu_serial),cfg.train.batch_size,cfg.model.temperature,
        #                              cfg.model.use_cosine_similarity,delta_threshold=50)
        #
        # loss = NTX_loss(pred, true)
        loss_fn = TemperatureContrastiveLoss(tau=0.1, lambda_inter=0.05)
        loss = loss_fn(pred_feature, region_te)
        return loss