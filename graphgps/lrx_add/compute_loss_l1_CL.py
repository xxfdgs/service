"""
@Name:  compute_loss_multi.py
@Auth:  rongxing
@Date:  2024/12/9-下午4:08
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
import warnings
from typing import Optional

from graphgps.loss.loss_region import region_losses
from graphgps.loss.Regression_aware_Metric_Learning import RAML_losses
from graphgps.loss.Label_dependent_Embedding_Regularization import LDER_losses
from graphgps.loss.WeightedLabelDependentEmbeddingRegularization import WLDER_losses
from graphgps.loss.weight_Regression_aware_Metric_Learning import WRAML_losses
import numpy as np

def compute_loss_l1_ntx(pred_1,true,pred_feature,batch,cur_epoch):
# def compute_loss_multi(pred, true, batch):
    """
    Compute loss and prediction score

    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Grou

    Returns: Loss, normalized prediction score

    """

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    pred_1 = pred_1.squeeze(-1) if pred_1.ndim > 1 else pred_1
    true_1 = true
    true_1 = true_1.squeeze(-1) if true_1.ndim > 1 else true_1
    # Try to load customized loss
    # list_all = [[pred_1, true_1, loss_g],[pred_2, true_2, loss_d],pred_3, true_3, loss_m]
    for func in register.loss_dict.values():
        # value_l1 = l1_losses(pred_1, true_1)   #### error
        loss_every = l1_every(pred_1, true_1)
        # value_l1 = torch.mean(loss_every, dtype=torch.float32)
        # value = value_l1

        ### l1
        # loss_l1 = torch.mean(value_l1, dtype=torch.float32)
        ### scale l1
        loss_l1_scale = torch.mean(torch.mul(loss_every, batch.scale), dtype=torch.float32)
        ### scale l1_rarity
        # loss_l1_rarity = torch.mean(torch.mul(loss_every, batch.tg_rarity), dtype=torch.float32)
        # Dynamic weight adjustment based on epoch
        # Exponential function to gradually increase weight over epochs
        # dy_weight = dynamic_weight(batch.scale,cur_epoch,cfg)
        # loss_l1_scale_dynamic = torch.mean(torch.mul(loss_every, dy_weight), dtype=torch.float32)
        # print(loss_l1_scale,loss_l1_scale_dynamic,loss_l1)
        ### similar_l1
        # loss_l1_similar = torch.mean(torch.mul(loss_every, batch.similarity), dtype=torch.float32)

        ###
        # value_region = 50 * region_losses(gps_feature, true_1, batch.y)
        # value_region = 50 * region_losses(middle_feature, true_1, batch.y)
        value_region = 500 * region_losses(pred_feature, true_1, batch.y)

        # value = value_l1 + value_region
        # value = 10 * (loss_l1_scale + value_region)
        value = loss_l1_scale + value_region #+ loss_l1_rarity + loss_l1_similar
        # mae_weight = get_dynamic_mae_weight(cur_epoch,cfg)
        # value = (get_dynamic_mae_weight(cur_epoch)) * loss_l1_scale + value_region
        # value = loss_l1_scale_dynamic + value_region
        # value = loss_l1_scale + value_region + loss_l1_similar
        # value = loss_l1
        return value, pred_1

    else:
        raise ValueError('Loss func {} not supported'.format(
            cfg.model.loss_fun))

def get_dynamic_mae_weight(cur_epoch, k=0.05, max_weight=5.0, dtype=torch.float32):
    # 计算当前的 mae_loss 权重，保持为 tensor
    # cur_epoch_tensor = torch.tensor(cur_epoch, dtype=dtype, device=torch.device(cfg['accelerator'], cfg['gpu_serial']))
    # weight = 1 + (max_weight - 1) * (1 - torch.exp(-k * cur_epoch_tensor))
    # weight = torch.round(weight * 100) / 100
    weight = round((1 + (max_weight - 1) * (1 - np.exp(-k * cur_epoch))), 2)
    # cut_epoch = 100
    # if cur_epoch<=cut_epoch:
    #     weight = 1.0
    # else:
    #     cur_epoch = cur_epoch - cut_epoch
    #     weight = round((1 + (max_weight - 1) * (1 - np.exp(-k * cur_epoch))),2)
    return weight

def l1_every(pred, true):
    # if cfg.model.loss_fun == 'l1_similar':
    reduction = 'none'
    size_average = None
    reduce = None
    loss = l1_loss_(pred, true,size_average, reduce, reduction)
    return loss

def dynamic_weight(target_values,cur_epoch,cfg):
    alpha = 0.02  # 可以根据需求调整该值
    weight_factors = target_values
    # for cur_epoch in range(1500):
        # weight_factor = np.exp(-alpha_rate * cur_epoch)
    # 计算weight_factor_exp  .to(torch.device(cfg.accelerator, cfg.gpu_serial))
    alpha_tensor = torch.tensor(alpha, dtype=torch.float64, device=(torch.device(cfg['accelerator'], cfg['gpu_serial'])))
    epoch_tensor = torch.tensor(cur_epoch, dtype=torch.float64, device=(torch.device(cfg['accelerator'], cfg['gpu_serial'])))

    # 使用公式对每个元素计算
    weight_factors = 1 + (target_values - 1) * (1 - torch.exp(-alpha_tensor * epoch_tensor))
    return weight_factors
    # for target_value in target_values:
    #
    #     weight_factor_exp = 1 + (target_value - 1) * (1 - torch.exp(torch.tensor(-alpha * cur_epoch).to(torch.device(cfg.accelerator, cfg.gpu_serial))))
    #     if cur_epoch %10 ==0:
    #         print('cur_epoch= ', cur_epoch, ' weight_factor=', weight_factor_exp, target_value)
    #     weight_factors.append(weight_factor_exp)
    # return torch.tensor(weight_factors).to(torch.device(cfg.accelerator, cfg.gpu_serial))

def l1_losses(pred, true):
    if cfg.model.loss_fun == 'l1':
        l1_loss = nn.L1Loss()
        loss = l1_loss(pred, true)
        return loss, pred
    elif cfg.model.loss_fun == 'l1_ntx':
        l1_loss = nn.L1Loss()
        loss = l1_loss(pred, true)
        return loss
    elif cfg.model.loss_fun == 'smoothl1':
        l1_loss = nn.SmoothL1Loss()
        loss = l1_loss(pred, true)
        return loss, pred

def l1_loss_similar(pred, true, batch):
    # if cfg.model.loss_fun == 'l1_similar':
    reduction = 'none'
    size_average = None
    reduce = None
    loss = l1_loss_(pred, true,size_average, reduce, reduction)
    loss_multily = torch.mul(loss, batch.similarity)
    loss_mean = torch.mean(loss_multily, dtype=torch.float32)
    return loss_mean, pred


def l1_loss_(input,target,size_average = None,reduce = None,reduction = 'none'):
# ) -> Tensor:
    r"""l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Function that takes the mean element-wise absolute value difference.

    See :class:`~torch.nn.L1Loss` for details.
    """
    reduction = 'none'
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    return torch._C._nn.l1_loss(expanded_input, expanded_target, get_enum(reduction))


# NB: Keep this file in sync with enums in aten/src/ATen/core/Reduction.h


def get_enum(reduction: str) -> int:
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'elementwise_mean':
        warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    return ret

# In order to support previous versions, accept boolean size_average and reduce
# and convert them into the new constants for now


# We use these functions in torch/legacy as well, in which case we'll silence the warning
def legacy_get_string(size_average: Optional[bool], reduce: Optional[bool], emit_warning: bool = True) -> str:
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret


def legacy_get_enum(size_average: Optional[bool], reduce: Optional[bool], emit_warning: bool = True) -> int:
    return get_enum(legacy_get_string(size_average, reduce, emit_warning))
