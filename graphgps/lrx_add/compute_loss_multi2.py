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
import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
import warnings
from typing import Optional

import numpy as np

def compute_loss_multi2(pred,true, batch):
    """
    pred, true: 1D Tensor, shape [B * property_num]
    对每个性质单独计算 loss，然后加合返回。

    order:
      - "batch_first": 展平顺序为 先batch再性质  (row-major)
      - "prop_first" : 展平顺序为 先性质再batch  (col-major)
    """
    property_num=2

    # 每个性质的 loss（标量）
    loss_list = []

    # 该性质在所有batch上的序列：prop_idx, prop_idx+P, prop_idx+2P, ...
    for prop_idx in range(property_num):
        pred_single = pred[prop_idx::property_num]   # [B]
        true_single = true[prop_idx::property_num]   # [B]

        loss_every = l1_every(pred_single, true_single)          # [B]
        loss_i = torch.mean(loss_every, dtype=torch.float32)     # 标量
        loss_list.append(loss_i)

    total_loss = torch.sum(torch.stack(loss_list), dtype=torch.float32)
    return total_loss, pred


def loss_dict(loss_l1,loss_l1_scale,loss_l1_rarity,loss_l1_similar,value_region):
    dict=[loss_l1,loss_l1_scale,loss_l1_rarity,loss_l1_similar,value_region]
    return dict



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

def l1_every(pred, true):
    # if cfg.model.loss_fun == 'l1_similar':
    reduction = 'none'
    size_average = None
    reduce = None
    loss = l1_loss_(pred, true,size_average, reduce, reduction)
    return loss


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
