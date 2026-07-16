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

def compute_loss_l1_ntx_3(pred_1,true,gps_feature,middle_feature,pred_feature,batch,cur_epoch):
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

    for func in register.loss_dict.values():

        loss_every = l1_every(pred_1, true_1)
        ### l1
        loss_l1 = torch.mean(loss_every, dtype=torch.float32)
        value = loss_l1


        return value, pred_1 #,dict_loss_sum

    else:
        raise ValueError('Loss func {} not supported'.format(
            cfg.model.loss_fun))

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
