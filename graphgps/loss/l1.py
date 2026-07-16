import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
import numpy as np
import torch
from typing import Optional
import warnings

@register_loss('l1_losses')
def l1_losses(pred, true):
    if cfg.model.loss_fun == 'l1_similar':
        # l1_loss = nn.L1Loss()
        reduction = 'none'
        size_average = None
        reduce = None
        # loss = l1_loss(pred, true)
        loss = l1_loss_(pred, true,size_average, reduce, reduction)
        # loss = l1_loss(pred, true)
        # loss_ = loss.detach().to('cpu', non_blocking=True)
        # loss_numpy = loss_.numpy()
        # mean_ = np.mean(loss_numpy)
        return loss, pred
    elif cfg.model.loss_fun == 'l1':
        l1_loss = nn.L1Loss()
        loss = l1_loss(pred, true)
        return loss, pred
    elif cfg.model.loss_fun == 'smoothl1':
        l1_loss = nn.SmoothL1Loss()
        loss = l1_loss(pred, true)
        return loss, pred

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
