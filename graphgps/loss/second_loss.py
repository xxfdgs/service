"""
@Name:  second_loss.py
@Auth:  rongxing
@Date:  2023/4/1-下午4:43
@IDE:   PyCharm
@PROJECT_NAME:   $ {PROJECT_NAME}
compute second loss which represent the differential value of between Tm pred and true
"""


def second_compute_loss(pred, true):
    bce_loss = nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
    mse_loss = nn.MSELoss(reduction=cfg.model.size_average)
    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true
    # Try to load customized loss

    # for func in register.loss_dict.values():
    #     value = func(pred, true)
    #     if value is not None:
    #         return value
    loss, pred = second_l1_losses(pred, true)
    return loss, pred


import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


# @register_loss('l1_losses')
def second_l1_losses(pred, true):
    # if cfg.model.loss_fun == 'l1':
    l1_loss = nn.L1Loss()
    loss = l1_loss(pred, true)
    return loss, pred
    # elif cfg.model.loss_fun == 'smoothl1':
    #     l1_loss = nn.SmoothL1Loss()
    #     loss = l1_loss(pred, true)
    #     return loss, pred
