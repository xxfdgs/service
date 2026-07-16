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
import warnings
from typing import Optional

import numpy as np

# 定义6个性质的常量
PROPERTY_NUM = 6

def compute_loss_l1_ntx_3_5component(pred_1, true, gps_feature, middle_feature, pred_feature, batch, cur_epoch):
    """
    Compute loss for 6 properties（返回true_1=[batch_size,6]，与pred_1格式对齐）
    Args:
        pred_1 (torch.tensor): 预测值，shape [batch_size, 6]
        true (torch.tensor): 真实值，shape [batch_size×6] (e.g. [24])
        gps_feature/middle_feature/pred_feature/batch/cur_epoch: 兼容原参数（未使用）

    Returns:
        total_loss: 6个性质的平均损失（scalar）
        pred_1: 预测值，保持[batch_size,6]格式
        true_1: 重塑后的真实值，[batch_size,6]格式（与pred_1完全对齐）
    """
    # 1. 验证pred_1维度为[batch_size,6]
    assert pred_1.ndim == 2 and pred_1.shape[-1] == PROPERTY_NUM, \
        f"pred_1 must be [batch_size, {PROPERTY_NUM}], current shape: {pred_1.shape}"
    batch_size = pred_1.shape[0]  # 获取真实batch_size

    # 2. 验证true维度为[batch_size×6]
    assert true.ndim == 1 and len(true) == batch_size * PROPERTY_NUM, \
        f"true must be [batch_size×{PROPERTY_NUM}] (e.g. [24]), current length: {len(true)}"

    # 3. 核心：将一维true重塑为[batch_size,6]的true_1（与pred_1格式对齐）
    # reshape顺序与pred_1.flatten()完全一致，保证数据一一对应
    true_1 = true.reshape(batch_size, PROPERTY_NUM)

    # 4. 将pred_1展平为一维，与原始true维度对齐用于计算
    pred_flat = pred_1.flatten()

    # 5. 初始化6个性质的L1损失列表
    loss_l1_list = []

    # 6. 一维间隔切片拆分6个性质计算损失（保证原计算逻辑不变）
    for prop_idx in range(PROPERTY_NUM):
        pred_single = pred_flat[prop_idx::PROPERTY_NUM]  # [batch_size]
        true_single = true[prop_idx::PROPERTY_NUM]        # [batch_size]

        # 沿用原单性质L1损失计算逻辑
        loss_every = l1_every(pred_single, true_single)
        loss_l1 = torch.mean(loss_every, dtype=torch.float32)
        # loss_l1_list.append(loss_l1)
        if prop_idx == 4 or prop_idx == 5:
            loss_l1_list.append(10 * loss_l1)
        else:
            loss_l1_list.append(loss_l1)


    # 7. 计算6个性质的总损失（均值）
    # total_loss = torch.mean(torch.stack(loss_l1_list), dtype=torch.float32)
    # 堆叠后求和
    total_loss = torch.sum(torch.stack(loss_l1_list), dtype=torch.float32)

    # 8. 返回：总损失 + 原格式pred_1 + 重塑后的true_1（[batch_size,6]）
    return total_loss,loss_l1_list, pred_1, true_1

def loss_dict(loss_l1, loss_l1_scale=None, loss_l1_rarity=None, loss_l1_similar=None, value_region=None):
    """
    仅返回包含6个性质L1损失的列表（dict=[loss_l1]）
    Args:
        loss_l1 (list): 6个性质的L1损失列表
    Returns:
        list: 仅包含loss_l1的列表（[loss_l1]）
    """
    assert len(loss_l1) == PROPERTY_NUM, f"loss_l1 must have {PROPERTY_NUM} elements, current length: {len(loss_l1)}"
    dict_list = [loss_l1]  # 严格按要求返回 [loss_l1]
    return dict_list

def l1_losses(pred, true):
    """
    适配混合维度：pred=[batch,6]、true=[batch×6]，返回true_1=[batch,6]
    Args:
        pred: [batch_size,6]
        true: [batch_size×6]
    Returns:
        total_loss: 总损失（均值）
        pred: 原格式预测值 [batch_size,6]
        true_1: 重塑后的真实值 [batch_size,6]（扩展返回，可选）
    """
    # 验证维度
    assert pred.ndim == 2 and pred.shape[-1] == PROPERTY_NUM, \
        f"pred must be [batch_size, {PROPERTY_NUM}], current shape: {pred.shape}"
    batch_size = pred.shape[0]
    assert true.ndim == 1 and len(true) == batch_size * PROPERTY_NUM, \
        f"true must be [batch_size×{PROPERTY_NUM}], current length: {len(true)}"

    # 重塑true为[batch_size,6]
    true_1 = true.reshape(batch_size, PROPERTY_NUM)

    # 展平pred为一维用于计算
    pred_flat = pred.flatten()

    loss_l1_list = []
    # 一维切片拆分6个性质
    for prop_idx in range(PROPERTY_NUM):
        pred_single = pred_flat[prop_idx::PROPERTY_NUM]
        true_single = true[prop_idx::PROPERTY_NUM]

        # 原单性质损失计算逻辑
        if cfg.model.loss_fun == 'l1':
            l1_loss = nn.L1Loss()
            loss = l1_loss(pred_single, true_single)
        elif cfg.model.loss_fun == 'l1_ntx':
            l1_loss = nn.L1Loss()
            loss = l1_loss(pred_single, true_single)
        elif cfg.model.loss_fun == 'smoothl1':
            l1_loss = nn.SmoothL1Loss()
            loss = l1_loss(pred_single, true_single)
        else:
            raise ValueError(f'Loss func {cfg.model.loss_fun} not supported')
        loss_l1_list.append(loss)

    total_loss = torch.mean(torch.stack(loss_l1_list), dtype=torch.float32)

    # 扩展返回true_1（与pred格式对齐）
    if cfg.model.loss_fun == 'l1' or cfg.model.loss_fun == 'smoothl1':
        return total_loss, pred, true_1
    elif cfg.model.loss_fun == 'l1_ntx':
        return total_loss, true_1  # 适配l1_ntx模式的返回扩展

def l1_loss_similar(pred, true, batch):
    """
    适配混合维度：pred=[batch,6]、true=[batch×6]，返回true_1=[batch,6]
    Args:
        pred: [batch_size,6]
        true: [batch_size×6]
        batch: 包含similarity的批处理数据（shape [batch_size]）
    Returns:
        total_loss: 总损失（均值）
        pred: 原格式预测值 [batch_size,6]
        true_1: 重塑后的真实值 [batch_size,6]
    """
    # 验证维度
    assert pred.ndim == 2 and pred.shape[-1] == PROPERTY_NUM, \
        f"pred must be [batch_size, {PROPERTY_NUM}], current shape: {pred.shape}"
    batch_size = pred.shape[0]
    assert true.ndim == 1 and len(true) == batch_size * PROPERTY_NUM, \
        f"true must be [batch_size×{PROPERTY_NUM}], current length: {len(true)}"
    assert batch.similarity.ndim == 1 and len(batch.similarity) == batch_size, \
        f"batch.similarity must be [batch_size], current shape: {batch.similarity.shape}"

    # 重塑true为[batch_size,6]
    true_1 = true.reshape(batch_size, PROPERTY_NUM)

    # 展平pred为一维
    pred_flat = pred.flatten()

    loss_mean_list = []
    # 一维切片拆分6个性质
    for prop_idx in range(PROPERTY_NUM):
        pred_single = pred_flat[prop_idx::PROPERTY_NUM]
        true_single = true[prop_idx::PROPERTY_NUM]

        # 原单性质加权损失计算逻辑
        reduction = 'none'
        size_average = None
        reduce = None
        loss = l1_loss_(pred_single, true_single, size_average, reduce, reduction)
        loss_multily = torch.mul(loss, batch.similarity)
        loss_mean = torch.mean(loss_multily, dtype=torch.float32)
        loss_mean_list.append(loss_mean)

    total_loss = torch.mean(torch.stack(loss_mean_list), dtype=torch.float32)
    return total_loss, pred, true_1

def l1_every(pred, true):
    """
    逐样本计算L1损失（适配一维/二维输入）
    Args:
        pred: [batch_size,6] | [batch_size×6] | [batch_size]
        true: [batch_size×6] | [batch_size]
    Returns:
        loss: 逐样本损失，shape与输入一致
    """
    # 若pred是二维，先展平为一维
    if pred.ndim == 2:
        pred = pred.flatten()
    reduction = 'none'
    size_average = None
    reduce = None
    loss = l1_loss_(pred, true, size_average, reduce, reduction)
    return loss

def l1_loss_(input, target, size_average=None, reduce=None, reduction='none'):
    r"""
    底层L1损失计算（适配一维输入，保留原逻辑）
    """
    reduction = 'none'
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    return torch._C._nn.l1_loss(expanded_input, expanded_target, get_enum(reduction))

def get_enum(reduction: str) -> int:
    """保留原逻辑：转换reduction字符串为枚举值"""
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
        ret = -1
        raise ValueError(f"{reduction} is not a valid value for reduction")
    return ret

def legacy_get_string(size_average: Optional[bool], reduce: Optional[bool], emit_warning: bool = True) -> str:
    """保留原逻辑：兼容旧版size_average/reduce参数"""
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
    """保留原逻辑：兼容旧版参数并转换为枚举值"""
    return get_enum(legacy_get_string(size_average, reduce, emit_warning))