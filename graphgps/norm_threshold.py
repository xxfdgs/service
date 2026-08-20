"""Small, testable primitives for O14-A threshold-aware Norm training.

The functions here deliberately operate on raw physical Norm values for every
threshold decision.  Regression losses themselves may still be computed in a
configured transformed/normalized target space by the caller.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score, roc_auc_score


def high_target(raw_target: torch.Tensor, threshold: float) -> torch.Tensor:
    """Scientific high-Norm label: strictly greater than the physical threshold."""
    return (raw_target.reshape(-1) > float(threshold)).to(dtype=raw_target.dtype)


def crossing_false_negative_loss(raw_prediction: torch.Tensor, raw_target: torch.Tensor,
                                 threshold: float) -> torch.Tensor:
    """Squared penalty only for true-high predictions falling below the threshold."""
    prediction, target = raw_prediction.reshape(-1), raw_target.reshape(-1)
    positive = target > float(threshold)
    if not bool(positive.any()):
        # Keep dtype/device and avoid a NaN-producing mean over an empty tensor.
        return prediction.sum() * 0.0
    return functional.relu(float(threshold) - prediction[positive]).square().mean()


def double_high_underprediction_loss(raw_prediction: torch.Tensor, raw_target: torch.Tensor,
                                     double_mask: torch.Tensor,
                                     threshold: float) -> torch.Tensor:
    """Penalty for regression shrinkage below the *true* double high target.

    Unlike ``crossing_false_negative_loss``, this remains informative after a
    prediction has crossed 1.0 but is still materially below its continuous
    value.  The caller supplies raw physical Norm values, so the threshold is
    never accidentally applied in z-scored or log-transformed space.
    """
    prediction, target = raw_prediction.reshape(-1), raw_target.reshape(-1)
    priority = (target > float(threshold)) & double_mask.reshape(-1).bool()
    if not bool(priority.any()):
        return prediction.sum() * 0.0
    return functional.relu(target[priority] - prediction[priority]).square().mean()


def regression_elementwise_loss(prediction: torch.Tensor, target: torch.Tensor,
                                loss_type: str, huber_beta: float) -> torch.Tensor:
    """Unreduced baseline-compatible regression loss in optimization space."""
    if loss_type == "mae":
        return torch.abs(prediction - target)
    if loss_type == "mse":
        return (prediction - target).square()
    if loss_type == "huber":
        return functional.smooth_l1_loss(prediction, target, beta=huber_beta, reduction="none")
    raise ValueError(f"Unsupported regression loss: {loss_type}")


def weighted_regression_loss(prediction: torch.Tensor, target: torch.Tensor,
                             raw_target: torch.Tensor, double_mask: torch.Tensor,
                             positive_weight: float, threshold: float,
                             loss_type: str, huber_beta: float) -> torch.Tensor:
    """Mean regression loss, optionally upweighting only double true-high samples."""
    values = regression_elementwise_loss(prediction.reshape(-1), target.reshape(-1),
                                         loss_type, huber_beta)
    if positive_weight == 1.0:
        return values.mean()
    weight = torch.ones_like(values)
    emphasized = ((raw_target.reshape(-1) > float(threshold))
                  & double_mask.reshape(-1).bool())
    weight = torch.where(emphasized, weight * float(positive_weight), weight)
    return (values * weight).sum() / weight.sum().clamp_min(torch.finfo(weight.dtype).eps)


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def threshold_decision_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               threshold: float) -> dict[str, float | int]:
    """Threshold confusion matrix and recall-focused regression decisions."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    truth = y_true > float(threshold)
    predicted = y_pred > float(threshold)
    tp = int(np.sum(truth & predicted)); tn = int(np.sum(~truth & ~predicted))
    fp = int(np.sum(~truth & predicted)); fn = int(np.sum(truth & ~predicted))
    precision = _rate(tp, tp + fp); recall = _rate(tp, tp + fn)
    specificity = _rate(tn, tn + fp)
    if math.isnan(recall):
        f1 = math.nan; f2 = math.nan
    else:
        # When true positives exist but no positive prediction is emitted,
        # use the standard zero F-score rather than an undefined value; this
        # is essential for comparing threshold-aware checkpoints.
        precision_for_f = 0.0 if math.isnan(precision) else precision
        denominator = precision_for_f + recall
        f1 = 0.0 if denominator == 0 else 2.0 * precision_for_f * recall / denominator
        denominator_f2 = 4.0 * precision_for_f + recall
        f2 = 0.0 if denominator_f2 == 0 else 5.0 * precision_for_f * recall / denominator_f2
    return {
        "n": int(len(y_true)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision_gt1": precision, "recall_gt1": recall,
        "specificity_gt1": specificity, "f1_gt1": f1, "f2_gt1": f2,
        "accuracy_gt1": _rate(tp + tn, len(y_true)),
        "false_negative_rate_gt1": _rate(fn, tp + fn),
        "false_positive_rate_gt1": _rate(fp, fp + tn),
    }


def classifier_metrics(y_true: np.ndarray, high_probability: np.ndarray,
                       threshold: float) -> dict[str, float | int]:
    """Evaluate the auxiliary high-Norm classifier without replacing regression."""
    y_true, high_probability = np.asarray(y_true, dtype=float), np.asarray(high_probability, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(high_probability)
    y_true, high_probability = y_true[valid], high_probability[valid]
    decision = threshold_decision_metrics(
        y_true, np.where(high_probability >= 0.5, float(threshold) + 1.0,
                         float(threshold)), threshold)
    truth = (y_true > float(threshold)).astype(int)
    if len(np.unique(truth)) < 2:
        auroc = math.nan; auprc = math.nan
    else:
        auroc = float(roc_auc_score(truth, high_probability))
        auprc = float(average_precision_score(truth, high_probability))
    return {**decision, "auroc_gt1": auroc, "auprc_gt1": auprc}
