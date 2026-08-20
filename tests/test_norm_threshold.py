import torch

from graphgps.norm_threshold import (
    crossing_false_negative_loss,
    double_high_underprediction_loss,
    high_target,
    weighted_regression_loss,
)


def test_high_target_is_strictly_greater_than_threshold():
    value = torch.tensor([0.99, 1.00, 1.01])
    assert high_target(value, 1.0).tolist() == [0.0, 0.0, 1.0]


def test_crossing_loss_only_penalizes_true_positive_crossings():
    assert crossing_false_negative_loss(torch.tensor([1.2]), torch.tensor([1.5]), 1.0).item() == 0.0
    assert crossing_false_negative_loss(torch.tensor([0.9]), torch.tensor([1.5]), 1.0).item() > 0.0
    assert crossing_false_negative_loss(torch.tensor([0.5]), torch.tensor([0.7]), 1.0).item() == 0.0


def test_double_high_underprediction_penalizes_continuous_shrinkage_only():
    prediction = torch.tensor([1.1, 0.7, 0.6])
    target = torch.tensor([1.5, 1.4, 1.8])
    double = torch.tensor([True, False, False])
    # The first point is above the decision boundary but still below its true
    # value; only it is eligible because the other high values are non-double.
    assert double_high_underprediction_loss(prediction, target, double, 1.0).item() > 0
    assert double_high_underprediction_loss(
        torch.tensor([1.6]), torch.tensor([1.5]), torch.tensor([True]), 1.0
    ).item() == 0.0


def test_no_positive_batch_is_finite_and_both_heads_receive_gradients():
    prediction = torch.tensor([0.2, 0.7], requires_grad=True)
    target = torch.tensor([0.1, 0.6])
    logits = torch.tensor([-0.2, 0.3], requires_grad=True)
    crossing = crossing_false_negative_loss(prediction, target, 1.0)
    assert torch.isfinite(crossing)
    # A positive batch confirms that the O14-A total loss reaches both heads.
    prediction2 = torch.tensor([0.8, 1.2], requires_grad=True)
    target2 = torch.tensor([1.4, 0.4])
    logits2 = torch.tensor([0.1, -0.1], requires_grad=True)
    regression = weighted_regression_loss(
        prediction2, target2, target2, torch.tensor([True, False]), 1.5, 1.0, "mae", 0.1)
    classification = torch.nn.functional.binary_cross_entropy_with_logits(
        logits2, high_target(target2, 1.0))
    total = regression + 0.5 * classification + crossing_false_negative_loss(prediction2, target2, 1.0)
    total.backward()
    assert prediction2.grad is not None and torch.count_nonzero(prediction2.grad)
    assert logits2.grad is not None and torch.count_nonzero(logits2.grad)
