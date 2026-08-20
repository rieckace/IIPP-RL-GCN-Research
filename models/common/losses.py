"""
losses.py

Loss functions for reinforcement learning training.
"""

import torch
import torch.nn.functional as F


def huber_loss(predicted: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Compute Huber (Smooth L1) loss.

    More robust than MSE for RL because it is less sensitive to outliers
    in the TD error, which are common during early training when Q-value
    estimates are noisy.

    L(x) = 0.5 * x^2            if |x| <= delta
          delta * |x| - 0.5 * delta^2  otherwise

    Args:
        predicted: Predicted Q-values, shape (batch_size,).
        target:    Target Q-values, shape (batch_size,).
        delta:     Threshold for switching between L2 and L1 behaviour.

    Returns:
        Scalar loss tensor.
    """
    return F.smooth_l1_loss(predicted, target, beta=delta)
