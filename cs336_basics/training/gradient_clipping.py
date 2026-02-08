"""
Gradient Clipping

Implements L2-norm gradient clipping for training stability.

Given all parameter gradients g, compute ||g||_2.
If ||g||_2 > M, scale each gradient by M / (||g||_2 + eps)
where eps = 1e-6 for numeric stability.
"""

import torch
from collections.abc import Iterable


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """
    Clip the combined L2 norm of all parameter gradients.

    Args:
        parameters: Collection of trainable parameters.
        max_l2_norm: Maximum allowed L2 norm (M).

    Modifies parameter gradients in-place.
    """
    # Implement gradient clipping
    # Steps:
    # 1. Convert parameters to a list (so we can iterate multiple times)
    # 2. Compute total L2 norm: sqrt(sum of squared norms of each p.grad)
    #    Hint: p.grad.data.norm(2).item() ** 2 for each param, then sqrt the sum
    # 3. If total_norm > max_l2_norm:
    #    scale = max_l2_norm / (total_norm + 1e-6)
    #    Multiply each p.grad.data by scale
    #    Hint: p.grad.data.mul_(scale)
    params = list(parameters)
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(scale)

