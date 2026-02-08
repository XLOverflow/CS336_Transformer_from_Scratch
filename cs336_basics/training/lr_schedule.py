"""
Learning Rate Schedule

Implements cosine annealing with linear warmup as used in LLaMA (Touvron et al., 2023).

Three phases:
    1. Warmup (t < T_w):         alpha_t = (t / T_w) * alpha_max
    2. Cosine annealing (T_w <= t <= T_c):
        alpha_t = alpha_min + 0.5 * (1 + cos((t - T_w) / (T_c - T_w) * pi)) * (alpha_max - alpha_min)
    3. Post-annealing (t > T_c): alpha_t = alpha_min
"""

import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Compute learning rate at iteration `it` under a cosine schedule with linear warmup.

    Args:
        it: Current iteration number.
        max_learning_rate: alpha_max, maximum learning rate.
        min_learning_rate: alpha_min, minimum/final learning rate.
        warmup_iters: T_w, number of warmup iterations.
        cosine_cycle_iters: T_c, total number of cosine annealing iterations.

    Returns:
        The learning rate at the given iteration.
    """
    if it < warmup_iters:
        # Phase 1: Linear warmup
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        # Phase 2: Cosine annealing
        cosine_progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_factor = 0.5 * (1 + math.cos(cosine_progress * math.pi))
        return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)
    else:
        # Phase 3: Post-annealing
        return min_learning_rate