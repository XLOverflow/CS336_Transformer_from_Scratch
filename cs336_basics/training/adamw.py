"""
AdamW Optimizer

Implements AdamW (Adam with decoupled weight decay) as described in
Loshchilov and Hutter, 2019 (Algorithm 2).

Algorithm:
    init(theta)
    m <- 0, v <- 0
    for t = 1, ..., T do
        g <- gradient of loss
        m <- beta1 * m + (1 - beta1) * g
        v <- beta2 * v + (1 - beta2) * g^2
        alpha_t <- alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
        theta <- theta - alpha_t * m / (sqrt(v) + eps)
        theta <- theta - alpha * lambda * theta   (weight decay)
    end for

Note: t starts at 1.
"""

import math
from collections.abc import Callable
from typing import Optional

import torch


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer.

    Args:
        params: Iterable of parameters or dicts defining parameter groups.
        lr (float): Learning rate (alpha). Default: 1e-3.
        betas (tuple): (beta1, beta2) for moment estimates. Default: (0.9, 0.999).
        eps (float): Numerical stability term. Default: 1e-8.
        weight_decay (float): Weight decay coefficient (lambda). Default: 0.0.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        # Validate hyperparameters and call super().__init__
        # Steps:
        # 1. Check lr >= 0, eps >= 0, 0 <= beta1 < 1, 0 <= beta2 < 1
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        # 2. Create defaults dict with lr, betas, eps, weight_decay
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        # 3. Call super().__init__(params, defaults)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.

        Iterate over all parameter groups and parameters.
        For each parameter with a gradient:
            1. Initialize state (t=0, m=zeros, v=zeros) if first step
            2. Increment t
            3. Update m: m = beta1 * m + (1 - beta1) * g
            4. Update v: v = beta2 * v + (1 - beta2) * g^2
            5. Compute bias-corrected lr: alpha_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
            6. Update params: p = p - alpha_t * m / (sqrt(v) + eps)
            7. Apply weight decay: p = p - lr * weight_decay * p

        Hints:
            - Use self.state[p] to get/set per-parameter state
            - Use torch.zeros_like(p.data) to initialize moment vectors
            - Use .mul_(), .add_(), .addcmul_(), .addcdiv_() for in-place ops
            - Use math.sqrt for the bias correction
        """
        # Implement AdamW step
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                state['t'] += 1
                t = state['t']
                g = p.grad.data
                m = state['m']
                v = state['v']
                # Update moments
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                # Compute bias-corrected learning rate
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                # Update parameters
                p.data.addcdiv_(m, v.sqrt().add(eps), value=-alpha_t)
                # Apply weight decay
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
        return loss