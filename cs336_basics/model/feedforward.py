"""
Feed-Forward Networks

Implements SwiGLU: a gated feed-forward network with SiLU activation.
"""

import torch
import torch.nn as nn
from typing import Optional
from .linear import Linear


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Implements: FFN(x) = (SiLU(W1 * x) ⊙ W3 * x) * W2
    where ⊙ is element-wise multiplication

    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Args:
        d_model (int): Input/output dimension
        d_ff (int): Hidden dimension (typically ~8/3 * d_model, rounded to multiple of 64)
        device: Device for parameters
        dtype: Data type for parameters

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)

    Examples:
        >>> ffn = SwiGLU(512, 1344)  # 1344 ≈ 8/3 * 512 rounded to multiple of 64
        >>> x = torch.randn(32, 128, 512)
        >>> output = ffn(x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # TODO: Initialize three linear layers
        # - W1: d_model -> d_ff (for SiLU branch)
        # - W2: d_ff -> d_model (output projection)
        # - W3: d_model -> d_ff (for gating branch)
        # Hint: Use Linear class
        raise NotImplementedError("Initialize W1, W2, W3 linear layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward network.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # TODO: Implement SwiGLU
        # Steps:
        # 1. Compute gate = SiLU(W1(x))
        #    SiLU(z) = z * sigmoid(z) = z / (1 + exp(-z))
        #    Hint: Use torch.sigmoid
        # 2. Compute value = W3(x)
        # 3. Element-wise multiply: gate ⊙ value
        # 4. Project back: W2(gate ⊙ value)
        raise NotImplementedError("Implement SwiGLU forward pass")
