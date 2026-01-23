"""
Linear Transformation Module

Implements a linear layer without bias: y = Wx
Following modern LLM practices (PaLM, LLaMA), we omit the bias term.
"""

import torch
import torch.nn as nn
from typing import Optional
from einops import einsum


class Linear(nn.Module):
    """
    Linear transformation layer without bias.

    Args:
        in_features (int): Size of input dimension
        out_features (int): Size of output dimension
        device (torch.device, optional): Device to store parameters
        dtype (torch.dtype, optional): Data type of parameters

    Shape:
        - Input: (*, in_features) where * means any number of batch dimensions
        - Output: (*, out_features)

    Examples:
        >>> linear = Linear(512, 2048)
        >>> x = torch.randn(32, 128, 512)  # (batch, seq_len, in_features)
        >>> out = linear(x)  # (32, 128, 2048)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight matrix W of shape (out_features, in_features)
        # Store as W (not W^T) for memory ordering reasons
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        # Truncated normal initialization: N(0, σ²=2/(in+out)), truncated at [-3σ, 3σ]
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation.

        Args:
            x: Input tensor of shape (*, in_features)

        Returns:
            Output tensor of shape (*, out_features)
        """
        # Compute y = xW^T using einsum
        # x: (..., in_features), weight: (out_features, in_features)
        # output: (..., out_features)
        return einsum(x, self.weight, "... i, o i -> ... o")
