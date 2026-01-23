"""
Linear Transformation Module

Implements a linear layer without bias: y = Wx
Following modern LLM practices (PaLM, LLaMA), we omit the bias term.
"""

import torch
import torch.nn as nn
from typing import Optional


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

        # TODO: Initialize weight matrix W of shape (out_features, in_features)
        # Use truncated normal initialization: N(0, 2/(in_features + out_features))
        # Hint: Use nn.Parameter and torch.nn.init.trunc_normal_
        # Remember to store as row-major W, not W^T
        raise NotImplementedError("Initialize weight parameter here")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation.

        Args:
            x: Input tensor of shape (*, in_features)

        Returns:
            Output tensor of shape (*, out_features)
        """
        # TODO: Implement forward pass
        # Compute y = xW^T (because we store W in row-major order)
        # Hint: Use einsum or @ operator for batched matrix multiplication
        raise NotImplementedError("Implement forward pass: y = xW^T")
