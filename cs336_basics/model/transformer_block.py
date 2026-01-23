"""
Transformer Block

Implements a pre-norm Transformer decoder block.
"""

import torch
import torch.nn as nn
from typing import Optional
from .normalization import RMSNorm
from .attention import MultiHeadSelfAttention
from .feedforward import SwiGLU


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer Block.

    Architecture:
        z = x + MultiHeadSelfAttention(RMSNorm(x))
        y = z + FFN(RMSNorm(z))

    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward hidden dimension
        rope: Shared RoPE module
        device: Device for parameters
        dtype: Data type for parameters

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)

    Examples:
        >>> from .positional_encoding import RotaryPositionalEmbedding
        >>> rope = RotaryPositionalEmbedding(10000, 64, 2048)
        >>> block = TransformerBlock(512, 8, 1344, rope)
        >>> x = torch.randn(32, 128, 512)
        >>> output = block(x)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # TODO: Initialize components
        # 1. Two RMSNorm layers (one before attention, one before FFN)
        # 2. MultiHeadSelfAttention
        # 3. SwiGLU feed-forward network
        raise NotImplementedError("Initialize norm layers, attention, and FFN")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer block.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # TODO: Implement pre-norm Transformer block
        # Steps:
        # 1. Attention sub-layer:
        #    z = x + self.attention(self.norm1(x))
        # 2. FFN sub-layer:
        #    y = z + self.ffn(self.norm2(z))
        raise NotImplementedError("Implement Transformer block forward pass")
