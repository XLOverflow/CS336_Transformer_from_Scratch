"""
Transformer Block

Implements a pre-norm Transformer decoder block.
"""

import torch
import torch.nn as nn
from typing import Optional
from .normalization import RMSNorm
from .attention import MultiHeadSelfAttention
from .feedforward import SwiGLU, SiLUFFN


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
        norm_type (str): "rmsnorm" (default) or "none" (identity)
        use_post_norm (bool): If True, use post-norm instead of pre-norm
        ffn_type (str): "swiglu" (default) or "silu"

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
        norm_type: str = "rmsnorm",
        use_post_norm: bool = False,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.use_post_norm = use_post_norm

        # Initialize components
        # 1. Two norm layers (one for attention, one for FFN)
        if norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        else:  # "none" - identity
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        # 2. MultiHeadSelfAttention
        self.mha = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        # 3. Feed-forward network
        if ffn_type == "silu":
            self.ffn = SiLUFFN(d_model, d_ff, device=device, dtype=dtype)
        else:  # "swiglu"
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer block.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        if self.use_post_norm:
            # Post-norm: z = norm(x + attn(x)), y = norm(z + ffn(z))
            z = self.norm1(x + self.mha(x))
            return self.norm2(z + self.ffn(z))
        else:
            # Pre-norm: z = x + attn(norm(x)), y = z + ffn(norm(z))
            z = x + self.mha(self.norm1(x))
            return z + self.ffn(self.norm2(z))
