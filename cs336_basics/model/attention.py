"""
Attention Mechanisms

Implements scaled dot-product attention and multi-head self-attention.
"""

import torch
import torch.nn as nn
from typing import Optional
from .linear import Linear
from .positional_encoding import RotaryPositionalEmbedding


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax implementation.

    Args:
        x: Input tensor
        dim: Dimension to apply softmax

    Returns:
        Softmax probabilities
    """
    # TODO: Implement numerically stable softmax
    # Steps:
    # 1. Subtract max value along dim for stability
    # 2. Compute exp
    # 3. Normalize by sum
    # Hint: Use x.max(dim=dim, keepdim=True) and torch.exp
    raise NotImplementedError("Implement numerically stable softmax")


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.

    Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        None (stateless operation)

    Shape:
        - Q: (batch, ..., seq_len_q, d_k)
        - K: (batch, ..., seq_len_k, d_k)
        - V: (batch, ..., seq_len_k, d_v)
        - mask: (seq_len_q, seq_len_k) boolean mask
        - Output: (batch, ..., seq_len_q, d_v)

    Examples:
        >>> attn = ScaledDotProductAttention()
        >>> q = torch.randn(32, 8, 128, 64)  # (batch, heads, seq_len, d_k)
        >>> k = v = torch.randn(32, 8, 128, 64)
        >>> output = attn(q, k, v)
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention.

        Args:
            q: Query tensor (..., seq_len_q, d_k)
            k: Key tensor (..., seq_len_k, d_k)
            v: Value tensor (..., seq_len_k, d_v)
            mask: Optional boolean mask (seq_len_q, seq_len_k)
                  True = attend, False = mask out

        Returns:
            Attention output (..., seq_len_q, d_v)
        """
        # TODO: Implement scaled dot-product attention
        # Steps:
        # 1. Compute attention scores: QK^T / sqrt(d_k)
        #    Hint: Use einsum or @ with appropriate transposes
        # 2. Apply mask: set scores to -inf where mask is False
        # 3. Apply softmax to get attention weights
        # 4. Compute weighted sum of values: attention_weights @ V
        # Hint: d_k = q.size(-1)
        raise NotImplementedError("Implement scaled dot-product attention")


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE and causal masking.

    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        rope: RoPE module (shared across layers)
        device: Device for parameters
        dtype: Data type for parameters

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope

        # TODO: Initialize projection matrices
        # - W_q: projects to query (d_model -> d_model)
        # - W_k: projects to key (d_model -> d_model)
        # - W_v: projects to value (d_model -> d_model)
        # - W_o: output projection (d_model -> d_model)
        # Hint: Use Linear class defined earlier
        raise NotImplementedError("Initialize Q, K, V, O projection matrices")

        # TODO: Create causal mask (lower triangular matrix)
        # This prevents attending to future positions
        # Hint: Use torch.triu or index comparisons
        raise NotImplementedError("Create and register causal mask buffer")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # TODO: Implement multi-head self-attention
        # Steps:
        # 1. Project to Q, K, V: shape (batch, seq_len, d_model)
        # 2. Reshape to separate heads: (batch, num_heads, seq_len, d_k)
        #    Hint: Use view and transpose
        # 3. Apply RoPE to Q and K (NOT V)
        #    Token positions: torch.arange(seq_len)
        # 4. Apply scaled dot-product attention with causal mask
        # 5. Reshape attention output: (batch, seq_len, d_model)
        # 6. Apply output projection
        raise NotImplementedError("Implement multi-head self-attention forward pass")
