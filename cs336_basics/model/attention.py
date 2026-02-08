"""
Attention Mechanisms

Implements scaled dot-product attention and multi-head self-attention.
"""

import torch
import torch.nn as nn
from typing import Optional
from .linear import Linear
from .positional_encoding import RotaryPositionalEmbedding
from einops import einsum


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax implementation.

    Args:
        x: Input tensor
        dim: Dimension to apply softmax

    Returns:
        Softmax probabilities
    """
    # Steps:
    # 1. Subtract max value along dim for stability
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max
    # 2. Compute exp
    exp_x = torch.exp(x_stable)
    # 3. Normalize by sum
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    # Hint: Use x.max(dim=dim, keepdim=True) and torch.exp
    return exp_x / sum_exp_x


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
        # Steps:
        # 1. Compute attention scores: QK^T / sqrt(d_k)
        #    Hint: Use einsum or @ with appropriate transposes
        d_k = q.size(-1)
        attn_scores = einsum(q, k, "b ... q d_k, b ... k d_k -> b ... q k") / (d_k ** 0.5)
        # 2. Apply mask: set scores to -inf where mask is False
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        # 3. Apply softmax to get attention weights
        attn_weights = softmax(attn_scores, dim=-1)
        # 4. Compute weighted sum of values: attention_weights @ V
        # Hint: d_k = q.size(-1)
        return einsum(attn_weights, v, "b ... q k, b ... k d_v -> b ... q d_v")


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
        rope: Optional[RotaryPositionalEmbedding] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        self.attention = ScaledDotProductAttention()

        # Initialize projection matrices
        # - W_q: projects to query (d_model -> d_model)
        # - W_k: projects to key (d_model -> d_model)
        # - W_v: projects to value (d_model -> d_model)
        # - W_o: output projection (d_model -> d_model)
        # Hint: Use Linear class defined earlier
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)

        # Create causal mask (lower triangular matrix)
        if rope is not None:
            max_seq_len = rope.max_seq_len
            self.register_buffer("causal_mask", torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)), persistent=False)
        else:
            self.causal_mask = None

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Implement multi-head self-attention
        # Steps:
        # 1. Project to Q, K, V: shape (batch, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Reshape to separate heads: (batch, num_heads, seq_len, d_k)
        #    Hint: Use view and transpose
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, num_heads, seq_len, d_k
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
        # 3. Apply RoPE to Q and K (NOT V)
        #    Token positions: torch.arange(seq_len)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        # 4. Apply scaled dot-product attention with causal mask
        if self.causal_mask is not None:
            mask = self.causal_mask[:seq_len, :seq_len]
        else:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attn_output = self.attention(Q, K, V, mask=mask)
        # 5. Reshape attention output: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        # 6. Apply output projection
        return self.W_o(attn_output)