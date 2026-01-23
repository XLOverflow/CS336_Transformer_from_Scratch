"""
Positional Encoding

Implements Rotary Position Embeddings (RoPE) for injecting positional information.
"""

import torch
import torch.nn as nn
from typing import Optional


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Applies rotation to query and key vectors based on their positions.
    This encodes relative position information without learnable parameters.

    Args:
        theta (float): Base for computing rotation angles (typically 10000)
        d_k (int): Dimension of key/query vectors
        max_seq_len (int): Maximum sequence length
        device (torch.device, optional): Device to store buffers

    Shape:
        - Input: (*, seq_len, d_k) where * means batch dimensions
        - token_positions: (*, seq_len) with position indices
        - Output: (*, seq_len, d_k)

    Examples:
        >>> rope = RotaryPositionalEmbedding(theta=10000, d_k=64, max_seq_len=2048)
        >>> q = torch.randn(32, 16, 128, 64)  # (batch, heads, seq_len, d_k)
        >>> positions = torch.arange(128).unsqueeze(0).expand(32, 16, -1)
        >>> q_rotated = rope(q, positions)
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # TODO: Precompute sin and cos values for efficiency
        # 1. Compute angles: theta_k = theta^(-2k/d_k) for k=0..d_k/2-1
        # 2. Compute positions: 0 to max_seq_len-1
        # 3. Compute rotation angles: position * theta_k (outer product)
        # 4. Compute and register sin/cos buffers
        # Hint: Use self.register_buffer with persistent=False
        # Shape of buffers: (max_seq_len, d_k//2)
        raise NotImplementedError("Precompute and register sin/cos buffers")

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rotary position embedding.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Position indices of shape (..., seq_len)

        Returns:
            Rotated tensor of same shape as x
        """
        # TODO: Apply RoPE rotation
        # Steps:
        # 1. Slice precomputed sin/cos using token_positions
        # 2. Reshape x into pairs: (x[0], x[1]), (x[2], x[3]), ...
        # 3. Apply rotation matrix to each pair:
        #    [x_even']   [cos  -sin] [x_even]
        #    [x_odd' ] = [sin   cos] [x_odd ]
        # 4. Concatenate rotated pairs back
        # Hint: Split x along last dim, apply rotation, stack/concatenate
        raise NotImplementedError("Implement RoPE rotation")
