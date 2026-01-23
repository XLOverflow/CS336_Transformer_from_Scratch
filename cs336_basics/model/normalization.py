"""
Normalization Layers

Implements Root Mean Square Layer Normalization (RMSNorm) as used in modern LLMs.
"""

import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes activations using RMS and applies learned gain parameters.
    Formula: output = (input / RMS(input)) * gain
    where RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        d_model (int): Hidden dimension size
        eps (float): Small constant for numerical stability (default: 1e-5)
        device (torch.device, optional): Device to store parameters
        dtype (torch.dtype, optional): Data type of parameters

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)

    Examples:
        >>> norm = RMSNorm(512)
        >>> x = torch.randn(32, 128, 512)
        >>> normalized = norm(x)  # Same shape as input
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # TODO: Initialize learnable gain parameter of shape (d_model,)
        # Initialize to ones
        # Hint: Use nn.Parameter with torch.ones
        raise NotImplementedError("Initialize gain parameter")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Normalized tensor of same shape
        """
        # TODO: Implement RMSNorm
        # Steps:
        # 1. Save original dtype
        # 2. Cast to float32 for numerical stability
        # 3. Compute RMS: sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
        # 4. Normalize: x / RMS
        # 5. Apply gain: normalized * self.gain
        # 6. Cast back to original dtype
        # Hint: Use torch.mean, torch.sqrt for RMS calculation
        raise NotImplementedError("Implement RMSNorm forward pass")
