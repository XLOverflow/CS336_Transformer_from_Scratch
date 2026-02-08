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

        # Initialize learnable gain parameter of shape (d_model,)
        # Initialize to ones
        # Hint: Use nn.Parameter with torch.ones
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Normalized tensor of same shape
        """
        # Implement RMSNorm
        # Steps:
        # 1. Save original dtype
        original_dtype = x.dtype
        # 2. Cast to float32 for numerical stability
        x = x.to(torch.float32)
        # 3. Compute RMS: sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 4. Normalize: x / RMS
        normalized = x / rms
        # 5. Apply gain: normalized * self.gain
        output = normalized * self.weight
        # 6. Cast back to original dtype
        # Hint: Use torch.mean, torch.sqrt for RMS calculation
        return output.to(original_dtype)
