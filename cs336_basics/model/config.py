"""
Model Configuration

Defines configuration dataclasses for different model sizes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """
    Configuration for Transformer Language Model.

    This follows the architecture described in the assignment:
    - Pre-norm Transformer blocks
    - RMSNorm for normalization
    - SwiGLU for feed-forward
    - RoPE for positional encoding
    - Multi-head causal self-attention
    """

    # Vocabulary and sequence
    vocab_size: int
    context_length: int

    # Model architecture
    d_model: int  # Hidden dimension
    num_layers: int  # Number of Transformer blocks
    num_heads: int  # Number of attention heads
    d_ff: int  # Feed-forward hidden dimension

    # RoPE parameters
    rope_theta: float = 10000.0

    # Compute
    device: Optional[str] = None
    dtype: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert self.d_ff % 64 == 0, \
            f"d_ff ({self.d_ff}) should be multiple of 64 for hardware efficiency"


# Predefined configurations from the assignment

@dataclass
class TinyStoriesConfig(TransformerConfig):
    """
    Small model for TinyStories dataset (~17M parameters).

    Based on assignment specifications:
    - 4 layers, 16 heads
    - d_model = 512, d_ff ≈ 8/3 * d_model
    - Context length: 256
    """
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344  # ≈ 8/3 * 512, rounded to multiple of 64


@dataclass
class GPT2SmallConfig(TransformerConfig):
    """GPT-2 Small (117M parameters) configuration."""
    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 2048  # 4 * d_model (using SiLU instead of SwiGLU)


@dataclass
class GPT2MediumConfig(TransformerConfig):
    """GPT-2 Medium (345M parameters) configuration."""
    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    d_ff: int = 2730  # ≈ 8/3 * 1024, rounded to multiple of 64


@dataclass
class GPT2LargeConfig(TransformerConfig):
    """GPT-2 Large (774M parameters) configuration."""
    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 1280
    num_layers: int = 36
    num_heads: int = 20
    d_ff: int = 3392  # ≈ 8/3 * 1280, rounded to multiple of 64


@dataclass
class GPT2XLConfig(TransformerConfig):
    """GPT-2 XL (1.5B parameters) configuration."""
    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 1600
    num_layers: int = 48
    num_heads: int = 25
    d_ff: int = 4224  # ≈ 8/3 * 1600, rounded to multiple of 64


def get_config(name: str) -> TransformerConfig:
    """
    Get predefined configuration by name.

    Args:
        name: Configuration name ('tinystories', 'gpt2-small', etc.)

    Returns:
        TransformerConfig instance

    Examples:
        >>> config = get_config('tinystories')
        >>> print(config.num_layers)
        4
    """
    configs = {
        'tinystories': TinyStoriesConfig(),
        'gpt2-small': GPT2SmallConfig(),
        'gpt2-medium': GPT2MediumConfig(),
        'gpt2-large': GPT2LargeConfig(),
        'gpt2-xl': GPT2XLConfig(),
    }

    if name not in configs:
        raise ValueError(
            f"Unknown config '{name}'. Available: {list(configs.keys())}"
        )

    return configs[name]
