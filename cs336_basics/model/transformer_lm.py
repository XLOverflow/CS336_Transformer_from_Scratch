"""
Transformer Language Model

Complete decoder-only Transformer for autoregressive language modeling.
"""

import torch
import torch.nn as nn
from typing import Optional
from .embedding import Embedding
from .normalization import RMSNorm
from .positional_encoding import RotaryPositionalEmbedding
from .transformer_block import TransformerBlock
from .linear import Linear


class TransformerLM(nn.Module):
    """
    Transformer Language Model (decoder-only).

    Architecture:
        1. Token Embedding
        2. N Transformer Blocks
        3. Final RMSNorm (for pre-norm architecture)
        4. Output Linear (LM head)
        5. Softmax to get probabilities

    Args:
        vocab_size (int): Size of vocabulary
        context_length (int): Maximum sequence length
        d_model (int): Model dimension
        num_layers (int): Number of Transformer blocks
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward hidden dimension
        rope_theta (float): RoPE theta parameter (default: 10000)
        device: Device for parameters
        dtype: Data type for parameters

    Shape:
        - Input: (batch_size, seq_len) LongTensor of token IDs
        - Output: (batch_size, seq_len, vocab_size) logits

    Examples:
        >>> model = TransformerLM(
        ...     vocab_size=10000,
        ...     context_length=256,
        ...     d_model=512,
        ...     num_layers=4,
        ...     num_heads=8,
        ...     d_ff=1344,
        ... )
        >>> token_ids = torch.randint(0, 10000, (32, 128))
        >>> logits = model(token_ids)  # (32, 128, 10000)
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        # TODO: Initialize components
        # 1. Token embedding: Embedding(vocab_size, d_model)
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # 2. Shared RoPE module: RotaryPositionalEmbedding(rope_theta, d_k, context_length)
        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(rope_theta, d_k, context_length, device=device)
        # 3. Transformer blocks: nn.ModuleList of TransformerBlock
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, rope=self.rope, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        # 4. Final normalization: RMSNorm(d_model)
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        # 5. Output projection (LM head): Linear(d_model, vocab_size)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the language model.

        Args:
            token_ids: Input token IDs (batch_size, seq_len)

        Returns:
            Logits over vocabulary (batch_size, seq_len, vocab_size)
        """
        # TODO: Implement forward pass
        # Steps:
        # 1. Embed tokens: x = self.token_embedding(token_ids)
        #    Shape: (batch_size, seq_len, d_model)
        x = self.token_embedding(token_ids)
        # 2. Pass through Transformer blocks sequentially
        #    for block in self.blocks:
        #        x = block(x)
        for block in self.blocks:
            x = block(x)
        # 3. Apply final normalization: x = self.final_norm(x)
        x = self.final_norm(x)
        # 4. Project to vocabulary: logits = self.lm_head(x)
        #    Shape: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)
        return logits

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            prompt_ids: Initial token IDs (batch_size, prompt_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold (optional)
            eos_token_id: End-of-sequence token ID (optional)

        Returns:
            Generated token IDs (batch_size, prompt_len + generated_len)
        """
        # TODO: Implement autoregressive generation
        # Steps:
        # 1. Start with prompt_ids
        # 2. For each step up to max_new_tokens:
        #    a. Get logits for current sequence
        #    b. Take logits at last position: logits[:, -1, :]
        #    c. Apply temperature scaling: logits / temperature
        #    d. (Optional) Apply top-p filtering
        #    e. Apply softmax to get probabilities
        #    f. Sample next token from distribution
        #    g. Append to sequence
        #    h. If generated eos_token_id, stop
        # 3. Return full sequence
        raise NotImplementedError("Implement text generation")

    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.

        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
