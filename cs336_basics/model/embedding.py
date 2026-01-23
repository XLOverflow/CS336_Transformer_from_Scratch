"""
Token Embedding Module

Maps integer token IDs to dense vector representations.
"""

import torch
import torch.nn as nn
from typing import Optional
from einops import einsum


class Embedding(nn.Module):
    """
    Token embedding layer.

    Args:
        num_embeddings (int): Size of the vocabulary
        embedding_dim (int): Dimension of embedding vectors (d_model)
        device (torch.device, optional): Device to store parameters
        dtype (torch.dtype, optional): Data type of parameters

    Shape:
        - Input: (*) LongTensor containing token indices
        - Output: (*, embedding_dim)

    Examples:
        >>> embedding = Embedding(10000, 512)
        >>> token_ids = torch.randint(0, 10000, (32, 128))  # (batch, seq_len)
        >>> embedded = embedding(token_ids)  # (32, 128, 512)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # TODO: Initialize embedding matrix of shape (num_embeddings, embedding_dim)
        # Use truncated normal initialization: N(0, 1) truncated at [-3, 3]
        # Hint: Use nn.Parameter and torch.nn.init.trunc_normal_
        self.embedding = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.embedding, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for token IDs.

        Args:
            token_ids: Tensor of token indices, shape (*)

        Returns:
            Embedded tokens of shape (*, embedding_dim)
        """
        return self.embedding[token_ids]
