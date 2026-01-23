"""
Token Embedding Module

Maps integer token IDs to dense vector representations.
"""

import torch
import torch.nn as nn
from typing import Optional


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
        raise NotImplementedError("Initialize embedding matrix here")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for token IDs.

        Args:
            token_ids: Tensor of token indices, shape (*)

        Returns:
            Embedded tokens of shape (*, embedding_dim)
        """
        # TODO: Implement embedding lookup
        # Index into the embedding matrix using token_ids
        # Hint: Use tensor indexing, e.g., self.weight[token_ids]
        raise NotImplementedError("Implement embedding lookup")
