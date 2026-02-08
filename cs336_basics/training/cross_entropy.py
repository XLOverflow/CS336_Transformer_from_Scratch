"""
Cross-Entropy Loss

Numerically stable cross-entropy loss for language modeling.
"""

import torch


def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the average cross-entropy loss.

    l_i = -log softmax(o_i)[x_{i+1}]

    Args:
        inputs: Unnormalized logits of shape (..., vocab_size).
            Batch dimensions come first, vocab_size is the last dimension.
        targets: Target class indices of shape (...).
            Each value is in [0, vocab_size).

    Returns:
        Scalar tensor: the average cross-entropy loss across all examples.
    """
    # Subtract max for numerical stability
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values

    # loss = log(sum(exp(shifted))) - shifted[target]
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    return (log_sum_exp - target_logits).mean()
