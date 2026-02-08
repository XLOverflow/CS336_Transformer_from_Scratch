"""
Data Loading

Implements batch sampling for language model training.

For a dataset of n tokens, randomly sample B start positions i,
and create:
    input  = dataset[i : i + context_length]
    target = dataset[i+1 : i+1 + context_length]
"""

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of input sequences and next-token targets.

    Args:
        dataset: 1D numpy array of integer token IDs.
        batch_size: Number of sequences to sample.
        context_length: Length of each sampled sequence.
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0').

    Returns:
        Tuple of (inputs, targets), both LongTensors of shape (batch_size, context_length)
        on the specified device.
    """
    # 1. Compute max valid start index
    max_start = len(dataset) - context_length - 1
    # 2. Randomly sample batch_size start indices
    starts = torch.randint(0, max_start + 1, (batch_size,))
    # 3. For each start index, slice input and target (offset by 1)
    inputs = torch.stack(
        [torch.from_numpy(dataset[i : i + context_length].astype(np.int64)) for i in starts]
    )
    targets = torch.stack(
        [torch.from_numpy(dataset[i + 1 : i + 1 + context_length].astype(np.int64)) for i in starts]
    )
    # 4. Move to device and return
    return inputs.to(device), targets.to(device)
