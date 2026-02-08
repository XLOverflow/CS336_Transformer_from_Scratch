"""
Checkpointing

Save and load model, optimizer, and training state.
Uses torch.save/torch.load with state_dict for serialization.
"""

import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Serialize model, optimizer, and iteration number to disk.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        iteration: Current training iteration number.
        out: Path or file-like object to write the checkpoint to.
    """
    # Steps:
    # 1. Create a dict with keys:
    #    "model_state_dict" -> model.state_dict()
    #    "optimizer_state_dict" -> optimizer.state_dict()
    #    "iteration" -> iteration
    # 2. Use torch.save(checkpoint_dict, out)
    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint_dict, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load a checkpoint and restore model and optimizer state.

    Args:
        src: Path or file-like object to read the checkpoint from.
        model: Model to restore state into.
        optimizer: Optimizer to restore state into.

    Returns:
        The iteration number from the checkpoint.
    """
    # Steps:
    # 1. Load checkpoint dict: torch.load(src, weights_only=False)
    # 2. Restore model: model.load_state_dict(checkpoint["model_state_dict"])
    # 3. Restore optimizer: optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # 4. Return checkpoint["iteration"]
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
