#!/usr/bin/env python3
"""
Training script for Transformer Language Model.

Usage with predefined config:
    python scripts/train_lm.py \
        --config tinystories \
        --train_data data/train_tokens.npy \
        --val_data data/val_tokens.npy

Usage with custom config (overrides predefined):
    python scripts/train_lm.py \
        --config tinystories \
        --d_model 768 --num_heads 12 --num_layers 6 \
        --train_data data/train_tokens.npy \
        --val_data data/val_tokens.npy

Usage with fully custom config (no predefined):
    python scripts/train_lm.py \
        --vocab_size 10000 --context_length 256 \
        --d_model 512 --num_heads 8 --num_layers 4 --d_ff 2048 \
        --train_data data/train_tokens.npy \
        --val_data data/val_tokens.npy
"""

import argparse
import math
import os
import time

import numpy as np
import torch
from dotenv import load_dotenv

# Load .env file (for WANDB_API_KEY, etc.)
load_dotenv()

try:
    import wandb
except ImportError:
    wandb = None

from cs336_basics.model import TransformerLM
from cs336_basics.model.config import get_config, TransformerConfig
from cs336_basics.training import (
    cross_entropy,
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM")

    # ---- Model config: predefined or custom ----
    parser.add_argument("--config", type=str, default=None,
                        choices=["tinystories", "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                        help="Predefined model config (can be overridden by custom args)")

    # Custom model hyperparameters (override predefined config if provided)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--rope_theta", type=float, default=None)

    # ---- Data ----
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training token IDs (.npy)")
    parser.add_argument("--val_data", type=str, required=True,
                        help="Path to validation token IDs (.npy)")

    # ---- Optimizer ----
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # ---- Training ----
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=500)

    # ---- Logging & checkpointing ----
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=1000)

    # ---- Wandb ----
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")

    # ---- Device & resume ----
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def build_model_config(args) -> TransformerConfig:
    """
    Build model config from args.
    Priority: custom args > predefined config > error
    """
    if args.config is not None:
        # Start from predefined config
        config = get_config(args.config)
        # Override with any custom args that were explicitly provided
        if args.vocab_size is not None:
            config.vocab_size = args.vocab_size
        if args.context_length is not None:
            config.context_length = args.context_length
        if args.d_model is not None:
            config.d_model = args.d_model
        if args.num_heads is not None:
            config.num_heads = args.num_heads
        if args.num_layers is not None:
            config.num_layers = args.num_layers
        if args.d_ff is not None:
            config.d_ff = args.d_ff
        if args.rope_theta is not None:
            config.rope_theta = args.rope_theta
        return config
    else:
        # Fully custom: all required fields must be provided
        required = ["vocab_size", "context_length", "d_model", "num_heads", "num_layers"]
        for field in required:
            if getattr(args, field) is None:
                raise ValueError(
                    f"--{field} is required when not using a predefined --config"
                )
        d_ff = args.d_ff if args.d_ff is not None else int(8 / 3 * args.d_model)
        # Round d_ff to multiple of 64
        d_ff = (d_ff + 63) // 64 * 64
        return TransformerConfig(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=d_ff,
            rope_theta=args.rope_theta or 10000.0,
        )


@torch.no_grad()
def evaluate(model, val_data, batch_size, context_length, device, eval_batches):
    """Evaluate model on validation set. Returns average loss."""
    model.eval()
    total_loss = 0.0
    for _ in range(eval_batches):
        inputs, targets = get_batch(val_data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        total_loss += loss.item()
    model.train()
    return total_loss / eval_batches


def train(args):
    """Main training loop."""
    # ---------------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------------
    config = build_model_config(args)
    device = args.device
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load datasets with memory mapping
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    print("=" * 60)
    print(f"Model config: {config}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")
    print(f"Device:       {device}")
    print("=" * 60)

    # Initialize wandb (reads WANDB_PROJECT, WANDB_ENTITY, WANDB_API_KEY from .env)
    if args.wandb:
        assert wandb is not None, "wandb not installed. Run: pip install wandb"
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "cs336-assignment1"),
            entity=os.environ.get("WANDB_ENTITY", None),
            config=vars(args),
        )

    # ---------------------------------------------------------------
    # 2. Create model
    # ---------------------------------------------------------------
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # ---------------------------------------------------------------
    # 3. Create optimizer
    # ---------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # ---------------------------------------------------------------
    # 4. Optionally resume from checkpoint
    # ---------------------------------------------------------------
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from checkpoint at step {start_step}")

    # ---------------------------------------------------------------
    # 5. Training loop
    # ---------------------------------------------------------------
    model.train()
    start_time = time.time()

    for step in range(start_step, args.max_steps):
        # a. Update learning rate
        lr = get_lr_cosine_schedule(step, args.lr, args.min_lr, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # b-h. Forward, backward, update
        inputs, targets = get_batch(train_data, args.batch_size, config.context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # i. Log training loss
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"step {step:>6d} | loss {loss.item():.4f} | lr {lr:.6f} | time {elapsed:.1f}s")
            if args.wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/perplexity": math.exp(loss.item()),
                    "train/lr": lr,
                    "train/step": step,
                    "train/wallclock": elapsed,
                })

        # j. Evaluate on validation set
        if step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, args.batch_size, config.context_length, device, args.eval_batches)
            print(f"  >>> val loss {val_loss:.4f} | val ppl {math.exp(val_loss):.2f}")
            if args.wandb:
                wandb.log({"val/loss": val_loss, "val/perplexity": math.exp(val_loss), "val/step": step})

        # k. Save checkpoint
        if step > 0 and step % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step}.pt")
            save_checkpoint(model, optimizer, step, ckpt_path)
            print(f"  >>> checkpoint saved to {ckpt_path}")

    # ---------------------------------------------------------------
    # 6. Final eval & save
    # ---------------------------------------------------------------
    val_loss = evaluate(model, val_data, args.batch_size, config.context_length, device, args.eval_batches)
    print("=" * 60)
    print(f"Training done. Final val loss: {val_loss:.4f} | ppl: {math.exp(val_loss):.2f}")

    ckpt_path = os.path.join(args.checkpoint_dir, "final.pt")
    save_checkpoint(model, optimizer, args.max_steps, ckpt_path)
    print(f"Final checkpoint saved to {ckpt_path}")

    if args.wandb:
        wandb.log({"val/loss": val_loss, "val/perplexity": math.exp(val_loss)})
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
