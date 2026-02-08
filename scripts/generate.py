#!/usr/bin/env python3
"""
Text generation script using a trained Transformer LM checkpoint.

Usage:
    python scripts/generate.py \
        --checkpoint checkpoints/final.pt \
        --config tinystories \
        --tokenizer_dir trained_tokenizers/tinystories \
        --prompt "Once upon a time" \
        --max_new_tokens 200 \
        --temperature 1.0 \
        --top_p 0.9
"""

import argparse
import torch
from cs336_basics.model import TransformerLM
from cs336_basics.model.config import get_config
from cs336_basics.tokenizers.tokenizer import BPETokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained LM")

    # ---- Model ----
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="tinystories",
                        choices=["tinystories", "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"])

    # ---- Tokenizer ----
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                        help="Directory containing vocab.json and merges.txt")

    # ---- Generation ----
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Input prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None,
                        help="Nucleus sampling threshold (e.g. 0.9)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")

    # ---- Ablation flags (must match training) ----
    parser.add_argument("--remove_rmsnorm", action="store_true")
    parser.add_argument("--use_post_norm", action="store_true")
    parser.add_argument("--remove_rope", action="store_true")
    parser.add_argument("--ffn_type", type=str, default="swiglu",
                        choices=["swiglu", "silu"])

    # ---- Device ----
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device

    # Load config
    config = get_config(args.config)

    # Build model with same ablation settings as training
    norm_type = "none" if args.remove_rmsnorm else "rmsnorm"
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=device,
        norm_type=norm_type,
        use_post_norm=args.use_post_norm,
        use_rope=not args.remove_rope,
        ffn_type=args.ffn_type,
    )

    # Load checkpoint (model weights only)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Load tokenizer
    import os
    vocab_path = os.path.join(args.tokenizer_dir, "vocab.json")
    merges_path = os.path.join(args.tokenizer_dir, "merges.txt")
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path)
    print(f"Loaded tokenizer from {args.tokenizer_dir}")

    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Generate
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print("=" * 60)

    with torch.no_grad():
        for i in range(args.num_samples):
            output_ids = model.generate(
                prompt_tensor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            generated_text = tokenizer.decode(output_ids[0].tolist())
            if args.num_samples > 1:
                print(f"\n--- Sample {i+1} ---")
            print(generated_text)

    print("=" * 60)


if __name__ == "__main__":
    main()
