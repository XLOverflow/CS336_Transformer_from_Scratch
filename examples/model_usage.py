"""
Transformer Model Usage Examples

This script demonstrates how to use the Transformer components.
"""

import torch
import sys
sys.path.append('..')

from cs336_basics.model import (
    Linear,
    Embedding,
    RMSNorm,
    RotaryPositionalEmbedding,
    ScaledDotProductAttention,
    MultiHeadSelfAttention,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
)
from cs336_basics.model.config import get_config, TinyStoriesConfig


def example_basic_components():
    """Example usage of basic components."""
    print("=" * 80)
    print("Basic Components Example")
    print("=" * 80)

    # Linear layer
    linear = Linear(512, 2048)
    x = torch.randn(32, 128, 512)  # (batch, seq_len, in_features)
    out = linear(x)
    print(f"Linear: {x.shape} -> {out.shape}")

    # Embedding
    embedding = Embedding(10000, 512)
    token_ids = torch.randint(0, 10000, (32, 128))
    embedded = embedding(token_ids)
    print(f"Embedding: {token_ids.shape} -> {embedded.shape}")

    # RMSNorm
    norm = RMSNorm(512)
    normalized = norm(x)
    print(f"RMSNorm: {x.shape} -> {normalized.shape}")


def example_attention():
    """Example usage of attention mechanisms."""
    print("\n" + "=" * 80)
    print("Attention Example")
    print("=" * 80)

    batch, seq_len, d_model = 32, 128, 512
    num_heads = 8

    # Scaled Dot-Product Attention
    attn = ScaledDotProductAttention()
    q = k = v = torch.randn(batch, num_heads, seq_len, d_model // num_heads)
    out = attn(q, k, v)
    print(f"Scaled Dot-Product Attention: {q.shape} -> {out.shape}")

    # Multi-Head Self-Attention
    rope = RotaryPositionalEmbedding(theta=10000, d_k=d_model // num_heads, max_seq_len=2048)
    mha = MultiHeadSelfAttention(d_model, num_heads, rope)
    x = torch.randn(batch, seq_len, d_model)
    out = mha(x)
    print(f"Multi-Head Self-Attention: {x.shape} -> {out.shape}")


def example_transformer_block():
    """Example usage of Transformer block."""
    print("\n" + "=" * 80)
    print("Transformer Block Example")
    print("=" * 80)

    d_model, num_heads, d_ff = 512, 8, 1344
    batch, seq_len = 32, 128

    rope = RotaryPositionalEmbedding(
        theta=10000,
        d_k=d_model // num_heads,
        max_seq_len=2048
    )

    block = TransformerBlock(d_model, num_heads, d_ff, rope)
    x = torch.randn(batch, seq_len, d_model)
    out = block(x)
    print(f"Transformer Block: {x.shape} -> {out.shape}")


def example_transformer_lm():
    """Example usage of complete Transformer LM."""
    print("\n" + "=" * 80)
    print("Transformer LM Example")
    print("=" * 80)

    # Using predefined config
    config = get_config('tinystories')
    print(f"Config: {config}")

    # Create model
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    )

    # Count parameters
    num_params = model.count_parameters()
    print(f"Number of parameters: {num_params:,}")

    # Forward pass
    batch_size, seq_len = 32, 128
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits = model(token_ids)
    print(f"Forward pass: {token_ids.shape} -> {logits.shape}")

    # Generate
    prompt = torch.randint(0, config.vocab_size, (1, 10))  # Single prompt
    generated = model.generate(
        prompt,
        max_new_tokens=50,
        temperature=1.0,
        top_p=0.9,
    )
    print(f"Generation: {prompt.shape} -> {generated.shape}")


def example_custom_config():
    """Example of creating custom configuration."""
    print("\n" + "=" * 80)
    print("Custom Configuration Example")
    print("=" * 80)

    from cs336_basics.model.config import TransformerConfig

    # Create custom config
    custom_config = TransformerConfig(
        vocab_size=32000,
        context_length=512,
        d_model=768,
        num_layers=6,
        num_heads=12,
        d_ff=2048,  # 8/3 * 768 ≈ 2048
        rope_theta=10000.0,
    )

    print(f"Custom config: {custom_config}")

    # Create model from custom config
    model = TransformerLM(
        vocab_size=custom_config.vocab_size,
        context_length=custom_config.context_length,
        d_model=custom_config.d_model,
        num_layers=custom_config.num_layers,
        num_heads=custom_config.num_heads,
        d_ff=custom_config.d_ff,
        rope_theta=custom_config.rope_theta,
    )

    print(f"Model parameters: {model.count_parameters():,}")


if __name__ == "__main__":
    # Run all examples
    try:
        example_basic_components()
        example_attention()
        example_transformer_block()
        example_transformer_lm()
        example_custom_config()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
    except NotImplementedError as e:
        print(f"\n⚠️  Some components not yet implemented: {e}")
        print("Please implement the missing components in the model files.")
