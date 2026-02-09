# CS336 Assignment 1: Building a Transformer Language Model from Scratch

> A comprehensive reflection on implementing a complete Transformer language model pipeline — from BPE tokenizer to text generation — trained on TinyStories and OpenWebText.

---

## Table of Contents

1. [Overview](#1-overview)
2. [BPE Tokenizer](#2-bpe-tokenizer)
3. [Transformer Architecture](#3-transformer-architecture)
4. [Training Infrastructure](#4-training-infrastructure)
5. [Training Loop](#5-training-loop)
6. [Text Generation](#6-text-generation)
7. [Experiments](#7-experiments)
8. [Reflections](#8-reflections)

---

## 1. Overview

This assignment implements a complete Transformer language model pipeline from scratch, without relying on high-level libraries like `torch.nn.Linear` or `torch.nn.Embedding`. The codebase covers:

- **Byte-Pair Encoding (BPE) tokenizer** with parallel pre-tokenization
- **Decoder-only Transformer** with RMSNorm, RoPE, SwiGLU, and causal multi-head attention
- **Training infrastructure**: AdamW optimizer, cosine LR schedule, gradient clipping, data loading, checkpointing
- **Autoregressive text generation** with temperature and nucleus (top-p) sampling
- **Experiments**: learning rate sweeps, batch size studies, architectural ablations, and OpenWebText training

**Full code available on GitHub**: [https://github.com/XLOverflow/CS336_Transformer_from_Scratch](https://github.com/XLOverflow/CS336_Transformer_from_Scratch)

### Project Structure

```
cs336_basics/
├── model/
│   ├── linear.py              # Linear (no bias)
│   ├── embedding.py           # Token Embedding
│   ├── normalization.py       # RMSNorm
│   ├── positional_encoding.py # RoPE
│   ├── attention.py           # Softmax, Scaled Dot-Product Attention, Multi-Head Self-Attention
│   ├── feedforward.py         # SwiGLU, SiLUFFN
│   ├── transformer_block.py   # Pre-norm / Post-norm Transformer Block
│   ├── transformer_lm.py      # Full Transformer LM with generate()
│   └── config.py              # Model configurations (TinyStories, GPT-2 family)
├── tokenizers/
│   ├── bpe.py                 # BPE trainer with parallel pre-tokenization
│   └── tokenizer.py           # BPE encode/decode with parallel encoding
└── training/
    ├── cross_entropy.py       # Numerically stable cross-entropy
    ├── adamw.py               # AdamW optimizer (from scratch)
    ├── lr_schedule.py         # Cosine annealing with linear warmup
    ├── gradient_clipping.py   # L2-norm gradient clipping
    ├── data_loader.py         # Random batch sampling
    └── checkpointing.py       # Save/load model checkpoints
```

---

## 2. BPE Tokenizer

### 2.1 Unicode Basics

**Q: What's the relationship between Unicode code points and UTF-8 encoding?**

Unicode assigns each character a unique **code point** (e.g., `U+0041` for 'A'). UTF-8 is a **variable-length encoding** that maps code points to 1–4 bytes:

| Code Point Range   | UTF-8 Bytes | Example                      |
| ------------------ | ----------- | ---------------------------- |
| U+0000 – U+007F    | 1 byte      | ASCII characters             |
| U+0080 – U+07FF    | 2 bytes     | Latin, Greek, Cyrillic       |
| U+0800 – U+FFFF    | 3 bytes     | CJK characters, most emoji   |
| U+10000 – U+10FFFF | 4 bytes     | Rare emoji, historic scripts |

UTF-8 is backwards-compatible with ASCII and self-synchronizing: you can always tell if a byte is the start of a character or a continuation byte.

**Q: Why use byte-level tokenization instead of character-level?**

Byte-level tokenization starts with a base vocabulary of 256 byte values, which can represent **any** text in any language without unknown tokens. Character-level tokenization would need to handle the full Unicode range (143,000+ characters) as the base vocabulary.

### 2.2 BPE Training Algorithm

The core BPE training process:

1. **Initialize vocabulary** with 256 byte values + special tokens (e.g., `<|endoftext|>`)
2. **Pre-tokenize** corpus using GPT-2 regex pattern to split text into "words"
3. **Iteratively merge** the most frequent adjacent byte pair, adding the merged token to the vocabulary
4. Repeat until reaching target vocabulary size

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

This regex pattern handles English contractions (`'s`, `'t`, `'ll`, etc.), words with optional leading space, numbers, punctuation, and whitespace.

### 2.3 Parallelization Strategy

Training a BPE tokenizer on large corpora (e.g., OpenWebText) is computationally expensive. My implementation uses **parallel pre-tokenization** with multiprocessing:

1. **Find chunk boundaries** aligned to `<|endoftext|>` tokens to avoid splitting documents
2. **Distribute chunks** across workers using `multiprocessing.Pool`
3. Each worker applies regex pre-tokenization and returns **frequency counts** (`Counter`)
4. **Merge frequency counts** incrementally in the main process to control memory usage
5. BPE merging runs sequentially (since each merge depends on the previous one)

Key optimizations:

- **Memory management**: Periodic garbage collection and index rebuilds every 5000 merges to reduce memory fragmentation
- **Incremental pair updates**: Instead of recomputing all pair frequencies from scratch after each merge, we maintain `pair_to_tuples` and `pair_freq` indices and update only the affected entries
- **Batch processing**: Workers process chunks in batches of 16 to control concurrent memory usage

### 2.4 Tokenizer Experiments

**Vocabulary size comparison on TinyStories:**

For the TinyStories dataset, I trained tokenizers with vocab_size = 10,000. The tokenizer successfully learns common English words and subword patterns. For example:

- Common words like "the", "and", "once" become single tokens
- Less common words are split into learned subword units
- `<|endoftext|>` is handled as a special token that doesn't participate in BPE merging

**Encoding**: The encoder applies BPE merges greedily — for each pre-tokenized word, it starts with individual bytes and repeatedly merges the highest-priority pair (earliest in the merge list) until no more merges are applicable.

**Decoding**: Simply concatenates the byte values for each token ID and decodes the result as UTF-8.

---

## 3. Transformer Architecture

<img src="https://raw.githubusercontent.com/XLOverflow/blog-image/main/image-20260208223722664.png" alt="image-20260208223722664" style="zoom:50%;" />

### 3.1 Linear Layer (No Bias)

Following modern LLM practices (PaLM, LLaMA), all linear layers omit the bias term:

$$
y = xW^T
$$

**Initialization**: Truncated normal distribution $\mathcal{N}(0, \sigma^2)$ where $\sigma = \sqrt{2 / (d_{in} + d_{out})}$, truncated at $[-3\sigma, 3\sigma]$.

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x):
        return einsum(x, self.weight, "... i, o i -> ... o")
```

### 3.2 Token Embedding

Simple lookup table mapping token IDs to dense vectors:

$$
\text{embed}(x) = E[x]
$$

where $E \in \mathbb{R}^{V \times d_{model}}$ is initialized with truncated normal $\mathcal{N}(0, 1)$.

### 3.3 RMSNorm

Root Mean Square Layer Normalization (Zhang & Sennrich, 2019), used in LLaMA instead of LayerNorm:

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}
$$

Key implementation detail: **cast to float32** for numerical stability before computing RMS, then cast back to the original dtype.

```python
def forward(self, x):
    original_dtype = x.dtype
    x = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
    normalized = x / rms
    return (normalized * self.weight).to(original_dtype)
```

### 3.4 Rotary Position Embedding (RoPE)

RoPE (Su et al., 2021) encodes **relative** position information by applying rotation to query and key vectors:

$$
\text{RoPE}(x, m) = \begin{pmatrix} x_0 \cos(m\theta_0) - x_1 \sin(m\theta_0) \\ x_0 \sin(m\theta_0) + x_1 \cos(m\theta_0) \\ \vdots \\ x_{d-2} \cos(m\theta_{d/2-1}) - x_{d-1} \sin(m\theta_{d/2-1}) \\ x_{d-2} \sin(m\theta_{d/2-1}) + x_{d-1} \cos(m\theta_{d/2-1}) \end{pmatrix}
$$

where $\theta_k = \theta_{\text{base}}^{-2k/d_k}$ for $k = 0, \ldots, d_k/2 - 1$.

Key properties:

- **No learnable parameters**: RoPE is purely computed from positions and frequencies
- Applied to Q and K only (not V)
- Captures **relative** positions: $q_m^T k_n$ depends only on $m - n$
- Shared across all layers (one RoPE module instance)

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        theta_k = theta ** (-2 * torch.arange(d_k // 2, device=device) / d_k)
        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)
        angles = positions * theta_k.unsqueeze(0)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
        self.register_buffer("cos", torch.cos(angles), persistent=False)

    def forward(self, x, token_positions):
        sin, cos = self.sin[token_positions], self.cos[token_positions]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
```

### 3.5 Softmax

Numerically stable softmax using the max-subtraction trick:

$$
\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

### 3.6 Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The $\sqrt{d_k}$ scaling prevents the dot products from growing too large in magnitude, which would push the softmax into regions with extremely small gradients.

Implementation uses `einops.einsum` for clarity and supports arbitrary batch dimensions:

```python
attn_scores = einsum(q, k, "b ... q d_k, b ... k d_k -> b ... q k") / (d_k ** 0.5)
attn_scores = attn_scores.masked_fill(~mask, float("-inf"))  # causal mask
attn_weights = softmax(attn_scores, dim=-1)
return einsum(attn_weights, v, "b ... q k, b ... k d_v -> b ... q d_v")
```

### 3.7 Multi-Head Self-Attention

Splits the model dimension into multiple heads for parallel attention:

$$
\text{MultiHead}(x) = W_O \cdot \text{Concat}(\text{head}_1, \ldots, \text{head}_h)
$$

$$
\text{head}_i = \text{Attention}(xW_Q^i, xW_K^i, xW_V^i)
$$

Process:

1. Project input to Q, K, V using separate linear layers
2. Reshape to separate heads: `(batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)`
3. Apply RoPE to Q and K
4. Apply scaled dot-product attention with causal mask (lower triangular)
5. Concatenate heads and project back

### 3.8 Feed-Forward Networks

<img src="https://raw.githubusercontent.com/XLOverflow/blog-image/main/image-20260208223809903.png" alt="image-20260208223809903" style="zoom: 50%;" />

**SwiGLU** (Shazeer, 2020): Gated FFN with SiLU activation
$$
\text{SwiGLU}(x) = W_2 \cdot (\text{SiLU}(W_1 x) \odot W_3 x)
$$

where $\text{SiLU}(x) = x \cdot \sigma(x)$ and $\odot$ is element-wise multiplication.

SwiGLU uses $d_{ff} \approx \frac{8}{3} d_{model}$ (rounded to multiple of 64) with 3 weight matrices, giving total parameters $\approx 3 \times d_{model} \times \frac{8}{3} d_{model} = 8 d_{model}^2$.

**SiLU FFN** (for ablation): Standard 2-layer FFN

$$
\text{SiLUFFN}(x) = W_2 \cdot \text{SiLU}(W_1 x)
$$

Uses $d_{ff} = 4 \times d_{model}$ with 2 weight matrices, giving total parameters $\approx 2 \times d_{model} \times 4 d_{model} = 8 d_{model}^2$. This matches SwiGLU's parameter count for fair ablation comparison.

### 3.9 Transformer Block

**Pre-norm** (default):
$$
z = x + \text{Attention}(\text{RMSNorm}(x))
$$

$$
y = z + \text{FFN}(\text{RMSNorm}(z))
$$

**Post-norm** (ablation):
$$
z = \text{RMSNorm}(x + \text{Attention}(x))
$$

$$
y = \text{RMSNorm}(z + \text{FFN}(z))
$$

Pre-norm is preferred in modern LLMs because it stabilizes training — the residual connection preserves the magnitude of the input, and normalization before the sublayer prevents the activations from growing unboundedly.

### 3.10 Full Transformer LM

The complete decoder-only architecture:

1. **Token Embedding**: `token_ids → (batch, seq_len, d_model)`
2. **N Transformer Blocks**: Apply self-attention + FFN with residual connections
3. **Final RMSNorm**: Normalize the output
4. **LM Head**: Linear projection to vocabulary logits `(batch, seq_len, vocab_size)`

### 3.11 Transformer Accounting: Parameters, Memory, FLOPs & Training Time

> Let $B$ = batch_size, $T$ = context_length, $d$ = d_model, $L$ = num_layers, $H$ = num_heads, $V$ = vocab_size, $d_{ff} = 4d$

#### 3.11.1 Parameter Count $P$

| Component                                  | Parameters                         |
| ------------------------------------------ | ---------------------------------- |
| Per-layer Attention ($W_Q, W_K, W_V, W_O$) | $4d^2$                             |
| Per-layer FFN (SwiGLU: $W_1, W_2, W_3$)    | $3 \times d \times d_{ff} = 12d^2$ |
| Per-layer RMSNorm ×2                       | $2d$                               |
| Token Embedding                            | $Vd$                               |
| LM Head                                    | $Vd$                               |
| Final RMSNorm                              | $d$                                |

$$
\boxed{P = L(16d^2 + 2d) + 2Vd + d}
$$

#### 3.11.2 Training Memory Analysis

During training, GPU memory consists of four parts (float32 = 4 bytes):

| Component       | Formula   | Description                                              |
| --------------- | --------- | -------------------------------------------------------- |
| Parameters      | $4P$      | 4 bytes per parameter                                    |
| Gradients       | $4P$      | Same size as parameters                                  |
| Optimizer (m+v) | $8P$      | AdamW stores 2 tensors with the same shape as parameters |
| Activations     | See below | Proportional to batch_size                               |

**Per-layer activation memory** (intermediate results saved for backpropagation):

| Component                     | Shape          | Element Count |
| ----------------------------- | -------------- | ------------- |
| RMSNorm inputs ×2             | $(B,T,d)$ ×2   | $2BTd$        |
| Q, K, V                       | $(B,T,d)$ ×3   | $3BTd$        |
| Softmax output                | $(B,H,T,T)$    | $BHT^2$       |
| Attention output              | $(B,T,d)$      | $BTd$         |
| W1 output (for SiLU backward) | $(B,T,d_{ff})$ | $4BTd$        |
| W3 output                     | $(B,T,d_{ff})$ | $4BTd$        |
| SiLU output                   | $(B,T,d_{ff})$ | $4BTd$        |
| Gate⊙Value = W2 input         | $(B,T,d_{ff})$ | $4BTd$        |

Per-layer activations ≈ $22BTd + BHT^2$

Plus non-layer components: embedding output ($BTd$) + logits ($BTV$) + cross-entropy softmax ($BTV$) ≈ $BTd + 2BTV$

$$
\text{Total activation memory} = 4 \times \left[L(22BTd + BHT^2) + BTd + 2BTV\right] \text{ bytes}
$$

$$
\boxed{\text{Peak Memory} = 16P + 4BT\left[L(22d + HT) + d + 2V\right]}
$$

#### 3.11.3 GPT-2 XL Concrete Example

$d=1600, L=48, H=25, T=1024, V=50257$

**(a) Detailed parameter count:**

Per-layer parameters:

| Component             | Parameters                                                   |
| --------------------- | ------------------------------------------------------------ |
| $W_Q, W_K, W_V, W_O$  | $4 \times d^2 = 4 \times 2{,}560{,}000 = 10{,}240{,}000$     |
| $W_1, W_2, W_3$ (FFN) | $3 \times d \times d_{ff} = 3 \times 10{,}240{,}000 = 30{,}720{,}000$ |
| 2 × RMSNorm           | $2 \times 1{,}600 = 3{,}200$                                 |
| **Per-layer total**   | **40,963,200**                                               |

Full model:

| Component                      | Parameters                                       |
| ------------------------------ | ------------------------------------------------ |
| 48 layers                      | $48 \times 40{,}963{,}200 = 1{,}966{,}233{,}600$ |
| Token Embedding ($V \times d$) | $80{,}411{,}200$                                 |
| LM Head ($V \times d$)         | $80{,}411{,}200$                                 |
| Final RMSNorm                  | $1{,}600$                                        |
| **Total**                      | **≈ 2.13B**                                      |

Parameter memory: $2.13\text{B} \times 4 \text{ bytes} \approx 8.51 \text{ GB}$

**(b) Memory analysis:**

**Model-related memory (fixed)**: $16P = 16 \times 2.13 \times 10^9 \approx 34.0 \text{ GB}$

**Activation memory (per batch element)**:
$$
L(22d + HT) + d + 2V = 48(22 \times 1600 + 25 \times 1024) + 1600 + 2 \times 50257
$$

$$
= 48(35200 + 25600) + 102114 = 48 \times 60800 + 102114 = 2{,}920{,}514
$$

$$
\text{Per batch element}: 4 \times 1024 \times 2{,}920{,}514 \approx 12.0 \text{ GB}
$$

**Maximum batch size on 80GB A100**:
$$
\text{Total memory}: 34.0 + 12.0 \times B \leq 80 \text{ GB}
$$

$$
B \leq (80 - 34) / 12 \approx 3.8 \rightarrow \boxed{B_{\max} = 3}
$$

#### 3.11.4 Why Forward Pass ≈ 2 × Parameters FLOPs/token?

Transformer computation is dominated by matrix multiplications. For a matmul $Y = X \times W$ where $W$ has shape $(d_{in}, d_{out})$:

- Each output element requires $d_{in}$ multiplications + $d_{in}$ additions = $2d_{in}$ FLOPs
- There are $d_{out}$ output elements (per token)
- Total FLOPs = $2 \times d_{in} \times d_{out}$ = **2 × parameter count**

**Per-layer matmul breakdown** (×$L$ layers), using GPT-2 XL numbers ($d=1600, T=1024, H=25, d_k=64, d_{ff}=6400$):

| Operation           | Dimensions                        | FLOPs Formula      | FLOPs      |
| ------------------- | --------------------------------- | ------------------ | ---------- |
| Q projection        | $(T,d) \times (d,d)$              | $2Td^2$            | 5.24B      |
| K projection        | same                              | $2Td^2$            | 5.24B      |
| V projection        | same                              | $2Td^2$            | 5.24B      |
| O projection        | same                              | $2Td^2$            | 5.24B      |
| $QK^T$ ($H$ heads)  | $H \times (T,d_k) \times (d_k,T)$ | $2T^2d$            | 3.36B      |
| attn_weights × V    | $H \times (T,T) \times (T,d_k)$   | $2T^2d$            | 3.36B      |
| FFN W1              | $(T,d) \times (d,d_{ff})$         | $2Td \cdot d_{ff}$ | 20.97B     |
| FFN W3              | same                              | $2Td \cdot d_{ff}$ | 20.97B     |
| FFN W2              | $(T,d_{ff}) \times (d_{ff},d)$    | $2Td \cdot d_{ff}$ | 20.97B     |
| **Per-layer total** |                                   |                    | **90.60B** |

**Model-level FLOPs:**

| Component                     | FLOPs             |
| ----------------------------- | ----------------- |
| 48 layers                     | 4,348.7B          |
| LM Head: $(T,d) \times (d,V)$ | 164.7B            |
| **Total**                     | **≈ 4.51 TFLOPs** |

#### 3.11.5 FLOPs Breakdown Across Model Sizes

**(c)** Per-layer, FFN accounts for ~69.5% (62.91B / 90.60B), making it the most compute-heavy component. Attention projections account for 23.1%, while attention scores ($QK^T$ + attn×V) are only 7.4%.

| Component                 | Small (12L, 768) | Medium (24L, 1024) | Large (36L, 1280) | XL (48L, 1600) |
| ------------------------- | ---------------- | ------------------ | ----------------- | -------------- |
| Attn projections          | 16.6%            | 20.0%              | 21.4%             | 22.3%          |
| Attn scores ($QK^T$ etc.) | 11.1%            | 10.0%              | 8.6%              | **7.1%**       |
| FFN                       | 49.7%            | 59.9%              | 64.2%             | **66.9%**      |
| LM Head                   | 22.6%            | 10.2%              | 5.8%              | 3.7%           |

**(d) Trend**: As models grow larger, FFN's share increases (50% → 67%) while LM Head's share drops significantly (23% → 4%). This is because the LM head has a fixed size per layer (tied to vocab_size), whereas FFN grows with both num_layers and d_model.

#### 3.11.6 Context Length Scaling: Why FlashAttention Matters

**(e)** Increasing context_length from 1024 to 16384:

| Component        | T=1024    | T=16384            |
| ---------------- | --------- | ------------------ |
| Attn projections | 22.3%     | 10.8%              |
| Attn scores      | **7.1%**  | **55.2%**          |
| FFN              | 66.9%     | 32.3%              |
| LM Head          | 3.7%      | 1.8%               |
| **Total FLOPs**  | **4.51T** | **≈ 149.5T (33×)** |

Total FLOPs increase ~33× (not 16×!) because attention scores scale as $O(T^2)$. When context length grows 16×, attention scores jump from 7.1% to 55.2%, becoming the dominant cost. This is precisely why long-context models require **FlashAttention** and other IO-aware attention optimizations — the quadratic attention cost overwhelms the linear FFN cost at long sequences.

#### 3.11.7 Why Backward Pass ≈ 2× Forward?

For each matmul $Y = XW$, backpropagation requires computing two gradients:

- $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \times W^T$ (one matmul)
- $\frac{\partial L}{\partial W} = X^T \times \frac{\partial L}{\partial Y}$ (one matmul)

That's **2 matmuls** for backward vs. 1 for forward. Therefore:

$$
\boxed{\text{Backward} \approx 2 \times \text{Forward}}
$$

$$
\text{Total = Forward + Backward} \approx 3 \times \text{Forward} = 6PBT
$$

#### 3.11.8 AdamW Per-Step FLOPs

Operations performed for each parameter:

| Operation    | Formula                                    | FLOPs/param                  |
| ------------ | ------------------------------------------ | ---------------------------- |
| Update m     | $m = \beta_1 m + (1-\beta_1)g$             | 3 (2 mul + 1 add)            |
| Update v     | $v = \beta_2 v + (1-\beta_2)g^2$           | 4 (3 mul + 1 add)            |
| Param update | $p = \alpha_t \cdot m/(\sqrt{v}+\epsilon)$ | 5 (sqrt, add, div, mul, sub) |
| Weight decay | $p -= lr \times \lambda \times p$          | 2 (mul + sub)                |

$$
\text{AdamW FLOPs} = 14P
$$

(The bias correction $\alpha_t$ is a scalar computation, negligible. Much smaller than forward/backward FLOPs.)

#### 3.11.9 GPT-2 XL Training Time Estimate

**Per-step FLOPs**:

- Forward ≈ $2P \times B \times T$ (each parameter does ~2 ops per token)
- Backward ≈ $2 \times$ Forward
- Total ≈ $3 \times$ Forward = $6PBT$

Substituting GPT-2 XL ($B=1024, T=1024$):
$$
\text{Per step} = 6 \times 2.13 \times 10^9 \times 1024 \times 1024 = 1.34 \times 10^{16} \text{ FLOPs/step}
$$

**400K steps total**: $400{,}000 \times 1.34 \times 10^{16} = 5.36 \times 10^{21}$ FLOPs

**Effective throughput**: 50% × 19.5 TFLOP/s = $9.75 \times 10^{12}$ FLOP/s

$$
\text{Time} = \frac{5.36 \times 10^{21}}{9.75 \times 10^{12}} \approx 5.5 \times 10^8 \text{ sec} \approx 6{,}360 \text{ days} \approx \boxed{17.4 \text{ years}}
$$

This explains why large-scale model training requires massive GPU parallelism — training GPT-2 XL on a single A100 would take 17 years!

---

## 4. Training Infrastructure

### 4.1 Cross-Entropy Loss

Numerically stable implementation using the log-sum-exp trick:

$$
\ell_i = -\log \text{softmax}(o_i)[x_{i+1}] = \log\left(\sum_j e^{o_j - o_{\max}}\right) - (o_{x_{i+1}} - o_{\max})
$$

```python
def cross_entropy(inputs, targets):
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()
```

### 4.2 AdamW Optimizer

Implementing AdamW (Loshchilov & Hutter, 2019) with **decoupled weight decay**:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{\alpha}_t = \alpha \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
$$

$$
\theta_t = \theta_{t-1} - \hat{\alpha}_t \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} - \alpha \lambda \theta_{t-1}
$$

Key distinction from L2 regularization: weight decay is applied as a separate step using the **base learning rate** $\alpha$, not the bias-corrected rate. This is the "decoupled" part of AdamW.

**AdamW Memory Accounting**: For each parameter, AdamW maintains 2 additional tensors ($m$ and $v$), so the optimizer state requires **2× the model parameters** in memory. With the model weights themselves, the total is **3× model size** (excluding gradients). Including gradients, it's **4× model size** in float32.

### 4.3 Cosine Learning Rate Schedule with Warmup

Three phases (following LLaMA):

1. **Linear warmup** ($t < T_w$): $\alpha_t = \frac{t}{T_w} \cdot \alpha_{\max}$
2. **Cosine annealing** ($T_w \leq t \leq T_c$): $\alpha_t = \alpha_{\min} + \frac{1}{2}(1 + \cos(\frac{t - T_w}{T_c - T_w} \cdot \pi)) \cdot (\alpha_{\max} - \alpha_{\min})$
3. **Constant minimum** ($t > T_c$): $\alpha_t = \alpha_{\min}$

**Purpose of warmup**: In the early stages of training, the model parameters are randomly initialized and gradients can be very noisy and large. A learning rate warmup prevents the optimizer from taking excessively large steps that could destabilize training or cause divergence. It gives Adam's moment estimates time to accumulate meaningful statistics before using the full learning rate.

### 4.4 Gradient Clipping

L2-norm gradient clipping for training stability:

$$
\text{If } \|g\|_2 > M: \quad g \leftarrow g \cdot \frac{M}{\|g\|_2 + \epsilon}
$$

where $M$ is the max allowed norm (typically 1.0) and $\epsilon = 10^{-6}$.

---

## 5. Training Loop

### 5.1 Data Loading

For a dataset of $n$ tokens, each batch randomly samples $B$ start positions and creates:

- **Input**: `dataset[i : i + context_length]`
- **Target**: `dataset[i+1 : i+1 + context_length]`

Data is stored as memory-mapped uint16 numpy arrays for efficient random access without loading the entire dataset into RAM.

### 5.2 Checkpointing

Checkpoints save:

- `model_state_dict`: All model parameters
- `optimizer_state_dict`: Optimizer states (moments, step count)
- `iteration`: Current training step

This enables resuming training from any checkpoint with full optimizer state recovery.

### 5.3 Training Configuration

**TinyStories (default experiments):**

| Parameter      | Value         |
| -------------- | ------------- |
| Vocab size     | 10,000        |
| Context length | 256           |
| d_model        | 512           |
| Layers         | 4             |
| Heads          | 16            |
| d_ff           | 1,344         |
| Learning rate  | 1e-3 (varies) |
| Batch size     | 256 (varies)  |
| Max steps      | 5,000         |
| Warmup steps   | 500           |
| Weight decay   | 0.1           |
| Gradient clip  | 1.0           |

---

## 6. Text Generation

### 6.1 Autoregressive Generation

The model generates text one token at a time:

1. Encode the prompt into token IDs
2. Feed through the model to get logits for the next token
3. Apply **temperature scaling**: `logits / temperature`
4. (Optional) Apply **top-p / nucleus sampling**: keep only tokens whose cumulative probability ≤ p
5. Sample from the resulting distribution
6. Append the sampled token and repeat

**Temperature** controls randomness:

- `T → 0`: Greedy (argmax), deterministic but repetitive
- `T = 1.0`: Standard sampling from the model's distribution
- `T > 1.0`: More random, more diverse but potentially less coherent

**Top-p (nucleus) sampling** (Holtzman et al., 2019): Instead of sampling from the full distribution, keep only the smallest set of tokens whose cumulative probability exceeds $p$, then renormalize. This dynamically adapts the number of candidate tokens based on the model's confidence.

### 6.2 Generated Samples

Example generations from the TinyStories model (temperature=0.8, top_p=0.9):

> **Prompt**: "Once upon a time"
>
> Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball on the ground. She picked it up and started to bounce it.
> "Look, Mommy!" she said. "I found a ball!"
> Her mommy smiled and said, "That's a great find, Lily..."

The model successfully learns:

- **Coherent narrative structure** with beginning, middle, and end
- **Correct grammar** and dialogue formatting
- **Character consistency** (names, pronouns)
- **Story conventions** typical of children's stories (morals, simple conflicts)

Note: The `<|endoftext|>` token sometimes appears mid-generation. This is not a bug — it's the **document separator** used in training data. The model learned that this token marks the boundary between stories and may generate new stories after it.

---

## 7. Experiments

All experiments use the TinyStories dataset with the configuration described in Section 5.3 unless otherwise noted. Results are logged via Weights & Biases.

### 7.1 Learning Rate Sweep

**Setup**: Fixed batch_size=256, max_steps=5000, warmup=500. Sweep lr ∈ {5e-4, 1e-3, 2e-3, 5e-3, 1e-2}.

| Learning Rate | Final Val Loss | Val Perplexity |
| ------------- | -------------- | -------------- |
| 1e-2          | **1.3004**     | **3.671**      |
| 5e-3          | 1.3171         | 3.733          |
| 2e-3          | 1.3567         | 3.883          |
| 1e-3          | 1.3974         | 4.045          |
| 5e-4          | 1.4930         | 4.450          |

**Analysis**: Higher learning rates consistently achieve lower loss within 5000 steps. The best learning rate is lr=1e-2 with val loss 1.3004 and perplexity 3.671. This is somewhat surprising — one might expect such a high learning rate to cause instability, but the combination of warmup, cosine annealing, gradient clipping, and RMSNorm provides sufficient regularization.

The trend is monotonic in this range: higher LR → lower loss. This suggests the model is still in the regime where it benefits from more aggressive optimization, likely because 5000 steps is relatively few for this model size.

<img src="https://raw.githubusercontent.com/XLOverflow/blog-image/main/image-20260208225124153.png" alt="image-20260208225124153" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/XLOverflow/blog-image/main/image-20260208224958480.png" alt="image-20260208224958480" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/XLOverflow/blog-image/main/image-20260208225043193.png" alt="image-20260208225043193" style="zoom:80%;" />

### 7.2 Batch Size Experiment

**Setup**: Fixed lr=1e-3, varying batch size with proportional step adjustments to maintain roughly the same number of token updates.

| Batch Size | Steps  | Final Val Loss | Val Perplexity |
| ---------- | ------ | -------------- | -------------- |
| 16         | 80,000 | **1.3264**     | **3.768**      |
| 64         | 20,000 | 1.3318         | 3.788          |
| 128        | 10,000 | 1.3560         | 3.881          |
| 512        | 2,500  | 1.4805         | 4.395          |

**Analysis**: Smaller batch sizes achieve better final loss when training for the same number of total tokens. The best result is batch_size=16 with val loss 1.3264.

This aligns with the "generalization gap" theory: smaller batches introduce more noise in gradient estimates, which acts as implicit regularization and can lead to flatter minima with better generalization. However, smaller batches are also more computationally expensive due to lower hardware utilization.

In practice, the choice of batch size involves a trade-off between:

- **Computational efficiency**: Larger batches better utilize GPU parallelism
- **Generalization**: Smaller batches tend to generalize better
- **Convergence speed**: Smaller batches need more steps but see the same number of token

<img src="https://raw.githubusercontent.com/XLOverflow/blog-image/main/image-20260208225239934.png" alt="image-20260208225239934" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/XLOverflow/blog-image/main/image-20260208225353926.png" alt="image-20260208225353926" style="zoom:80%;" />

### 7.3 Ablation Studies

**Setup**: Fixed lr=1e-3, batch_size=256, max_steps=5000. Each ablation modifies one aspect of the baseline architecture.

| Configuration                                     | Final Val Loss | Val Perplexity | Δ Loss  |
| ------------------------------------------------- | -------------- | -------------- | ------- |
| **Baseline** (pre-norm + RMSNorm + RoPE + SwiGLU) | **1.3974**     | **4.045**      | —       |
| Post-norm (instead of pre-norm)                   | 1.4095         | 4.094          | +0.0121 |
| No RMSNorm (Identity normalization)               | 1.4400         | 4.221          | +0.0426 |
| SiLU FFN (instead of SwiGLU)                      | 1.4649         | 4.327          | +0.0675 |
| No RoPE (NoPE — no positional encoding)           | 1.4712         | 4.354          | +0.0738 |

**Analysis by component importance** (most to least critical):

1. **RoPE** (Δ = +0.074): The most impactful component. Without positional encoding, the model has no way to distinguish token order. Remarkably, NoPE still achieves reasonable perplexity (4.354), suggesting that the model can partially infer order from semantic context and causal masking alone. But positional information clearly provides a significant boost.

2. **SwiGLU** (Δ = +0.068): Replacing SwiGLU with SiLU FFN (matched parameter count) hurts by 0.068 in loss. The gating mechanism in SwiGLU provides finer control over information flow through the FFN, leading to better representation learning.

3. **RMSNorm** (Δ = +0.043): Removing normalization entirely degrades performance, confirming that normalization is important for training stability and representation quality. Without it, activations can grow unboundedly through the residual connections.

4. **Pre-norm vs Post-norm** (Δ = +0.012): The smallest difference. Post-norm slightly underperforms pre-norm, consistent with the literature showing that pre-norm is more training-stable. However, the gap is small for this model size and training duration.

<img src="https://raw.githubusercontent.com/XLOverflow/blog-image/main/image-20260208230125952.png" alt="image-20260208230125952" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/XLOverflow/blog-image/main/image-20260208230058900.png" alt="image-20260208230146528" style="zoom:100%;" />

### 7.4 OpenWebText (OWT) Training

**Setup**: GPT-2 Small architecture (117M parameters) trained on OpenWebText.

| Parameter      | Value       |
| -------------- | ----------- |
| Config         | GPT-2 Small |
| Vocab size     | 50,257      |
| Context length | 1,024       |
| d_model        | 768         |
| Layers         | 12          |
| Heads          | 12          |
| d_ff           | 2,048       |
| Batch size     | 8           |
| Max steps      | 10,000      |
| LR             | 1e-3        |

| Metric               | Value  |
| -------------------- | ------ |
| Final Val Loss       | 3.9364 |
| Final Val Perplexity | 51.236 |

**Analysis**: The OWT training achieves a validation perplexity of ~51, which is reasonable for 10K steps of training on a 117M parameter model. For reference:

- GPT-2 (117M) trained for 300K steps achieves perplexity ~30 on WebText
- Our model has seen far fewer tokens but shows clear learning (loss decreasing throughout training)

The main bottleneck was **GPU memory**: batch_size=64 with context_length=1024 caused OOM on a single 80GB A100. Reducing to batch_size=8 resolved this, but it means each step processes fewer tokens. Gradient accumulation could be used to simulate larger effective batch sizes without additional memory cost.

<img src="https://raw.githubusercontent.com/XLOverflow/blog-image/main/image-20260208230235210.png" alt="image-20260208230235210" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/XLOverflow/blog-image/main/image-20260208230218497.png" alt="image-20260208230218497" style="zoom:80%;" />

---

## 8. Reflections

### 8.1 What I Learned

**Implementing from scratch matters.** Building every component from `nn.Parameter` and `torch.empty` forces you to understand the exact data flow, shapes, and numerical considerations at each step. For example:

- **RMSNorm precision**: Without casting to float32, the mean-of-squares computation can overflow in float16/bfloat16, leading to NaN losses. This is a subtle bug that wouldn't be caught by unit tests in float32.

- **Weight initialization**: The truncated normal initialization significantly impacts training stability. Too wide → gradient explosion; too narrow → vanishing gradients. The $\sigma = \sqrt{2/(d_{in} + d_{out})}$ formula (Glorot-like) keeps the variance roughly constant across layers.

- **Causal mask efficiency**: Pre-computing the causal mask as a buffer and slicing it per forward pass is much more efficient than creating it fresh each time, especially for long sequences.

**Tokenizer training is the hidden bottleneck.** BPE training on large corpora requires careful memory management:

- Pre-tokenization can generate millions of unique byte sequences
- Pair frequency tables can grow to tens of GB
- Without incremental index updates, each merge iteration would be O(corpus_size)
- Parallel pre-tokenization provides near-linear speedup, but BPE merging remains sequential

**Hyperparameter sensitivity varies by component.** Learning rate has the largest impact on training dynamics, while architectural choices (pre-norm vs post-norm) can have surprisingly small effects for small models. This suggests that for quick experiments, spending time on LR tuning is more valuable than architectural variations.

### 8.2 Design Decisions

1. **Shared RoPE module**: Instead of each attention layer creating its own RoPE, a single instance is shared across all layers. This saves memory and ensures consistent positional encoding.

2. **Ablation support via constructor arguments**: Rather than creating separate model classes for each ablation, the `TransformerBlock` and `TransformerLM` accept configuration flags (`norm_type`, `use_post_norm`, `use_rope`, `ffn_type`). This keeps the codebase DRY while supporting all experimental variations.

3. **Memory-mapped data loading**: Using `np.memmap` for training data avoids loading the entire dataset into RAM. Random batch sampling then reads only the needed slices, making it feasible to train on datasets much larger than available memory.

4. **Parallel tokenizer encoding**: The `encode_parallel` method splits text at `<|endoftext|>` boundaries, encodes chunks in parallel, saves intermediate results to disk, and merges them. This supports resume and avoids OOM on large texts.

### 8.3 Things I Would Do Differently

- **Learning rate warmup tuning**: I used a fixed 500 warmup steps across all experiments. Tuning this per-configuration (e.g., proportional to total steps) might improve results.
- **Gradient accumulation**: For the OWT experiment, implementing gradient accumulation would allow using an effective batch size of 64+ while staying within GPU memory limits with batch_size=8 per step.
- **Mixed precision training**: Using `torch.cuda.amp` with bfloat16 would reduce memory usage and increase throughput, potentially enabling larger batch sizes or more steps.
- **KV-cache for generation**: The current generation implementation recomputes all attention scores from scratch for each new token. A KV-cache would store past key-value pairs, reducing generation cost from O(n²) to O(n) per token.

---

*This blog post documents the implementation of CS336 Assignment 1 (Spring 2025). All code was written from scratch in PyTorch, with experiments run on NVIDIA H100 GPUs via the Pittsburgh Supercomputing Center (PSC).*
