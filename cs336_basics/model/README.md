# Transformer Language Model 实现指南

本目录包含了一个完整的Transformer语言模型的骨架实现，基于CS336 Assignment 1的第三章要求。

## 📁 文件结构

```
model/
├── __init__.py              # 模块导出
├── config.py                # 模型配置（已完成）
├── linear.py                # 线性层（待实现）
├── embedding.py             # 词嵌入层（待实现）
├── normalization.py         # RMSNorm（待实现）
├── positional_encoding.py   # RoPE位置编码（待实现）
├── attention.py             # 注意力机制（待实现）
├── feedforward.py           # SwiGLU前馈网络（待实现）
├── transformer_block.py     # Transformer块（待实现）
├── transformer_lm.py        # 完整的Transformer LM（待实现）
└── README.md                # 本文件
```

## 🎯 实现顺序建议

建议按照以下顺序实现各个组件，每个组件实现后都可以通过简单的测试验证：

### 1. 基础组件（第1-2天）

#### 1.1 Linear (`linear.py`)
- **难度**: ⭐
- **关键点**:
  - 矩阵存储为 `W` (out_features, in_features)
  - 前向传播: `y = xW^T`
  - 初始化: truncated normal `N(0, 2/(d_in + d_out))`

#### 1.2 Embedding (`embedding.py`)
- **难度**: ⭐
- **关键点**:
  - 简单的索引查找
  - 初始化: truncated normal `N(0, 1)`

#### 1.3 RMSNorm (`normalization.py`)
- **难度**: ⭐⭐
- **关键点**:
  - 上转换到 float32 避免数值溢出
  - RMS = sqrt(mean(x^2) + eps)
  - 输出 = (x / RMS) * gain

### 2. 位置编码（第3天）

#### 2.1 RoPE (`positional_encoding.py`)
- **难度**: ⭐⭐⭐
- **关键点**:
  - 预计算 sin/cos 值存储为 buffer
  - 角度计算: `theta_k = theta^(-2k/d_k)`
  - 旋转应用到相邻的元素对

### 3. 注意力机制（第4-5天）

#### 3.1 Softmax (`attention.py`)
- **难度**: ⭐⭐
- **关键点**:
  - 减去最大值以保证数值稳定性

#### 3.2 ScaledDotProductAttention (`attention.py`)
- **难度**: ⭐⭐⭐
- **关键点**:
  - 计算 `QK^T / sqrt(d_k)`
  - 应用causal mask（未来位置设为 -inf）
  - Softmax后与V相乘

#### 3.3 MultiHeadSelfAttention (`attention.py`)
- **难度**: ⭐⭐⭐⭐
- **关键点**:
  - Q, K, V投影后reshape为多头
  - 只对Q和K应用RoPE（不对V）
  - 创建causal mask防止看到未来

### 4. 前馈网络（第6天）

#### 4.1 SwiGLU (`feedforward.py`)
- **难度**: ⭐⭐
- **关键点**:
  - SiLU(x) = x * sigmoid(x)
  - 门控: SiLU(W1*x) ⊙ W3*x
  - 输出: W2 * (门控结果)

### 5. 组装模型（第7-8天）

#### 5.1 TransformerBlock (`transformer_block.py`)
- **难度**: ⭐⭐
- **关键点**:
  - Pre-norm架构
  - 残差连接

#### 5.2 TransformerLM (`transformer_lm.py`)
- **难度**: ⭐⭐⭐
- **关键点**:
  - 共享RoPE模块
  - 最后的RMSNorm
  - 生成函数的实现

## 🔧 实现技巧

### 使用 einsum 简化代码

```python
# 不推荐：多次 reshape 和 transpose
x = x.view(batch, seq, heads, d_k).transpose(1, 2)

# 推荐：使用 einsum
from einops import rearrange
x = rearrange(x, 'batch seq (heads d_k) -> batch heads seq d_k', heads=num_heads)
```

### 处理批次维度

所有操作都应该支持任意数量的批次维度：
```python
# 输入可能是: (batch, seq, d) 或 (batch, heads, seq, d)
# 使用 ... 来处理: (..., d) -> (..., d_out)
```

### 数值稳定性

1. **RMSNorm**: 始终上转换到 float32
2. **Softmax**: 减去最大值
3. **注意力**: 缩放因子 1/sqrt(d_k)

## ✅ 测试策略

### 单元测试
```python
def test_linear():
    linear = Linear(512, 2048)
    x = torch.randn(32, 128, 512)
    out = linear(x)
    assert out.shape == (32, 128, 2048)
```

### 梯度检查
```python
# 确保梯度可以正确反向传播
x = torch.randn(2, 4, 512, requires_grad=True)
model = TransformerBlock(512, 8, 1344, rope)
out = model(x)
loss = out.sum()
loss.backward()
assert x.grad is not None
```

### 过拟合小批次
```python
# 在单个批次上过拟合，确保模型可以学习
batch = next(iter(dataloader))
for _ in range(100):
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
# Loss应该接近0
```

## 📊 模型大小参考

基于 TinyStories 配置 (vocab=10000, ctx=256, d=512, layers=4, heads=16):

| 组件 | 参数量 |
|------|--------|
| Token Embedding | 5.12M |
| Transformer Blocks | ~12M |
| Output Linear | 5.12M |
| **总计** | **~17M** |

## 🚀 工业标准实践

本实现遵循现代LLM的最佳实践：

1. **Pre-norm架构** - 更稳定的训练（LLaMA, GPT-3）
2. **RMSNorm** - 比LayerNorm更简单高效
3. **SwiGLU** - 比ReLU性能更好
4. **RoPE** - 相对位置编码，无需学习
5. **无bias** - 跟随PaLM, LLaMA的做法
6. **Causal mask** - 防止看到未来token

## 📚 参考资源

- **原始论文**: Vaswani et al., "Attention is All You Need" (2017)
- **Pre-norm**: Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020)
- **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- **SwiGLU**: Shazeer, "GLU Variants Improve Transformer" (2020)
- **现代实现**: LLaMA, GPT-3, PaLM

## 🐛 常见问题

### Q: 为什么要用 W 而不是 W^T 存储？
A: PyTorch使用行优先存储，存储 W 可以更高效地进行 xW^T 计算。

### Q: RoPE应该应用到哪些向量？
A: 只应用到 Query 和 Key，不应用到 Value。

### Q: Pre-norm 和 Post-norm 有什么区别？
A: Pre-norm在子层之前归一化，Post-norm在子层之后。Pre-norm训练更稳定。

### Q: 为什么 d_ff 要是64的倍数？
A: GPU的tensor核心在处理64的倍数时效率最高。

## 💡 优化建议

1. **使用 torch.compile()**: PyTorch 2.0+
2. **Flash Attention**: 更高效的注意力实现
3. **混合精度训练**: torch.cuda.amp
4. **梯度累积**: 模拟更大的batch size

## 🎓 学习检查清单

- [ ] 理解注意力机制的数学原理
- [ ] 掌握 einsum 操作
- [ ] 理解 RoPE 如何编码位置信息
- [ ] 知道何时需要 causal mask
- [ ] 理解残差连接的作用
- [ ] 能够计算模型参数量和FLOPs

## 📝 实现后的扩展

完成实现后，可以进行以下扩展：

1. **实现训练循环** (Assignment §4-5)
   - Cross-entropy loss
   - AdamW optimizer
   - Learning rate scheduling
   - Gradient clipping

2. **实现数据加载** (Assignment §5)
   - 使用 mmap 处理大文件
   - Batch sampling
   - Checkpointing

3. **实现文本生成** (Assignment §6)
   - Temperature scaling
   - Top-p (nucleus) sampling
   - Beam search

祝你实现顺利！🚀
