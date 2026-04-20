# Deconstructing FlashAttention: A Compiler-Centric Analysis of Attention Kernel Optimization 

## Introduction

### The Problem: Attention is Slow

Transformer models power modern AI (GPT, Claude, etc.), and their core operation is **attention**. The standard attention computation follows this formula:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Breaking this down into steps:

1. **MatMul**: Multiply query (Q) by key transpose (K^T)
2. **Scale**: Divide by √d_k (dimension size)
3. **Mask**: Apply causal mask (prevent looking at future tokens)
4. **Softmax**: Normalize to probabilities
5. **MatMul**: Multiply result by values (V)

**The bottleneck:** Running these as separate operations means repeatedly writing intermediate results to slow GPU memory (HBM - High Bandwidth Memory). For a 1024-token sequence, this creates gigabytes of memory traffic.

### FlashAttention's Solution

FlashAttention (Dao et al., 2022-2025) demonstrated dramatic speedups by **fusing** all operations into a single GPU kernel. Instead of:

```
GPU Memory → Load Q,K → Compute QK^T → Write to Memory
           → Load QK^T → Scale → Write to Memory
           → Load scaled → Mask → Write to Memory
           → Load masked → Softmax → Write to Memory
```

FlashAttention does:

```
GPU Memory → Load tile of Q,K → Compute+Scale+Mask+Softmax in fast SRAM → Write final result
```

**Key insight:** Keep intermediate values in fast on-chip memory (SRAM) instead of slow off-chip memory (HBM).

**The catch:** FlashAttention is written in hand-optimized CUDA code - thousands of lines specific to NVIDIA GPUs.

### The Research Question

**Can we express FlashAttention's optimizations as reusable compiler passes instead of hand-written kernels?**

**Why this matters:**

- **Portability:** Compiler passes work across GPU vendors (NVIDIA, AMD, Intel)
- **Maintainability:** High-level passes are easier to understand and modify than CUDA
- **Reusability:** Same optimization strategy can apply to other operations beyond attention
- **Transparency:** Hand-written kernels are opaque; compiler passes are inspectable

**Our goal is NOT to beat FlashAttention's performance.** Instead, we ask:

1. Which FlashAttention optimizations can be expressed as compiler transformations?
2. How much performance can we recover using only compiler infrastructure?
3. What gaps remain, and what would compilers need to close them?
