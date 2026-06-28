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

---

## Build

**Prerequisites:** LLVM/MLIR built from source, CMake 3.20+, Ninja.

```bash
git clone https://github.com/luisroayerdi/MLIR-FlashAttention.git
cd MLIR-FlashAttention
mkdir build && cd build
cmake .. -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir -G Ninja
ninja
```

The compiler driver binary is produced at `build/bin/attention-opt`.

---

## Inspecting the IR Pipeline

The fastest way to understand the project is to watch the IR transform at each pass. All commands run from the repo root.

```bash
OPT=build/bin/attention-opt
```

### Stage 0 — Input: unfused attention

The raw 5-op sequence before any compiler transformation:

```bash
cat test/Attention/fusion.mlir
```

You will see `linalg.generic` (QK^T), `linalg.generic` (scale), `linalg.generic` (mask), `linalg.softmax`, and `linalg.matmul` (PV) as separate operations writing through intermediate `memref` buffers.

### Stage 1 — After Pass 1: Fusion

```bash
$OPT test/Attention/fusion.mlir --fusion-pass 2>/dev/null
```

The 5-op sequence collapses into a single `attention.fused` op. No intermediate buffers between the ops; the fused op carries Q, K, V, scale, mask, and output as explicit operands.

### Stage 2 — After Pass 2: Tiling

```bash
$OPT test/Attention/tiling.mlir --tiling-pass="tile-size=32" 2>/dev/null
```

`attention.fused` expands into nested `affine.for` loops implementing the online softmax algorithm. No `attention.fused` remains — output is standard affine + linalg + memref IR.

### Full pipeline (Fusion → Tiling)

```bash
$OPT test/Attention/fusion.mlir --fusion-pass --tiling-pass="tile-size=32" 2>/dev/null
```

### See all IR stages in one run

```bash
$OPT test/Attention/fusion.mlir \
  --fusion-pass \
  --tiling-pass="tile-size=32" \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-all \
  2>&1 | less
```

MLIR prints a snapshot before and after every pass. Use `q` to exit, `/attention` to search.

### Save intermediate IR to files

```bash
$OPT test/Attention/fusion.mlir --fusion-pass 2>/dev/null > /tmp/after_fusion.mlir
$OPT /tmp/after_fusion.mlir --tiling-pass="tile-size=32" 2>/dev/null > /tmp/after_tiling.mlir
```

This lets you inspect or hand-edit the IR between passes.

---

## Running the Tests

```bash
# Pass 1: verify attention.fused is produced, linalg.softmax/matmul are gone
$OPT test/Attention/fusion.mlir --fusion-pass 2>/dev/null | \
  FileCheck test/Attention/fusion.mlir && echo "PASS"

# Pass 2: verify affine.for loops produced, attention.fused is gone
$OPT test/Attention/tiling.mlir --tiling-pass="tile-size=32" 2>/dev/null | \
  FileCheck test/Attention/tiling.mlir && echo "PASS"
```

`FileCheck` is in your LLVM tools directory (e.g. `/opt/homebrew/opt/llvm/bin/FileCheck` on macOS with Homebrew LLVM).
