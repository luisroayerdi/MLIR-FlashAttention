# Deconstructing FlashAttention: A Compiler-Centric Analysis of Attention Kernel Optimization

## Research Presentation Document

---

## 1. Introduction

### 1.1 The Problem: Attention is Slow

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

### 1.2 FlashAttention's Solution

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

### 1.3 The Research Question

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

## 2. Background

### 2.1 MLIR: Multi-Level Intermediate Representation

MLIR is a compiler framework built by Google for ML workloads. Key idea: **progressive lowering** through multiple abstraction levels.

**MLIR Dialects** (different abstraction levels):

```
High Level:    linalg (linear algebra operations)
                 ↓
Mid Level:     vector (explicit vectorization)
                 ↓
Low Level:     nvgpu (NVIDIA GPU specifics)
                 ↓
Final:         LLVM (machine code generation)
```

**Example - Matrix Multiply Lowering:**

**Level 1 (linalg):**

```mlir
linalg.matmul ins(%A, %B) outs(%C)  // Abstract: "multiply matrices"
```

**Level 2 (scf - loops):**

```mlir
for i in 0..M:
  for j in 0..N:
    for k in 0..K:
      C[i,j] += A[i,k] * B[k,j]  // Explicit loops
```

**Level 3 (vector):**

```mlir
vector.load %A  // Load 8 elements at once
vector.fma      // Fused multiply-add
```

**Level 4 (nvgpu):**

```mlir
nvgpu.ldmatrix  // Use tensor core instructions
```

Each level is a **transformation pass** that converts one representation to another.
### 2.2 FlashAttention Evolution

**FlashAttention 1 (2022):**

- Core innovation: Tiling + fusion
- Keep tiles in SRAM, avoid HBM writes
- ~2-4x faster than standard attention

**FlashAttention 2 (2023):**

- Better parallelization across GPU cores
- Reduced non-matrix-multiply overhead
- ~2x faster than FA1

**FlashAttention 3 (2024):**

- Warp-specialization (different thread groups do different work)
- Asynchronous memory operations
- Ping-pong buffering
- ~1.5-2x faster than FA2

**FlashAttention 4 (2025):**

- Dynamic load balancing
- Adaptive tile sizing
- Better handling of variable sequence lengths

**Key observation:** Each version adds optimizations that are increasingly **low-level** and **hardware-specific**.

---

## 3. Research Methodology

### 3.1 Research Design

This is an **empirical study** that builds a compiler artifact and measures its capabilities.

**Research approach:**

1. **Decompose** FlashAttention optimizations into individual techniques
2. **Implement** each as a standalone MLIR compiler pass
3. **Evaluate** which optimizations are expressible and their performance impact
4. **Analyze** what gaps remain between compiler-generated and hand-tuned code

**Output:** A taxonomy of attention optimizations by compiler expressibility.

### 3.2 Compiler Pipeline Architecture

We build a **modular MLIR pipeline** where each optimization is a separate pass:

```
Input: linalg IR (unfused attention operations)
  ↓
Pass 1: Fusion (merge 4 ops)
  ↓
Pass 2: Memory-aware tiling (fit in SRAM)
  ↓
Pass 3: Vectorization (use SIMD)
  ↓
Pass 4: Causal mask specialization (boundary tiles)
  ↓
Pass 5: Backend lowering (nvgpu dialect)
  ↓
Output: Executable GPU kernel
```

Each pass allow us to measure impact individually.
### 3.3 Pass Descriptions

#### **Pass 1: Operation Fusion**

**What it does:** Merge MatMul → Scale → Mask → Softmax into a single operation.

**Input IR:**

```mlir
%qk = linalg.matmul ins(%Q, %K)
%scaled = linalg.generic { arith.divf %qk, %scale }
%masked = linalg.generic { arith.select %mask, %scaled, %neg_inf }
%probs = linalg.softmax ins(%masked)
```

**Output IR:**

```mlir
%probs = custom.fused_attention ins(%Q, %K, %scale, %mask)
```

**Why this matters:** Single operation allow us to use a single kernel launch and therefore have no intermediate memory writes.

**Implementation strategy:**

- Pattern matching at linalg level
- Create custom `fused_attention` operation
- Preserve semantics

**Validation:**

- Unit test: compare fused vs unfused output numerically
- Test edge cases: empty sequences, all-masked tokens

---
#### **Pass 2: Memory-Aware Tiling**

**What it does:** Break large matrices into tiles that fit in fast SRAM. This due to the impact tiling has on memory access patterns, if we have a poor tiling our kernel will also be slow.

**Example:** For 1024×1024 attention on A100 GPU:

- SRAM size: 192 KB per SM (Streaming Multiprocessor)
- Tile size: 128×128 (fits in SRAM with room for intermediate values)

**Algorithm:**

```
1. Measure available SRAM
2. Calculate max tile size given op memory footprint
3. Insert tiling loops in IR
4. Ensure tile boundaries align with tensor core dimensions (16×16)
```

**Input IR:**

```mlir
linalg.matmul ins(%Q, %K) outs(%output)  // 1024×1024
```

**Output IR:**

```mlir
for tile_i in 0..8:     // 1024/128 = 8 tiles
  for tile_j in 0..8:
    %Q_tile = extract_tile(%Q, tile_i)  // 128×1024
    %K_tile = extract_tile(%K, tile_j)  // 1024×128
    linalg.matmul ins(%Q_tile, %K_tile) outs(%output_tile)
```

**Parameters to explore:**

- Static tile size (compile-time constant)
- Dynamic tile size (runtime-determined)
- Multi-level tiling (L1 cache, SRAM, HBM hierarchy)

**Validation:**

- Memory traffic profiling (via NVIDIA nsight compute)
- SRAM utilization percentage
- Compare against optimal tile size from roofline model

---

#### **Pass 3: Vectorization**

**What it does:** Convert scalar operations to vector operations (SIMD - Single Instruction, Multiple Data).

**Example - Scalar version:**

```mlir
for i in 0..1024:
  output[i] = input[i] + 1.0  // One operation per iteration
```

**Vectorized version:**

```mlir
for i in 0..128:  // 1024/8 = 128 iterations
  %vec = vector.load input[i*8 : i*8+8]  // Load 8 elements
  %result = vector.addf %vec, %ones      // 8 adds in one instruction
  vector.store %result, output[i*8 : i*8+8]
```

**Importance:** Modern GPUs execute 32-128 operations per instruction via vectorization.

**Implementation:**

- Detect vectorizable loops
- Insert vector.load / vector.store
- Handle remainders (when size not divisible by vector width)

**Validation:**

- Check assembly: verify vector instructions generated
- Measure throughput improvement

---

#### **Pass 4: Causal Mask Specialization**

**What it does:** Handle causal masking efficiently by generating specialized code for different tile types.

**The problem:** In autoregressive attention (GPT-style), tokens can't attend to future positions:

```
Attention matrix (4×4 example):
[✓ ✗ ✗ ✗]   ✓ = allowed
[✓ ✓ ✗ ✗]   ✗ = masked
[✓ ✓ ✓ ✗]
[✓ ✓ ✓ ✓]
```

When tiled, we get three cases:

```
Diagonal tiling (128×128 tiles on 1024×1024):

[Full  Masked Masked ...]   Full = all elements valid
[Full  Bound  Masked ...]   Bound = partial mask (straddles diagonal)
[Full  Full   Bound  ...]   Masked = all elements masked
[...]
```

**Approach:** Generate three kernel variants:

1. **Full tile kernel:** No mask checks (fast path)
2. **Masked tile kernel:** Skip computation entirely (early exit)
3. **Boundary tile kernel:** Only check mask at tile edges

**IR transformation:**

```mlir
// Before: generic kernel with mask checks
for tile in tiles:
  if mask[tile]: compute(tile)

// After: specialized kernels
for tile in full_tiles:
  compute_full(tile)          // No branching
for tile in boundary_tiles:
  compute_boundary(tile)      // Minimal branching
// Skip masked tiles entirely
```

**Importance:** Branching (if statements) kills GPU performance. Specialization removes branches.

**Validation:**

- Correctness: output matches masked PyTorch attention
- Performance: measure speedup vs generic masking
- Coverage: ensure all tiles classified correctly

---

#### **Pass 5: Backend Lowering (nvgpu)**

**What it does:** Convert to NVIDIA-specific instructions (tensor cores, async copy).

**Tensor cores:** Special hardware units on NVIDIA GPUs (Ampere/Hopper) that do 16×16×16 matrix multiply in one instruction.

**Standard GPU computation:**

```
for i, j, k in nested_loops:
  C[i,j] += A[i,k] * B[k,j]  // Thousands of scalar operations
```

**Tensor core computation:**

```
tensor_core_mma(A_tile, B_tile, C_tile)  // 4096 operations in one instruction
```

**Pass transformation:**

```mlir
// Before: generic matmul
linalg.matmul ins(%A, %B) outs(%C)

// After: nvgpu tensor core ops
%A_frag = nvgpu.ldmatrix %A  // Load into tensor core format
%B_frag = nvgpu.ldmatrix %B
%C_frag = nvgpu.mma %A_frag, %B_frag, %C_acc  // Tensor core multiply
nvgpu.stmatrix %C_frag, %C  // Store result
```

**Importance:** Tensor cores are 8-16x faster than standard GPU cores for matmul.

**Challenges:**

- Alignment requirements (operands must be 16×16 aligned)
- Data layout transformations (row-major → tensor core layout)
- Async memory operations (overlap compute and memory transfer)

**Validation:**

- Check generated PTX assembly includes `mma` instructions
- Measure achieved TFLOPS vs theoretical peak

---

### 3.4 Evaluation Metrics

We measure **four dimensions:**

#### **1. Correctness**

- **Numerical accuracy:** Max absolute error vs PyTorch reference
- **Edge case handling:** Empty sequences, all-masked, numerical stability
- **Test suite:** 100+ test cases covering sequence lengths, batch sizes

#### **2. Performance**

- **Throughput:** Tokens/second for various sequence lengths
- **Memory bandwidth:** GB/s measured via profiler
- **Kernel launch overhead:** Time spent launching kernels vs computing
- **Roofline analysis:** How close to theoretical hardware limits

#### **3. Compiler Analysis**

- **Pass effectiveness:** Speedup from each individual pass
- **Compile time:** How long each pass takes
- **IR complexity:** Number of operations before/after each pass

#### **4. Expressiveness**

- **Which optimizations are implementable?** Taxonomy table
- **What's left on the table?** Gap between our impl and FlashAttention
- **What would it take to close the gap?** Compiler feature recommendations

---

### 3.5 Baseline Comparisons

We compare against four baselines:

**Baseline 1: Unfused PyTorch**

```python
qk = torch.matmul(Q, K.T)
scaled = qk / math.sqrt(d_k)
masked = scaled.masked_fill(mask, float('-inf'))
probs = F.softmax(masked, dim=-1)
output = torch.matmul(probs, V)
```

- **Purpose:** Measures cost of NOT fusing
- **Expected result:** Our worst-case lower bound

**Baseline 2: torch.compile (Triton)**

```python
@torch.compile
def attention(Q, K, V, mask):
    # Same ops as above
```

- **Purpose:** State-of-art automatic fusion
- **Expected result:** Our competitive target

**Baseline 3: FlashAttention-2 (via Triton)**

```python
from flash_attn import flash_attn_func
output = flash_attn_func(Q, K, V, causal=True)
```

- **Purpose:** Hand-optimized reference
- **Expected result:** Upper bound (we won't beat this)

**Baseline 4: Our MLIR Pipeline**

- Multiple variants (pass ablation study)
- Incrementally enabling each optimization

---

### 3.6 Experimental Setup

**Hardware:**

- **Primary:** NVIDIA A100 GPU (80GB, Ampere architecture)
- **Secondary:** NVIDIA H100 (if available, Hopper architecture)
- **Fallback:** RTX 4090 (consumer GPU, limited SRAM)

**Software:**

- MLIR: Latest stable release (or main branch)
- PyTorch 2.x with torch.compile
- CUDA 12.x
- FlashAttention-2/3 installed via pip

**Workloads:**

```
Sequence lengths: [512, 1024, 2048, 4096, 8192]
Batch sizes: [1, 8, 16, 32]
Head dimension: 64 (standard for models)
Number of heads: 16
```

**Measurement protocol:**

- Warmup: 10 iterations (exclude from timing)
- Measurement: 100 iterations, report median
- Profiling: Use NVIDIA nsight compute for detailed metrics
- Reproducibility: Fix random seeds, report variance

---

## 4. Implementation Plan

### 4.1 Development Phases (Currently working on Phase 3)

**Phase 1: Foundation

- Set up MLIR build environment
- Implement simple matmul lowering (validation)
- Create testing infrastructure
- **Deliverable:** MatMul executes on CPU

**Phase 2: Fusion Pass**

- Implement pattern matching for 4-op sequence
- Create fused_attention custom operation
- Lower fused op to loops

**Phase 3: Tiling Pass**

- Implement tile size calculation
- Insert tiling loops in IR
- Handle boundary conditions

**Phase 4: GPU Lowering**

- Port to nvgpu dialect
- Implement tensor core lowering
- Optimize memory access patterns
- **Deliverable:** GPU executable, profile shows tensor core usage

**Phase 5: Mask**

- Implement tile classification
- Generate three kernel variants
- Integrate into pipeline
- **Deliverable:** Causal attention working, speedup measured

**Phase 6: Evaluation**

- Run full benchmark suite
- Generate comparison tables
- Profile all baselines
- **Deliverable:** Complete data set

**Phase 7: Analysis**

- Analyze results
- Write report
- Create presentation
- **Deliverable:** Final document

---

## 5. Success Criteria

### 5.1 Minimum Viable Success

**What must work:**

- Fusion pass implemented and tested
- Tiling pass implemented and tested
- Pipeline generates executable code (CPU or GPU)
- Outperforms unfused baseline measurably
- Correctness validated against PyTorch

**Outcome:** Demonstrates compiler approach is viable, even if incomplete.

---

### 5.2 Target Success

**Additional achievements:**

- GPU lowering working
- Mask specialization pass implemented
- Comparison table complete
- Gap analysis with concrete recommendations

---

### 5.3 Stretch Goals

**Aspirational goals:**

- Matches torch.compile performance
- One FA3 optimization implemented
- Multi-backend support (NVIDIA + AMD)
- Pass upstreamed to MLIR mainline

---

## 6. Related Work

### 6.1 Hand-Optimized Kernels

**FlashAttention series (Dao et al.)**

- Establishes performance targets
- Our work makes these optimizations compiler-expressible

**Triton (OpenAI)**

- Python DSL for GPU kernels
- Higher-level than CUDA, lower-level than MLIR
- Comparison point for "how good can DSLs get?"

**CuDNN Attention (NVIDIA)**

- Closed-source, vendor-optimized
- Black box we can't inspect or modify

---

### 6.2 ML Compilers

**XLA (Google)**

- Older compiler for TensorFlow
- Has fusion passes but not MLIR-based
- Shows compiler approach works at scale

**IREE (Google)**

- MLIR-based, most similar to our work
- **Key difference:** We focus on _deconstructing FA optimizations_, they focus on general ML compilation
- Our work could contribute passes back to IREE

**TVM (Apache)**

- Compiler with auto-tuning
- Different philosophy: search-based optimization vs expert-encoded passes
