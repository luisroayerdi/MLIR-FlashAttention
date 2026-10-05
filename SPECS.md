# SPECS.md

What this project studies and what the system must do.

- §1 and §2 change only by maintainer decision; the reviewer agent may
  comment on them.
- §3 and §4 change through issues triaged by the maintainer (see
  `CONTRIBUTING.md`). A PR may only move a feature it implements from
  `planned` to `implemented`.
- Feature status: `implemented`, `planned`, or `dropped` (kept, with reason).
- Work is tracked in GitHub issues, each referencing the section it
  implements.

## 1. Purpose

**Goal: establish, with evidence, what an MLIR compiler needs to know about
attention to generate FlashAttention-quality kernels.**

Attention is optimized in MLIR-based compilers in several different ways,
each shaped by its project's needs. IREE represents it as a named operator
with an online-softmax decomposition. Triton leaves the algorithm to the
kernel author. Upstream MLIR has no attention operator. These approaches
make different choices about what information the compiler keeps, and no
one has measured what each choice buys.

This project builds FlashAttention's optimizations (fusion, online-softmax
tiling, vectorization, mask specialization, GPU mapping) as separate MLIR
passes on upstream dialects, and uses them to measure:

- **What each transformation is worth**, on several GPU architectures, with
  hardware counters showing why.
- **What information each transformation needs** from the representation:
  for example, that the computation is attention at all, or that a mask has
  structure (causal, sliding-window) the compiler can reason about to skip
  work.
- **How the pass-based kernel compares** with IREE, Triton, TVM, and the
  hand-written FlashAttention and cuDNN kernels on the same hardware.

The result is a grounded answer to which attention information is worth
preserving through an MLIR compiler, reported to the upstream community
along with the concrete gaps found while building the pipeline.

## 2. Scope

**In scope**
- Attention forward pass, including grouped-query attention and decode
  (short query, long key/value).
- Passes over upstream dialects, with `attention.fused` as the entry op.
- Recognizing attention in the forms frontends produce, not only one
  hand-written pattern.
- NVIDIA GPUs: A100 (sm_80) and H100 (sm_90); Ada (sm_89) for continuity
  with Phase 1.
- Baselines run as published, through one shared harness.

**Out of scope**
- Backward pass, training, multi-GPU.
- New dialect ops beyond `attention.fused`.
- Reimplementing any baseline.
- ALiBi, soft-capping, paged KV caches.
- Non-attention workloads.
- AMD GPUs, for now.
- Beating FlashAttention.

## 3. System

### 3.1 Input: `attention.fused`

| Feature | Status |
|---|---|
| Operands: Q, K, V, scale, optional mask, output | implemented |
| Single-head shapes `[N, d]`, static, N divisible by tile size | implemented |
| Batched multi-head shapes `[B, H, N, d]` | planned |
| Grouped-query attention (fewer K/V heads than Q heads) | planned |
| Decode shapes (query length 1 to a few, long key/value) | planned |
| f32 | implemented |
| f16/bf16 inputs with f32 accumulation | planned |
| Mask as an explicit N×N tensor | implemented |
| Mask as affine conditions on indices (causal, sliding-window, prefix-LM), no tensor read | planned |

### 3.2 Pass pipeline

Each pass runs independently or composed, in this order, so any prefix of
the pipeline can be measured.

| Pass | Feature | Status |
|---|---|---|
| Fusion | Unfused matmul, scale, mask, softmax, matmul on memrefs becomes `attention.fused` | implemented |
| | Recognizes the forms torch-mlir produces from PyTorch attention, and common variants (scale before or after the matmul, additive or select mask, transposed K) | planned |
| Tiling | `attention.fused` becomes tiled loops with online softmax; tile size is an option | implemented |
| Vectorization | Tile bodies lowered to vector ops | implemented |
| | Compiles at production scale (N = 16k, d = 128) | planned |
| Mask specialization | Tiles classified as masked, unmasked, or boundary for a causal mask | implemented |
| | Causal precondition verified, not assumed | planned |
| | Affine mask families from §3.1; non-affine masks fall back to per-element masking | planned |
| GPU mapping | One GPU block per query tile | implemented |
| | Intra-tile work mapped to threads (warp-cooperative) | planned |
| | Q/K/V tiles staged in shared memory | planned |
| | Both matmuls on tensor cores (`nvgpu.mma.sync`, sm_80/sm_89) | planned |
| | Parallelism over the sequence dimension (FlashAttention-2 work distribution) | planned |

### 3.3 Correctness

| Feature | Status |
|---|---|
| FileCheck test per pass and for the composed pipeline | implemented |
| NumPy reference, f32, CPU and GPU execution | implemented |
| fp64 reference with documented f16/bf16 tolerances | planned |

### 3.4 Measurement

| Feature | Status |
|---|---|
| CPU timing inside the compiled program (Phase 1) | implemented |
| GPU timing with CUDA events, warmup, L2 flush; compile time reported separately | planned |
| Shapes per the FlashAttention-2 benchmark protocol | planned |
| Throughput (TFLOP/s, FlashAttention-2 FLOP convention) and fraction of peak | planned |
| Hardware counters (Nsight Compute): DRAM bytes, occupancy, tensor-pipe use | planned |
| Every reported number generated by a script from raw results | planned |

### 3.5 Baselines

| Feature | Status |
|---|---|
| FlashAttention-2 (A100), FlashAttention-3 (H100) | planned |
| PyTorch SDPA dispatching to cuDNN | planned |
| Triton fused-attention tutorial kernel | planned |
| IREE | planned |
| TVM, with tuning budget and any library dispatch reported | planned |
| Pinned versions for every baseline | planned |

## 4. Evaluation

The experiments that answer §1 together, and the records they produce.

| Output | Answers | Status |
|---|---|---|
| Ablation: each pipeline prefix on each target GPU, with counters | What each transformation is worth | planned |
| Mask study: mask families × sequence lengths; executed-tile fraction and speedup over per-element masking | What information transformations need | planned |
| Recognition study: which input forms each system turns into a fused kernel | What information transformations need | planned |
| Comparison, best performance: every system in its recommended configuration | How the kernel compares | planned |
| Comparison, automation: every system given the same unfused input, no hand-written kernels or library calls | How the kernel compares | planned |
| Upstream gaps record in `docs/LITERATURE.md`: observed failure, reproducer, upstream status, link | Report to upstream | planned |
| Literature ledger in `docs/LITERATURE.md`, each entry dated | All | planned |
