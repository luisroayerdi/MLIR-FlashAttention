# MLIR Attention Pipeline — Design Document

**Version:** 1.0  
**Date:** April 2026  
**Status:** Approved for implementation

---

## 1. Architecture Overview

### 1.1 Transformation Pipeline

| Pass | Flag | Input IR | Output IR | FA Technique |
|------|------|----------|-----------|--------------|
| 1 — Fusion | `--fusion-pass` | linalg ops (matmul, generic, softmax) | `attention.fused` | FA1: Op Fusion |
| 2 — Tiling | `--tiling-pass` | `attention.fused` | affine.for + linalg (online softmax) | FA1: Tiling + Online Softmax |
| 3 — Vectorization | `--vectorization-pass` | affine.for + scalar linalg | affine.for + vector ops | SIMD |
| 4 — Mask Specialization | `--mask-specialization-pass` | affine.for + generic masking | Specialized kernel variants | Domain-specific |
| 5 — GPU Lowering | `--gpu-lowering-pass` | linalg.matmul | nvgpu.mma | FA2: Tensor Cores |

### 1.2 IR Transformation Chain

```
// Stage 0: Input (unfused)
%qk  = linalg.matmul ins(%Q, %K)        // [seq_q x seq_k]
%sc  = linalg.generic { divf %qk, %s }  // scale
%msk = linalg.generic { select ... }    // causal mask
%p   = linalg.generic { softmax %msk }  // attention weights [seq_q x seq_k]
%out = linalg.matmul ins(%p, %V)        // [seq_q x head_dim]

    ↓  Pass 1: Fusion

// Stage 1: Fused high-level op
%out = attention.fused ins(%Q, %K, %V, %scale)
                       mask(%mask)
                       outs(%output)

    ↓  Pass 2: Tiling (expands attention.fused + introduces online softmax)

// Stage 2: Tiled linalg with explicit online softmax accumulation
affine.for %i = 0 to %seq_q step 128 {
  // Accumulators for this Q-tile (live across K-tile iterations)
  %O_acc  = memref.alloca [128 x head_dim]  // output accumulator
  %m_acc  = memref.alloca [128]             // running max per row
  %l_acc  = memref.alloca [128]             // running sum per row
  affine.for %j = 0 to %seq_k step 128 {
    // QK^T + scale + mask
    linalg.matmul  Q_tile, K_tile -> S_tile
    linalg.generic { divf S, scale; select mask }
    // Online softmax update
    linalg.generic { m_new = max(m_acc, rowmax(S_tile)) }
    linalg.generic { P = exp(S - m_new); l_new = exp(m-m_new)*l + rowsum(P) }
    linalg.generic { O_acc = exp(m-m_new)*O_acc + P @ V_tile }
    // Update accumulators
  }
  linalg.generic { O_tile = O_acc / l_acc }  // final rescale
}

    ↓  Pass 3: Vectorization

// Stage 3: vector.load / vector.addf / vector.store replacing scalar linalg
for %i ... {
  %vec = vector.load %input[%i*8 : +8]
  %res = vector.addf %vec, %const_vec
  vector.store %res, %output[%i*8 : +8]
}

    ↓  Pass 4: Mask Specialization

// Stage 4: Dispatch on tile type
if tile_i * TILE > tile_j * TILE:  // full tile (below diagonal)
  <no mask checks>
elif tile_i * TILE < tile_j * TILE:  // masked tile (above diagonal)
  <skip computation>
else:  // boundary tile (straddles diagonal)
  <per-element mask check>

    ↓  Pass 5: GPU Lowering (deferred until hardware available)

// Stage 5: nvgpu dialect
%A_frag = nvgpu.ldmatrix %A
%B_frag = nvgpu.ldmatrix %B
%C_frag = nvgpu.mma %A_frag, %B_frag, %C_acc
nvgpu.stmatrix %C_frag, %C
```

---

## 2. Dialect Extension

### 2.1 `attention.fused` Operation

**Location:** `include/Attention/AttentionOps.td`

```tablegen
def Attention_FusedOp : Attention_Op<"fused", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Fused multi-head attention: output = softmax(Q @ K^T * scale + mask) @ V";
    let description = [{
        Computes the full attention operation in a single fused op.
        Includes V so tiling can introduce online softmax without materializing
        the full seq_q x seq_k attention weight matrix.

        Semantics:
          S     = Q @ K^T                        [seq_q x seq_k]
          S     = S * scale + mask               (optional mask: -inf for masked positions)
          P     = softmax(S, axis=seq_k)
          output = P @ V                          [seq_q x head_dim]
    }];

    let arguments = (ins
        MemRefOf<[F32]>:$Q,             // [seq_q x head_dim]
        MemRefOf<[F32]>:$K,             // [seq_k x head_dim]
        MemRefOf<[F32]>:$V,             // [seq_k x head_dim]
        F32:$scale,                     // 1/sqrt(head_dim), computed at runtime
        Optional<MemRefOf<[I1]>>:$mask, // [seq_q x seq_k]; absent means no masking
        MemRefOf<[F32]>:$output         // [seq_q x head_dim]; written in-place
    );
    let results = (outs MemRefOf<[F32]>:$result);

    let assemblyFormat = [{
        `ins` `(` $Q `,` $K `,` $V `:` type($Q) `,` type($K) `,` type($V) `)`
        `scale` `(` $scale `:` type($scale) `)`
        (`mask` `(` $mask^ `:` type($mask) `)`)?
        `outs` `(` $output `:` type($output) `)`
        attr-dict `->` type($result)
    }];
}
```

**Design rationale:**
- V is included so that the tiling pass can tile the complete attention computation and introduce online softmax accumulation across K/V tiles. Without V, the NxN attention matrix must be fully materialized between passes.
- `scale` is an SSA operand (not an attribute) because it is derived from `head_dim` at runtime (i.e., `1/sqrt(d_k)`), not a compile-time constant.
- `mask` is `Optional` — absent means unmasked attention; Pass 4 will further specialize masked tiles.
- `output` is an `outs` buffer (written in-place). The result is aliased to `output` for SSA tracking, following the `linalg` convention.

### 2.2 Dialect Driver Changes

`attention-opt/attention-opt.cpp` must register all needed dialects:

```cpp
// Replace selective registration with:
mlir::registerAllDialects(registry);
```

**Reason:** Passes 1–4 depend on `linalg`, `memref`, `affine`, `arith`, `vector`, and `func` dialects. Registering all dialects avoids incremental omissions during development.

### 2.3 Library Dependencies

Add to `lib/Attention/CMakeLists.txt`:

```cmake
LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
  MLIRLinalgDialect       # new
  MLIRMemRefDialect       # new
  MLIRAffineDialect       # new
  MLIRArithDialect        # new
  MLIRVectorDialect       # new (Pass 3)
  MLIRTransformUtils      # new (pattern rewriting)
```

---

## 3. Pass 1: Operation Fusion (`--fusion-pass`)

### 3.1 Goal

Recognize the 5-op attention sequence and replace it with a single `attention.fused` op.

### 3.2 Pattern Matching

The fusion pass uses a `RewritePattern` on `linalg.matmul` (the final PV matmul is the anchor). Walking backwards through SSA def-use chains, it verifies the following sequence:

```
linalg.matmul(%Q, %K)          → %qk
linalg.generic(divf, %qk, %s)  → %scaled    (scale op)
linalg.generic(select, %scaled) → %masked   (mask op; optional)
linalg.generic(softmax, %masked) → %probs   (softmax op)
linalg.matmul(%probs, %V)       → %out      (anchor; fusion root)
```

### 3.3 Algorithm (Pseudocode)

```
pattern FuseAttention matches linalg.matmul(%probs, %V) → %out:
  if defining_op(%probs) is NOT linalg.softmax:
    return failure

  %masked = input of softmax
  if defining_op(%masked) is linalg.generic with select:
    has_mask = true
    %scaled = input of mask op
  else:
    has_mask = false
    %scaled = %masked

  if defining_op(%scaled) is NOT linalg.generic with divf:
    return failure

  %qk = input of scale op
  if defining_op(%qk) is NOT linalg.matmul:
    return failure

  (%Q, %K) = operands of %qk matmul
  (%scale) = scale operand from scale op
  (%mask)  = mask operand if has_mask else absent

  verify no uses of %qk, %scaled, %masked, %probs outside this chain

  output = allocate memref [seq_q x head_dim]
  %result = attention.fused ins(%Q, %K, %V, %scale)
                             mask(%mask)   // optional
                             outs(%output) -> memref

  replace %out with %result
  erase original 5 ops
```

### 3.4 IR Example

**Input:**
```mlir
%qk     = linalg.matmul ins(%Q, %K : memref<1024x64xf32>, memref<1024x64xf32>)
                         outs(%qk_buf : memref<1024x1024xf32>) -> memref<1024x1024xf32>
%scaled = linalg.generic ... { divf %qk, %scale } ...
%masked = linalg.generic ... { select %causal_mask, %scaled, %neg_inf } ...
%probs  = linalg.generic ... { exp / sum for softmax } ...
%out    = linalg.matmul ins(%probs, %V : ...)
                         outs(%out_buf : memref<1024x64xf32>) -> memref<1024x64xf32>
```

**Output:**
```mlir
%out = attention.fused ins(%Q, %K, %V : memref<1024x64xf32>, ...)
                        scale(%scale : f32)
                        mask(%causal_mask : memref<1024x1024xi1>)
                        outs(%out_buf : memref<1024x64xf32>)
                        -> memref<1024x64xf32>
```

### 3.5 Known Limitations

- Only handles f32 (f16/bf16 deferred to future pass extension).
- Requires the exact 5-op sequence with single-use intermediates; non-standard softmax implementations will not match.
- The mask operand must be a `memref<*xi1>` (boolean); additive bias masks are out of scope.

---

## 4. Pass 2: Memory-Aware Tiling (`--tiling-pass`)

### 4.1 Goal

Tile `attention.fused` into loops over Q-tiles and K/V-tiles, expanding the inner computation into linalg ops with an explicit online softmax accumulation scheme. After this pass, no `attention.fused` ops remain — output is pure affine + linalg, CPU-runnable.

Online softmax is introduced here (not in Pass 1) because it is a property of the tiled execution: each K/V tile updates running (max, sum) state rather than requiring the full attention row to be computed first.

### 4.2 Tile Size Calculation

```python
# Default target: A100 SRAM = 192 KB
# Tiles needed: Q_tile, K_tile, V_tile, S_tile, O_tile, P_tile
# Conservative: 3 working memrefs of float32
tile_size = sqrt(SRAM_bytes / (3 * sizeof(float32)))
tile_size = round_to_multiple(tile_size, 16)  # tensor core alignment
# → tile_size = 128 for 192KB SRAM

# For CPU correctness testing: tile_size is a pass option (default 64)
```

The pass exposes a `tile-size` option (default: 128). For CPU testing, any power-of-2 size works; 128 is preserved as the default to match A100 constraints.

### 4.3 Algorithm: Online Softmax Tiling

The online softmax algorithm (Milakov & Gimelshein, 2018; used in FA1) avoids materializing the full attention row by maintaining running statistics:

```
// For each Q-tile (outer loop):
initialize O_acc[TILE x D]  ← 0     // output accumulator
initialize m_acc[TILE]      ← -inf  // running row maximum
initialize l_acc[TILE]      ← 0     // running row sum (of exp)

for each K-tile, V-tile (inner loop):
    // 1. Compute attention scores for this tile
    S_tile[TILE x TILE] = Q_tile @ K_tile^T   // matmul
    S_tile = S_tile * scale                    // scale
    if mask:
        S_tile[i,j] = -inf  where mask[q_base+i, k_base+j] == True

    // 2. Online softmax update
    m_new[i]  = max(m_acc[i], max_j(S_tile[i,:]))
    // Rescale previous accumulator to new max
    alpha[i]  = exp(m_acc[i] - m_new[i])
    P_tile[i,j] = exp(S_tile[i,j] - m_new[i])   // unnormalized probs for this tile
    l_new[i]  = alpha[i] * l_acc[i] + sum_j(P_tile[i,:])

    // 3. Update output accumulator
    O_acc[i,:] = alpha[i] * O_acc[i,:] + P_tile @ V_tile  // matmul

    // 4. Advance running state
    m_acc = m_new
    l_acc = l_new

// Final rescale
O_tile[i,:] = O_acc[i,:] / l_acc[i]
write O_tile to output
```

### 4.4 MLIR IR Structure (Post-Tiling)

```mlir
affine.for %i = 0 to %seq_q step 128 {
  // Tile-local accumulators (stack-allocated)
  %O_acc = memref.alloca() : memref<128x64xf32>
  %m_acc = memref.alloca() : memref<128xf32>
  %l_acc = memref.alloca() : memref<128xf32>

  // Initialize
  linalg.fill ins(%neg_inf) outs(%m_acc)
  linalg.fill ins(%zero)    outs(%l_acc)
  linalg.fill ins(%zero)    outs(%O_acc)

  affine.for %j = 0 to %seq_k step 128 {
    %Q_tile = memref.subview %Q[%i, 0][128, 64][1, 1]
    %K_tile = memref.subview %K[%j, 0][128, 64][1, 1]
    %V_tile = memref.subview %V[%j, 0][128, 64][1, 1]

    // Step 1: Score tile
    %S_buf  = memref.alloca() : memref<128x128xf32>
    linalg.matmul ins(%Q_tile, %K_tile) outs(%S_buf)
    linalg.generic { arith.mulf %s, %scale }  // scale

    // Optional: apply mask tile
    // (mask subview: [%i, %j][128, 128])

    // Step 2: Online softmax update (linalg.generic over rows)
    %m_new = memref.alloca() : memref<128xf32>
    %P_buf = memref.alloca() : memref<128x128xf32>
    linalg.generic ... { compute m_new, P_buf, l_new, update O_acc }

    // Step 3: Update m_acc, l_acc
  }

  // Final rescale and write output tile
  linalg.generic { arith.divf %O_acc, %l_acc }
  %out_tile = memref.subview %output[%i, 0][128, 64][1, 1]
  memref.copy %O_acc, %out_tile
}
```

### 4.5 Design Decisions

- **Full expansion:** The tiling pass removes all `attention.fused` ops. Output is standard affine/linalg/memref, directly runnable by `mlir-cpu-runner` without additional lowering passes. This is the only way to reach the CPU validation checkpoint with the 2-pass command in §5.2 of Requirements.
- **Stack allocation for accumulators:** `memref.alloca` for tile-local buffers avoids heap allocation overhead and keeps memory visible to the compiler for optimization.
- **Loop order (Q outer, K inner):** Q tiles write to disjoint output rows (no K-loop synchronization needed). K/V tiles are streamed through, which is the FA1 access pattern.
- **Remainder handling:** For sequence lengths not divisible by tile size, the pass inserts a remainder loop (or uses `affine.if` guard). Initial implementation: assert divisibility; remainder loop added in follow-up.

### 4.6 Known Limitations

- Static tile size: suboptimal on H100 (different SRAM). Tile size is a pass option to mitigate this.
- Remainder handling is deferred (initial implementation requires `seq_len % tile_size == 0`).
- Multi-head (batch) dimension is not yet in scope; input is assumed `[seq x head_dim]` per head.

---

## 5. Pass 3: Vectorization (`--vectorization-pass`)

### 5.1 Goal

Replace scalar linalg loops in the tile body with SIMD vector operations using MLIR's `vector` dialect.

### 5.2 Algorithm

```
for each affine.for loop in the tile body:
  if loop is vectorizable (no loop-carried dependencies, stride-1 access):
    VF = 8  // for f32: 256-bit AVX2 vector width
    new_bound = ceil(original_bound / VF)
    replace:
      %val = memref.load %buf[%i]
      %res = arith.addf %val, %c
      memref.store %res, %out[%i]
    with:
      %vec = vector.load %buf[%i*VF : +VF] : vector<8xf32>
      %res = vector.addf %vec, %c_splat
      vector.store %res, %out[%i*VF : +VF]
    insert remainder loop for indices [VF*(new_bound) : original_bound]
```

Vector factor is a pass option (default: 8 for f32 / AVX2). The pass uses `mlir::vectorize` infrastructure where available and falls back to manual rewriting for patterns not covered.

### 5.3 Known Limitations

- Initially targets only element-wise linalg.generic patterns; matmul vectorization relies on upstream MLIR linalg vectorization.
- Remainder loop generation is a stub; initial tests require loop bounds divisible by VF.

---

## 6. Pass 4: Causal Mask Specialization (`--mask-specialization-pass`)

### 6.1 Goal

Classify tiles by position relative to the causal mask diagonal and generate specialized code paths, eliminating runtime mask checks in the common case.

### 6.2 Tile Classification

Given tile position `(tile_i, tile_j)` and tile size `T`:

```
q_start = tile_i * T,  q_end = q_start + T - 1
k_start = tile_j * T,  k_end = k_start + T - 1

FULL     if q_start >= k_end    // entire tile is below diagonal: no mask needed
MASKED   if k_start > q_end     // entire tile is above diagonal: skip
BOUNDARY if neither             // straddles diagonal: per-element check
```

### 6.3 Generated Code Structure

```mlir
%is_full    = arith.cmpi sge, %q_start, %k_end
%is_masked  = arith.cmpi sgt, %k_start, %q_end
affine.if %is_masked {
  // skip: do nothing (zero contribution)
} else {
  affine.if %is_full {
    // full tile: call inner loop with no mask check
    call @inner_full(%Q_tile, %K_tile, %V_tile, ...)
  } else {
    // boundary tile: call inner loop with per-element mask
    call @inner_boundary(%Q_tile, %K_tile, %V_tile, %mask_tile, ...)
  }
}
```

The three kernel variants are outlined functions produced by the pass and inlined by a subsequent canonicalization step.

### 6.4 Known Limitations

- Requires Pass 2 (Tiling) to have already created the affine.for structure; must run after `--tiling-pass`.
- Only handles square causal masks; rectangular cross-attention masks out of scope.

---

## 7. Pass 5: GPU Backend Lowering (`--gpu-lowering-pass`)

> **Status:** Design complete. Implementation deferred until GPU hardware (university HPCC or cloud) is available. CPU testing (Passes 1–4) is the immediate goal.

### 7.1 Goal

Lower `linalg.matmul` ops inside the tiled loops to NVIDIA tensor core operations via the `nvgpu` dialect.

### 7.2 Prerequisites

- LLVM built with NVPTX backend (`-DLLVM_TARGETS_TO_BUILD=NVPTX`).
- Tile size must be a multiple of 16 (tensor core fragment size). The default 128 satisfies this.

### 7.3 Algorithm

```
for each linalg.matmul in tiled body:
  if operand shapes are multiples of 16:
    insert layout transformations for tensor core fragment layout
    %A_frag = nvgpu.ldmatrix %A_shared_tile
    %B_frag = nvgpu.ldmatrix %B_shared_tile
    %C_frag = nvgpu.mma.sync %A_frag, %B_frag, %C_acc
              shape = [16, 16, 16] dtype = f32
    nvgpu.stmatrix %C_frag, %C_shared_tile
  else:
    leave as linalg.matmul (no-op)
```

### 7.4 Design Decisions

- Targets `nvgpu.mma.sync` with `m16n8k16` fragments (A100 primary). H100 fragments (`m16n8k16` with bf16) to be evaluated on hardware.
- Shared memory promotion (global → `gpu.shared` → ldmatrix) is part of this pass; it inserts `memref.alloca` in `gpu.private` and `gpu.shared` address spaces.

### 7.5 Known Limitations (Anticipated)

- `nvgpu` dialect may require additional conversion passes before PTX emission.
- Register pressure with 128×128 tiles may cause spills; tunable via tile-size option.

---

## 8. Build & Test Plan

### 8.1 Build

```bash
cd build
cmake .. -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir -G Ninja
ninja
```

### 8.2 CPU Correctness Testing (Primary Milestone)

The correctness pipeline runs Passes 1 and 2 and feeds the output to `mlir-cpu-runner`:

```bash
./build/bin/attention-opt test.mlir --fusion-pass --tiling-pass | \
  mlir-cpu-runner -e main -entry-point-result=void \
  -shared-libs=/path/to/mlir/lib/libmlir_runner_utils.so
```

Each pass also testable in isolation:

```bash
# Pass 1: verify fusion
./build/bin/attention-opt test/fusion.mlir --fusion-pass | FileCheck test/fusion.mlir

# Pass 2: verify tiled + expanded IR, then run numerics
./build/bin/attention-opt test/tiling.mlir --fusion-pass --tiling-pass | \
  mlir-cpu-runner -e main -entry-point-result=void
```

Numerical correctness validated against PyTorch reference:

```bash
python3 test/numerical/validate.py --seq-len=256 --batch=1 --head-dim=64
```

**Performance benchmarking on CPU is intentionally not a goal** — the hardware target is GPU. CPU testing validates behavioral correctness only.

### 8.3 GPU Testing (Deferred)

When HPCC/cloud GPU is available:

```bash
# Build with NVPTX target
cmake .. -DMLIR_DIR=... -DLLVM_TARGETS_TO_BUILD=NVPTX -G Ninja
ninja

# Run GPU benchmark
python3 benchmarks/mlir/run_gpu.py --seq-len=1024 --batch=16 --iterations=100

# Profile
ncu --set full --export profile.ncu-rep python3 benchmarks/mlir/run_gpu.py
```

---

## 9. File Layout

```
MLIR-Scheduling-Kernel/
├── Design.md                          ← this file
├── Requirements.md
├── TRADEOFFS.md                       ← updated per pass during implementation
├── include/Attention/
│   ├── AttentionDialect.td            ← add attention dialect def
│   ├── AttentionOps.td                ← add attention.fused op
│   ├── AttentionPasses.td             ← add FusionPass, TilingPass, VectorizationPass,
│   │                                     MaskSpecializationPass, GPULoweringPass
│   ├── AttentionDialect.h
│   ├── AttentionOps.h
│   └── AttentionPasses.h
├── lib/Attention/
│   ├── AttentionDialect.cpp
│   ├── AttentionOps.cpp
│   ├── FusionPass.cpp                 ← Pass 1
│   ├── TilingPass.cpp                 ← Pass 2
│   ├── VectorizationPass.cpp          ← Pass 3
│   ├── MaskSpecializationPass.cpp     ← Pass 4
│   ├── GPULoweringPass.cpp            ← Pass 5 (deferred)
│   └── CMakeLists.txt
├── attention-opt/
│   └── attention-opt.cpp              ← update: registerAllDialects
├── test/Attention/
│   ├── fusion.mlir
│   ├── tiling.mlir
│   ├── vectorization.mlir
│   └── mask_specialization.mlir
├── test/numerical/
│   └── validate.py
└── benchmarks/
    ├── baselines/
    │   ├── unfused_pytorch.py
    │   ├── torch_compile.py
    │   └── flash_attn2.py
    ├── mlir/
    │   └── run_gpu.py
    └── ablation.py
```

---

## 10. Tradeoffs Summary

| Decision | Chosen | Alternative | Reason |
|----------|--------|-------------|--------|
| Include V in `attention.fused` | Yes (full attention) | No (QK+softmax only) | Required for online softmax tiling; without V, NxN matrix cannot be avoided |
| `scale` as SSA operand | SSA operand (F32) | Attribute (F32Attr) | Computed at runtime from `head_dim`; not a compile-time constant |
| Online softmax location | Pass 2 (Tiling) | Pass 1 (Fusion) | Online softmax is a property of tiled execution, not fusion |
| Tiling pass output | Pure linalg (no attention.fused) | Keep attention.fused on tiles | Required for CPU-runnable output with 2-pass command |
| Tile size | Static 128 (pass option) | Dynamic | Simplicity; 128 optimal for A100; tunable for other hardware |
| Mask in fused op | Optional (absent = no mask) | Always required | Enables unmasked attention without dummy memrefs |
| Driver dialect registration | `registerAllDialects` | Selective | Avoids incremental omissions across 5 passes |
| GPU testing | Deferred to HPCC/cloud | Block on local GPU | No GPU available; CPU correctness is achievable now |
