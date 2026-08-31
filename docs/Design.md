# MLIR Attention Pipeline — Design Document

**Version:** 1.1  
**Date:** April 2026 (corrected July 2026)  
**Status:** Approved for implementation. §2.1, §3.2–3.4, §4.4 corrected to match the
as-built Pass 1–2 implementation (`FusedOp` has no SSA result; QK^T is matched as
`linalg.generic`, not `linalg.matmul`; scale uses `mulf` not `divf`; fusion matching
walks buffer DPS writers, not SSA def-use chains). §5 corrected to match the as-built
Pass 3 implementation (whole-tile vectorization via MLIR's built-in linalg vectorizer,
not the manual VF=8/remainder-loop scheme the original pseudocode implied). §6.3
corrected to match the as-built Pass 4 implementation (inline `affine.if` with
`IntegerSet` conditions and direct block cloning, not `arith.cmpi` + outlined/inlined
kernel functions). See `TRADEOFFS.md` for the original discovery of these
discrepancies.

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
%qk  = linalg.generic { mulf/addf, Q x K^T }  // QK^T [seq_q x seq_k] (parallel/parallel/reduction; not a named matmul — K is read transposed via indexing maps)
%sc  = linalg.generic { mulf %qk, %scale }    // scale
%msk = linalg.generic { select ... }          // causal mask
%p   = linalg.softmax dimension(1) ins(%msk)  // attention weights [seq_q x seq_k]
%out = linalg.matmul ins(%p, %V)              // [seq_q x head_dim] ← fusion anchor

    ↓  Pass 1: Fusion

// Stage 1: Fused high-level op (no SSA result — writes %output in place)
attention.fused ins(%Q, %K, %V, %scale)
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
    // No SSA results — pure side-effecting write to `output` (DPS style, like linalg).

    let assemblyFormat = [{
        `ins` `(` $Q `,` $K `,` $V `:` type($Q) `,` type($K) `,` type($V) `)`
        `scale` `(` $scale `:` type($scale) `)`
        (`mask` `(` $mask^ `:` type($mask) `)`)?
        `outs` `(` $output `:` type($output) `)`
        attr-dict
    }];

    let hasVerifier = 1;
}
```

**Design rationale:**
- V is included so that the tiling pass can tile the complete attention computation and introduce online softmax accumulation across K/V tiles. Without V, the NxN attention matrix must be fully materialized between passes.
- `scale` is an SSA operand (not an attribute) because it is derived from `head_dim` at runtime (i.e., `1/sqrt(d_k)`), not a compile-time constant.
- `mask` is `Optional` — absent means unmasked attention; Pass 4 will further specialize masked tiles.
- `output` is an `outs` buffer (written in-place). The op has no SSA results at all — callers reference the written data through `output` directly, matching `linalg`'s destination-passing-style (DPS) convention for ops with no result.

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

The fusion pass uses a `RewritePattern` on `linalg.matmul` (the final PV matmul is the anchor). All operands are memrefs, not tensors, so a buffer can in principle be written by more than one op — backward matching cannot rely on pure SSA def-use chains. Instead, each step asks "which op is the *unique* writer of this buffer?" via `DestinationStyleOpInterface::getDpsInits()`, bailing out (pattern fails to match) if a buffer has more than one writer:

```
linalg.generic(mulf/addf, %Q, %K)      → %qk      (QK^T; parallel/parallel/reduction —
                                                    not a named matmul, since K is read
                                                    transposed via indexing maps rather
                                                    than physically transposed in memory)
linalg.generic(mulf, %qk, %scale)      → %scaled  (scale op)
linalg.generic(select, %scaled, %mask) → %masked  (mask op; optional — identified by
                                                    having 2 DPS inputs instead of 1)
linalg.softmax(%masked)                 → %probs   (named softmax op)
linalg.matmul(%probs, %V)               → %out     (anchor; fusion root)
```

### 3.3 Algorithm (Pseudocode)

```
pattern FuseAttention matches linalg.matmul(%probs, %V) → %out:
  if unique DPS writer of %probs is NOT linalg.softmax:
    return failure

  %masked = input of softmax
  if unique DPS writer of %masked is linalg.generic with 2 DPS inputs (select):
    has_mask = true
    %scaled = first DPS input of mask op
  else:
    has_mask = false
    %scaled = %masked

  if unique DPS writer of %scaled is NOT linalg.generic (scale; mulf):
    return failure

  %qk = first DPS input of scale op
  if unique DPS writer of %qk is NOT linalg.generic with ≥2 DPS inputs (QK^T):
    return failure

  (%Q, %K) = DPS inputs of %qk generic
  (%scale) = scale SSA value extracted from scale op's body — the first arith.mulf
             operand that is NOT a block argument local to the generic's own body
             (distinguishes the outer captured scale from the per-element block args)
  (%mask)  = mask buffer if has_mask else absent

  insert attention.fused ins(%Q, %K, %V) scale(%scale) [mask(%mask)] outs(%outBuf)
  erase pvMatmul, softmax, [maskOp], scaleOp, qkGeneric
  // No SSA result to thread through: attention.fused has no results, so %outBuf
  // continues to be used downstream exactly as it was before fusion.
```

### 3.4 IR Example

**Input:** (memref-based; ops with `outs` write in place and return no SSA result)
```mlir
linalg.generic { iterator_types = ["parallel","parallel","reduction"], ... }  // QK^T
  ins(%Q, %K : memref<1024x64xf32>, memref<1024x64xf32>)
  outs(%qk_buf : memref<1024x1024xf32>) { ^bb0(...): arith.mulf/addf ... }

linalg.generic { ... } ins(%qk_buf : memref<1024x1024xf32>)      // scale
  outs(%sc_buf : memref<1024x1024xf32>) { ^bb0(...): arith.mulf %in, %scale ... }

linalg.generic { ... } ins(%sc_buf, %causal_mask : ..., memref<1024x1024xi1>)  // mask
  outs(%mk_buf : memref<1024x1024xf32>) { ^bb0(...): arith.select ... }

linalg.softmax dimension(1) ins(%mk_buf : memref<1024x1024xf32>)
  outs(%probs_buf : memref<1024x1024xf32>)

linalg.matmul ins(%probs_buf, %V : memref<1024x1024xf32>, memref<1024x64xf32>)  // PV; anchor
  outs(%out_buf : memref<1024x64xf32>)
```

**Output:**
```mlir
attention.fused ins(%Q, %K, %V : memref<1024x64xf32>, memref<1024x64xf32>, memref<1024x64xf32>)
                scale(%scale : f32)
                mask(%causal_mask : memref<1024x1024xi1>)
                outs(%out_buf : memref<1024x64xf32>)
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

    // Step 2: Online softmax update — implemented as several distinct
    // linalg.generic ops (each a small parallel or row-reduction op), not
    // one combined generic:
    //   m_tile  = rowReduce(max, S_tile)              // this tile's row max
    //   m_new   = elementwise max(m_acc, m_tile)
    //   alpha   = elementwise exp(m_acc - m_new)       // rescale factor
    //   P_tile  = elementwise exp(S_tile - m_new)      // row-broadcast subtract
    //   P_sum   = rowReduce(add, P_tile)
    //   l_new   = elementwise alpha * l_acc + P_sum
    //   O_acc   = rowBroadcast(alpha * O_acc)          // rescale previous output
    //   O_acc  += P_tile @ V_tile                       // matmul-style generic

    // Step 3: memref.copy m_new → m_acc, l_new → l_acc
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

Replace the scalar `linalg.generic` / `linalg.fill` / `memref.copy` ops that
`TilingPass` emits inside each tile body with `vector`-dialect operations.

### 5.2 Algorithm (as-built)

`TilingPass` output has no raw scalar `affine.for` loops with `memref.load`/
`memref.store` for Claude Code to pattern-match against — every per-element
and per-row computation in the tile body is already a `linalg.generic` (or
`linalg.fill`/`memref.copy`) with explicit affine indexing maps over a
*statically shaped* tile (§4.2–4.4). That is exactly the input MLIR's own
linalg vectorizer (`mlir::linalg::vectorize` / `mlir::linalg::vectorizeCopy`,
`mlir/Dialect/Linalg/Transforms/Transforms.h`) is built to consume, so Pass 3
is a thin driver over that utility rather than a hand-rolled rewrite:

```
walk the function body, collecting every linalg.generic / linalg.fill /
memref.copy op (the vocabulary TilingPass emits) into a worklist

for each op in the worklist:
  if op is memref.copy:
    linalg::vectorizeCopy(rewriter, op)   // -> vector.transfer_read + vector.transfer_write
    continue
  if not linalg::hasVectorizationImpl(op):
    continue                              // leave unvectorizable ops as-is (best effort)
  result = linalg::vectorize(rewriter, op)
  if succeeded(result):
    rewriter.replaceOp(op, result.replacements)   // erases the original scalar op;
                                                     // see "gotcha" below
```

**No manual VF / remainder loop.** Because every tile dimension is a
compile-time constant (`TilingPass` requires static shapes — §4.5), the
vectorizer is invoked with no explicit `inputVectorSizes`, which makes it
default to the op's own full static iteration-space shape as the vector
shape (e.g. a `[T,T]` elementwise op becomes one `vector<TxTxf32>`
read/compute/write, not `T*T/8` chunks of `vector<8xf32>`). There is no
remainder to handle — remainder-handling only matters for shapes not evenly
divisible by a chosen VF, and here the "VF" *is* the tile size by
construction. Decomposing these tile-wide vectors into hardware-register-
sized (e.g. AVX2 `vector<8xf32>`) chunks with loops is left to the standard
downstream `--convert-vector-to-scf` / `--convert-vector-to-llvm` lowering
passes, not to this pass — the same division of labor MLIR uses everywhere
else structured-op vectorization is applied.

**Gotcha (non-obvious MLIR behavior):** `linalg::vectorize()` builds the
`vector.transfer_read`/arithmetic/`vector.transfer_write` replacement but does
**not** erase or replace the original op itself — for buffer (memref) DPS ops
with zero SSA results, `result.replacements` comes back empty, and it is the
caller's job to call `rewriter.replaceOp(op, result.replacements)` (which
degrades to a plain erase when `replacements` is empty). Skipping this step
leaves the original scalar op *coexisting* with the new vector code, silently
overwriting the vectorized result right after it runs — FileCheck for
`vector.transfer_write` would still pass, and CPU output would still be
numerically correct (the scalar op recomputes the same value), but none of
the intended vectorization would actually be happening. Confirmed against
upstream usage: `transform::VectorizeOp` (`LinalgTransformOps.cpp`) follows
the same `vectorize()` → `replaceOp()` pattern.

Requirements.md §4.3's VF=8/remainder-loop pseudocode is illustrative for a
*generic* vectorization pass over raw scalar loops; it does not describe
`TilingPass`'s actual (linalg-generic-based) output shape, the same kind of
simplification already flagged for Passes 1–2's Requirements snippets.

### 5.3 Known Limitations

- Full-tile vectorization means the emitted `vector<TxTxf32>`-shaped ops are
  virtual/architecture-agnostic until a later `--convert-vector-to-scf` /
  `--convert-vector-to-llvm` pipeline decomposes them to real hardware
  registers; this pass does not target AVX2/NEON specifically, unlike the
  VF=8 framing in Requirements.md.
- **Mask (`i1` memref) operands are deliberately never vectorized.**
  `memref<...xi1>` stores one byte per boolean; `vector<...xi1>` lowers to a
  bit-packed `llvm.load`. Vectorizing the mask-select `linalg.generic` reads
  through that layout mismatch and silently produces wrong results (found via
  the numerical suite: masked configs failed with ~1.0 max error while
  unmasked configs passed at ~1e-6). `VectorizationPass` detects any op with
  an `i1`-element memref operand and leaves it scalar. See TRADEOFFS.md.
- Best-effort otherwise: any op `linalg::hasVectorizationImpl` rejects, or
  where `vectorize()`/`vectorizeCopy()` fails its internal preconditions
  (e.g. a dynamic shape), is left as scalar `linalg`/`memref` IR rather than
  causing a pass failure.
- `test/numerical/pipeline.py` now wires Pass 3 into the numerical/CPU-
  benchmark harness behind an opt-in `--vectorize` flag (both `validate.py`
  and `benchmark.py`); the default suites remain scalar-only, matching
  Pass 1–2's existing default behavior. `_LOWER_FLAGS` grew from 11 to 15
  entries to support this — see TRADEOFFS.md for exactly which three passes
  were needed and why (`--convert-vector-to-scf`,
  `--lower-vector-multi-reduction`, `--convert-ub-to-llvm`), found the same
  empirical way the original 11-flag pipeline was.
- **Full-tile vectorization does not scale to CPU JIT compilation at
  production tile sizes.** Because there is no hardware-width chunking (see
  above), the QK^T/PV reduction ops lower to an LLVM array-of-vectors
  aggregate whose element count is `tile_size² × head_dim`. This JIT-compiles
  in seconds up to ~4,096 elements but hangs (multi-minute, multi-GB RSS)
  at 8,192+ — which includes Pass 1–2's own `tile=32`/`head_dim=64`
  production-scale benchmark config. `benchmark.py --vectorize --suite`
  therefore uses a separate, smaller `VECTORIZED_SUITE` (`tile-size` 8–16,
  `head-dim` 16) rather than `DEFAULT_SUITE`. Closing this gap is future
  work (`inputVectorSizes`-based hardware-width chunking); see TRADEOFFS.md.

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

### 6.3 Generated Code Structure (as-built)

`MaskSpecializationPass` is an `OpRewritePattern<affine::AffineForOp>` matching
the K/V inner tile loop TilingPass emits (identified as: a loop whose parent
op is another `affine.for` — the Q outer loop — and whose body directly
contains a `linalg.generic` with an `i1`-typed memref operand — the
mask-select op from §4.4's "3. Optional mask" step). A K-loop with no such op
(unmasked attention) is left untouched.

Rather than `arith.cmpi` feeding a generic conditional plus outlined/inlined
kernel functions, the pass builds `affine.if` directly against an
`IntegerSet` over the two loop induction variables `(i, j)` — this is the
MLIR-idiomatic way to express a loop-bound-relative condition (no separate
comparison op needed; the tile size `T`, read from the K-loop's constant
step, is folded into the set as a literal coefficient), and skips the
outline/inline round-trip entirely by moving/cloning the K-loop's existing
body ops directly into the new control-flow structure:

```mlir
// MASKED: k_start > q_end  <=>  j - i - T >= 0
#masked = affine_set<(i, j) : (j - i - T >= 0)>
// FULL:   q_start >= k_end <=>  i - j - T + 1 >= 0
#full   = affine_set<(i, j) : (i - j - T + 1 >= 0)>

affine.if #masked(%i, %j) {
  // empty: skip the tile entirely (online-softmax state unchanged)
} else {
  affine.if #full(%i, %j) {
    // clone of the original body, with the mask subview + arith.select
    // dropped and every consumer of the select's output (S_masked)
    // rewired to read the pre-mask score buffer (S_tile) instead
  } else {
    // the original body, moved here unchanged (per-element arith.select
    // against the mask tile, exactly as TilingPass emitted it)
  }
}
```

The rewiring for the FULL branch is done via `IRMapping`: before cloning,
`S_masked` (the mask-select op's DPS init) is pre-mapped to whatever `S_tile`
(the op's non-mask DPS input) resolves to in the clone, then the mask
subview and select ops themselves are skipped during the clone walk — every
downstream op that referenced `S_masked` in the original body automatically
picks up the cloned `S_tile` instead when `rewriter.clone()` remaps its
operands. See TRADEOFFS.md for why the outlined-function framing was
dropped and for a correctness precondition this design implies.

### 6.4 Known Limitations

- Requires Pass 2 (Tiling) to have already created the affine.for structure; must run after `--tiling-pass`.
- Only handles square causal masks; rectangular cross-attention masks out of scope.
- **Correctness precondition, not verified by the pass:** the FULL/MASKED
  classification is only valid if the mask is actually causal
  (`mask[i,j] == true` iff `j > i`). The pass has no way to inspect the
  mask's runtime contents — it classifies tiles purely from loop-induction-
  variable position — so passing a non-causal boolean mask to
  `attention.fused` and then running `--mask-specialization-pass` would
  silently produce wrong results rather than erroring. See TRADEOFFS.md.

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

*(§8.2 above is illustrative — see §8.2.1 for the as-built Phase 2 testing this project actually runs, including CPU speedup measurement, which the project has in practice not treated as out of scope.)*

### 8.2.1 Phase 2: Integration & CPU Testing (as-built)

Requirements.md §9.2 frames Phase 2 ("Integration tests", "CPU benchmarks",
"CHECKPOINT: CPU validation must pass") as the step after all four Phase 1
passes exist. As-built, this project ran numerical/CPU-speedup validation
incrementally per-pass (§4–§7 tradeoffs already reference each pass's own
`test/numerical/` results) rather than deferring it to a single batch step —
so what remained for Phase 2, once Pass 4 was done, was specifically
*cross-pass* integration coverage: verifying all four passes compose
correctly in one pipeline invocation, not just pairwise.

**IR-level integration test** — `test/Attention/integration.mlir` starts
from the raw unfused 5-op sequence (unlike the other per-pass test files,
which start from pre-fused or pre-tiled input) and runs the complete
`--fusion-pass --tiling-pass --vectorization-pass --mask-specialization-pass`
pipeline in one invocation, checking that `linalg.softmax`/`attention.fused`
are gone, the K-loop body is vectorized (`vector.transfer_read` present),
and the mask-specialization dispatch (`affine.if`) wraps the vectorized
BOUNDARY branch's still-scalar `arith.select` correctly.

**Numerical integration** — already covered before this subsection was
written: `validate.py --suite --vectorize --mask-specialize` runs the
complete pipeline (Pass 3 and Pass 4 flags compose in `pipeline.py`'s
`run_module`) and passes 5/5 with error magnitudes identical to every
narrower combination.

**CPU benchmark: the §5.4 Go/No-Go checkpoint** — `benchmark.py
--full-pipeline` is the new addition: it benchmarks the complete four-pass
pipeline against the unfused baseline, gated at Requirements.md §5.4's own
`>1.5x` "PROCEED" threshold (not §5.2's `>1.2x`, which Passes 1–2 alone
already clear) — this is the literal "CHECKPOINT: CPU validation must pass"
gate before Phase 3 (GPU Lowering) could begin. Because Pass 3 is in the
mix, it inherits Pass 3's small-scale JIT ceiling (reuses
`VECTORIZED_SUITE`'s shapes — see TRADEOFFS.md). Result: 3/3 configs pass at
5.0x–7.6x speedup, comfortably clearing `>1.5x` — the masked config benefits
from both Pass 3 and Pass 4 together and shows the largest speedup (7.6x).

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

