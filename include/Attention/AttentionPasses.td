//===- AttentionPasses.td - Attention dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ATTENTION_PASS
#define ATTENTION_PASS

include "mlir/Pass/PassBase.td"

def FusionPass : Pass<"fusion-pass", "::mlir::func::FuncOp"> {
  let summary = "Fuse linalg attention ops into attention.fused";
  let description = [{
    Matches the 5-op unfused attention sequence:
      linalg.generic (QK^T) -> linalg.generic (scale) ->
      linalg.generic (mask, optional) -> linalg.softmax -> linalg.matmul (PV)
    and replaces it with a single attention.fused op.

    Pattern is anchored on the final linalg.matmul (the PV matmul).
    The scale value is extracted from the body of the scale generic.
  }];
}

def TilingPass : Pass<"tiling-pass", "::mlir::func::FuncOp"> {
  let summary = "Tile attention.fused with online softmax accumulation";
  let description = [{
    Expands each attention.fused op into nested affine.for loops with
    explicit online softmax (Milakov & Gimelshein 2018) accumulation.

    After this pass no attention.fused ops remain; output is standard
    affine + linalg + memref IR that mlir-cpu-runner can lower.

    Loop structure:
      for i in [0, seq_q, tile_size]:        # Q outer loop
        init O_acc, m_acc, l_acc
        for j in [0, seq_k, tile_size]:      # K/V inner loop
          compute S_tile = Q_tile @ K_tile^T * scale  (+ optional mask)
          online softmax update of m_acc, l_acc, O_acc
        O_tile = O_acc / l_acc               # final rescale
        write O_tile to output
  }];
  let options = [
    Option<"tileSize", "tile-size", "int64_t", "128",
           "Tile size for both Q and K/V sequence dimensions. "
           "Default 128 targets A100 SRAM (192 KB). "
           "Use 32 or 64 for CPU correctness testing.">
  ];
}

def VectorizationPass : Pass<"vectorization-pass", "::mlir::func::FuncOp"> {
  let summary = "Vectorize the scalar linalg ops TilingPass emits into the vector dialect";
  let description = [{
    Walks the function for linalg.generic, linalg.fill, and memref.copy ops
    (the op vocabulary TilingPass emits inside each tile body) and vectorizes
    each one in place using MLIR's built-in linalg vectorizer
    (linalg::vectorize / linalg::vectorizeCopy), replacing the scalar
    element-wise/reduction body with a vector.transfer_read /
    vector-dialect-arithmetic / vector.transfer_write sequence over the op's
    full static iteration-space shape.

    This vectorizes each tile "all at once" rather than chunking by a fixed
    vector width with a scalar remainder loop: since TilingPass already
    requires static, tile-sized shapes, there is no remainder to handle, and
    decomposition to hardware-width vectors is left to the standard
    convert-vector-to-scf / convert-vector-to-llvm lowering passes that run
    downstream of this one.

    Ops the vectorizer cannot handle are left unchanged; this pass never
    fails the build over an unvectorizable op.
  }];
}

def MaskSpecializationPass : Pass<"mask-specialization-pass", "::mlir::func::FuncOp"> {
  let summary = "Specialize each K/V tile body by position relative to the causal diagonal";
  let description = [{
    Must run after --tiling-pass. For each K/V inner tile loop TilingPass
    emits that applies a boolean mask (identified by the linalg.generic doing
    arith.select against an i1 memref operand), wraps that loop's per-iteration
    body in a two-level affine.if dispatching on the Q-tile / K-tile induction
    variables relative to the causal diagonal:

      MASKED   (k_start > q_end):  skip the tile entirely (do nothing --
                                     online-softmax state unchanged)
      FULL     (q_start >= k_end): run the same computation as the unmasked
                                     path (skip the mask subview + select)
      BOUNDARY (neither):          the original per-element masked computation,
                                     unchanged

    PRECONDITION: this classification is only correct for a causal mask
    (mask[i,j] == true iff j > i). It is not verified against the mask's
    actual contents -- see TRADEOFFS.md.

    A K/V loop with no mask-select op (i.e. unmasked attention) is left
    untouched.
  }];
}

def GPULoweringPass : Pass<"gpu-lowering-pass", "::mlir::ModuleOp"> {
  let summary = "Stage A: wrap TilingPass's Q-tile loop in gpu.launch and outline it";
  let description = [{
    Stage A of Pass 5 (GPU Backend Lowering, Design.md Section 7). Must run
    after --tiling-pass: it operates on the affine.for loop nest that pass
    emits, not on attention.fused.

    For each function, finds the top-level affine.for loop -- TilingPass's
    outer Q-tile loop, by construction the only loop at that nesting level
    -- and maps it onto a 1-D grid of GPU blocks, one block per Q tile,
    each block running with a single thread:
    mlir::convertAffineLoopNestToGPULaunch(forOp, numBlockDims=1,
    numThreadDims=0) wraps the loop body -- including the untouched inner
    K/V tile loop, which stays a sequential affine.for inside the kernel --
    in a gpu.launch op, replacing the induction variable with
    blockIdx.x * tile_size. gpu-kernel-outlining (MLIR's existing pass, run
    here as a nested pipeline) then moves that gpu.launch body into a
    gpu.func in a new gpu.module, replacing the call site with
    gpu.launch_func.

    No tensor cores, no shared-memory promotion, no custom
    intrinsic-insertion logic: this stage is pipeline wiring over existing
    MLIR GPU infrastructure (Design.md Section 7.4), not a novel
    pattern-matching pass. Stage B (matmul -> nvgpu.mma.sync) is separate
    and not yet implemented.

    PRECONDITION (documented, not verified): the top-level loop must
    actually be safe to run in parallel across blocks. This holds for
    TilingPass's output by construction -- each Q-tile iteration owns
    freshly-allocated accumulators and writes a disjoint output subview --
    but is not checked here: convertAffineLoopNestToGPULaunch itself
    performs no dependence analysis, by its own documented contract, and
    the responsibility for parallelization-safety falls on the caller.
  }];
}

#endif // ATTENTION_PASS
