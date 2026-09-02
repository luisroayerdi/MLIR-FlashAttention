// Pass 5, Stage B (Design.md §7.6): standalone tensor-core microbenchmark,
// deliberately NOT wired into attention-opt's --fusion-pass/--tiling-pass/
// .../--gpu-lowering-pass pipeline -- see TRADEOFFS.md "GPU lowering: Stage B
// rescoped..." for why integrating tensor cores into the actual fused/tiled
// kernel is deferred, and why this standalone form was chosen instead.
//
// This closely mirrors an existing MLIR upstream test that MLIR itself
// hardware-executes and numerically checks as part of LLVM's own CI on
// Ampere-class GPUs:
//   mlir/test/Integration/GPU/CUDA/TensorCore/sm80/transform-mma-sync-matmul-f32.mlir
// (structural check) and
//   mlir/test/Dialect/NVGPU/transform-matmul-to-nvvm.mlir
// (the same recipe, target-independent). The matmul shape (16x4 @ 4x8 ->
// 16x8, f32) is not arbitrary: it is exactly the tf32 mma.sync recipe's
// native fragment shape (mlir/lib/Dialect/NVGPU/TransformOps/
// NVGPUTransformOps.cpp's getIndexCalculators only supports this shape or
// 16x8x16 f16) -- one instruction, no retiling required.
//
// RUN line 1 (local, $0, no GPU/NVPTX needed -- this is what's actually
// verified today): the transform is target-independent IR, so we can check
// it produces nvgpu.mma.sync with this build's existing tools.
//
// RUN: attention-opt %s -transform-interpreter | FileCheck %s --check-prefix=CHECK-MMA-SYNC

// CHECK-MMA-SYNC-LABEL: func.func @matmul_tensorcore
// CHECK-MMA-SYNC:       nvgpu.mma.sync(%{{.*}}) {mmaShape = [16, 8, 4], tf32Enabled}
// CHECK-MMA-SYNC-SAME:    : (vector<2x1xf32>, vector<1x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>
// (Matches upstream's own check exactly -- see transform-matmul-to-nvvm.mlir
// -- which likewise doesn't assert the leftover transform schedule's
// "linalg.matmul" match-pattern *string* is gone, only that the rewrite
// fired. The linalg.matmul *op* itself is erased by the rewrite
// (RewriteMatmulAsMmaSyncOp::applyToOne calls rewriter.eraseOp) --
// -test-transform-dialect-erase-schedule (RUN line 2) removes the
// leftover schedule text too, for the real execution path.)

// RUN line 2 (Stage 2, GPU instance only -- requires an LLVM build with
// NVPTX + the MLIR CUDA runtime enabled, per Design.md §7.2; not runnable
// against this Mac build). cubin-chip=sm_89 targets our actual RTX 4090
// (Ada), not upstream's sm_80 (Ampere) -- tf32 mma.sync is a stable
// Ampere-generation PTX primitive Ada stays backward-compatible with, so
// this substitution is architectural continuity (see TRADEOFFS.md).
//
// RUN-GPU: mlir-opt %s \
// RUN-GPU:   -transform-interpreter -test-transform-dialect-erase-schedule \
// RUN-GPU:   -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_89 cubin-features=+ptx78 cubin-format=bin" \
// -e matmul_tensorcore: mlir-runner's -e defaults to "main" (see
// mlir/lib/ExecutionEngine/JitRunner.cpp), but this file's entry function
// is @matmul_tensorcore, not @main (unlike the upstream test it mirrors,
// which does use @main) -- found live, "Error: entry point not found".
//
// RUN-GPU: | mlir-runner \
// RUN-GPU:   --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_runner_utils \
// RUN-GPU:   -e matmul_tensorcore --entry-point-result=void \
// RUN-GPU: | FileCheck %s --check-prefix=CHECK-RESULT

!lhs_t = memref<16x4xf32>
!rhs_t = memref<4x8xf32>
!res_t = memref<16x8xf32>

func.func @compute_linspace_val(%r: index, %c: index, %strideC: index) -> f32 {
  %ri = arith.index_cast %r : index to i32
  %ci = arith.index_cast %c : index to i32
  %si = arith.index_cast %strideC : index to i32
  %prod = arith.muli %ri, %si : i32
  %sum = arith.addi %ci, %prod : i32
  %f = arith.sitofp %sum : i32 to f32
  return %f : f32
}

func.func @matmul_tensorcore() {
  %lhs = memref.alloc() : !lhs_t
  %rhs = memref.alloc() : !rhs_t
  %res = memref.alloc() : !res_t

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %M = memref.dim %res, %c0 : !res_t
  %N = memref.dim %res, %c1 : !res_t
  %K = memref.dim %lhs, %c1 : !lhs_t
  %f0 = arith.constant 0.000000e+00 : f32

  // Deterministic linspace init (matches the upstream reference this test
  // mirrors) so the CHECK-RESULT numbers below are reproducible.
  scf.for %r = %c0 to %M step %c1 {
    scf.for %c = %c0 to %K step %c1 {
      %v = func.call @compute_linspace_val(%r, %c, %K) : (index, index, index) -> f32
      memref.store %v, %lhs[%r, %c] : !lhs_t
    }
  }
  scf.for %r = %c0 to %K step %c1 {
    scf.for %c = %c0 to %N step %c1 {
      %v = func.call @compute_linspace_val(%r, %c, %N) : (index, index, index) -> f32
      memref.store %v, %rhs[%r, %c] : !rhs_t
    }
  }
  scf.for %r = %c0 to %M step %c1 {
    scf.for %c = %c0 to %N step %c1 {
      memref.store %f0, %res[%r, %c] : !res_t
    }
  }

  // gpu.host_register before gpu.launch, on every memref the kernel touches
  // (matches the upstream test this mirrors) -- registers this host
  // allocation as CUDA managed/pinned memory so the device can actually
  // dereference it. Getting this order wrong is exactly the kind of
  // silent-on-FileCheck, broken-on-real-hardware bug Design.md Section 7.6
  // is about: a plain gpu.launch reading an unregistered host pointer from
  // device code is invalid, but nothing here catches that short of
  // executing on a real GPU.
  %ulhs = memref.cast %lhs : !lhs_t to memref<*xf32>
  %urhs = memref.cast %rhs : !rhs_t to memref<*xf32>
  %ures = memref.cast %res : !res_t to memref<*xf32>
  gpu.host_register %ulhs : memref<*xf32>
  gpu.host_register %urhs : memref<*xf32>
  gpu.host_register %ures : memref<*xf32>

  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%bxs = %c32, %bys = %c1, %bzs = %c1) {
    linalg.matmul ins(%lhs, %rhs : !lhs_t, !rhs_t) outs(%res : !res_t)
    gpu.terminator
  }

  call @printMemrefF32(%ures) : (memref<*xf32>) -> ()
  // CHECK-RESULT: [112, 118, 124, 130, 136, 142, 148, 154],
  // CHECK-RESULT: [304, 326, 348, 370, 392, 414, 436, 458],
  // CHECK-RESULT: [496, 534, 572, 610, 648, 686, 724, 762],
  // CHECK-RESULT: [688, 742, 796, 850, 904, 958, 1012, 1066],
  // CHECK-RESULT: [880, 950, 1020, 1090, 1160, 1230, 1300, 1370],
  // CHECK-RESULT: [1072, 1158, 1244, 1330, 1416, 1502, 1588, 1674],
  // CHECK-RESULT: [1264, 1366, 1468, 1570, 1672, 1774, 1876, 1978],
  // CHECK-RESULT: [1456, 1574, 1692, 1810, 1928, 2046, 2164, 2282],
  // CHECK-RESULT: [1648, 1782, 1916, 2050, 2184, 2318, 2452, 2586],
  // CHECK-RESULT: [1840, 1990, 2140, 2290, 2440, 2590, 2740, 2890],
  // CHECK-RESULT: [2032, 2198, 2364, 2530, 2696, 2862, 3028, 3194],
  // CHECK-RESULT: [2224, 2406, 2588, 2770, 2952, 3134, 3316, 3498],
  // CHECK-RESULT: [2416, 2614, 2812, 3010, 3208, 3406, 3604, 3802],
  // CHECK-RESULT: [2608, 2822, 3036, 3250, 3464, 3678, 3892, 4106],
  // CHECK-RESULT: [2800, 3030, 3260, 3490, 3720, 3950, 4180, 4410],
  // CHECK-RESULT: [2992, 3238, 3484, 3730, 3976, 4222, 4468, 4714]
  return
}

func.func private @printMemrefF32(memref<*xf32>)

// ── Timed benchmark (added after correctness was confirmed on real
// hardware) ──────────────────────────────────────────────────────────────
//
// @matmul_tensorcore above is correctness-only (single-shot, FileCheck'd
// against known-correct values) -- it has no timing. This adds a second,
// separate entry point that does, following this project's existing CPU
// convention exactly (bench_codegen.py's warmup + rtclock()-bracketed
// timed loop, both calling the same function via func.call inside
// scf.for -- see TRADEOFFS.md "bare call/return don't parse inside
// scf.for"). @matmul_tensorcore is untouched.
//
// The kernel-launch logic is factored into its own function
// (@matmul_tensorcore_once) so both the warmup and timed loops invoke the
// exact same, single gpu.launch site -- gpu-kernel-outlining outlines it
// once; the loop is what causes it to actually launch repeatedly at
// runtime. Re-zeroing %res with an explicit scf.for/memref.store loop
// (matching @matmul_tensorcore's own zero-init above, not linalg.fill)
// before each call matches how @attention_unfused's own body
// re-initializes its intermediates on every call in bench_codegen.py --
// linalg.matmul accumulates onto its output (C += A@B), so without this,
// results would grow across iterations (harmless for a pure timing run,
// but sloppy and inconsistent with the rest of this project's benchmark
// code). Explicit loop, not linalg.fill: found live that linalg.fill here
// has no lowering path at all -- unlike gpu_matmul_no_tensorcore.mlir's
// RUN-GPU-TIMED, this pipeline deliberately never runs
// -convert-linalg-to-loops (it would risk touching linalg.matmul before
// the transform below gets to rewrite it to nvgpu.mma.sync), so a
// leftover linalg.fill survived all the way into mlir-runner's input,
// which doesn't have the linalg dialect registered and failed to parse it.
//
// RUN-GPU-TIMED: mlir-opt %s \
// RUN-GPU-TIMED:   -transform-interpreter -test-transform-dialect-erase-schedule \
// RUN-GPU-TIMED:   -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_89 cubin-features=+ptx78 cubin-format=bin" \
// RUN-GPU-TIMED: | mlir-runner \
// RUN-GPU-TIMED:   --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_runner_utils \
// RUN-GPU-TIMED:   --shared-libs=%mlir_c_runner_utils \
// RUN-GPU-TIMED:   -e main --entry-point-result=void

func.func private @rtclock() -> f64
func.func private @printF64(f64)
func.func private @printNewline()

func.func @matmul_tensorcore_once(%lhs: !lhs_t, %rhs: !rhs_t, %res: !res_t) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %M = memref.dim %res, %c0 : !res_t
  %N = memref.dim %res, %c1 : !res_t
  scf.for %r = %c0 to %M step %c1 {
    scf.for %c = %c0 to %N step %c1 {
      memref.store %f0, %res[%r, %c] : !res_t
    }
  }
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%bxs = %c32, %bys = %c1, %bzs = %c1) {
    linalg.matmul ins(%lhs, %rhs : !lhs_t, !rhs_t) outs(%res : !res_t)
    gpu.terminator
  }
  return
}

func.func @main() {
  %lhs = memref.alloc() : !lhs_t
  %rhs = memref.alloc() : !rhs_t
  %res = memref.alloc() : !res_t

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %res, %c0 : !res_t
  %N = memref.dim %res, %c1 : !res_t
  %K = memref.dim %lhs, %c1 : !lhs_t

  scf.for %r = %c0 to %M step %c1 {
    scf.for %c = %c0 to %K step %c1 {
      %v = func.call @compute_linspace_val(%r, %c, %K) : (index, index, index) -> f32
      memref.store %v, %lhs[%r, %c] : !lhs_t
    }
  }
  scf.for %r = %c0 to %K step %c1 {
    scf.for %c = %c0 to %N step %c1 {
      %v = func.call @compute_linspace_val(%r, %c, %N) : (index, index, index) -> f32
      memref.store %v, %rhs[%r, %c] : !rhs_t
    }
  }

  // gpu.host_register once, before any launches -- see @matmul_tensorcore
  // above for why the ordering matters.
  %ulhs = memref.cast %lhs : !lhs_t to memref<*xf32>
  %urhs = memref.cast %rhs : !rhs_t to memref<*xf32>
  %ures = memref.cast %res : !res_t to memref<*xf32>
  gpu.host_register %ulhs : memref<*xf32>
  gpu.host_register %urhs : memref<*xf32>
  gpu.host_register %ures : memref<*xf32>

  // Same warmup=5/timed=50 convention as bench_codegen.py's CPU benchmark.
  %warmup = arith.constant 5 : index
  %iters = arith.constant 50 : index

  scf.for %i = %c0 to %warmup step %c1 {
    func.call @matmul_tensorcore_once(%lhs, %rhs, %res) : (!lhs_t, !rhs_t, !res_t) -> ()
  }

  %t0 = func.call @rtclock() : () -> f64
  scf.for %i = %c0 to %iters step %c1 {
    func.call @matmul_tensorcore_once(%lhs, %rhs, %res) : (!lhs_t, !rhs_t, !res_t) -> ()
  }
  %t1 = func.call @rtclock() : () -> f64

  %elapsed = arith.subf %t1, %t0 : f64
  func.call @printF64(%elapsed) : (f64) -> ()
  func.call @printNewline() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.rewrite_matmul_as_mma_sync %matmul : (!transform.any_op) -> ()
    transform.yield
  }
}
