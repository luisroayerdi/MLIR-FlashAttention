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
// RUN-GPU: attention-opt %s \
// RUN-GPU:   -transform-interpreter -test-transform-dialect-erase-schedule \
// RUN-GPU:   -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_89 cubin-features=+ptx78 cubin-format=bin" \
// RUN-GPU: | mlir-runner \
// RUN-GPU:   --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_runner_utils \
// RUN-GPU:   --entry-point-result=void \
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
  // CHECK-RESULT: [112, 119, 126, 133, 140, 147, 154, 161],
  // CHECK-RESULT: [312, 335, 358, 381, 404, 427, 450, 473],
  // CHECK-RESULT: [512, 551, 590, 629, 668, 707, 746, 785],
  // CHECK-RESULT: [712, 767, 822, 877, 932, 987, 1042, 1097],
  // CHECK-RESULT: [912, 983, 1054, 1125, 1196, 1267, 1338, 1409],
  // CHECK-RESULT: [1112, 1199, 1286, 1373, 1460, 1547, 1634, 1721],
  // CHECK-RESULT: [1312, 1415, 1518, 1621, 1724, 1827, 1930, 2033],
  // CHECK-RESULT: [1512, 1631, 1750, 1869, 1988, 2107, 2226, 2345],
  // CHECK-RESULT: [1712, 1847, 1982, 2117, 2252, 2387, 2522, 2657],
  // CHECK-RESULT: [1912, 2063, 2214, 2365, 2516, 2667, 2818, 2969],
  // CHECK-RESULT: [2112, 2279, 2446, 2613, 2780, 2947, 3114, 3281],
  // CHECK-RESULT: [2312, 2495, 2678, 2861, 3044, 3227, 3410, 3593],
  // CHECK-RESULT: [2512, 2711, 2910, 3109, 3308, 3507, 3706, 3905],
  // CHECK-RESULT: [2712, 2927, 3142, 3357, 3572, 3787, 4002, 4217],
  // CHECK-RESULT: [2912, 3143, 3374, 3605, 3836, 4067, 4298, 4529],
  // CHECK-RESULT: [3112, 3359, 3606, 3853, 4100, 4347, 4594, 4841]
  return
}

func.func private @printMemrefF32(memref<*xf32>)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.rewrite_matmul_as_mma_sync %matmul : (!transform.any_op) -> ()
    transform.yield
  }
}
