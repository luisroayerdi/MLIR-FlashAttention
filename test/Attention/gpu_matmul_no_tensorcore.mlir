// Pass 5, Stage B (Design.md §7.6) comparison baseline: the exact same
// matmul shape/init as gpu_tensor_core_matmul.mlir, lowered WITHOUT the
// tensor-core rewrite -- single-thread-per-block, matching Stage A's own
// launch philosophy (§7.4: "one thread per block, no intra-tile
// parallelism"). Stage 2's wall-clock comparison is this file vs.
// gpu_tensor_core_matmul.mlir at the identical shape.
//
// RUN line 1 (local, $0, no GPU/NVPTX needed): gpu-kernel-outlining is
// target-independent IR -- verified today with this build's existing tools.
//
// RUN: attention-opt %s -gpu-kernel-outlining | FileCheck %s --check-prefix=CHECK-OUTLINE

// CHECK-OUTLINE: module attributes {gpu.container_module}
// CHECK-OUTLINE: gpu.launch_func @matmul_naive_kernel::@matmul_naive_kernel
// CHECK-OUTLINE-NOT: gpu.launch blocks
// CHECK-OUTLINE: gpu.module @matmul_naive_kernel
// CHECK-OUTLINE: gpu.func @matmul_naive_kernel
// CHECK-OUTLINE: linalg.matmul
// No tensor-core ops here -- that's the point of the comparison.
// CHECK-OUTLINE-NOT: nvgpu.mma.sync

// RUN line 2 (Stage 2, GPU instance only -- see gpu_tensor_core_matmul.mlir
// for why this can't run against this Mac build, and for why mlir-opt, not
// attention-opt, is used here). Same expected numeric result as the
// tensor-core version: same matmul, same inputs, only the execution
// strategy differs.
//
// -convert-linalg-to-loops is required before -gpu-lower-to-nvvm-pipeline:
// that pipeline has no linalg-lowering step of its own (upstream expects
// linalg.matmul already rewritten, e.g. by the tensor-core transform in
// gpu_tensor_core_matmul.mlir) -- without it linalg.matmul survives into
// later passes with its body partially LLVM-dialect-converted underneath
// it and fails to verify. Lowers to a single-thread sequential loop nest,
// matching this file's single-thread launch above.
//
// RUN-GPU: mlir-opt %s \
// RUN-GPU:   -gpu-kernel-outlining -convert-linalg-to-loops \
// RUN-GPU:   -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_89 cubin-features=+ptx78 cubin-format=bin" \
// -e matmul_naive: mlir-runner's -e defaults to "main", but this file's
// entry function is @matmul_naive -- see gpu_tensor_core_matmul.mlir for
// the same gotcha, found live there first.
//
// RUN-GPU: | mlir-runner \
// RUN-GPU:   --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_runner_utils \
// RUN-GPU:   -e matmul_naive --entry-point-result=void \
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

func.func @matmul_naive() {
  %lhs = memref.alloc() : !lhs_t
  %rhs = memref.alloc() : !rhs_t
  %res = memref.alloc() : !res_t

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %res, %c0 : !res_t
  %N = memref.dim %res, %c1 : !res_t
  %K = memref.dim %lhs, %c1 : !lhs_t
  %f0 = arith.constant 0.000000e+00 : f32

  // Same deterministic linspace init as gpu_tensor_core_matmul.mlir.
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
  // -- see gpu_tensor_core_matmul.mlir for why this ordering matters.
  %ulhs = memref.cast %lhs : !lhs_t to memref<*xf32>
  %urhs = memref.cast %rhs : !rhs_t to memref<*xf32>
  %ures = memref.cast %res : !res_t to memref<*xf32>
  gpu.host_register %ulhs : memref<*xf32>
  gpu.host_register %urhs : memref<*xf32>
  gpu.host_register %ures : memref<*xf32>

  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%bxs = %c1, %bys = %c1, %bzs = %c1) {
    linalg.matmul ins(%lhs, %rhs : !lhs_t, !rhs_t) outs(%res : !res_t)
    gpu.terminator
  }
  gpu.host_register %ures : memref<*xf32>
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
