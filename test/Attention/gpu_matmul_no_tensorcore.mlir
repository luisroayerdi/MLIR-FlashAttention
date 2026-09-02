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
// for why this can't run against this Mac build). Same expected numeric
// result as the tensor-core version: same matmul, same inputs, only the
// execution strategy differs.
//
// RUN-GPU: attention-opt %s \
// RUN-GPU:   -gpu-kernel-outlining \
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
