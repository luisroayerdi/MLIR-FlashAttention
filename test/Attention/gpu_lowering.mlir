// RUN: attention-opt %s --tiling-pass="tile-size=32" --gpu-lowering-pass | FileCheck %s
//
// Verify Pass 5 Stage A: TilingPass's top-level Q-tile loop is wrapped in a
// gpu.launch (one block per Q tile) and outlined into a gpu.func, while the
// inner K/V loop stays an ordinary sequential affine.for inside the kernel.
// Shapes: seq_q=64, seq_k=64, head_dim=32 (multiples of tile size 32).

// CHECK: module attributes {gpu.container_module}

// CHECK-LABEL: func.func @tiled_attention
// CHECK:       gpu.launch_func @tiled_attention_kernel::@tiled_attention_kernel
// CHECK-SAME:    blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})
// CHECK-NOT:   gpu.launch blocks
// CHECK-NOT:   attention.fused

// CHECK:       gpu.module @tiled_attention_kernel
// CHECK-NEXT:  gpu.func @tiled_attention_kernel
// CHECK:       gpu.block_id x
// Inner K/V loop is preserved, unmapped, inside the kernel body.
// CHECK:       affine.for
// CHECK:       linalg.generic
// CHECK:       gpu.return

func.func @tiled_attention(
    %Q      : memref<64x32xf32>,
    %K      : memref<64x32xf32>,
    %V      : memref<64x32xf32>,
    %scale  : f32,
    %output : memref<64x32xf32>) {

  attention.fused
    ins(%Q, %K, %V : memref<64x32xf32>, memref<64x32xf32>, memref<64x32xf32>)
    scale(%scale : f32)
    outs(%output : memref<64x32xf32>)

  return
}
