// RUN: attention-opt %s --tiling-pass="tile-size=32" | FileCheck %s
//
// Verify Pass 2: attention.fused is fully expanded into affine.for loops with
// linalg.generic ops.  Tile size 32 avoids large stack allocations on CPU.
// Shapes: seq_q=64, seq_k=64, head_dim=32 (multiples of tile size 32).

// CHECK-LABEL: func.func @tiled_attention
// CHECK:       affine.for
// CHECK:       linalg.generic
// CHECK-NOT:   attention.fused

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

// ── Masked variant ────────────────────────────────────────────────────────
// RUN: attention-opt %s --tiling-pass="tile-size=32" | FileCheck %s --check-prefix=MASK

// MASK-LABEL: func.func @tiled_attention_masked
// MASK:       affine.for
// MASK:       arith.select
// MASK-NOT:   attention.fused

func.func @tiled_attention_masked(
    %Q      : memref<64x32xf32>,
    %K      : memref<64x32xf32>,
    %V      : memref<64x32xf32>,
    %scale  : f32,
    %mask   : memref<64x64xi1>,
    %output : memref<64x32xf32>) {

  attention.fused
    ins(%Q, %K, %V : memref<64x32xf32>, memref<64x32xf32>, memref<64x32xf32>)
    scale(%scale : f32)
    mask(%mask : memref<64x64xi1>)
    outs(%output : memref<64x32xf32>)

  return
}
