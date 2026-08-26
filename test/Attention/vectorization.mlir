// RUN: attention-opt %s --tiling-pass="tile-size=32" --vectorization-pass | FileCheck %s
//
// Verify Pass 3: the scalar linalg.generic/linalg.fill/memref.copy ops that
// Pass 2 (Tiling) emits inside each tile body are converted to vector-dialect
// form. Shapes: seq_q=64, seq_k=64, head_dim=32 (multiples of tile size 32).

// CHECK-LABEL: func.func @tiled_attention
// CHECK:       affine.for
// CHECK:       vector.transfer_read
// CHECK:       vector.transfer_write
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
// RUN: attention-opt %s --tiling-pass="tile-size=32" --vectorization-pass | FileCheck %s --check-prefix=MASK

// MASK-LABEL: func.func @tiled_attention_masked
// MASK:       affine.for
// MASK:       vector.transfer_read
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
