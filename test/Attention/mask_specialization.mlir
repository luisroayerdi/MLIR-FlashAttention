// RUN: attention-opt %s --tiling-pass="tile-size=32" --mask-specialization-pass | FileCheck %s
//
// Verify Pass 4: the K/V tile loop's mask-select computation is wrapped in a
// two-level affine.if dispatching on tile position relative to the causal
// diagonal (Design.md 6). Shapes: seq_q=64, seq_k=64, head_dim=32 (2x2 tile
// grid at tile size 32), matching test/Attention/tiling.mlir.

// CHECK-LABEL: func.func @tiled_attention
// CHECK:       affine.for
// CHECK-NOT:   affine.if
// CHECK-NOT:   attention.fused
// CHECK-LABEL: func.func @tiled_attention_masked

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

// ── Masked variant: the mask-select computation gets specialized ───────────
// RUN: attention-opt %s --tiling-pass="tile-size=32" --mask-specialization-pass | FileCheck %s --check-prefix=MASK

// MASK-LABEL: func.func @tiled_attention_masked
// MASK:       affine.for
// MASK:       affine.for
// MASK:       affine.if
// MASK:       } else {
// MASK:         affine.if
// MASK-NOT:      arith.select
// MASK:         } else {
// MASK:           arith.select
// MASK:         }
// MASK:       }
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
