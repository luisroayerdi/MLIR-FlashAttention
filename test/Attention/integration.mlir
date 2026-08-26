// RUN: attention-opt %s --fusion-pass --tiling-pass="tile-size=32" --vectorization-pass --mask-specialization-pass | FileCheck %s
//
// Requirements.md 9.2 Phase 2 "Integration tests": verify all four Phase 1
// passes compose correctly in a single pipeline invocation, starting from
// the raw unfused 5-op sequence (not pre-fused/pre-tiled input, unlike the
// per-pass test files). Shapes: seq_q=64, seq_k=64, head_dim=32 (2x2 tile
// grid at tile size 32), matching the other per-pass test files.

// CHECK-LABEL: func.func @attention_unfused
// CHECK-NOT:   linalg.softmax
// CHECK-NOT:   attention.fused
// CHECK:       affine.for
// CHECK:       affine.for
// CHECK:       affine.if
// CHECK:       } else {
// CHECK:         affine.if
// CHECK-NOT:      arith.select
// CHECK:         } else {
// CHECK:           vector.transfer_read
// CHECK:           arith.select
// CHECK:         }
// CHECK:       }

func.func @attention_unfused(
    %Q      : memref<64x32xf32>,
    %K      : memref<64x32xf32>,
    %V      : memref<64x32xf32>,
    %scale  : f32,
    %mask   : memref<64x64xi1>,
    %output : memref<64x32xf32>) {

  %qk = memref.alloc() : memref<64x64xf32>
  %sc = memref.alloc() : memref<64x64xf32>
  %mk = memref.alloc() : memref<64x64xf32>
  %p  = memref.alloc() : memref<64x64xf32>

  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%qk : memref<64x64xf32>)

  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%Q, %K : memref<64x32xf32>, memref<64x32xf32>)
    outs(%qk   : memref<64x64xf32>) {
  ^bb0(%q : f32, %k : f32, %acc : f32):
    %prod = arith.mulf %q, %k : f32
    %sum  = arith.addf %acc, %prod : f32
    linalg.yield %sum : f32
  }

  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%qk : memref<64x64xf32>)
    outs(%sc : memref<64x64xf32>) {
  ^bb0(%in : f32, %out : f32):
    %r = arith.mulf %in, %scale : f32
    linalg.yield %r : f32
  }

  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%sc, %mask : memref<64x64xf32>, memref<64x64xi1>)
    outs(%mk      : memref<64x64xf32>) {
  ^bb0(%score : f32, %m : i1, %out : f32):
    %ninf = arith.constant -3.4028235e+38 : f32
    %r    = arith.select %m, %ninf, %score : f32
    linalg.yield %r : f32
  }

  linalg.softmax dimension(1)
    ins(%mk : memref<64x64xf32>)
    outs(%p : memref<64x64xf32>)

  linalg.fill ins(%zero : f32) outs(%output : memref<64x32xf32>)
  linalg.matmul
    ins(%p, %V   : memref<64x64xf32>, memref<64x32xf32>)
    outs(%output : memref<64x32xf32>)

  memref.dealloc %qk : memref<64x64xf32>
  memref.dealloc %sc : memref<64x64xf32>
  memref.dealloc %mk : memref<64x64xf32>
  memref.dealloc %p  : memref<64x64xf32>

  return
}
