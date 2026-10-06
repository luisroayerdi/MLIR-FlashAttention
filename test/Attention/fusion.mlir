// RUN: attention-opt %s --fusion-pass | FileCheck %s
//
// Verify Pass 1: the 5-op unfused sequence is collapsed into attention.fused.
// Shapes are small (128 × 64) so the test is quick on CPU.

// CHECK-LABEL: func.func @attention_unfused
// CHECK:       attention.fused
// CHECK-NOT:   linalg.softmax
// CHECK-NOT:   linalg.matmul

func.func @attention_unfused(
    %Q      : memref<128x64xf32>,
    %K      : memref<128x64xf32>,
    %V      : memref<128x64xf32>,
    %scale  : f32,
    %mask   : memref<128x128xi1>,
    %output : memref<128x64xf32>) {

  %qk = memref.alloc() : memref<128x128xf32>
  %sc = memref.alloc() : memref<128x128xf32>
  %mk = memref.alloc() : memref<128x128xf32>
  %p  = memref.alloc() : memref<128x128xf32>

  // Initialise accumulator buffer for QK^T
  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%qk : memref<128x128xf32>)

  // ── QK^T generic: Q[i,k] * K[j,k] → qk[i,j] ─────────────────────────
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%Q, %K : memref<128x64xf32>, memref<128x64xf32>)
    outs(%qk   : memref<128x128xf32>) {
  ^bb0(%q : f32, %k : f32, %acc : f32):
    %prod = arith.mulf %q, %k : f32
    %sum  = arith.addf %acc, %prod : f32
    linalg.yield %sum : f32
  }

  // ── Scale: sc[i,j] = qk[i,j] * scale ────────────────────────────────
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%qk : memref<128x128xf32>)
    outs(%sc : memref<128x128xf32>) {
  ^bb0(%in : f32, %out : f32):
    %r = arith.mulf %in, %scale : f32
    linalg.yield %r : f32
  }

  // ── Mask: mk[i,j] = mask[i,j] ? -inf : sc[i,j] ──────────────────────
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%sc, %mask : memref<128x128xf32>, memref<128x128xi1>)
    outs(%mk      : memref<128x128xf32>) {
  ^bb0(%score : f32, %m : i1, %out : f32):
    %ninf = arith.constant -3.4028235e+38 : f32
    %r    = arith.select %m, %ninf, %score : f32
    linalg.yield %r : f32
  }

  // ── Softmax (row-wise) ────────────────────────────────────────────────
  linalg.softmax dimension(1)
    ins(%mk : memref<128x128xf32>)
    outs(%p : memref<128x128xf32>)

  // ── PV matmul: output = probs @ V ─────────────────────────────────────
  linalg.fill ins(%zero : f32) outs(%output : memref<128x64xf32>)
  linalg.matmul
    ins(%p, %V   : memref<128x128xf32>, memref<128x64xf32>)
    outs(%output : memref<128x64xf32>)

  memref.dealloc %qk : memref<128x128xf32>
  memref.dealloc %sc : memref<128x128xf32>
  memref.dealloc %mk : memref<128x128xf32>
  memref.dealloc %p  : memref<128x128xf32>

  return
}

// Unmasked: softmax reads the scaled buffer directly.

// CHECK-LABEL: func.func @attention_unfused_nomask
// CHECK:       attention.fused
// CHECK-NOT:   linalg.softmax
// CHECK-NOT:   linalg.matmul

func.func @attention_unfused_nomask(
    %Q : memref<16x8xf32>, %K : memref<16x8xf32>, %V : memref<16x8xf32>,
    %scale : f32, %output : memref<16x8xf32>) {
  %qk = memref.alloc() : memref<16x16xf32>
  %sc = memref.alloc() : memref<16x16xf32>
  %p  = memref.alloc() : memref<16x16xf32>
  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%qk : memref<16x16xf32>)
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%Q, %K : memref<16x8xf32>, memref<16x8xf32>)
    outs(%qk : memref<16x16xf32>) {
  ^bb0(%q : f32, %k : f32, %acc : f32):
    %prod = arith.mulf %q, %k : f32
    %sum  = arith.addf %acc, %prod : f32
    linalg.yield %sum : f32
  }
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%qk : memref<16x16xf32>) outs(%sc : memref<16x16xf32>) {
  ^bb0(%in : f32, %out : f32):
    %r = arith.mulf %in, %scale : f32
    linalg.yield %r : f32
  }
  linalg.softmax dimension(1)
    ins(%sc : memref<16x16xf32>) outs(%p : memref<16x16xf32>)
  linalg.fill ins(%zero : f32) outs(%output : memref<16x8xf32>)
  linalg.matmul ins(%p, %V : memref<16x16xf32>, memref<16x8xf32>)
    outs(%output : memref<16x8xf32>)
  memref.dealloc %qk : memref<16x16xf32>
  memref.dealloc %sc : memref<16x16xf32>
  memref.dealloc %p  : memref<16x16xf32>
  return
}

// Negative: a second reader of an intermediate buffer (%p) would read
// uninitialized memory after fusion, so the pass must not fuse.

// CHECK-LABEL: func.func @intermediate_has_other_reader
// CHECK-NOT:   attention.fused
// CHECK:       linalg.softmax
// CHECK:       memref.copy
// CHECK-NOT:   attention.fused

func.func @intermediate_has_other_reader(
    %Q : memref<16x8xf32>, %K : memref<16x8xf32>, %V : memref<16x8xf32>,
    %scale : f32, %output : memref<16x8xf32>, %extra : memref<16x16xf32>) {
  %qk = memref.alloc() : memref<16x16xf32>
  %sc = memref.alloc() : memref<16x16xf32>
  %p  = memref.alloc() : memref<16x16xf32>
  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%qk : memref<16x16xf32>)
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%Q, %K : memref<16x8xf32>, memref<16x8xf32>)
    outs(%qk : memref<16x16xf32>) {
  ^bb0(%q : f32, %k : f32, %acc : f32):
    %prod = arith.mulf %q, %k : f32
    %sum  = arith.addf %acc, %prod : f32
    linalg.yield %sum : f32
  }
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%qk : memref<16x16xf32>) outs(%sc : memref<16x16xf32>) {
  ^bb0(%in : f32, %out : f32):
    %r = arith.mulf %in, %scale : f32
    linalg.yield %r : f32
  }
  linalg.softmax dimension(1)
    ins(%sc : memref<16x16xf32>) outs(%p : memref<16x16xf32>)
  linalg.fill ins(%zero : f32) outs(%output : memref<16x8xf32>)
  linalg.matmul ins(%p, %V : memref<16x16xf32>, memref<16x8xf32>)
    outs(%output : memref<16x8xf32>)
  memref.copy %p, %extra : memref<16x16xf32> to memref<16x16xf32>
  memref.dealloc %qk : memref<16x16xf32>
  memref.dealloc %sc : memref<16x16xf32>
  memref.dealloc %p  : memref<16x16xf32>
  return
}

// Negative: attention.fused accepts only f32, so f16 input is not fused.

// CHECK-LABEL: func.func @f16_not_fused
// CHECK-NOT:   attention.fused
// CHECK:       linalg.softmax
// CHECK:       linalg.matmul
// CHECK-NOT:   attention.fused

func.func @f16_not_fused(
    %Q : memref<16x8xf16>, %K : memref<16x8xf16>, %V : memref<16x8xf16>,
    %scale : f16, %output : memref<16x8xf16>) {
  %qk = memref.alloc() : memref<16x16xf16>
  %sc = memref.alloc() : memref<16x16xf16>
  %p  = memref.alloc() : memref<16x16xf16>
  %zero = arith.constant 0.0 : f16
  linalg.fill ins(%zero : f16) outs(%qk : memref<16x16xf16>)
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%Q, %K : memref<16x8xf16>, memref<16x8xf16>)
    outs(%qk : memref<16x16xf16>) {
  ^bb0(%q : f16, %k : f16, %acc : f16):
    %prod = arith.mulf %q, %k : f16
    %sum  = arith.addf %acc, %prod : f16
    linalg.yield %sum : f16
  }
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%qk : memref<16x16xf16>) outs(%sc : memref<16x16xf16>) {
  ^bb0(%in : f16, %out : f16):
    %r = arith.mulf %in, %scale : f16
    linalg.yield %r : f16
  }
  linalg.softmax dimension(1)
    ins(%sc : memref<16x16xf16>) outs(%p : memref<16x16xf16>)
  linalg.fill ins(%zero : f16) outs(%output : memref<16x8xf16>)
  linalg.matmul ins(%p, %V : memref<16x16xf16>, memref<16x8xf16>)
    outs(%output : memref<16x8xf16>)
  memref.dealloc %qk : memref<16x16xf16>
  memref.dealloc %sc : memref<16x16xf16>
  memref.dealloc %p  : memref<16x16xf16>
  return
}
