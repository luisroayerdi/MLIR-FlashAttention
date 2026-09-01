"""Generates the two MLIR modules compared by the CPU benchmark (§5.2):

  - `emit_baseline_module`: a naive unfused attention function using only
    `linalg.generic`/`linalg.matmul`/`linalg.fill` (softmax expanded into its
    four explicit steps: rowmax, exp(x - rowmax), rowsum, divide). Unlike
    codegen.py's `@attention_unfused`, this does NOT use `linalg.softmax` --
    that op has no `--convert-linalg-to-loops` lowering (verified empirically;
    see TRADEOFFS.md), so it cannot be executed directly. This module is
    lowered straight to LLVM with no attention-opt passes applied at all --
    it IS the unfused baseline.

  - `emit_fused_input_module`: the same `@attention_unfused` (linalg.softmax
    form) codegen.py already produces, wrapped in a timed loop instead of a
    single call + print. This is fed through
    `attention-opt --fusion-pass --tiling-pass` before lowering, so it
    measures the Pass 1+2 output.

Both wrap the repeated calls in an untimed warmup loop followed by a timed
loop, using `rtclock`/`printF64` from mlir_c_runner_utils so the elapsed time
is measured by native code *after* JIT compilation, inside a single process
run -- no per-call subprocess overhead.
"""

import numpy as np

from codegen import (Shapes, call_args_text, globals_block,
                      gpu_host_register_block, load_globals_block, shapes_of,
                      unfused_func_text)


def _timed_main(s: Shapes, scale: float, func_name: str,
                 warmup_iters: int, timed_iters: int, gpu: bool = False) -> str:
    gpu_registers = f"\n{gpu_host_register_block(s)}\n" if gpu else ""
    return f"""
  func.func private @rtclock() -> f64
  func.func private @printF64(f64)
  func.func private @printNewline()

  func.func @main() {{
{load_globals_block(s, scale)}
    %output = memref.alloc() : {s.outT}
{gpu_registers}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %warmup = arith.constant {warmup_iters} : index
    %iters  = arith.constant {timed_iters} : index

    scf.for %i = %c0 to %warmup step %c1 {{
      func.call @{func_name}({call_args_text(s)}, %output) : {s.call_sig_types}
      scf.yield
    }}

    %t0 = call @rtclock() : () -> f64
    scf.for %i = %c0 to %iters step %c1 {{
      func.call @{func_name}({call_args_text(s)}, %output) : {s.call_sig_types}
      scf.yield
    }}
    %t1 = call @rtclock() : () -> f64

    %elapsed = arith.subf %t1, %t0 : f64
    call @printF64(%elapsed) : (f64) -> ()
    call @printNewline() : () -> ()

    memref.dealloc %output : {s.outT}
    return
  }}
"""


def emit_fused_input_module(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                             scale: float, mask: np.ndarray | None,
                             warmup_iters: int, timed_iters: int,
                             gpu: bool = False) -> str:
    """`@attention_unfused` (linalg.softmax form) + a timed calling loop.
    Feed through `attention-opt --fusion-pass --tiling-pass` before lowering
    (add `--gpu-lowering-pass` too when `gpu=True` -- this only controls
    whether the module's own `@main` registers its memrefs with
    gpu.host_register before the timed loop; see codegen.py's
    gpu_host_register_block)."""
    s = shapes_of(Q, K, V, mask)
    return f"""module {{
{globals_block(Q, K, V, mask, s)}
{unfused_func_text(s)}
{_timed_main(s, scale, "attention_unfused", warmup_iters, timed_iters, gpu=gpu)}
}}
"""


def _baseline_func_text(s: Shapes) -> str:
    mask_arg = f",\n    %mask   : {s.maskT}" if s.has_mask else ""
    mask_alloc = f"\n  %mk = memref.alloc() : {s.qkT}" if s.has_mask else ""
    mask_dealloc = f"\n  memref.dealloc %mk : {s.qkT}" if s.has_mask else ""

    mask_generic = ""
    scores_buf = "%sc"
    if s.has_mask:
        mask_generic = f"""
  linalg.generic {{
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  }} ins(%sc, %mask : {s.qkT}, {s.maskT})
    outs(%mk      : {s.qkT}) {{
  ^bb0(%score : f32, %m : i1, %out : f32):
    %mninf = arith.constant -3.4028235e+38 : f32
    %r    = arith.select %m, %mninf, %score : f32
    linalg.yield %r : f32
  }}"""
        scores_buf = "%mk"

    rowT = f"memref<{s.seq_q}xf32>"

    return f"""
func.func @attention_baseline(
    %Q      : {s.qT},
    %K      : {s.kT},
    %V      : {s.kT},
    %scale  : f32{mask_arg},
    %output : {s.outT}) {{

  %qk  = memref.alloc() : {s.qkT}
  %sc  = memref.alloc() : {s.qkT}{mask_alloc}
  %rmx = memref.alloc() : {rowT}
  %ex  = memref.alloc() : {s.qkT}
  %rsm = memref.alloc() : {rowT}
  %p   = memref.alloc() : {s.qkT}

  %zero  = arith.constant 0.0 : f32
  %ninf  = arith.constant -3.4028235e+38 : f32
  linalg.fill ins(%zero : f32) outs(%qk : {s.qkT})

  // QK^T
  linalg.generic {{
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  }} ins(%Q, %K : {s.qT}, {s.kT})
    outs(%qk   : {s.qkT}) {{
  ^bb0(%q : f32, %k : f32, %acc : f32):
    %prod = arith.mulf %q, %k : f32
    %sum  = arith.addf %acc, %prod : f32
    linalg.yield %sum : f32
  }}

  // scale
  linalg.generic {{
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  }} ins(%qk : {s.qkT})
    outs(%sc : {s.qkT}) {{
  ^bb0(%in : f32, %out : f32):
    %r = arith.mulf %in, %scale : f32
    linalg.yield %r : f32
  }}
{mask_generic}

  // row max
  linalg.fill ins(%ninf : f32) outs(%rmx : {rowT})
  linalg.generic {{
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  }} ins({scores_buf} : {s.qkT})
    outs(%rmx : {rowT}) {{
  ^bb0(%e : f32, %acc : f32):
    %m = arith.maximumf %e, %acc : f32
    linalg.yield %m : f32
  }}

  // exp(x - rowmax)
  linalg.generic {{
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  }} ins({scores_buf}, %rmx : {s.qkT}, {rowT})
    outs(%ex : {s.qkT}) {{
  ^bb0(%e : f32, %m : f32, %out : f32):
    %d = arith.subf %e, %m : f32
    %x = math.exp %d : f32
    linalg.yield %x : f32
  }}

  // row sum
  linalg.fill ins(%zero : f32) outs(%rsm : {rowT})
  linalg.generic {{
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  }} ins(%ex : {s.qkT})
    outs(%rsm : {rowT}) {{
  ^bb0(%e : f32, %acc : f32):
    %a = arith.addf %acc, %e : f32
    linalg.yield %a : f32
  }}

  // divide by row sum -> probs
  linalg.generic {{
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  }} ins(%ex, %rsm : {s.qkT}, {rowT})
    outs(%p : {s.qkT}) {{
  ^bb0(%e : f32, %sum : f32, %out : f32):
    %r = arith.divf %e, %sum : f32
    linalg.yield %r : f32
  }}

  // PV matmul
  linalg.fill ins(%zero : f32) outs(%output : {s.outT})
  linalg.matmul
    ins(%p, %V   : {s.qkT}, {s.kT})
    outs(%output : {s.outT})

  memref.dealloc %qk  : {s.qkT}
  memref.dealloc %sc  : {s.qkT}{mask_dealloc}
  memref.dealloc %rmx : {rowT}
  memref.dealloc %ex  : {s.qkT}
  memref.dealloc %rsm : {rowT}
  memref.dealloc %p   : {s.qkT}

  return
}}
"""


def emit_baseline_module(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                          scale: float, mask: np.ndarray | None,
                          warmup_iters: int, timed_iters: int) -> str:
    """Naive unfused attention (no linalg.softmax) + a timed calling loop.
    Lowered straight to LLVM -- no attention-opt passes applied."""
    s = shapes_of(Q, K, V, mask)
    return f"""module {{
{globals_block(Q, K, V, mask, s)}
{_baseline_func_text(s)}
{_timed_main(s, scale, "attention_baseline", warmup_iters, timed_iters)}
}}
"""
