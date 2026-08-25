"""Generates a self-contained MLIR module for a concrete attention test case.

The generated `@attention_unfused` function reproduces, verbatim in structure,
the 5-op unfused pattern FusionPass matches (see test/Attention/fusion.mlir):

    linalg.generic  ins(%Q, %K)          outs(%qk)    // QK^T
    linalg.generic  ins(%qk)             outs(%sc)    // scale
    linalg.generic  ins(%sc, %mask)      outs(%mk)    // mask  (optional)
    linalg.softmax  ins(%mk or %sc)      outs(%p)     // softmax
    linalg.matmul   ins(%p, %V)          outs(%out)   // PV

`@main` wraps it with concrete `memref.global` constant inputs, calls it, and
prints the output via `printMemrefF32` (from mlir_c_runner_utils) so the
result can be captured from stdout and parsed by pipeline.py.
"""

import numpy as np


def _format_dense(arr: np.ndarray) -> str:
    """Recursively format a numpy array as an MLIR dense-elements literal body."""
    if arr.ndim == 1:
        if arr.dtype == bool:
            return "[" + ", ".join("true" if x else "false" for x in arr) + "]"
        return "[" + ", ".join(repr(float(x)) for x in arr) + "]"
    return "[" + ", ".join(_format_dense(row) for row in arr) + "]"


def _mlir_type(shape: tuple[int, ...], elem: str) -> str:
    return "memref<" + "x".join(str(d) for d in shape) + "x" + elem + ">"


def emit_module(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float,
                 mask: np.ndarray | None = None) -> str:
    seq_q, head_dim = Q.shape
    seq_k, _ = K.shape
    assert K.shape == V.shape == (seq_k, head_dim)
    if mask is not None:
        assert mask.shape == (seq_q, seq_k)

    qkT = _mlir_type((seq_q, seq_k), "f32")
    qT = _mlir_type((seq_q, head_dim), "f32")
    kT = _mlir_type((seq_k, head_dim), "f32")
    outT = _mlir_type((seq_q, head_dim), "f32")
    maskT = _mlir_type((seq_q, seq_k), "i1")

    mask_arg = f",\n    %mask   : {maskT}" if mask is not None else ""
    mask_alloc = f"\n  %mk = memref.alloc() : {qkT}" if mask is not None else ""
    mask_dealloc = f"\n  memref.dealloc %mk : {qkT}" if mask is not None else ""

    mask_generic = ""
    softmax_input = "%sc"
    if mask is not None:
        mask_generic = f"""
  linalg.generic {{
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  }} ins(%sc, %mask : {qkT}, {maskT})
    outs(%mk      : {qkT}) {{
  ^bb0(%score : f32, %m : i1, %out : f32):
    %ninf = arith.constant -3.4028235e+38 : f32
    %r    = arith.select %m, %ninf, %score : f32
    linalg.yield %r : f32
  }}"""
        softmax_input = "%mk"

    unfused_func = f"""
func.func @attention_unfused(
    %Q      : {qT},
    %K      : {kT},
    %V      : {kT},
    %scale  : f32{mask_arg},
    %output : {outT}) {{

  %qk = memref.alloc() : {qkT}
  %sc = memref.alloc() : {qkT}{mask_alloc}
  %p  = memref.alloc() : {qkT}

  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%qk : {qkT})

  linalg.generic {{
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  }} ins(%Q, %K : {qT}, {kT})
    outs(%qk   : {qkT}) {{
  ^bb0(%q : f32, %k : f32, %acc : f32):
    %prod = arith.mulf %q, %k : f32
    %sum  = arith.addf %acc, %prod : f32
    linalg.yield %sum : f32
  }}

  linalg.generic {{
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  }} ins(%qk : {qkT})
    outs(%sc : {qkT}) {{
  ^bb0(%in : f32, %out : f32):
    %r = arith.mulf %in, %scale : f32
    linalg.yield %r : f32
  }}
{mask_generic}

  linalg.softmax dimension(1)
    ins({softmax_input} : {qkT})
    outs(%p : {qkT})

  linalg.fill ins(%zero : f32) outs(%output : {outT})
  linalg.matmul
    ins(%p, %V   : {qkT}, {kT})
    outs(%output : {outT})

  memref.dealloc %qk : {qkT}
  memref.dealloc %sc : {qkT}{mask_dealloc}
  memref.dealloc %p  : {qkT}

  return
}}
"""

    mask_global = ""
    mask_call_arg = ""
    mask_main_load = ""
    if mask is not None:
        mask_global = (
            f'\n  memref.global "private" constant @Mask_data : {maskT} '
            f"= dense<{_format_dense(mask)}>"
        )
        mask_main_load = f"\n    %mask = memref.get_global @Mask_data : {maskT}"
        mask_call_arg = ", %mask"

    call_sig_types = f"({qT}, {kT}, {kT}, f32{', ' + maskT if mask is not None else ''}, {outT}) -> ()"

    module = f"""module {{
  memref.global "private" constant @Q_data : {qT} = dense<{_format_dense(Q)}>
  memref.global "private" constant @K_data : {kT} = dense<{_format_dense(K)}>
  memref.global "private" constant @V_data : {kT} = dense<{_format_dense(V)}>{mask_global}

  func.func private @printMemrefF32(memref<*xf32>)
{unfused_func}
  func.func @main() {{
    %Q = memref.get_global @Q_data : {qT}
    %K = memref.get_global @K_data : {kT}
    %V = memref.get_global @V_data : {kT}{mask_main_load}
    %scale = arith.constant {scale!r} : f32
    %output = memref.alloc() : {outT}

    call @attention_unfused(%Q, %K, %V, %scale{mask_call_arg}, %output)
      : {call_sig_types}

    %cast = memref.cast %output : {outT} to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %output : {outT}
    return
  }}
}}
"""
    return module
