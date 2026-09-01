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

The shape/global/function-text helpers here are also reused by
bench_codegen.py to build the CPU benchmarking modules, so that the fused
input IR fed to the benchmark is byte-for-byte the same pattern the
numerical-correctness harness already validated.
"""

from dataclasses import dataclass

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


@dataclass
class Shapes:
    seq_q: int
    seq_k: int
    head_dim: int
    has_mask: bool

    @property
    def qkT(self) -> str:
        return _mlir_type((self.seq_q, self.seq_k), "f32")

    @property
    def qT(self) -> str:
        return _mlir_type((self.seq_q, self.head_dim), "f32")

    @property
    def kT(self) -> str:
        return _mlir_type((self.seq_k, self.head_dim), "f32")

    @property
    def outT(self) -> str:
        return _mlir_type((self.seq_q, self.head_dim), "f32")

    @property
    def maskT(self) -> str:
        return _mlir_type((self.seq_q, self.seq_k), "i1")

    @property
    def call_sig_types(self) -> str:
        mask_part = f", {self.maskT}" if self.has_mask else ""
        return f"({self.qT}, {self.kT}, {self.kT}, f32{mask_part}, {self.outT}) -> ()"


def shapes_of(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
              mask: np.ndarray | None) -> Shapes:
    seq_q, head_dim = Q.shape
    seq_k, _ = K.shape
    assert K.shape == V.shape == (seq_k, head_dim)
    if mask is not None:
        assert mask.shape == (seq_q, seq_k)
    return Shapes(seq_q, seq_k, head_dim, mask is not None)


def globals_block(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                   mask: np.ndarray | None, s: Shapes) -> str:
    """memref.global constant declarations for Q, K, V, and (optionally) mask."""
    mask_global = ""
    if mask is not None:
        mask_global = (
            f'\n  memref.global "private" constant @Mask_data : {s.maskT} '
            f"= dense<{_format_dense(mask)}>"
        )
    return (
        f'  memref.global "private" constant @Q_data : {s.qT} = dense<{_format_dense(Q)}>\n'
        f'  memref.global "private" constant @K_data : {s.kT} = dense<{_format_dense(K)}>\n'
        f'  memref.global "private" constant @V_data : {s.kT} = dense<{_format_dense(V)}>'
        f"{mask_global}"
    )


def load_globals_block(s: Shapes, scale: float) -> str:
    """Statements loading Q/K/V/mask/scale from globals into SSA values,
    matching the names used by `call_args_text`."""
    mask_load = f"\n    %mask = memref.get_global @Mask_data : {s.maskT}" if s.has_mask else ""
    return (
        f"    %Q = memref.get_global @Q_data : {s.qT}\n"
        f"    %K = memref.get_global @K_data : {s.kT}\n"
        f"    %V = memref.get_global @V_data : {s.kT}{mask_load}\n"
        f"    %scale = arith.constant {float(scale)!r} : f32"
    )


def call_args_text(s: Shapes) -> str:
    mask_arg = ", %mask" if s.has_mask else ""
    return f"%Q, %K, %V, %scale{mask_arg}"


def gpu_host_register_block(s: Shapes, output_name: str = "output") -> str:
    """gpu.host_register calls for every memref the kernel touches (Q, K, V,
    mask if present, output), each `memref.cast` to unranked first --
    matching the pattern MLIR's own upstream tensor-core test uses (see
    test/Attention/gpu_tensor_core_matmul.mlir).

    Must be emitted BEFORE the call/loop that Pass 5 Stage A
    (--gpu-lowering-pass) will wrap in gpu.launch -- host_register makes a
    host allocation dereferenceable from device code; doing it after the
    kernel has already run doesn't fail loudly, it silently reads
    uninitialized/invalid device memory. See TRADEOFFS.md "GPU execution:
    gpu.host_register ordering" for how this was first gotten wrong in the
    Stage B microbenchmark files.
    """
    entries = [("Q", s.qT, "f32"), ("K", s.kT, "f32"), ("V", s.kT, "f32")]
    if s.has_mask:
        entries.append(("mask", s.maskT, "i1"))
    entries.append((output_name, f"memref<{s.seq_q}x{s.head_dim}xf32>", "f32"))

    lines = []
    for name, ty, elem in entries:
        u = f"%u{name}"
        lines.append(f"    {u} = memref.cast %{name} : {ty} to memref<*x{elem}>")
        lines.append(f"    gpu.host_register {u} : memref<*x{elem}>")
    return "\n".join(lines)


def unfused_func_text(s: Shapes) -> str:
    """The `@attention_unfused` function body: QK^T -> scale -> mask (optional)
    -> linalg.softmax -> PV matmul. This is the exact pattern FusionPass
    matches; it is NOT directly lowerable via --convert-linalg-to-loops
    (linalg.softmax has no loop lowering), so this text is only ever used as
    input to attention-opt --fusion-pass --tiling-pass, never executed as-is.
    """
    mask_arg = f",\n    %mask   : {s.maskT}" if s.has_mask else ""
    mask_alloc = f"\n  %mk = memref.alloc() : {s.qkT}" if s.has_mask else ""
    mask_dealloc = f"\n  memref.dealloc %mk : {s.qkT}" if s.has_mask else ""

    mask_generic = ""
    softmax_input = "%sc"
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
    %ninf = arith.constant -3.4028235e+38 : f32
    %r    = arith.select %m, %ninf, %score : f32
    linalg.yield %r : f32
  }}"""
        softmax_input = "%mk"

    return f"""
func.func @attention_unfused(
    %Q      : {s.qT},
    %K      : {s.kT},
    %V      : {s.kT},
    %scale  : f32{mask_arg},
    %output : {s.outT}) {{

  %qk = memref.alloc() : {s.qkT}
  %sc = memref.alloc() : {s.qkT}{mask_alloc}
  %p  = memref.alloc() : {s.qkT}

  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%qk : {s.qkT})

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

  linalg.softmax dimension(1)
    ins({softmax_input} : {s.qkT})
    outs(%p : {s.qkT})

  linalg.fill ins(%zero : f32) outs(%output : {s.outT})
  linalg.matmul
    ins(%p, %V   : {s.qkT}, {s.kT})
    outs(%output : {s.outT})

  memref.dealloc %qk : {s.qkT}
  memref.dealloc %sc : {s.qkT}{mask_dealloc}
  memref.dealloc %p  : {s.qkT}

  return
}}
"""


def emit_module(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float,
                 mask: np.ndarray | None = None, gpu: bool = False) -> str:
    s = shapes_of(Q, K, V, mask)
    gpu_registers = f"\n{gpu_host_register_block(s)}\n" if gpu else ""

    module = f"""module {{
{globals_block(Q, K, V, mask, s)}

  func.func private @printMemrefF32(memref<*xf32>)
{unfused_func_text(s)}
  func.func @main() {{
{load_globals_block(s, scale)}
    %output = memref.alloc() : {s.outT}
{gpu_registers}
    call @attention_unfused({call_args_text(s)}, %output)
      : {s.call_sig_types}

    %cast = memref.cast %output : {s.outT} to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %output : {s.outT}
    return
  }}
}}
"""
    return module
