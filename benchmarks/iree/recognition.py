#!/usr/bin/env python3
"""Which attention input forms IREE turns into iree_linalg_ext.attention.

Exports each form below from PyTorch with iree-turbine, compiles it with
iree-compile for llvm-cpu, and reports whether iree_linalg_ext.attention
(or online_attention) appears after two phases:
    input              torch-to-IREE input conversion (frontend lowering)
    dispatch-creation  after global optimization; what codegen receives

Prints a Markdown table and the package versions used.

Usage (Python 3.10-3.13; iree-base-compiler has no 3.14 wheels):
    python3.13 -m venv benchmarks/iree/.venv
    benchmarks/iree/.venv/bin/pip install -r benchmarks/iree/requirements.txt
    benchmarks/iree/.venv/bin/python benchmarks/iree/recognition.py
"""

import math
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import torch
import iree.turbine.aot as aot

B, H, N, D = 1, 4, 128, 64
SCALE = 1.0 / math.sqrt(D)
PHASES = ["input", "dispatch-creation"]
OPS = ("iree_linalg_ext.attention", "iree_linalg_ext.online_attention")


def _softmax_pv(scores, v):
    return torch.softmax(scores, dim=-1) @ v


class Sdpa(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class ScaleAfter(torch.nn.Module):
    def forward(self, q, k, v):
        return _softmax_pv((q @ k.transpose(-2, -1)) * SCALE, v)


class ScaleBefore(torch.nn.Module):
    def forward(self, q, k, v):
        return _softmax_pv((q * SCALE) @ k.transpose(-2, -1), v)


class TransposedK(torch.nn.Module):
    # K arrives already transposed, [B, H, D, N].
    def forward(self, q, kt, v):
        return _softmax_pv((q @ kt) * SCALE, v)


class AdditiveMask(torch.nn.Module):
    def forward(self, q, k, v, bias):
        return _softmax_pv((q @ k.transpose(-2, -1)) * SCALE + bias, v)


class SelectMask(torch.nn.Module):
    def forward(self, q, k, v, mask):
        s = (q @ k.transpose(-2, -1)) * SCALE
        return _softmax_pv(s.masked_fill(mask, float("-inf")), v)


def _forms():
    q = torch.randn(B, H, N, D)
    causal = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    bias = torch.zeros(N, N).masked_fill(causal, float("-inf"))
    return [
        ("SDPA (`F.scaled_dot_product_attention`)", Sdpa(), (q, q, q)),
        ("Hand-written, scale after QKᵀ", ScaleAfter(), (q, q, q)),
        ("Hand-written, scale before QKᵀ (on Q)", ScaleBefore(), (q, q, q)),
        ("Hand-written, K pre-transposed `[B,H,D,N]`", TransposedK(),
         (q, torch.randn(B, H, D, N), q)),
        ("Hand-written, additive mask (`+ bias`)", AdditiveMask(),
         (q, q, q, bias)),
        ("Hand-written, select mask (`masked_fill`)", SelectMask(),
         (q, q, q, causal)),
    ]


def _count_attention(mlir_path: Path, phase: str) -> int:
    compiler = Path(sys.executable).parent / "iree-compile"
    result = subprocess.run(
        [str(compiler), str(mlir_path), f"--compile-to={phase}",
         "--iree-hal-target-device=local",
         "--iree-hal-local-target-device-backends=llvm-cpu"],
        capture_output=True, text=True, check=True)
    return sum(result.stdout.count(op) for op in OPS)


def main() -> None:
    print(f"torch {version('torch')}, "
          f"iree-base-compiler {version('iree-base-compiler')}, "
          f"iree-turbine {version('iree-turbine')}; "
          f"f32, B={B} H={H} N={N} D={D}, llvm-cpu\n")
    print("| Form | " + " | ".join(f"after `{p}`" for p in PHASES) + " |")
    print("|---|" + "---|" * len(PHASES))
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, module, args) in enumerate(_forms()):
            path = Path(tmp) / f"form{i}.mlir"
            aot.export(module, args=args).save_mlir(str(path))
            cells = ["yes" if _count_attention(path, p) else "no"
                     for p in PHASES]
            print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
