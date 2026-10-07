#!/usr/bin/env python3
"""Whether IREE skips fully masked K/V tiles for causal attention on CUDA.

Exports causal SDPA from PyTorch with iree-turbine in two forms (is_causal,
and an explicit boolean mask), compiles each with iree-compile for CUDA, and
inspects the attention dispatch's K/V loop (the scf.for carrying the online
softmax state) just before SCF is lowered to control flow. Block skipping
would show up as loop bounds that depend on the workgroup (query tile) id,
or as a conditional inside the loop body.

Prints, per form: the mask operand iree_linalg_ext.attention receives, the
K/V loop's bounds and trip count, whether the bounds use the workgroup id,
conditionals in the loop body, and load ops from the mask in the loop body.

Needs an iree-compile with the CUDA backend: the Linux iree-base-compiler
wheels have it, the macOS wheel doesn't. On Linux:
    python3 -m venv benchmarks/iree/.venv
    benchmarks/iree/.venv/bin/pip install -r benchmarks/iree/requirements.txt
    benchmarks/iree/.venv/bin/python benchmarks/iree/causal_skipping.py
On macOS, the same inside Docker (CPU-only torch avoids ~3 GB of CUDA
libraries; compiling for CUDA doesn't need them):
    docker run --rm -v "$PWD":/repo -w /repo python:3.13-slim sh -c \\
      "pip install -q -r benchmarks/iree/requirements.txt \\
         --extra-index-url https://download.pytorch.org/whl/cpu &&
       python benchmarks/iree/causal_skipping.py"
"""

import re
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import torch
import iree.turbine.aot as aot

B, H, N, D = 1, 4, 1024, 64
CUDA_TARGET = "sm_80"


class IsCausal(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True)


class ExplicitMask(torch.nn.Module):
    def forward(self, q, k, v, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask)


def _forms():
    q, k, v = (torch.randn(B, H, N, D) for _ in range(3))
    # [1, 1, N, N]: a 2-D [N, N] mask fails torch-to-IREE conversion in
    # iree-base-compiler 3.12.0 (tensor.collapse_shape rank error).
    tril = torch.tril(torch.ones(N, N, dtype=torch.bool))[None, None]
    return [
        ("SDPA `is_causal=True`", IsCausal(), (q, k, v)),
        ("SDPA bool `attn_mask` `[1,1,N,N]`", ExplicitMask(), (q, k, v, tril)),
    ]


def _iree_compile(mlir: Path, *flags: str) -> subprocess.CompletedProcess:
    compiler = Path(sys.executable).parent / "iree-compile"
    return subprocess.run([str(compiler), str(mlir), *flags],
                          capture_output=True, text=True, check=True)


def _mask_operand(mlir: Path) -> str:
    out = _iree_compile(mlir, "--compile-to=input").stdout
    line = next(l for l in out.splitlines() if "iree_linalg_ext.attention" in l)
    types = re.search(r"ins\(.*: (.*)\) outs", line).group(1).split(", ")
    return types[4] if len(types) > 4 else "none"


def _attention_dispatch_before_cf(mlir: Path) -> list[str]:
    """Last dump of the attention dispatch before convert-scf-to-cf."""
    err = _iree_compile(
        mlir, "--iree-hal-target-device=cuda",
        f"--iree-cuda-target={CUDA_TARGET}", "--mlir-disable-threading",
        "--mlir-print-ir-before=convert-scf-to-cf",
        "-o", str(mlir.with_suffix(".vmfb"))).stderr
    dumps = [d.splitlines() for d in err.split("// -----// IR Dump Before")]
    return [d for d in dumps
            if any("func.func" in l and "_attention_" in l for l in d)][-1]


def _kv_loop(lines: list[str]) -> tuple[str, list[str]]:
    """The scf.for whose body updates the running max (arith.maximumf)."""
    for i, line in enumerate(lines):
        if "scf.for" not in line:
            continue
        indent = len(line) - len(line.lstrip())
        end = next(j for j in range(i + 1, len(lines))
                   if lines[j].strip().startswith("}")
                   and len(lines[j]) - len(lines[j].lstrip()) == indent)
        body = lines[i + 1:end]
        if any("arith.maximumf" in l for l in body):
            return line.strip(), body
    raise RuntimeError("no K/V loop found in the attention dispatch")


def _report(name: str, mlir: Path) -> str:
    mask_type = _mask_operand(mlir)
    lines = _attention_dispatch_before_cf(mlir)
    header, body = _kv_loop(lines)
    lb, ub, step = re.search(r"= (\S+) to (\S+) step (\S+)", header).groups()
    consts = {m.group(1): int(m.group(2)) for m in (
        re.search(r"(%\S+) = arith\.constant (-?\d+) :", l) for l in lines)
        if m}
    trips = ((consts[ub] - consts[lb]) // consts[step]
             if {lb, ub, step} <= consts.keys() else "dynamic")
    # A bound that isn't a constant would be computed from the workgroup id.
    wg_bounds = "no" if {lb, ub} <= consts.keys() else "yes"
    conds = sum(1 for l in body if re.search(r"\bscf\.if\b", l))
    # The mask is the dispatch's only operand with an N x N shape.
    mask_loads = sum(1 for l in body
                     if "load" in l and f"x{N}x{N}x" in l)
    return (f"| {name} | `{mask_type}` | `{lb} to {ub} step {step}` "
            f"({trips} iterations) | {wg_bounds} | {conds} | {mask_loads} |")


def main() -> None:
    print(f"torch {version('torch')}, "
          f"iree-base-compiler {version('iree-base-compiler')}, "
          f"iree-turbine {version('iree-turbine')}; "
          f"f32, B={B} H={H} N={N} D={D}, cuda {CUDA_TARGET}\n")
    print("| Form | Mask operand of `iree_linalg_ext.attention` "
          "| K/V loop | Bounds use workgroup id | `scf.if` in loop body "
          "| Mask load ops in loop body |")
    print("|---|---|---|---|---|---|")
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, module, args) in enumerate(_forms()):
            path = Path(tmp) / f"form{i}.mlir"
            aot.export(module, args=args).save_mlir(str(path))
            print(_report(name, path))


if __name__ == "__main__":
    main()
