"""Drives the MLIR pipeline: attention-opt (Pass 1 + Pass 2) -> standard MLIR
lowering to LLVM dialect -> mlir-runner JIT execution -> parsed numpy output.
"""

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_LOWER_FLAGS = [
    # Vector-dialect lowering (no-ops when the input has no vector ops, i.e.
    # Pass 1+2 output without --vectorization-pass -- see TRADEOFFS.md
    # "Vectorization pass: wired into the numerical/benchmark harness via an
    # opt-in `vectorize` flag").
    "--convert-vector-to-scf",       # multi-dim vector.transfer_* -> loops of 1-D ones
    "--lower-vector-multi-reduction",  # vector.multi_reduction -> vector.reduction/shuffle
    "--convert-linalg-to-loops",
    "--lower-affine",
    "--convert-scf-to-cf",
    "--expand-strided-metadata",
    "--lower-affine",  # expand-strided-metadata emits affine.apply for offsets
    "--convert-cf-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-math-to-llvm",
    "--convert-vector-to-llvm",
    "--convert-ub-to-llvm",  # ub.poison (vector.transfer_read padding) -> llvm.mlir.poison
    "--finalize-memref-to-llvm",
    "--convert-func-to-llvm",
    "--reconcile-unrealized-casts",
]

_DATA_RE = re.compile(r"data =\s*\n(.*)", re.DOTALL)


def _discover_llvm_build_dir() -> Path:
    """Read MLIR_DIR out of the CMake cache and derive the LLVM build dir."""
    cache = REPO_ROOT / "build" / "CMakeCache.txt"
    text = cache.read_text()
    m = re.search(r"^MLIR_DIR:\w+=(.*)$", text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"Could not find MLIR_DIR in {cache}")
    mlir_dir = Path(m.group(1).strip())
    # mlir_dir is .../build/lib/cmake/mlir
    return mlir_dir.parents[2]


@dataclass
class Toolchain:
    attention_opt: Path
    mlir_opt: Path
    mlir_runner: Path
    shared_libs: list[Path]

    @staticmethod
    def discover() -> "Toolchain":
        llvm_build = _discover_llvm_build_dir()
        return Toolchain(
            attention_opt=REPO_ROOT / "build" / "bin" / "attention-opt",
            mlir_opt=llvm_build / "bin" / "mlir-opt",
            mlir_runner=llvm_build / "bin" / "mlir-runner",
            shared_libs=[
                llvm_build / "lib" / "libmlir_runner_utils.dylib",
                llvm_build / "lib" / "libmlir_c_runner_utils.dylib",
            ],
        )

    def check(self) -> None:
        missing = [str(p) for p in
                   [self.attention_opt, self.mlir_opt, self.mlir_runner, *self.shared_libs]
                   if not p.exists()]
        if missing:
            raise RuntimeError("Missing required tool(s):\n  " + "\n  ".join(missing))


def _run(cmd: list[str], stdin: str | None = None) -> str:
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n--- stdout ---\n{proc.stdout}"
            f"\n--- stderr ---\n{proc.stderr}"
        )
    return proc.stdout


def run_module(module_text: str, tile_size: int, tools: Toolchain,
                vectorize: bool = False) -> list:
    """Run fusion+tiling(+vectorization), lower to LLVM, JIT-execute, and
    return the parsed nested-list output printed by printMemrefF32."""
    passes = [str(tools.attention_opt), "-",
              "--fusion-pass", f"--tiling-pass=tile-size={tile_size}"]
    if vectorize:
        passes.append("--vectorization-pass")
    fused_tiled = _run(passes, stdin=module_text)

    lowered = _run([str(tools.mlir_opt), "-", *_LOWER_FLAGS], stdin=fused_tiled)

    shared_libs_arg = "--shared-libs=" + ",".join(str(p) for p in tools.shared_libs)
    output = _run(
        [str(tools.mlir_runner), "-", "-e", "main", "-entry-point-result=void",
         shared_libs_arg],
        stdin=lowered,
    )

    m = _DATA_RE.search(output)
    if not m:
        raise RuntimeError(f"Could not find printed memref data in output:\n{output}")
    literal = m.group(1).strip()
    return ast.literal_eval(literal)


def _run_timed(module_text: str, tools: Toolchain) -> float:
    """Lower an already-fully-formed module (no attention-opt passes applied)
    straight to LLVM, JIT-execute it, and return the elapsed seconds printed
    by the module's own rtclock()/printF64() timing (see bench_codegen.py)."""
    lowered = _run([str(tools.mlir_opt), "-", *_LOWER_FLAGS], stdin=module_text)

    shared_libs_arg = "--shared-libs=" + ",".join(str(p) for p in tools.shared_libs)
    output = _run(
        [str(tools.mlir_runner), "-", "-e", "main", "-entry-point-result=void",
         shared_libs_arg],
        stdin=lowered,
    )
    try:
        return float(output.strip())
    except ValueError as e:
        raise RuntimeError(f"Could not parse timing output:\n{output}") from e


def run_baseline_timed(module_text: str, tools: Toolchain) -> float:
    """Lower+run a bench_codegen.emit_baseline_module output (no attention-opt
    passes -- it's the naive unfused baseline). Returns total elapsed seconds
    for the timed loop (not divided by iteration count)."""
    return _run_timed(module_text, tools)


def run_fused_timed(module_text: str, tile_size: int, tools: Toolchain,
                     vectorize: bool = False) -> float:
    """Run fusion+tiling(+vectorization) on a bench_codegen.emit_fused_input_module
    output, then lower+run. Returns total elapsed seconds for the timed loop
    (not divided by iteration count)."""
    passes = [str(tools.attention_opt), "-",
              "--fusion-pass", f"--tiling-pass=tile-size={tile_size}"]
    if vectorize:
        passes.append("--vectorization-pass")
    fused_tiled = _run(passes, stdin=module_text)
    return _run_timed(fused_tiled, tools)
