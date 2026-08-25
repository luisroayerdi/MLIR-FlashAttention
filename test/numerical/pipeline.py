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
    "--convert-linalg-to-loops",
    "--lower-affine",
    "--convert-scf-to-cf",
    "--expand-strided-metadata",
    "--lower-affine",  # expand-strided-metadata emits affine.apply for offsets
    "--convert-cf-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-math-to-llvm",
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


def run_module(module_text: str, tile_size: int, tools: Toolchain) -> list:
    """Run fusion+tiling, lower to LLVM, JIT-execute, and return the parsed
    nested-list output printed by printMemrefF32."""
    fused_tiled = _run(
        [str(tools.attention_opt), "-",
         "--fusion-pass", f"--tiling-pass=tile-size={tile_size}"],
        stdin=module_text,
    )

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
