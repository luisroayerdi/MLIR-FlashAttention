"""Drives the MLIR pipeline: attention-opt (Pass 1 + Pass 2) -> standard MLIR
lowering to LLVM dialect -> mlir-runner JIT execution -> parsed numpy output.
"""

import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# .dylib on this Mac dev environment; .so on the Linux GPU instance this
# module also needs to run on (Stage 2 -- see Design.md 7.2/NOTES.md). Picked
# by platform, not hardcoded, since GPU execution only ever happens on Linux.
_SHARED_LIB_EXT = ".dylib" if sys.platform == "darwin" else ".so"

# RTX 4090 (Ada) is this project's actual GPU target -- see Design.md 7.2.
# Matches the cubin-chip used in test/Attention/gpu_tensor_core_matmul.mlir's
# and gpu_matmul_no_tensorcore.mlir's documented RUN-GPU lines.
_GPU_CUBIN_CHIP = "sm_89"
_GPU_CUBIN_FEATURES = "+ptx76"

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
    # None on a CPU-only build (e.g. this Mac dev environment); populated
    # when the LLVM build this Toolchain points at was built with
    # -DMLIR_ENABLE_CUDA_RUNNER=ON (Design.md 7.2) -- only the GPU-executing
    # functions below (run_module_gpu, run_fused_timed_gpu) need it, so its
    # absence does not fail check(), only check_gpu().
    cuda_runtime_lib: Path | None = None

    @staticmethod
    def discover() -> "Toolchain":
        llvm_build = _discover_llvm_build_dir()
        cuda_lib = llvm_build / "lib" / f"libmlir_cuda_runtime{_SHARED_LIB_EXT}"
        return Toolchain(
            attention_opt=REPO_ROOT / "build" / "bin" / "attention-opt",
            mlir_opt=llvm_build / "bin" / "mlir-opt",
            mlir_runner=llvm_build / "bin" / "mlir-runner",
            shared_libs=[
                llvm_build / "lib" / f"libmlir_runner_utils{_SHARED_LIB_EXT}",
                llvm_build / "lib" / f"libmlir_c_runner_utils{_SHARED_LIB_EXT}",
            ],
            cuda_runtime_lib=cuda_lib if cuda_lib.exists() else None,
        )

    def check(self) -> None:
        missing = [str(p) for p in
                   [self.attention_opt, self.mlir_opt, self.mlir_runner, *self.shared_libs]
                   if not p.exists()]
        if missing:
            raise RuntimeError("Missing required tool(s):\n  " + "\n  ".join(missing))

    def check_gpu(self) -> None:
        """Like check(), plus the CUDA runtime shared lib GPU execution
        needs. Raises with a clear, actionable message on a CPU-only build
        (e.g. this Mac dev environment) rather than letting a GPU-targeted
        run fail confusingly deeper in the pipeline."""
        self.check()
        if self.cuda_runtime_lib is None:
            raise RuntimeError(
                "GPU execution requested, but libmlir_cuda_runtime"
                f"{_SHARED_LIB_EXT} was not found next to this build's other "
                "MLIR shared libs. This LLVM build needs "
                "-DMLIR_ENABLE_CUDA_RUNNER=ON and NVPTX in "
                "-DLLVM_TARGETS_TO_BUILD -- see Design.md Section 7.2. Not "
                "available on this Mac dev environment; only on the Stage 2 "
                "GPU instance."
            )


def _run(cmd: list[str], stdin: str | None = None) -> str:
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n--- stdout ---\n{proc.stdout}"
            f"\n--- stderr ---\n{proc.stderr}"
        )
    return proc.stdout


def run_module(module_text: str, tile_size: int, tools: Toolchain,
                vectorize: bool = False, mask_specialize: bool = False) -> list:
    """Run fusion+tiling(+vectorization)(+mask-specialization), lower to LLVM,
    JIT-execute, and return the parsed nested-list output printed by
    printMemrefF32."""
    passes = [str(tools.attention_opt), "-",
              "--fusion-pass", f"--tiling-pass=tile-size={tile_size}"]
    if vectorize:
        passes.append("--vectorization-pass")
    if mask_specialize:
        passes.append("--mask-specialization-pass")
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
                     vectorize: bool = False, mask_specialize: bool = False) -> float:
    """Run fusion+tiling(+vectorization)(+mask-specialization) on a
    bench_codegen.emit_fused_input_module output, then lower+run. Returns
    total elapsed seconds for the timed loop (not divided by iteration count)."""
    passes = [str(tools.attention_opt), "-",
              "--fusion-pass", f"--tiling-pass=tile-size={tile_size}"]
    if vectorize:
        passes.append("--vectorization-pass")
    if mask_specialize:
        passes.append("--mask-specialization-pass")
    fused_tiled = _run(passes, stdin=module_text)
    return _run_timed(fused_tiled, tools)


# ── GPU (Stage 2) execution ─────────────────────────────────────────────────
#
# Design.md 7.6 / TRADEOFFS.md: the NVVM pipeline used here
# (-gpu-lower-to-nvvm-pipeline) is missing --lower-vector-multi-reduction and
# --convert-ub-to-llvm, the two extra passes vectorized output needs on CPU
# too (see TRADEOFFS.md "Vectorization pass: wired into the numerical/
# benchmark harness..."). So none of the functions below support
# vectorize=True -- combining GPU execution with Pass 3's output is a known,
# documented gap, not silently wrong. mask_specialize alone is fine: Pass 4
# only introduces affine.if, which -gpu-lower-to-nvvm-pipeline's own
# --lower-affine already handles.
#
# None of this is runnable against this Mac build (no NVPTX, no CUDA
# runtime) -- see Toolchain.check_gpu().


def _gpu_nvvm_pipeline_flag() -> str:
    return (f"-gpu-lower-to-nvvm-pipeline=cubin-chip={_GPU_CUBIN_CHIP} "
            f"cubin-features={_GPU_CUBIN_FEATURES} cubin-format=bin")


def _gpu_shared_libs_arg(tools: Toolchain) -> str:
    assert tools.cuda_runtime_lib is not None  # caller must have called check_gpu()
    return "--shared-libs=" + ",".join(
        str(p) for p in [*tools.shared_libs, tools.cuda_runtime_lib])


def run_module_gpu(module_text: str, tile_size: int, tools: Toolchain,
                    mask_specialize: bool = False) -> list:
    """GPU counterpart of run_module: fusion+tiling(+mask-specialization)+
    Stage A (--gpu-lowering-pass), then GPU lowering/execution instead of CPU
    mlir-runner. No `vectorize` parameter -- see this module's GPU-section
    comment. `module_text` must already contain gpu.host_register calls for
    every memref the kernel touches, positioned before the call/loop Stage A
    will wrap in gpu.launch (codegen.py's emit_module(..., gpu=True) does
    this) -- see that function's docstring for why the ordering matters."""
    passes = [str(tools.attention_opt), "-",
              "--fusion-pass", f"--tiling-pass=tile-size={tile_size}"]
    if mask_specialize:
        passes.append("--mask-specialization-pass")
    passes.append("--gpu-lowering-pass")
    gpu_lowered = _run(passes, stdin=module_text)

    nvvm_lowered = _run(
        [str(tools.mlir_opt), "-", _gpu_nvvm_pipeline_flag()], stdin=gpu_lowered)

    output = _run(
        [str(tools.mlir_runner), "-", "-e", "main", "-entry-point-result=void",
         _gpu_shared_libs_arg(tools)],
        stdin=nvvm_lowered,
    )

    m = _DATA_RE.search(output)
    if not m:
        raise RuntimeError(f"Could not find printed memref data in output:\n{output}")
    literal = m.group(1).strip()
    return ast.literal_eval(literal)


def _run_timed_gpu(module_text: str, tools: Toolchain) -> float:
    """GPU counterpart of _run_timed: `module_text` is already a fully-formed,
    already-attention-opt'd module (including --gpu-lowering-pass) with
    gpu.host_register calls already in place."""
    nvvm_lowered = _run(
        [str(tools.mlir_opt), "-", _gpu_nvvm_pipeline_flag()], stdin=module_text)

    output = _run(
        [str(tools.mlir_runner), "-", "-e", "main", "-entry-point-result=void",
         _gpu_shared_libs_arg(tools)],
        stdin=nvvm_lowered,
    )
    try:
        return float(output.strip())
    except ValueError as e:
        raise RuntimeError(f"Could not parse timing output:\n{output}") from e


def run_fused_timed_gpu(module_text: str, tile_size: int, tools: Toolchain,
                         mask_specialize: bool = False) -> float:
    """GPU counterpart of run_fused_timed: fusion+tiling(+mask-specialization)
    + Stage A (--gpu-lowering-pass), then GPU lowering/execution.
    `module_text` must come from
    bench_codegen.emit_fused_input_module(..., gpu=True). No `vectorize`
    parameter -- see the module-level comment above."""
    passes = [str(tools.attention_opt), "-",
              "--fusion-pass", f"--tiling-pass=tile-size={tile_size}"]
    if mask_specialize:
        passes.append("--mask-specialization-pass")
    passes.append("--gpu-lowering-pass")
    gpu_lowered = _run(passes, stdin=module_text)
    return _run_timed_gpu(gpu_lowered, tools)
