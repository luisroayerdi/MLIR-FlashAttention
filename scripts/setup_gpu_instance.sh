#!/usr/bin/env bash
# Provisions a rented Ubuntu+CUDA GPU instance (RTX 4090 spot on RunPod or
# Vast.ai, per NOTES.md's "3-stage compute plan") to build and run this
# project's Pass 5 Stage A/B GPU work -- Design.md 7.2's prerequisites, made
# runnable instead of prose. Point of this script: setup should be scripted
# and reviewed before paying for instance time, not improvised live over
# SSH (NOTES.md's own framing for this checklist item).
#
# Safe to re-run: every step is built on `cmake`/`ninja`'s own incremental
# behavior and `git fetch` + `checkout`, not custom "already done" tracking
# -- if a run gets interrupted (SSH drop, spot preemption), just run it
# again.
#
# Usage:
#   ./scripts/setup_gpu_instance.sh [options]
#
# Options (all also settable as environment variables of the same name):
#   --llvm-dir DIR     Where to clone/build llvm-project (default: ~/llvm-project)
#   --repo-dir DIR     Where to clone/build this project   (default: ~/MLIR-FlashAttention)
#   --jobs N           Parallel build jobs                 (default: nproc)
#   --cubin-chip CHIP  Target GPU compute capability        (default: sm_89, RTX 4090/Ada
#                      -- Design.md 7.2; pass e.g. sm_80 for an A100 Stage 3 instance)
#   --hardware-label L Provenance label for the final `analyze_ablation.py
#                      --collect --hardware` command (e.g. "RTX 4090
#                      (Vast.ai)") -- no default provider assumed
#   --skip-sanity      Skip the CPU-only regression check at the end
#   -h, --help         Show this help and exit
#
# What it does, in order:
#   1. Preflight: confirm nvidia-smi/nvcc actually work on this instance --
#      fail fast and clearly rather than partway through a long LLVM build.
#      Prints GPU model + driver + CUDA version, which Requirements.md 5.3's
#      provenance requirement wants recorded alongside every result anyway.
#   2. Install build dependencies via apt.
#   3. Clone this project's own llvm-project fork, pinned to the exact
#      commit this project's Mac development was built and verified
#      against (not just "whatever llvm-project HEAD happens to be today"),
#      and build it with NVPTX + the MLIR CUDA runner enabled -- the two
#      things Design.md 7.2 lists as needed only for Stage 2/3, not for the
#      IR-level authoring/FileCheck-testing already done locally.
#   4. Clone/update this repo and build attention-opt against that build.
#   5. Set up the Python venv test/numerical/requirements.txt needs.
#   6. Run the existing CPU-only regression suite (attention-opt's own
#      FileCheck tests + validate.py --suite) as a build sanity check --
#      these need no GPU at all, so a failure here means the build itself
#      is broken, not that anything GPU-specific is.
#   7. Print the exact next-step commands (matching Design.md/
#      Requirements.md/TRADEOFFS.md's own documented commands) rather than
#      running them -- Stage 2 execution is a deliberate, separate step.

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────
# Pinned to the exact commit this project's Mac development was verified
# against (see the "LLVM Build Environment" reference and
# `git -C llvm-project rev-parse HEAD` on the Mac dev machine). Override via
# LLVM_COMMIT= if intentionally moving to a newer commit -- doing so is a
# real risk, not a no-op: this project's passes call MLIR APIs
# (mlir::convertAffineLoopNestToGPULaunch, transform.nvgpu.rewrite_matmul_as_
# mma_sync, -gpu-lower-to-nvvm-pipeline's exact option names) that could
# shift under an unpinned checkout.
LLVM_REPO_URL="${LLVM_REPO_URL:-https://github.com/luisroayerdi/llvm-project.git}"
LLVM_COMMIT="${LLVM_COMMIT:-3699735d6c63d36643d0aaf79f0e131054f48d8c}"
PROJECT_REPO_URL="${PROJECT_REPO_URL:-https://github.com/luisroayerdi/MLIR-FlashAttention.git}"

LLVM_DIR="${LLVM_DIR:-$HOME/llvm-project}"
REPO_DIR="${REPO_DIR:-$HOME/MLIR-FlashAttention}"
# nproc is standard on the Ubuntu instance this targets; the fallback chain
# just lets --help/argument parsing work when poking at this script on a
# machine without it (e.g. macOS during review).
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
CUBIN_CHIP="${CUBIN_CHIP:-sm_89}"  # RTX 4090 (Ada) -- Design.md 7.2
# Provenance label for analyze_ablation.py --collect (Requirements.md 5.3) --
# no default provider assumed; pass --hardware-label or set the env var to
# whichever marketplace this instance is actually on.
HARDWARE_LABEL="${HARDWARE_LABEL:-RTX 4090 (unset -- pass --hardware-label)}"
SKIP_SANITY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llvm-dir) LLVM_DIR="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --cubin-chip) CUBIN_CHIP="$2"; shift 2 ;;
    --hardware-label) HARDWARE_LABEL="$2"; shift 2 ;;
    --skip-sanity) SKIP_SANITY=1; shift ;;
    -h|--help) sed -n '2,49p' "$0"; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. Preflight ─────────────────────────────────────────────────────────
log "Preflight: checking this instance actually has a GPU + CUDA toolkit"

command -v nvidia-smi >/dev/null 2>&1 || die \
  "nvidia-smi not found -- this doesn't look like a GPU instance, or the" \
  "driver isn't installed. Check the RunPod/Vast.ai template."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || die "nvidia-smi found but failed to run -- driver problem?"

command -v nvcc >/dev/null 2>&1 || die \
  "nvcc not found -- this instance needs a CUDA toolkit image (standard on" \
  "RunPod/Vast.ai CUDA templates; see NOTES.md 'Software needed on the" \
  "cloud instance')."
nvcc --version | tail -1

echo "GPU + CUDA toolkit look present. Provenance for later result bundles" \
     "(Requirements.md 5.3): save the two blocks above alongside every" \
     "collected number."

# test/numerical/*.py use PEP 604 "X | None" annotations (no
# `from __future__ import annotations`), which crash on import under Python
# <3.10 -- and that's an eager, module-import-time crash, not something a
# later step can route around. Ubuntu 20.04 images ship python3 3.8 by
# default; Ubuntu 22.04+ ships >=3.10. If this fails, relaunch with a newer
# base image rather than trying to patch Python versions on this one.
command -v python3 >/dev/null 2>&1 || die "python3 not found."
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die \
  "python3 is $(python3 --version 2>&1 | awk "{print \$2}"), but" \
  "test/numerical/*.py needs >=3.10 (PEP 604 \"X | None\" annotations," \
  "evaluated eagerly on import -- no __future__ import guards them). Pick" \
  "an Ubuntu 22.04+ base image instead of 20.04 and relaunch."

# ── 2. Build dependencies ───────────────────────────────────────────────
log "Installing build dependencies (apt)"
if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=""; fi
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build git python3 python3-venv python3-pip \
  ccache lld

# LLVM/MLIR need CMake >=3.20 (mlir/CMakeLists.txt); Ubuntu 20.04's apt
# cmake is 3.16, too old. Self-heal via pip's prebuilt cmake wheel rather
# than just erroring -- reliable across arbitrary Vast.ai host images,
# unlike chasing a Kitware apt repo per-distro. Compared with `sort -V`
# (version-aware), not as a bare float -- "3.9" vs "3.10" compares backwards
# as floating point (3.9 > 3.10) despite 3.9 being the older version.
CMAKE_VERSION="$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
OLDEST="$(printf '%s\n%s\n' "$CMAKE_VERSION" "3.20.0" | sort -V | head -1)"
if [[ "$OLDEST" != "3.20.0" ]]; then
  log "System cmake ($CMAKE_VERSION) is older than the 3.20 LLVM/MLIR needs -- installing a newer one via pip"
  pip3 install --quiet --user 'cmake>=3.20'
  export PATH="$HOME/.local/bin:$PATH"
  hash -r
  echo "Now using: $(command -v cmake) ($(cmake --version | head -1))"
fi

# ── 3. LLVM/MLIR, built with NVPTX + the CUDA runner ────────────────────
log "Fetching llvm-project ($LLVM_REPO_URL @ $LLVM_COMMIT) into $LLVM_DIR"
# Shallow, single-commit fetch, not a full clone -- llvm-project's full
# history is tens of GB and we only ever need this one pinned commit. This
# is billed instance time; a full clone would be pure waste. GitHub
# supports fetching an arbitrary commit SHA directly for public repos.
if [[ ! -d "$LLVM_DIR/.git" ]]; then
  mkdir -p "$LLVM_DIR"
  git -C "$LLVM_DIR" init -q
  git -C "$LLVM_DIR" remote add origin "$LLVM_REPO_URL"
fi
git -C "$LLVM_DIR" fetch --depth 1 origin "$LLVM_COMMIT"
git -C "$LLVM_DIR" checkout FETCH_HEAD

CMAKE_LAUNCHER_ARGS=()
if command -v ccache >/dev/null 2>&1; then
  CMAKE_LAUNCHER_ARGS=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi
LINKER_ARGS=()
if command -v ld.lld >/dev/null 2>&1; then
  LINKER_ARGS=(-DLLVM_USE_LINKER=lld)
fi

# LLVM_INCLUDE_EXAMPLES=OFF: mlir/examples/Hello does not build cleanly at
# the pinned commit above (its own commit message: "Failed attempt to run a
# simple mlir example") and this project never needs it -- attention-opt
# only depends on MLIR's libraries/tools, not its example dialects. Found
# the hard way: this build failed ~91% through without it. Confirmed
# against mlir/CMakeLists.txt's actual `if(LLVM_INCLUDE_EXAMPLES)
# add_subdirectory(examples)` -- LLVM_BUILD_EXAMPLES is a real but
# different option that does NOT gate this (first guess, wrong; this is
# the one that actually stops the examples subdirectory from being added
# to the build graph at all).
log "Configuring LLVM/MLIR (Release, NVPTX + MLIR_ENABLE_CUDA_RUNNER=ON)"
cmake -S "$LLVM_DIR/llvm" -B "$LLVM_DIR/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
  -DMLIR_ENABLE_CUDA_RUNNER=ON \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  "${CMAKE_LAUNCHER_ARGS[@]}" "${LINKER_ARGS[@]}"

log "Building LLVM/MLIR (-j $JOBS) -- this is the slow step, expect 20-60+ minutes"
cmake --build "$LLVM_DIR/build" -j "$JOBS"

[[ -f "$LLVM_DIR/build/bin/mlir-opt" ]] || die "mlir-opt was not produced -- LLVM build failed."
[[ -f "$LLVM_DIR/build/lib/libmlir_cuda_runtime.so" ]] || die \
  "libmlir_cuda_runtime.so was not produced -- MLIR_ENABLE_CUDA_RUNNER didn't" \
  "take effect (CUDA toolkit not found by CMake?). Check the cmake configure" \
  "output above for CUDAToolkit_* lines."
echo "LLVM/MLIR build OK: mlir-opt and libmlir_cuda_runtime.so both present."

# ── 4. This project, built against that LLVM ────────────────────────────
log "Cloning/updating $PROJECT_REPO_URL into $REPO_DIR"
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" fetch origin
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$PROJECT_REPO_URL" "$REPO_DIR"
fi

log "Configuring + building attention-opt"
cmake -S "$REPO_DIR" -B "$REPO_DIR/build" -G Ninja \
  -DMLIR_DIR="$LLVM_DIR/build/lib/cmake/mlir" \
  "${CMAKE_LAUNCHER_ARGS[@]}" "${LINKER_ARGS[@]}"
cmake --build "$REPO_DIR/build" -j "$JOBS" --target attention-opt

[[ -f "$REPO_DIR/build/bin/attention-opt" ]] || die "attention-opt was not produced."
echo "attention-opt build OK."

# ── 5. Python venv (test/numerical/requirements.txt) ────────────────────
log "Setting up the Python venv"
python3 -m venv "$REPO_DIR/.venv"
# shellcheck disable=SC1091
source "$REPO_DIR/.venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "$REPO_DIR/test/numerical/requirements.txt"
deactivate
echo "Python venv OK: $REPO_DIR/.venv"

# ── 6. CPU-only sanity check ─────────────────────────────────────────────
if [[ "$SKIP_SANITY" -eq 0 ]]; then
  log "Running the CPU-only regression suite as a build sanity check" \
      "(no GPU involved -- a failure here means the build is broken, not" \
      "that anything GPU-specific is)"
  FC="$LLVM_DIR/build/bin/FileCheck"
  OPT="$REPO_DIR/build/bin/attention-opt"
  for f in "$REPO_DIR"/test/Attention/*.mlir; do
    # dummy.mlir is dead scaffold from the project's first commit -- it
    # references a fictional "attention.foo" op that was never real, and
    # genuinely fails on its own merits (not a script bug). Skip it here
    # rather than touch the actual tracked test suite under time pressure.
    [[ "$(basename "$f")" == "dummy.mlir" ]] && continue
    grep "^// RUN:" "$f" | sed -e 's|^// RUN: ||' \
      -e "s|attention-opt|$OPT|g" -e "s|FileCheck|$FC|g" -e "s|%s|$f|g" \
      | while IFS= read -r cmd; do bash -c "$cmd" >/dev/null; done
  done
  echo "attention-opt FileCheck suite: OK"

  # shellcheck disable=SC1091
  source "$REPO_DIR/.venv/bin/activate"
  ( cd "$REPO_DIR/test/numerical" && python3 validate.py --suite )
  deactivate
  echo "validate.py --suite (CPU): OK"
else
  log "Skipping CPU sanity check (--skip-sanity)"
fi

# ── 7. Next steps (printed, not run) ─────────────────────────────────────
log "Build complete. Next steps (Stage 2 -- these actually cost nothing" \
    "further to run, but are the point of renting this instance, so run" \
    "them deliberately, not automatically from this script):"
cat <<EOF

  cd $REPO_DIR
  source .venv/bin/activate

  # Correctness (Requirements.md 5.3): GPU execution vs. the numpy reference
  python3 test/numerical/validate.py --gpu --suite

  # Wall-clock speedup (Pass 5 Stage A, no tensor cores)
  python3 test/numerical/benchmark.py --gpu --suite

  # Stage B: standalone tensor-core microbenchmark vs. the same shape with
  # no tensor cores (Design.md 7.6) -- exactly each file's own RUN-GPU:
  # comment, using attention-opt (not plain mlir-opt) since that's what
  # those comments themselves specify -- it registers every dialect/pass
  # needed via registerAllDialects/registerAllExtensions.
  build/bin/attention-opt test/Attention/gpu_tensor_core_matmul.mlir \\
    -transform-interpreter -test-transform-dialect-erase-schedule \\
    -gpu-lower-to-nvvm-pipeline="cubin-chip=$CUBIN_CHIP cubin-features=+ptx78 cubin-format=bin" \\
    | $LLVM_DIR/build/bin/mlir-runner \\
      --shared-libs=$LLVM_DIR/build/lib/libmlir_cuda_runtime.so \\
      --shared-libs=$LLVM_DIR/build/lib/libmlir_runner_utils.so \\
      --entry-point-result=void

  build/bin/attention-opt test/Attention/gpu_matmul_no_tensorcore.mlir \\
    -gpu-kernel-outlining \\
    -gpu-lower-to-nvvm-pipeline="cubin-chip=$CUBIN_CHIP cubin-features=+ptx78 cubin-format=bin" \\
    | $LLVM_DIR/build/bin/mlir-runner \\
      --shared-libs=$LLVM_DIR/build/lib/libmlir_cuda_runtime.so \\
      --shared-libs=$LLVM_DIR/build/lib/libmlir_runner_utils.so \\
      --entry-point-result=void

  # Record it (NOTES.md's "Experiment execution workflow" step 3 --
  # provenance to save alongside the numbers above, before disconnecting):
  nvidia-smi
  nvcc --version
  git -C $REPO_DIR rev-parse HEAD

  # Append to results/ablation.csv with provenance built in:
  python3 benchmarks/analyze_ablation.py --collect --hardware "$HARDWARE_LABEL"

EOF
