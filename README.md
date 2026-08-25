# Deconstructing FlashAttention: A Compiler-Centric Analysis of Attention Kernel Optimization 

## Introduction

### The Problem: Attention is Slow

Transformer models power modern AI (GPT, Claude, etc.), and their core operation is **attention**. The standard attention computation follows this formula:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Breaking this down into steps:

1. **MatMul**: Multiply query (Q) by key transpose (K^T)
2. **Scale**: Divide by √d_k (dimension size)
3. **Mask**: Apply causal mask (prevent looking at future tokens)
4. **Softmax**: Normalize to probabilities
5. **MatMul**: Multiply result by values (V)

**The bottleneck:** Running these as separate operations means repeatedly writing intermediate results to slow GPU memory (HBM - High Bandwidth Memory). For a 1024-token sequence, this creates gigabytes of memory traffic.

### FlashAttention's Solution

FlashAttention (Dao et al., 2022-2025) demonstrated dramatic speedups by **fusing** all operations into a single GPU kernel. Instead of:

```
GPU Memory → Load Q,K → Compute QK^T → Write to Memory
           → Load QK^T → Scale → Write to Memory
           → Load scaled → Mask → Write to Memory
           → Load masked → Softmax → Write to Memory
```

FlashAttention does:

```
GPU Memory → Load tile of Q,K → Compute+Scale+Mask+Softmax in fast SRAM → Write final result
```

**Key insight:** Keep intermediate values in fast on-chip memory (SRAM) instead of slow off-chip memory (HBM).

**The catch:** FlashAttention is written in hand-optimized CUDA code - thousands of lines specific to NVIDIA GPUs.

### The Research Question

**Can we express FlashAttention's optimizations as reusable compiler passes instead of hand-written kernels?**

**Why this matters:**

- **Portability:** Compiler passes work across GPU vendors (NVIDIA, AMD, Intel)
- **Maintainability:** High-level passes are easier to understand and modify than CUDA
- **Reusability:** Same optimization strategy can apply to other operations beyond attention
- **Transparency:** Hand-written kernels are opaque; compiler passes are inspectable

**Our goal is NOT to beat FlashAttention's performance.** Instead, we ask:

1. Which FlashAttention optimizations can be expressed as compiler transformations?
2. How much performance can we recover using only compiler infrastructure?
3. What gaps remain, and what would compilers need to close them?

---

## Build

**Prerequisites:** LLVM/MLIR built from source, CMake 3.20+, Ninja.

```bash
git clone https://github.com/luisroayerdi/MLIR-FlashAttention.git
cd MLIR-FlashAttention
mkdir build && cd build
cmake .. -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir -G Ninja
ninja
```

The compiler driver binary is produced at `build/bin/attention-opt`.

---

## Inspecting the IR Pipeline

The fastest way to understand the project is to watch the IR transform at each pass. All commands run from the repo root.

```bash
OPT=build/bin/attention-opt
```

### Stage 0 — Input: unfused attention

The raw 5-op sequence before any compiler transformation:

```bash
cat test/Attention/fusion.mlir
```

You will see `linalg.generic` (QK^T), `linalg.generic` (scale), `linalg.generic` (mask), `linalg.softmax`, and `linalg.matmul` (PV) as separate operations writing through intermediate `memref` buffers.

### Stage 1 — After Pass 1: Fusion

```bash
$OPT test/Attention/fusion.mlir --fusion-pass 2>/dev/null
```

The 5-op sequence collapses into a single `attention.fused` op. No intermediate buffers between the ops; the fused op carries Q, K, V, scale, mask, and output as explicit operands.

### Stage 2 — After Pass 2: Tiling

```bash
$OPT test/Attention/tiling.mlir --tiling-pass="tile-size=32" 2>/dev/null
```

`attention.fused` expands into nested `affine.for` loops implementing the online softmax algorithm. No `attention.fused` remains — output is standard affine + linalg + memref IR.

### Full pipeline (Fusion → Tiling)

```bash
$OPT test/Attention/fusion.mlir --fusion-pass --tiling-pass="tile-size=32" 2>/dev/null
```

### See all IR stages in one run

```bash
$OPT test/Attention/fusion.mlir \
  --fusion-pass \
  --tiling-pass="tile-size=32" \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-all \
  2>&1 | less
```

MLIR prints a snapshot before and after every pass. Use `q` to exit, `/attention` to search.

### Save intermediate IR to files

```bash
$OPT test/Attention/fusion.mlir --fusion-pass 2>/dev/null > /tmp/after_fusion.mlir
$OPT /tmp/after_fusion.mlir --tiling-pass="tile-size=32" 2>/dev/null > /tmp/after_tiling.mlir
```

This lets you inspect or hand-edit the IR between passes.

---

## Running the Tests

```bash
# Pass 1: verify attention.fused is produced, linalg.softmax/matmul are gone
$OPT test/Attention/fusion.mlir --fusion-pass 2>/dev/null | \
  FileCheck test/Attention/fusion.mlir && echo "PASS"

# Pass 2: verify affine.for loops produced, attention.fused is gone
$OPT test/Attention/tiling.mlir --tiling-pass="tile-size=32" 2>/dev/null | \
  FileCheck test/Attention/tiling.mlir && echo "PASS"
```

`FileCheck` is in your LLVM tools directory (e.g. `/opt/homebrew/opt/llvm/bin/FileCheck` on macOS with Homebrew LLVM).

FileCheck only verifies IR *shape* (the right ops appear in the right structure) — it says nothing about whether the numbers produced are correct. That's what the numerical validation harness below is for.

---

## Numerical Validation

`test/numerical/` runs Pass 1 + Pass 2 output through a full lowering-to-LLVM pipeline, JIT-executes it via `mlir-runner`, and compares the result against an independent numpy reference implementation of attention — element-wise, against the tolerances in Requirements.md §5.1 (`max_error < 1e-5`, `mean_error < 1e-6`, `>99.9%` of elements within tolerance).

**Setup (one-time):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r test/numerical/requirements.txt
```

**Run the default suite** (small shapes, with and without a causal mask, single-tile and multi-tile):

```bash
source .venv/bin/activate
cd test/numerical && python3 validate.py --suite
```

**Run a single configuration:**

```bash
python3 test/numerical/validate.py --seq-q 16 --seq-k 16 --head-dim 8 --tile-size 4 --mask
```

`seq-q` and `seq-k` must be divisible by `tile-size` — `TilingPass` does not yet handle remainder tiles (see TRADEOFFS.md). There is currently no batch dimension in `attention.fused`, so each run validates one `[seq, head_dim]` case at a time.

The tool paths (`attention-opt`, `mlir-opt`, `mlir-runner`, runner-utils shared libs) are auto-discovered from `build/CMakeCache.txt` — no configuration needed as long as `build/` has been configured per the Build section above.

---

## CPU Execution Benchmarking

`test/numerical/benchmark.py` implements Requirements.md §5.2 (CPU Validation — the pre-GPU-hardware checkpoint): it times the naive unfused baseline against the Pass 1+2 (fusion+tiling) output, both JIT-executed via `mlir-runner`, and checks the fused version is `>1.2x` faster. Timing is done *inside* the compiled program via `rtclock()`/`printF64()` (bracketing many repeated calls after an untimed warmup loop), so process startup and JIT-compile time don't pollute the measurement. Each config runs several independent trials; results are reported as median ± stdev, with a warning if stdev exceeds 5% of the median.

It also re-runs the numerical correctness check (`validate.run_case`) for the same shape before timing — per §5.2's "if fails: STOP, do not proceed to GPU" rule, a config that isn't numerically correct is reported as failed without being benchmarked.

**Run the default suite** (seq lengths 128–512, head_dim 64, tile 32, with and without a causal mask):

```bash
source .venv/bin/activate
cd test/numerical && python3 benchmark.py --suite
```

**Run a single configuration:**

```bash
python3 test/numerical/benchmark.py --seq-q 512 --seq-k 512 --head-dim 64 --tile-size 32 --trials 5
```

Requirements.md §5.2 also calls for `perf stat -e cycles,instructions,cache-misses` profiling — that's Linux-only and unavailable on macOS, so this harness reports wall-clock speedup only (the actual quantity the acceptance gate checks). See TRADEOFFS.md.

**Current result:** all 4 default-suite configs pass, with measured speedups of 1.36x–1.48x — comfortably clearing the §5.2 `>1.2x` threshold and approaching the §5.4 `>1.5x` Go/No-Go bar for GPU work.

---

## Project Documents

The project is driven by three documents with different roles. **Requirements and Design are the stable foundation** — they define what the project is trying to do and why. TRADEOFFS is a living record of every non-obvious implementation decision made along the way.

### [Requirements.md](Requirements.md) — What and Why

Defines the research question, success criteria, and pass specifications. Reading this first gives the full picture of what each pass is supposed to accomplish and how performance will be measured. Key sections:

- **§1 Problem Statement** — why hand-written CUDA is the wrong approach for research
- **§2 Success Criteria** — minimum viable (Passes 1–2 on CPU), target (GPU + tensor cores), stretch (FA2 optimizations)
- **§3 FlashAttention Techniques** — which FA1/FA2 techniques each pass implements and why
- **§4 Pass Specifications** — per-pass input IR, output IR, algorithm, and test commands
- **§6 Baselines** — PyTorch unfused, torch.compile, FlashAttention-2 (the ablation study targets)
- **§9 Development Workflow** — the propose → confirm → implement protocol

> The IR snippets in Requirements.md are illustrative. Where they conflict with Design.md, Design.md takes precedence.

### [Design.md](Design.md) — How

The technical design approved before implementation began. Contains the exact IR transformation at each stage, the `attention.fused` op definition, per-pass algorithms in pseudocode, and the rationale for every structural decision. This is the reference for what the code is supposed to produce.

Key sections:

- **§1 Architecture Overview** — the full transformation chain from unfused linalg to nvgpu
- **§2 Dialect Extension** — `attention.fused` op definition (TableGen + rationale)
- **§3 Pass 1: Fusion** — pattern matching algorithm and IR examples
- **§4 Pass 2: Tiling** — online softmax algorithm and full post-tiling IR structure
- **§5–7 Passes 3–5** — vectorization, mask specialization, GPU lowering (designed; Passes 3–4 not yet implemented; Pass 5 deferred until GPU hardware)

### [TRADEOFFS.md](TRADEOFFS.md) — Decisions Made During Implementation

Updated continuously as implementation reveals decisions not fully resolved by Design.md. Each entry records what was decided, why, and what it costs. This is the right place to look when the code does something that seems surprising relative to the design.

Current entries cover: why V is in `attention.fused`, why scale is an SSA operand, the memref-based pattern matching strategy, how the scale value is extracted from the linalg.generic body, why the tiling pass fully expands rather than tiling-in-place, and dialect loading requirements.

---

## Implementation Status

| Pass | Flag | Status |
|------|------|--------|
| 1 — Fusion | `--fusion-pass` | ✅ Implemented, FileCheck passing |
| 2 — Tiling | `--tiling-pass` | ✅ Implemented, FileCheck passing |
| 3 — Vectorization | `--vectorization-pass` | Not yet implemented |
| 4 — Mask Specialization | `--mask-specialization-pass` | Not yet implemented |
| 5 — GPU Lowering | `--gpu-lowering-pass` | Deferred (requires GPU hardware) |

Current milestone: CPU Validation checkpoint (Requirements.md §5.2) passing — numerical correctness (§5.1) and CPU execution speedup (§5.2) both hold for Passes 1–2 (`test/numerical/`, 4/4 default-suite benchmark configs, 1.36x–1.48x speedup vs. unfused). Minimum Viable success criteria (§2) are now met. Next step: Pass 3 (Vectorization).
