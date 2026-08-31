# MLIR-FlashAttention

## Introduction

### The Problem

Attention is the core, and most expensive, operation in transformer models:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Computed naively, this is five separate operations:

1. **MatMul**: Q · K^T
2. **Scale**: divide by √d_k
3. **Mask**: apply the causal mask
4. **Softmax**: normalize to probabilities
5. **MatMul**: multiply by V

Each operation writes its result to GPU HBM (High Bandwidth Memory) and reads it back for the next step. That round trip, not the matmuls themselves, is the bottleneck.

### Fusing the Bottleneck Away

FlashAttention (Dao et al., 2022-2025) removes this traffic by fusing all five operations into one kernel and keeping intermediate values in fast on-chip SRAM instead of writing them back to HBM:

```
GPU Memory → Load Q,K → Compute QK^T → Write to Memory
           → Load QK^T → Scale → Write to Memory
           → Load scaled → Mask → Write to Memory
           → Load masked → Softmax → Write to Memory
```

becomes

```
GPU Memory → Load tile of Q,K → Compute+Scale+Mask+Softmax in SRAM → Write final result
```

The catch: this speedup comes from thousands of lines of hand optimized CUDA, specific to one hardware vendor and opaque to anyone trying to modify it.

### Our Approach

Applying FlashAttention's memory-aware optimizations to new hardware requires compiler infrastructure that doesn't exist in general purpose compilers. This project asks: **can FlashAttention's optimizations be expressed as reusable MLIR compiler passes instead of hand-written kernels, and how much performance can we recover that way?**

This matters because compiler passes are portable across GPU vendors, inspectable where hand-written kernels are opaque, and reusable beyond attention itself. The goal is not to beat FlashAttention's hand-tuned performance. It's to answer three narrower questions: which of FlashAttention's optimizations can be expressed as compiler transformations, how much performance compiler infrastructure alone can recover, and what gaps remain.

This project answers it with a custom MLIR dialect and a pipeline of compiler passes, one per FlashAttention optimization. A fusion pass collapses attention's five-operation sequence into a single fused IR op. A tiling pass expands that op into memory-aware loops implementing online softmax, the transformation that removes the HBM round trips described above. Fusion reduces the five-op IR sequence to one op, and tiling eliminates four intermediate buffers from GPU memory, roughly 16MB of HBM traffic removed per attention layer at a 1024-token sequence, from the compiler transformation alone. A vectorization pass and a causal mask specialization pass build on top. All four passes are implemented and verified, on CPU, against automated IR tests and numerical correctness. Details below.

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

## IR Pipeline

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

### Stage 3 — After Pass 3: Vectorization

```bash
$OPT test/Attention/vectorization.mlir --tiling-pass="tile-size=32" --vectorization-pass 2>/dev/null
```

Every `linalg.generic`/`linalg.fill`/`memref.copy` op inside the tile body becomes a `vector.transfer_read` / vector-dialect-arithmetic / `vector.transfer_write` sequence over the op's own full tile shape (e.g. `vector<32x32xf32>`). No `linalg.generic`, `linalg.fill`, or `memref.copy` remains.

### Stage 4 — After Pass 4: Mask Specialization

Only has an effect on masked attention (nothing to specialize otherwise), so this uses the masked function in `mask_specialization.mlir` rather than `fusion.mlir`:

```bash
$OPT test/Attention/mask_specialization.mlir --tiling-pass="tile-size=32" --mask-specialization-pass 2>/dev/null
```

The K/V tile loop's per-element `arith.select` masking gets wrapped in a two-level `affine.if` dispatching on tile position relative to the causal diagonal: fully-masked tiles are skipped entirely, fully-unmasked tiles run the same computation without the mask check, and only tiles that straddle the diagonal keep the original per-element `arith.select`.

### Full pipeline (Fusion → Tiling → Vectorization)

```bash
$OPT test/Attention/fusion.mlir --fusion-pass --tiling-pass="tile-size=32" --vectorization-pass 2>/dev/null
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

# Pass 3: verify vector.transfer_read/write produced, no linalg.generic/fill/memref.copy remain
$OPT test/Attention/vectorization.mlir --tiling-pass="tile-size=32" --vectorization-pass 2>/dev/null | \
  FileCheck test/Attention/vectorization.mlir && echo "PASS"

# Pass 4: verify the masked K-loop body is wrapped in a two-level affine.if
$OPT test/Attention/mask_specialization.mlir --tiling-pass="tile-size=32" --mask-specialization-pass 2>/dev/null | \
  FileCheck test/Attention/mask_specialization.mlir && echo "PASS"

# Integration: all four passes together, starting from raw unfused input
$OPT test/Attention/integration.mlir --fusion-pass --tiling-pass="tile-size=32" \
  --vectorization-pass --mask-specialization-pass 2>/dev/null | \
  FileCheck test/Attention/integration.mlir && echo "PASS"
```

`FileCheck` is in your LLVM tools directory (e.g. `/opt/homebrew/opt/llvm/bin/FileCheck` on macOS with Homebrew LLVM).

FileCheck only verifies IR *shape* (the right ops appear in the right structure) — it says nothing about whether the numbers produced are correct. That's what the numerical validation harness below is for.

---

## Numerical Validation

`test/numerical/` runs Pass 1 + Pass 2 output through a full lowering-to-LLVM pipeline, JIT-executes it via `mlir-runner`, and compares the result against an independent numpy reference implementation of attention — element-wise, against the tolerances in Requirements.md §5.1 (`max_error < 1e-5`, `mean_error < 1e-6`, `>99.9%` of elements within tolerance). Add `--vectorize` to also run Pass 3 (`--vectorization-pass`) and `--mask-specialize` to also run Pass 4 (`--mask-specialization-pass`, only affects masked configs) — both validate their output against the same reference and can be combined.

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
python3 validate.py --suite --vectorize   # same suite, also through Pass 3
python3 validate.py --suite --mask-specialize   # same suite, also through Pass 4
python3 validate.py --suite --vectorize --mask-specialize   # all four passes together
```

**Run a single configuration:**

```bash
python3 test/numerical/validate.py --seq-q 16 --seq-k 16 --head-dim 8 --tile-size 4 --mask
python3 test/numerical/validate.py --seq-q 16 --seq-k 16 --head-dim 8 --tile-size 4 --mask --vectorize
python3 test/numerical/validate.py --seq-q 16 --seq-k 16 --head-dim 8 --tile-size 4 --mask --mask-specialize
```

`seq-q` and `seq-k` must be divisible by `tile-size` — `TilingPass` does not yet handle remainder tiles (see Design.md §4.6). There is currently no batch dimension in `attention.fused`, so each run validates one `[seq, head_dim]` case at a time.

The tool paths (`attention-opt`, `mlir-opt`, `mlir-runner`, runner-utils shared libs) are auto-discovered from `build/CMakeCache.txt` — no configuration needed as long as `build/` has been configured per the Build section above.

---

## CPU Execution Benchmarking

`test/numerical/benchmark.py` implements Requirements.md §5.2 (CPU Validation — the pre-GPU-hardware checkpoint): it times the naive unfused baseline against the Pass 1+2 (fusion+tiling) output, both JIT-executed via `mlir-runner`, and checks the fused version is `>1.2x` faster. Timing is done *inside* the compiled program via `rtclock()`/`printF64()` (bracketing many repeated calls after an untimed warmup loop), so process startup and JIT-compile time don't pollute the measurement. Each config runs several independent trials; results are reported as median ± stdev, with a warning if stdev exceeds 5% of the median. Add `--vectorize` to time the Pass 1+2+3 (fusion+tiling+vectorization) output instead of Pass 1+2 alone.

Pass 4's own Requirements.md §4.4 performance target ("1.15-1.3x vs generic masking") is a *different* comparison — Pass 1+2 generic per-element masking vs. Pass 1+2+4 specialized masking, not vs. the unfused baseline. `--mask-specialize` switches `benchmark.py` to that comparison entirely (always uses a causal mask; `--vectorize` is ignored if both are passed).

`--full-pipeline` is a third mode (overrides both of the above): benchmarks all four passes together against the unfused baseline, gated at Requirements.md §5.4's own Go/No-Go `>1.5x` threshold instead of §5.2's `>1.2x` — this is the literal Phase 2 (§9.2) "CHECKPOINT: CPU validation must pass" gate for the complete Phase 1 pipeline.

It also re-runs the numerical correctness check (`validate.run_case`) for the same shape before timing — per §5.2's "if fails: STOP, do not proceed to GPU" rule, a config that isn't numerically correct is reported as failed without being benchmarked.

**Run the default suite** (seq lengths 128–512, head_dim 64, tile 32, with and without a causal mask):

```bash
source .venv/bin/activate
cd test/numerical && python3 benchmark.py --suite
python3 benchmark.py --suite --vectorize        # uses a smaller suite -- see caveat below
python3 benchmark.py --suite --mask-specialize  # generic-vs-specialized masking comparison
python3 benchmark.py --suite --full-pipeline    # all four passes vs. unfused, >1.5x Go/No-Go gate
```

**Run a single configuration:**

```bash
python3 test/numerical/benchmark.py --seq-q 512 --seq-k 512 --head-dim 64 --tile-size 32 --trials 5
python3 test/numerical/benchmark.py --seq-q 64 --seq-k 64 --head-dim 16 --tile-size 16 --trials 5 --vectorize
python3 test/numerical/benchmark.py --seq-q 512 --seq-k 512 --head-dim 64 --tile-size 32 --trials 5 --mask-specialize
python3 test/numerical/benchmark.py --seq-q 64 --seq-k 64 --head-dim 16 --tile-size 16 --trials 5 --full-pipeline
```

**`--vectorize` / `--full-pipeline` scale caveat:** both use a separate, smaller `VECTORIZED_SUITE` (`tile-size` 8–16, `head-dim` 16), not `DEFAULT_SUITE`. Full-tile vectorization (Design.md §5.2) has no hardware-width chunking, so JIT compile time blows up once `tile_size² × head_dim` exceeds ~4,096 — this includes `DEFAULT_SUITE`'s own `tile=32`/`head_dim=64` production-scale config. Don't pass `--tile-size 32 --head-dim 64` with either flag for a single-case run either — it will hang (multi-minute, multi-GB RSS) rather than error (see Design.md §5.3).

Requirements.md §5.2 also calls for `perf stat -e cycles,instructions,cache-misses` profiling — that's Linux-only and unavailable on macOS, so this harness reports wall-clock speedup only (the actual quantity the acceptance gate checks).

**Current result:** all 4 default-suite configs pass (Pass 1–2, no vectorization), with measured speedups of 1.36x–1.49x — comfortably clearing the §5.2 `>1.2x` threshold. With `--vectorize` (Pass 1–2–3, small scale — see caveat above), all 3 `VECTORIZED_SUITE` configs pass at 4.8x–6.5x speedup, clearing Pass 3's own §4.3 "1.5-2x vs scalar" target. With `--mask-specialize` (Pass 1–2 vs. Pass 1–2–4, `tile=32`/`head_dim=64`, no scale caveat here since Pass 4 doesn't vectorize anything), both `MASK_SPEC_SUITE` configs pass at 1.77x–1.87x speedup vs. generic masking — comfortably clearing Pass 4's own §4.4 "1.15-1.3x" target. With `--full-pipeline` (all four passes, small scale), all 3 `VECTORIZED_SUITE` configs pass at 5.0x–7.6x speedup — clearing the §5.4 `>1.5x` Go/No-Go bar for proceeding to GPU work.

---

## Project Documents

The project is driven by two documents: Requirements defines what the project is trying to do and why, Design defines how it's built and stays corrected against what actually shipped.

### [Requirements.md](docs/Requirements.md) — What and Why

Defines the research question, success criteria, and pass specifications. Reading this first gives the full picture of what each pass is supposed to accomplish and how performance will be measured. Key sections:

- **§1 Problem Statement** — why hand-written CUDA is the wrong approach for research
- **§2 Success Criteria** — minimum viable (Passes 1–2 on CPU), target (GPU + tensor cores), stretch (FA2 optimizations)
- **§3 FlashAttention Techniques** — which FA1/FA2 techniques each pass implements and why
- **§4 Pass Specifications** — per-pass input IR, output IR, algorithm, and test commands
- **§6 Baselines** — PyTorch unfused, torch.compile, FlashAttention-2 (the ablation study targets)
- **§8 Deliverables** — what the research report/analysis is expected to produce

> The IR snippets in Requirements.md are illustrative. Where they conflict with Design.md, Design.md takes precedence.

### [Design.md](docs/Design.md) — How

The technical design approved before implementation began. Contains the exact IR transformation at each stage, the `attention.fused` op definition, per-pass algorithms in pseudocode, and the rationale for every structural decision. This is the reference for what the code is supposed to produce.

Key sections:

- **§1 Architecture Overview** — the full transformation chain from unfused linalg to nvgpu
- **§2 Dialect Extension** — `attention.fused` op definition (TableGen + rationale)
- **§3 Pass 1: Fusion** — pattern matching algorithm and IR examples
- **§4 Pass 2: Tiling** — online softmax algorithm and full post-tiling IR structure
- **§5–7 Passes 3–5** — vectorization, mask specialization, GPU lowering (Passes 3–4 implemented; Pass 5 deferred until GPU hardware)

---

## Implementation Status

| Pass | Flag | Status |
|------|------|--------|
| 1 — Fusion | `--fusion-pass` | ✅ Implemented, FileCheck passing |
| 2 — Tiling | `--tiling-pass` | ✅ Implemented, FileCheck passing |
| 3 — Vectorization | `--vectorization-pass` | ✅ Implemented, FileCheck + numerically validated + CPU-speedup validated (small scale — see below) |
| 4 — Mask Specialization | `--mask-specialization-pass` | ✅ Implemented, FileCheck + numerically validated + CPU-speedup validated |
| 5 — GPU Lowering | `--gpu-lowering-pass` | Deferred (requires GPU hardware) |
