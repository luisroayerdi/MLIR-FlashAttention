#!/usr/bin/env python3
"""CPU execution benchmark (Requirements.md Section 5.2 - Pre-Hardware Checkpoint).

For a given shape, times the naive unfused baseline (linalg ops only, no
attention-opt passes) against the Pass 1+2 (fusion+tiling) output, both
JIT-executed via mlir-runner. Timing is done with rtclock()/printF64() from
mlir_c_runner_utils *inside* the compiled program, bracketing a loop of N
calls after an untimed warmup loop -- so process-startup and JIT-compile time
are excluded from the measurement (see bench_codegen.py).

Runs several independent trials (fresh subprocess + fresh JIT compile each
time) and reports median/stddev, matching the statistical protocol
Requirements.md Section 5.3 already uses for GPU timing.

Acceptance criteria (Section 5.2):
    - both variants execute without error
    - numerical correctness validated (reuses validate.run_case)
    - speedup vs unfused > 1.2x
"perf stat" (cycles/instructions/cache-misses) from Section 5.2 is Linux-only
and unavailable on macOS; this harness reports wall-clock speedup only (the
actual quantity the acceptance criterion gates on). See TRADEOFFS.md.

Usage:
    python3 benchmark.py --seq-q 256 --seq-k 256 --head-dim 64 --tile-size 32
    python3 benchmark.py --suite
"""

import argparse
import statistics
import sys
from dataclasses import dataclass

import numpy as np

from bench_codegen import emit_baseline_module, emit_fused_input_module
from pipeline import Toolchain, run_baseline_timed, run_fused_timed, run_fused_timed_gpu
from validate import run_case

SPEEDUP_THRESHOLD = 1.2  # Requirements.md 5.2 acceptance criterion
GO_NO_GO_THRESHOLD = 1.5  # Requirements.md 5.4 Go/No-Go "PROCEED if >1.5x speedup vs unfused"
MASK_SPEC_SPEEDUP_THRESHOLD = 1.15  # Requirements.md 4.4 "1.15-1.3x vs generic masking"
VARIANCE_WARN_FRACTION = 0.05  # Requirements.md 5.3 "flag variance >5%"


@dataclass
class BenchResult:
    """Numeric result of one bench_case()/bench_mask_specialization_case()
    call, for callers (e.g. analyze_ablation.py) that need the actual
    numbers rather than just pass/fail."""
    ok: bool
    speedup: float
    a_med: float     # baseline_med (bench_case) or generic_med (mask-spec)
    b_med: float      # fused_med (bench_case) or specialized_med (mask-spec)
    a_stdev: float
    b_stdev: float


def _trial_times(module_fn, run_fn, trials: int, timed_iters: int) -> list[float]:
    """Run `trials` independent subprocess invocations, each with its own
    warmup + timed loop, and return per-call elapsed seconds for each trial."""
    per_call = []
    for _ in range(trials):
        module_text = module_fn()
        total = run_fn(module_text)
        per_call.append(total / timed_iters)
    return per_call


def bench_case(seq_q: int, seq_k: int, head_dim: int, tile_size: int,
                seed: int, use_mask: bool, tools: Toolchain,
                trials: int = 5, warmup_iters: int = 5,
                timed_iters: int = 50, vectorize: bool = False,
                mask_specialize: bool = False, gpu: bool = False,
                threshold: float = SPEEDUP_THRESHOLD) -> bool:
    if seq_q % tile_size or seq_k % tile_size:
        raise ValueError(
            f"seq_q ({seq_q}) and seq_k ({seq_k}) must be divisible by "
            f"tile_size ({tile_size}); TilingPass only supports full tiles."
        )
    if gpu and vectorize:
        raise ValueError(
            "gpu=True does not yet support vectorize=True -- see "
            "pipeline.py's GPU section comment and TRADEOFFS.md."
        )

    tags = []
    if vectorize:
        tags.append("vectorized")
    if mask_specialize:
        tags.append("mask-specialized")
    if gpu:
        tags.append("GPU")
    label = (f"seq_q={seq_q} seq_k={seq_k} head_dim={head_dim} "
             f"tile={tile_size} mask={use_mask}"
             f"{' ' + ','.join(tags) if tags else ''}")

    correct = run_case(seq_q, seq_k, head_dim, tile_size, seed, use_mask,
                        tools, verbose=False, vectorize=vectorize,
                        mask_specialize=mask_specialize, gpu=gpu)
    if not correct:
        print(f"FAIL  {label}: numerical correctness check failed "
              f"(see validate.py) -- skipping timing, per 5.2 'STOP' rule")
        return BenchResult(False, 0.0, 0.0, 0.0, 0.0, 0.0)

    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((seq_q, head_dim), dtype=np.float32)
    K = rng.standard_normal((seq_k, head_dim), dtype=np.float32)
    V = rng.standard_normal((seq_k, head_dim), dtype=np.float32)
    scale = float(1.0 / np.sqrt(head_dim))
    mask = np.triu(np.ones((seq_q, seq_k), dtype=bool), k=1) if use_mask else None

    # The baseline side always runs on CPU, gpu=True or not -- Requirements.md
    # 6.4's ablation table reports every rung's speedup against the same
    # fixed CPU-unfused reference point (Design.md 7.6), not a GPU-executed
    # baseline. Only the "fused" side switches execution target.
    fused_module_fn = lambda: emit_fused_input_module(  # noqa: E731
        Q, K, V, scale, mask, warmup_iters, timed_iters, gpu=gpu)
    if gpu:
        fused_run_fn = lambda m: run_fused_timed_gpu(  # noqa: E731
            m, tile_size, tools, mask_specialize=mask_specialize)
    else:
        fused_run_fn = lambda m: run_fused_timed(  # noqa: E731
            m, tile_size, tools, vectorize=vectorize, mask_specialize=mask_specialize)

    try:
        baseline_times = _trial_times(
            lambda: emit_baseline_module(Q, K, V, scale, mask, warmup_iters, timed_iters),
            lambda m: run_baseline_timed(m, tools),
            trials, timed_iters,
        )
        fused_times = _trial_times(fused_module_fn, fused_run_fn, trials, timed_iters)
    except RuntimeError as e:
        print(f"FAIL  {label}: execution error\n{e}")
        return BenchResult(False, 0.0, 0.0, 0.0, 0.0, 0.0)

    base_med = statistics.median(baseline_times)
    fused_med = statistics.median(fused_times)
    base_stdev = statistics.stdev(baseline_times) if trials > 1 else 0.0
    fused_stdev = statistics.stdev(fused_times) if trials > 1 else 0.0
    speedup = base_med / fused_med

    ok = speedup > threshold

    variance_note = ""
    for name, med, sd in (("baseline", base_med, base_stdev), ("fused", fused_med, fused_stdev)):
        if med > 0 and sd / med > VARIANCE_WARN_FRACTION:
            variance_note += f" [WARN: {name} stdev/median = {sd / med:.1%} > 5%]"

    status = "PASS" if ok else "FAIL"
    print(f"{status}  {label}  "
          f"baseline={base_med * 1e6:.2f}us (+/-{base_stdev * 1e6:.2f})  "
          f"fused={fused_med * 1e6:.2f}us (+/-{fused_stdev * 1e6:.2f})  "
          f"speedup={speedup:.3f}x (need >{threshold}x){variance_note}")
    return BenchResult(ok, speedup, base_med, fused_med, base_stdev, fused_stdev)


def bench_mask_specialization_case(seq_q: int, seq_k: int, head_dim: int,
                                    tile_size: int, seed: int, tools: Toolchain,
                                    trials: int = 5, warmup_iters: int = 5,
                                    timed_iters: int = 50) -> bool:
    """Requirements.md 4.4's own performance target: speedup of Pass 4
    (--mask-specialization-pass) over Pass 1+2's generic per-element masking,
    both already fusion+tiling'd -- NOT the unfused-baseline comparison
    bench_case() does. Always uses a causal mask (Pass 4 is a no-op
    otherwise)."""
    if seq_q % tile_size or seq_k % tile_size:
        raise ValueError(
            f"seq_q ({seq_q}) and seq_k ({seq_k}) must be divisible by "
            f"tile_size ({tile_size}); TilingPass only supports full tiles."
        )

    label = f"seq_q={seq_q} seq_k={seq_k} head_dim={head_dim} tile={tile_size}"

    correct = run_case(seq_q, seq_k, head_dim, tile_size, seed, True, tools,
                        verbose=False, mask_specialize=True)
    if not correct:
        print(f"FAIL  {label}: numerical correctness check failed "
              f"(see validate.py) -- skipping timing, per 5.2 'STOP' rule")
        return BenchResult(False, 0.0, 0.0, 0.0, 0.0, 0.0)

    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((seq_q, head_dim), dtype=np.float32)
    K = rng.standard_normal((seq_k, head_dim), dtype=np.float32)
    V = rng.standard_normal((seq_k, head_dim), dtype=np.float32)
    scale = float(1.0 / np.sqrt(head_dim))
    mask = np.triu(np.ones((seq_q, seq_k), dtype=bool), k=1)

    try:
        generic_times = _trial_times(
            lambda: emit_fused_input_module(Q, K, V, scale, mask, warmup_iters, timed_iters),
            lambda m: run_fused_timed(m, tile_size, tools, mask_specialize=False),
            trials, timed_iters,
        )
        specialized_times = _trial_times(
            lambda: emit_fused_input_module(Q, K, V, scale, mask, warmup_iters, timed_iters),
            lambda m: run_fused_timed(m, tile_size, tools, mask_specialize=True),
            trials, timed_iters,
        )
    except RuntimeError as e:
        print(f"FAIL  {label}: execution error\n{e}")
        return BenchResult(False, 0.0, 0.0, 0.0, 0.0, 0.0)

    generic_med = statistics.median(generic_times)
    spec_med = statistics.median(specialized_times)
    generic_stdev = statistics.stdev(generic_times) if trials > 1 else 0.0
    spec_stdev = statistics.stdev(specialized_times) if trials > 1 else 0.0
    speedup = generic_med / spec_med

    ok = speedup > MASK_SPEC_SPEEDUP_THRESHOLD

    variance_note = ""
    for name, med, sd in (("generic", generic_med, generic_stdev), ("specialized", spec_med, spec_stdev)):
        if med > 0 and sd / med > VARIANCE_WARN_FRACTION:
            variance_note += f" [WARN: {name} stdev/median = {sd / med:.1%} > 5%]"

    status = "PASS" if ok else "FAIL"
    print(f"{status}  {label}  "
          f"generic={generic_med * 1e6:.2f}us (+/-{generic_stdev * 1e6:.2f})  "
          f"specialized={spec_med * 1e6:.2f}us (+/-{spec_stdev * 1e6:.2f})  "
          f"speedup={speedup:.3f}x (need >{MASK_SPEC_SPEEDUP_THRESHOLD}x){variance_note}")
    return BenchResult(ok, speedup, generic_med, spec_med, generic_stdev, spec_stdev)


DEFAULT_SUITE = [
    # (seq_q, seq_k, head_dim, tile_size, use_mask)
    (128, 128, 64, 32, False),
    (256, 256, 64, 32, False),
    (512, 512, 64, 32, False),
    (256, 256, 64, 32, True),
]

# --vectorize --suite uses this smaller suite instead of DEFAULT_SUITE.
#
# Why: VectorizationPass vectorizes each tile op to its own full static
# shape (Design.md 5.2 "no manual VF/remainder loop") -- there is no decomposition
# to hardware-width vectors before JIT. The array-of-vectors LLVM lowering this
# produces for the QK^T/PV reduction ops scales with tile_size^2 * head_dim;
# empirically this is fast up to ~4096 (e.g. tile=16,head_dim=16: 7.0x speedup,
# completes in seconds) but the JIT hangs (multiple minutes, multi-GB RSS, killed
# rather than waited out) once it reaches 8192 (tile=16,head_dim=32) or the
# DEFAULT_SUITE's production scale (tile=32,head_dim=64 => 65536). See
# TRADEOFFS.md "Vectorization pass: full-tile vectorization does not scale to
# CPU JIT compilation at production tile sizes".
VECTORIZED_SUITE = [
    # (seq_q, seq_k, head_dim, tile_size, use_mask)
    (32, 32, 16, 8, False),
    (64, 64, 16, 16, False),
    (32, 32, 16, 8, True),
]

# --full-pipeline --suite: Requirements.md 9.2 Phase 2 "CPU benchmarks" /
# 5.4 Go/No-Go checkpoint, run against the complete Phase 1 pipeline (Pass
# 1+2+3+4 together) rather than any single pass in isolation. Reuses
# VECTORIZED_SUITE's shapes/scale (Pass 3 is in the mix, so the same JIT
# scale ceiling applies -- see that suite's comment); its one masked config
# also exercises Pass 4.
FULL_PIPELINE_SUITE = VECTORIZED_SUITE

# --mask-specialize --suite uses this suite (via bench_mask_specialization_case,
# which always uses a causal mask -- there is nothing to specialize otherwise).
# Larger tile grids than DEFAULT_SUITE's masked config (8x8 tiles at seq=256)
# so more FULL/MASKED tiles get skipped relative to BOUNDARY-only ones,
# giving Pass 4's own effect more room to show up against the unfused-vs-fused
# noise floor.
MASK_SPEC_SUITE = [
    # (seq_q, seq_k, head_dim, tile_size)
    (256, 256, 64, 32),
    (512, 512, 64, 32),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-q", type=int)
    parser.add_argument("--seq-k", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask", action="store_true", help="use a causal mask")
    parser.add_argument("--vectorize", action="store_true",
                         help="also apply --vectorization-pass (Pass 3) to the "
                              "fused variant being timed")
    parser.add_argument("--mask-specialize", action="store_true",
                         help="benchmark Pass 4 (--mask-specialization-pass) "
                              "against generic per-element masking instead of "
                              "the default unfused-vs-fused comparison "
                              "(Requirements.md 4.4's own speedup target); "
                              "always uses a causal mask; --vectorize is "
                              "ignored if both are passed")
    parser.add_argument("--full-pipeline", action="store_true",
                         help="Requirements.md 9.2 Phase 2 / 5.4 Go/No-Go "
                              "checkpoint: benchmark all four passes together "
                              "(fusion+tiling+vectorization+mask-specialization) "
                              "against the unfused baseline, gated at the >1.5x "
                              "Go/No-Go threshold instead of 5.2's >1.2x; "
                              "overrides --vectorize/--mask-specialize")
    parser.add_argument("--gpu", action="store_true",
                         help="Pass 5 Stage A (Requirements.md 5.3): run the "
                              "fused side via --gpu-lowering-pass and GPU "
                              "execution instead of CPU mlir-runner (the "
                              "unfused baseline always stays CPU -- see "
                              "Design.md 7.6). Only against an LLVM build "
                              "with NVPTX + the CUDA runtime (Stage 2 "
                              "hardware only, not this Mac build). Not yet "
                              "compatible with --vectorize or --full-pipeline.")
    parser.add_argument("--trials", type=int, default=5,
                         help="independent subprocess trials per config")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--timed-iters", type=int, default=50)
    parser.add_argument("--suite", action="store_true",
                         help="run the default sweep of configs instead of a "
                              "single case")
    args = parser.parse_args()

    if args.gpu and args.full_pipeline:
        print("error: --gpu does not yet support --full-pipeline (implies "
              "--vectorize, which --gpu does not yet support -- see "
              "pipeline.py's GPU section comment)", file=sys.stderr)
        return 2

    tools = Toolchain.discover()
    try:
        tools.check_gpu() if args.gpu else tools.check()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.full_pipeline:
        if args.suite:
            results = [
                bench_case(sq, sk, hd, ts, args.seed, mask, tools,
                           args.trials, args.warmup_iters, args.timed_iters,
                           vectorize=True, mask_specialize=True,
                           threshold=GO_NO_GO_THRESHOLD)
                for sq, sk, hd, ts, mask in FULL_PIPELINE_SUITE
            ]
            n_pass = sum(r.ok for r in results)
            print(f"\n{n_pass}/{len(results)} configs met the "
                  f">{GO_NO_GO_THRESHOLD}x Go/No-Go threshold")
            return 0 if all(r.ok for r in results) else 1

        seq_q = args.seq_q or 64
        seq_k = args.seq_k or 64
        head_dim = args.head_dim or 16
        tile_size = args.tile_size or 16
        result = bench_case(seq_q, seq_k, head_dim, tile_size, args.seed, args.mask,
                             tools, args.trials, args.warmup_iters, args.timed_iters,
                             vectorize=True, mask_specialize=True,
                             threshold=GO_NO_GO_THRESHOLD)
        return 0 if result.ok else 1

    if args.mask_specialize:
        if args.suite:
            results = [
                bench_mask_specialization_case(sq, sk, hd, ts, args.seed, tools,
                                                args.trials, args.warmup_iters,
                                                args.timed_iters)
                for sq, sk, hd, ts in MASK_SPEC_SUITE
            ]
            n_pass = sum(r.ok for r in results)
            print(f"\n{n_pass}/{len(results)} configs met the "
                  f">{MASK_SPEC_SPEEDUP_THRESHOLD}x mask-specialization threshold")
            return 0 if all(r.ok for r in results) else 1

        seq_q = args.seq_q or 256
        seq_k = args.seq_k or 256
        head_dim = args.head_dim or 64
        tile_size = args.tile_size or 32
        result = bench_mask_specialization_case(seq_q, seq_k, head_dim, tile_size,
                                                  args.seed, tools, args.trials,
                                                  args.warmup_iters, args.timed_iters)
        return 0 if result.ok else 1

    if args.suite:
        suite = VECTORIZED_SUITE if args.vectorize else DEFAULT_SUITE
        results = [
            bench_case(sq, sk, hd, ts, args.seed, mask, tools,
                       args.trials, args.warmup_iters, args.timed_iters,
                       vectorize=args.vectorize, gpu=args.gpu)
            for sq, sk, hd, ts, mask in suite
        ]
        n_pass = sum(r.ok for r in results)
        print(f"\n{n_pass}/{len(results)} configs met the >{SPEEDUP_THRESHOLD}x "
              f"CPU validation threshold")
        return 0 if all(r.ok for r in results) else 1

    seq_q = args.seq_q or 256
    seq_k = args.seq_k or 256
    head_dim = args.head_dim or 64
    tile_size = args.tile_size or 32
    result = bench_case(seq_q, seq_k, head_dim, tile_size, args.seed, args.mask,
                         tools, args.trials, args.warmup_iters, args.timed_iters,
                         vectorize=args.vectorize, gpu=args.gpu)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
