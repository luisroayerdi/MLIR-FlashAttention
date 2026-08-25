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

import numpy as np

from bench_codegen import emit_baseline_module, emit_fused_input_module
from pipeline import Toolchain, run_baseline_timed, run_fused_timed
from validate import run_case

SPEEDUP_THRESHOLD = 1.2  # Requirements.md 5.2 acceptance criterion
VARIANCE_WARN_FRACTION = 0.05  # Requirements.md 5.3 "flag variance >5%"


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
                timed_iters: int = 50) -> bool:
    if seq_q % tile_size or seq_k % tile_size:
        raise ValueError(
            f"seq_q ({seq_q}) and seq_k ({seq_k}) must be divisible by "
            f"tile_size ({tile_size}); TilingPass only supports full tiles."
        )

    label = (f"seq_q={seq_q} seq_k={seq_k} head_dim={head_dim} "
             f"tile={tile_size} mask={use_mask}")

    correct = run_case(seq_q, seq_k, head_dim, tile_size, seed, use_mask,
                        tools, verbose=False)
    if not correct:
        print(f"FAIL  {label}: numerical correctness check failed "
              f"(see validate.py) -- skipping timing, per 5.2 'STOP' rule")
        return False

    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((seq_q, head_dim), dtype=np.float32)
    K = rng.standard_normal((seq_k, head_dim), dtype=np.float32)
    V = rng.standard_normal((seq_k, head_dim), dtype=np.float32)
    scale = float(1.0 / np.sqrt(head_dim))
    mask = np.triu(np.ones((seq_q, seq_k), dtype=bool), k=1) if use_mask else None

    try:
        baseline_times = _trial_times(
            lambda: emit_baseline_module(Q, K, V, scale, mask, warmup_iters, timed_iters),
            lambda m: run_baseline_timed(m, tools),
            trials, timed_iters,
        )
        fused_times = _trial_times(
            lambda: emit_fused_input_module(Q, K, V, scale, mask, warmup_iters, timed_iters),
            lambda m: run_fused_timed(m, tile_size, tools),
            trials, timed_iters,
        )
    except RuntimeError as e:
        print(f"FAIL  {label}: execution error\n{e}")
        return False

    base_med = statistics.median(baseline_times)
    fused_med = statistics.median(fused_times)
    base_stdev = statistics.stdev(baseline_times) if trials > 1 else 0.0
    fused_stdev = statistics.stdev(fused_times) if trials > 1 else 0.0
    speedup = base_med / fused_med

    ok = speedup > SPEEDUP_THRESHOLD

    variance_note = ""
    for name, med, sd in (("baseline", base_med, base_stdev), ("fused", fused_med, fused_stdev)):
        if med > 0 and sd / med > VARIANCE_WARN_FRACTION:
            variance_note += f" [WARN: {name} stdev/median = {sd / med:.1%} > 5%]"

    status = "PASS" if ok else "FAIL"
    print(f"{status}  {label}  "
          f"baseline={base_med * 1e6:.2f}us (+/-{base_stdev * 1e6:.2f})  "
          f"fused={fused_med * 1e6:.2f}us (+/-{fused_stdev * 1e6:.2f})  "
          f"speedup={speedup:.3f}x (need >{SPEEDUP_THRESHOLD}x){variance_note}")
    return ok


DEFAULT_SUITE = [
    # (seq_q, seq_k, head_dim, tile_size, use_mask)
    (128, 128, 64, 32, False),
    (256, 256, 64, 32, False),
    (512, 512, 64, 32, False),
    (256, 256, 64, 32, True),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-q", type=int)
    parser.add_argument("--seq-k", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask", action="store_true", help="use a causal mask")
    parser.add_argument("--trials", type=int, default=5,
                         help="independent subprocess trials per config")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--timed-iters", type=int, default=50)
    parser.add_argument("--suite", action="store_true",
                         help="run the default sweep of configs instead of a "
                              "single case")
    args = parser.parse_args()

    tools = Toolchain.discover()
    try:
        tools.check()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.suite:
        results = [
            bench_case(sq, sk, hd, ts, args.seed, mask, tools,
                       args.trials, args.warmup_iters, args.timed_iters)
            for sq, sk, hd, ts, mask in DEFAULT_SUITE
        ]
        n_pass = sum(results)
        print(f"\n{n_pass}/{len(results)} configs met the >{SPEEDUP_THRESHOLD}x "
              f"CPU validation threshold")
        return 0 if all(results) else 1

    seq_q = args.seq_q or 256
    seq_k = args.seq_k or 256
    head_dim = args.head_dim or 64
    tile_size = args.tile_size or 32
    ok = bench_case(seq_q, seq_k, head_dim, tile_size, args.seed, args.mask,
                     tools, args.trials, args.warmup_iters, args.timed_iters)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
