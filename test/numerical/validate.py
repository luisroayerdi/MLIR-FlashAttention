#!/usr/bin/env python3
"""Numerical validation harness (Requirements.md Section 5.1).

Generates random Q/K/V (and optionally a mask), runs them through the real
MLIR pipeline (attention-opt --fusion-pass --tiling-pass, lowered to LLVM and
JIT-executed via mlir-runner), and compares the result against a plain numpy
reference implementation of attention.

    max_error  < 1e-5
    mean_error < 1e-6
    fraction of elements within 1e-5 > 0.999

Usage:
    python3 validate.py --seq-q 8 --seq-k 8 --head-dim 4 --tile-size 4
    python3 validate.py --suite            # run the default config sweep
"""

import argparse
import sys

import numpy as np

from codegen import emit_module
from pipeline import Toolchain, run_module
from reference import attention_reference

MAX_ERROR_TOL = 1e-5
MEAN_ERROR_TOL = 1e-6
WITHIN_TOL_FRACTION = 0.999


def run_case(seq_q: int, seq_k: int, head_dim: int, tile_size: int,
             seed: int, use_mask: bool, tools: Toolchain, verbose: bool = True) -> bool:
    if seq_q % tile_size or seq_k % tile_size:
        raise ValueError(
            f"seq_q ({seq_q}) and seq_k ({seq_k}) must be divisible by "
            f"tile_size ({tile_size}); TilingPass only supports full tiles "
            f"(see Design.md 4.6 Known Limitations)."
        )

    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((seq_q, head_dim), dtype=np.float32)
    K = rng.standard_normal((seq_k, head_dim), dtype=np.float32)
    V = rng.standard_normal((seq_k, head_dim), dtype=np.float32)
    scale = float(1.0 / np.sqrt(head_dim))

    mask = None
    if use_mask:
        # Causal mask: query i cannot attend to key j > i.
        mask = np.triu(np.ones((seq_q, seq_k), dtype=bool), k=1)

    expected = attention_reference(Q, K, V, scale, mask)

    module_text = emit_module(Q, K, V, scale, mask)
    actual = np.array(run_module(module_text, tile_size, tools), dtype=np.float32)

    if actual.shape != expected.shape:
        print(f"FAIL  seq_q={seq_q} seq_k={seq_k} head_dim={head_dim} "
              f"tile={tile_size} mask={use_mask}: shape mismatch "
              f"{actual.shape} vs {expected.shape}")
        return False

    abs_err = np.abs(actual - expected)
    max_error = float(np.max(abs_err))
    mean_error = float(np.mean(abs_err))
    within_tol = float(np.sum(abs_err < MAX_ERROR_TOL) / abs_err.size)

    ok = (max_error < MAX_ERROR_TOL and mean_error < MEAN_ERROR_TOL
          and within_tol > WITHIN_TOL_FRACTION)

    status = "PASS" if ok else "FAIL"
    if verbose or not ok:
        print(f"{status}  seq_q={seq_q} seq_k={seq_k} head_dim={head_dim} "
              f"tile={tile_size} mask={use_mask} seed={seed}  "
              f"max_err={max_error:.3e} mean_err={mean_error:.3e} "
              f"within_tol={within_tol:.4f}")
    return ok


DEFAULT_SUITE = [
    # (seq_q, seq_k, head_dim, tile_size, use_mask)
    (8, 8, 4, 4, False),    # multiple Q-tiles x multiple K-tiles, no mask
    (8, 8, 4, 8, False),    # single tile (degenerate, sanity check)
    (16, 16, 8, 4, False),  # more tiles per dimension
    (8, 8, 4, 4, True),     # causal mask, multiple tiles
    (4, 4, 4, 4, True),     # causal mask, single tile
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-q", type=int)
    parser.add_argument("--seq-k", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask", action="store_true", help="use a causal mask")
    parser.add_argument("--suite", action="store_true",
                         help="run the default sweep of small configs instead "
                              "of a single case")
    args = parser.parse_args()

    tools = Toolchain.discover()
    try:
        tools.check()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.suite:
        results = [
            run_case(sq, sk, hd, ts, args.seed, mask, tools)
            for sq, sk, hd, ts, mask in DEFAULT_SUITE
        ]
        n_pass = sum(results)
        print(f"\n{n_pass}/{len(results)} configs passed")
        return 0 if all(results) else 1

    seq_q = args.seq_q or 8
    seq_k = args.seq_k or 8
    head_dim = args.head_dim or 4
    tile_size = args.tile_size or 4
    ok = run_case(seq_q, seq_k, head_dim, tile_size, args.seed, args.mask, tools)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
