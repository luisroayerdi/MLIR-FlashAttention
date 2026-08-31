#!/usr/bin/env python3
"""Ablation study (Requirements.md Section 6.4): progressively enable
passes and measure the speedup delta each contributes, at one fixed shape,
against the unfused baseline.

Reuses benchmark.py's bench_case() for the actual timing (Section 5.2/5.3
protocol -- fresh subprocess + fresh JIT per trial, median/stdev over
--trials runs) rather than re-implementing timing logic. The default shape
(seq=32x32, head_dim=16, tile=8, causal mask) is VECTORIZED_SUITE's existing
masked config from benchmark.py -- already validated, and small enough that
Vectorization's JIT-compile-time ceiling (Design.md Section 5.3) doesn't
apply to any ladder step.

Two independently runnable modes; default (neither flag) does both:
    --collect   run the ladder, append results to results/ablation.csv
                with provenance (commit hash, timestamp, hardware label) --
                see NOTES.md "how do I prove these came from real hardware"
    --plot      read that CSV, print a summary table, write a matplotlib
                bar chart to results/ablation.png

Usage:
    python3 benchmarks/analyze_ablation.py
    python3 benchmarks/analyze_ablation.py --collect --hardware "RTX 4090 (RunPod)"
    python3 benchmarks/analyze_ablation.py --plot
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "test" / "numerical"))

from benchmark import bench_case  # noqa: E402
from pipeline import Toolchain  # noqa: E402

RESULTS_CSV = REPO_ROOT / "results" / "ablation.csv"
RESULTS_PNG = REPO_ROOT / "results" / "ablation.png"

CSV_FIELDS = [
    "timestamp", "commit", "hardware", "shape", "step",
    "speedup", "baseline_us", "fused_us",
    "baseline_stdev_pct", "fused_stdev_pct",
]

# Cumulative ladder, matching pass implementation order (Requirements.md
# Section 6.4): each step is a strict superset of the previous step's
# passes, all measured against the same unfused baseline via bench_case().
LADDER = [
    ("Fusion+Tiling", dict(vectorize=False, mask_specialize=False)),
    ("+ Vectorization", dict(vectorize=True, mask_specialize=False)),
    ("+ Mask Specialization", dict(vectorize=True, mask_specialize=True)),
]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def collect(seq_q: int, seq_k: int, head_dim: int, tile_size: int,
            hardware: str, trials: int, warmup_iters: int,
            timed_iters: int) -> None:
    tools = Toolchain.discover()
    tools.check()

    shape = (f"seq={seq_q}x{seq_k} head_dim={head_dim} tile={tile_size} "
             f"mask=True")
    timestamp = datetime.now(timezone.utc).isoformat()
    commit = _git_commit()

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for step_name, flags in LADDER:
            result = bench_case(
                seq_q, seq_k, head_dim, tile_size, seed=0, use_mask=True,
                tools=tools, trials=trials, warmup_iters=warmup_iters,
                timed_iters=timed_iters, **flags,
            )
            if result.speedup == 0.0 and not result.ok:
                print(f"error: {step_name} failed correctness/execution -- "
                      f"not writing a row for it", file=sys.stderr)
                continue

            writer.writerow({
                "timestamp": timestamp,
                "commit": commit,
                "hardware": hardware,
                "shape": shape,
                "step": step_name,
                "speedup": f"{result.speedup:.4f}",
                "baseline_us": f"{result.a_med * 1e6:.4f}",
                "fused_us": f"{result.b_med * 1e6:.4f}",
                "baseline_stdev_pct": f"{(result.a_stdev / result.a_med * 100) if result.a_med else 0:.2f}",
                "fused_stdev_pct": f"{(result.b_stdev / result.b_med * 100) if result.b_med else 0:.2f}",
            })

    print(f"Wrote {len(LADDER)} rows to {RESULTS_CSV}")


def _latest_rows() -> dict:
    """Read the CSV and keep only the most recent row per (hardware, step)
    -- later appends (re-runs) supersede earlier ones for that combo."""
    if not RESULTS_CSV.exists():
        return {}
    latest = {}
    with RESULTS_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["hardware"], row["step"])
            latest[key] = row  # later rows in file order win
    return latest


def plot() -> None:
    latest = _latest_rows()
    if not latest:
        print(f"error: no data in {RESULTS_CSV} -- run --collect first",
              file=sys.stderr)
        sys.exit(1)

    step_order = [name for name, _ in LADDER]
    hardware_list = sorted({hw for hw, _ in latest})

    print(f"{'hardware':<22} {'step':<24} {'speedup':>9}  "
          f"{'baseline (us)':>15}  {'fused (us)':>13}  shape / commit")
    for hw in hardware_list:
        for step in step_order:
            row = latest.get((hw, step))
            if row is None:
                continue
            print(f"{hw:<22} {step:<24} {row['speedup']:>8}x  "
                  f"{row['baseline_us']:>15}  {row['fused_us']:>13}  "
                  f"{row['shape']} @ {row['commit']}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    n_hw = len(hardware_list)
    width = 0.8 / max(n_hw, 1)
    x = range(len(step_order))

    for i, hw in enumerate(hardware_list):
        speedups = [float(latest[(hw, s)]["speedup"]) if (hw, s) in latest else 0.0
                    for s in step_order]
        offsets = [xi + (i - (n_hw - 1) / 2) * width for xi in x]
        bars = ax.bar(offsets, speedups, width=width, label=hw)
        ax.bar_label(bars, fmt="%.2fx", padding=2, fontsize=8)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
               label="unfused baseline (1.0x)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(step_order, rotation=15, ha="right")
    ax.set_ylabel("speedup vs. unfused baseline")
    ax.set_title("Ablation: cumulative speedup vs. unfused baseline")
    ax.legend(fontsize=8)
    fig.tight_layout()

    RESULTS_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_PNG, dpi=150)
    print(f"Wrote {RESULTS_PNG}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--seq-q", type=int, default=32)
    parser.add_argument("--seq-k", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--hardware", default="CPU (local)",
                         help="label for the results row, e.g. "
                              "'RTX 4090 (RunPod)' when run on a GPU instance")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--timed-iters", type=int, default=50)
    args = parser.parse_args()

    do_collect = args.collect or not (args.collect or args.plot)
    do_plot = args.plot or not (args.collect or args.plot)

    if do_collect:
        collect(args.seq_q, args.seq_k, args.head_dim, args.tile_size,
                args.hardware, args.trials, args.warmup_iters,
                args.timed_iters)
    if do_plot:
        plot()
    return 0


if __name__ == "__main__":
    sys.exit(main())
