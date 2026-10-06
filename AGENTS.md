# AGENTS.md

Guidance for AI agents working in this repository. Humans: see
`CONTRIBUTING.md` — the process is the same for everyone.

Never push to `main` or merge a PR.
Always build and run the tests below before opening a PR, unless it is
docs-only (changes only `*.md` files).
Never change project scope directly: propose it in a GitHub issue.

## Project

MLIR-FlashAttention establishes what an MLIR compiler needs to know about
attention to generate FlashAttention-quality kernels. It builds
FlashAttention's optimizations (fusion, online-softmax tiling,
vectorization, mask specialization, GPU mapping) as separate passes on
upstream dialects, drawing on IREE's attention operator, Triton's tile-level
code generation, and the FlashAttention kernels, and measures what each
transformation is worth, what information it needs from the
representation, and how the result compares with existing compilers and
kernels. Findings are reported to the upstream MLIR community.

## Read first

1. `SPECS.md` — research questions, scope, work items. The source of truth.
2. `CONTRIBUTING.md` — workflow, PR format, review checklist, coding standards.
3. `docs/LITERATURE.md` — what already exists. Check it before building anything.
4. `docs/DECISIONS.md` — why things are the way they are.

## Build

Requires an LLVM/MLIR build from source (`$LLVM_BUILD` below).

```bash
cmake -S . -B build -G Ninja -DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir
ninja -C build attention-opt
```

If CMake reports a wrong source directory, `build/` is stale: delete it and
reconfigure.

## Test

```bash
# IR structure (one per pass; see README for the full list)
build/bin/attention-opt test/Attention/fusion.mlir --fusion-pass \
  | $LLVM_BUILD/bin/FileCheck test/Attention/fusion.mlir

# Numerical correctness against the NumPy reference
source .venv/bin/activate
python3 test/numerical/validate.py --suite
```

Tool paths for `test/numerical/` are discovered from `build/CMakeCache.txt`.

## Layout

- `include/Attention/`, `lib/Attention/` — dialect and passes
- `attention-opt/` — driver
- `test/Attention/` — FileCheck tests; `test/numerical/` — correctness and timing harness
- `benchmarks/` — analysis scripts; `scripts/` — machine setup

## Rules

- Work only on GitHub issues labeled `ready`. One issue per PR.
- Propose new work, spec changes, or problems found in `main` as a GitHub
  issue. The maintainer decides what becomes `ready` and what enters
  `SPECS.md`.
- In a PR, the only `SPECS.md` edit allowed is moving a feature you
  implemented from `planned` to `implemented`.
- Before adding a capability, find the closest existing system in
  `docs/LITERATURE.md`. Prefer upstream MLIR or an existing tool over new code.
- Write the smallest change that does what the issue asks. No speculative
  options, abstractions, or files the item does not require.
- Do not add dialect ops or reimplement a baseline unless an item asks for it.
- Every pass change ships with a FileCheck test; every change that affects
  numerics is checked with `validate.py`.
- Numbers in docs come from scripts, never typed by hand.
- Record non-obvious design decisions in `docs/DECISIONS.md`.
- Do not claim novelty. Cite `docs/LITERATURE.md` entries instead.

## Roles

`.claude/agents/implementer.md` and `.claude/agents/reviewer.md` define the
two agent roles. Both follow `CONTRIBUTING.md`.
