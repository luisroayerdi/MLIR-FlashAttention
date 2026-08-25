# TRADEOFFS.md — MLIR Attention Pipeline

Living document. One entry per non-obvious design decision.
Updated as passes are implemented.

---

## Dialect: `attention.fused` includes V

**Decision:** `attention.fused` takes Q, K, V, scale, mask, output — the full
attention computation, not just QK + softmax.

**Why:** Without V in the fused op, the NxN attention weight matrix must be
fully materialised between the softmax and the PV matmul. That breaks the core
FA1 promise: tiling can never avoid the O(N²) materialisation unless P@V is
inside the same fused region so the tiling pass can tile over K/V together.

**Cost:** The op is slightly wider than the Requirements IR snippet suggests.
The Requirements snippets are acknowledged as illustrative/simplified.

---

## Dialect: `scale` is an SSA f32 operand, not an attribute

**Decision:** `FusedOp::scale` is `F32:$scale` (a runtime SSA value), not
`F32Attr:$scale` (a compile-time constant).

**Why:** `1/sqrt(head_dim)` is derived from a runtime-determined head dimension
in general use. Making it an attribute would force constant-folding of head_dim
before fusion, coupling the fusion pass to a specific call pattern.

**Cost:** The assembler format is slightly more verbose
(`scale(%val : f32)` instead of `scale = 0.125`).

---

## Dialect: mask is Optional, not required

**Decision:** `mask` is `Optional<MemRefOf<[I1]>>:$mask` with `AttrSizedOperandSegments`.

**Why:** Many attention variants (e.g., bidirectional encoder attention) have no
mask. Requiring a dummy all-false memref would force an allocation that a
constant-folding pass might not eliminate.

**Cost:** Requires `AttrSizedOperandSegments` trait, which stores a hidden
`operand_segment_sizes` attribute in the IR. Slightly more verbose C++ to
check `if (Value m = getMask())`.

---

## Fusion pass: memref-based pattern matching via buffer SSA values

**Decision:** The fusion pass matches the 5-op sequence by tracing which ops
write to which memref SSA values (using `DestinationStyleOpInterface::getDpsInits()`),
rather than requiring tensor semantics with pure SSA results.

**Why:** The rest of the pipeline uses memrefs for explicit memory control.
Switching to tensors for fusion then bufferizing adds a dependency on
one-shot-bufferize, which requires bufferization interfaces for every custom op.

**Cost:** The pattern assumes each intermediate buffer is written by exactly one
`linalg.GenericOp` or `linalg.SoftmaxOp`. Functions with aliased buffers or
multiple writers will not match. This is acceptable for a research prototype
with structured input IR.

---

## Fusion pass: scale extracted from linalg.generic body

**Decision:** The scale value is read from the body of the scale generic by
finding the first `arith.mulf` whose one operand is not a *local* block argument
of the generic's own body.

**Why:** The scale is an outer SSA value captured by the region. There is no
dedicated operand slot for it in `linalg.generic` (it's not in `ins`).

**Non-obvious MLIR behavior:** In MLIR, function arguments are `BlockArgument`s
— they are the block arguments of the function's entry block. A naive
`!isa<BlockArgument>(v)` check incorrectly rejects function-argument scale
values. The correct check is whether the argument's parent block is the
linalg.generic body itself (`ba.getParentBlock() == body`), distinguishing
local per-element args from outer captured values.

**Cost:** Fragile: only works if the scale generic body is exactly
`arith.mulf(block_arg, outer_scale)`. If the scale generic applies additional
operations (e.g., negation), extraction fails. Acceptable for the structured
test input; a production pass would need a more robust body analysis.

---

## Fusion pass: QK^T matched as linalg.generic, not linalg.matmul

**Decision:** The pattern matches the QK^T step as a `linalg.generic` with
`(parallel, parallel, reduction)` iterator types, not as a named `linalg::MatmulOp`.

**Why:** The test IR and real unfused attention code represent QK^T as a
`linalg.generic` with explicit indexing maps (transposing K on the fly). A
`linalg.matmul` would require the K matrix to already be transposed in memory,
adding an explicit transpose op before fusion. The generic form is more
natural for the input IR the fusion pass targets.

**Cost:** Design.md §3.2 describes the anchor as `linalg.matmul(%Q, %K)` for
the QK step — this is inaccurate. The anchor is the PV `linalg.matmul`; the
QK step is found via `findGenericWriterOf`. Design.md's algorithm pseudocode
needs updating to reflect this.

---

## Tiling pass: fully expands `attention.fused` (no remaining fused ops)

**Decision:** Pass 2 both tiles AND lowers `attention.fused` to linalg/affine.
After Pass 2 no `attention.fused` ops exist.

**Why:** The Requirements pipeline is
`--fusion-pass --tiling-pass | mlir-cpu-runner`.
`mlir-cpu-runner` cannot interpret `attention.fused`; if tiling only added loops
but kept the inner `attention.fused` intact, a third lowering pass would be needed.
Keeping the lowering inside the tiling pass matches the two-pass pipeline exactly.

**Cost:** The tiling pass does two conceptual things (tiling + lowering).
If we later want to tile without lowering (e.g., to inspect the tiled IR before
expansion), we would need to split it into two passes.

---

## Tiling pass: online softmax is part of tiling, not fusion

**Decision:** The online softmax accumulation (running max `m` and sum `l`) is
introduced by Pass 2 (Tiling), not Pass 1 (Fusion).

**Why:** Online softmax is a property of *tiled* execution — it only makes sense
when the attention row is processed in pieces. Introducing it at fusion time
would force the fused op to carry extra state even for the untiled (single-tile)
case, complicating the op definition.

**Cost:** The fusion pass output (`attention.fused`) does not itself implement
online softmax. Users of the fused op without tiling get standard (non-online)
softmax semantics; correctness requires tiling before CPU execution.

---

## Tiling pass: static shapes only (initial implementation)

**Decision:** The tiling pass asserts static memref shapes and returns
`emitOpError` for dynamic shapes.

**Why:** Affine loops (`affine.for`) require compile-time-constant bounds.
Supporting dynamic shapes would require `scf.for` loops plus runtime checks
that `seq % tile_size == 0`, adding significant complexity.

**Cost:** All test inputs must have shapes that are multiples of `tile_size`.
Dynamic sequence lengths (common in production transformers with padding) are
not supported; this is a known limitation for future work.

---

## Tiling pass: `memref.alloca` for tile-local buffers

**Decision:** All tile-local working buffers (O_acc, m_acc, l_acc, S_tile,
P_tile, etc.) use `memref.alloca` (stack allocation).

**Why:** `memref.alloca` avoids the need for explicit `memref.dealloc` calls
inside loop bodies, keeping generated IR simpler. LLVM typically hoists allocas
to the function entry, so the effective lifetime is the entire function.

**Cost:** The stack frame grows with tile size. At `tile_size=32` the total
allocation is ≈17 KB, well within typical 8 MB stack limits. At `tile_size=128`
it is ≈161 KB — still within limits but close to the danger zone on some systems.
For CPU correctness testing, use `--tile-size=32` or `--tile-size=64`.

---

## Driver: `registerAllDialects`

**Decision:** `attention-opt` calls `mlir::registerAllDialects(registry)` rather
than registering dialects individually.

**Why:** The 5-pass pipeline uses linalg, memref, affine, arith, math, vector,
func, and (future) nvgpu dialects. Enumerating them all would require updating
the driver every time a new dialect is added to a pass.

**Cost:** The binary links and registers dialects it may not use (e.g., SPIRV,
OpenMP). This is a minor binary-size increase acceptable for a research tool.
Production tooling would use selective registration.

---

## GPU lowering: deferred until hardware is available

**Decision:** Pass 5 (GPU Backend Lowering / nvgpu / tensor cores) is designed
but not yet implemented.

**Why:** No GPU hardware is available in the current development environment.
The target is NVIDIA A100 via university HPCC or cloud compute. CPU correctness
validation (Passes 1–4) is achievable now; GPU performance measurement requires
an LLVM build with NVPTX backend (`-DLLVM_TARGETS_TO_BUILD=NVPTX`) and actual
hardware to profile.

**Cost:** Pass 5 performance numbers cannot be collected until HPCC access is
established. The ablation study will be incomplete until then.

---

## Tile size default: 128

**Decision:** Default `--tile-size=128`.

**Why:** 128×128 tiles of f32 fit comfortably in A100 SRAM (192 KB:
128×128×4 bytes = 64 KB per tile, 3 tiles needed ≈ 192 KB). This is the
hardware-optimal size for the primary target.

**Cost:** On CPU (correctness testing only), 128 is unnecessarily large and
puts pressure on the stack. The `--tile-size=32` option should be used for CPU
tests. The default is intentionally kept at 128 to avoid forgetting to change
it when moving to GPU.

---

## Pass implementations declare getDependentDialects

**Decision:** Both `FusionPassImpl` and `TilingPassImpl` override
`getDependentDialects()` to explicitly declare the dialects whose ops they
create at runtime.

**Why:** MLIR only loads a dialect into the `MLIRContext` when it is
encountered in the input IR being parsed. If the input contains no
`attention.fused` ops (as is the case for `fusion.mlir`, which contains only
standard linalg ops), the attention dialect is never loaded — and
`FusedOp::create` crashes with "op not known in MLIRContext". The same applies
to `arith`, `affine`, `math`, and `memref` ops created by the tiling pass on
an input that only contains `attention.fused`. `getDependentDialects` is the
MLIR-idiomatic hook for declaring "this pass creates ops from these dialects;
load them before the pass runs."

**Cost:** Each pass must be kept in sync with the set of dialects it creates
ops from. Missing a dialect produces a runtime crash rather than a compile-time
error.

---

## Numerical validation: numpy reference instead of PyTorch

**Decision:** `test/numerical/reference.py` implements attention directly in
numpy (`(Q @ K.T) * scale`, masked softmax, `P @ V`) rather than calling
PyTorch.

**Why:** Requirements.md §5.1 specifies PyTorch as the reference, but standard
scaled-dot-product attention is a fixed, unambiguous formula — a numpy
implementation is mathematically identical ground truth, and PyTorch was not
available in the dev environment. Numpy is also sufficient for Minimum Viable
correctness checking; the actual PyTorch *baseline* (unfused, wall-clock timed,
§6.1) is a separate, later concern from correctness validation and does need
real PyTorch when that work starts.

**Cost:** If a PyTorch-specific numerical quirk (e.g. its softmax's exact
reduction order) ever mattered, this reference wouldn't catch it. Not expected
to matter for standard attention.

---

## Numerical validation: execution via mlir-runner + explicit lowering pipeline

**Decision:** `test/numerical/pipeline.py` runs `attention-opt --fusion-pass
--tiling-pass`, pipes the result through a fixed sequence of eleven `mlir-opt`
conversion passes down to the `llvm` dialect, then JIT-executes it with
`mlir-runner` (this LLVM build's name for `mlir-cpu-runner`) linked against
`libmlir_runner_utils`/`libmlir_c_runner_utils`, and parses the
`printMemrefF32` stdout dump back into a numpy array via `ast.literal_eval`.

**Why:** MLIR Python bindings are not enabled in this LLVM build
(`MLIR_ENABLE_BINDINGS_PYTHON=0`), so there is no in-process way to hand numpy
arrays to the JIT and get numpy arrays back. Piping through the command-line
tools and parsing the pretty-printed memref dump is the only path available
without rebuilding LLVM.

**The pass order matters and was found empirically:** `--convert-linalg-to-loops
--lower-affine --convert-scf-to-cf --expand-strided-metadata --lower-affine
--convert-cf-to-llvm --convert-arith-to-llvm --convert-math-to-llvm
--finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts`.
Two subtleties: `--expand-strided-metadata` (needed to lower the
`memref.subview` ops the tiling pass emits for tile slicing) itself emits new
`affine.apply`/`arith` ops, so `--lower-affine` must run a second time *after*
it, and `--convert-arith-to-llvm` must come after that second `--lower-affine`
rather than alongside the first batch of conversions.

**Cost:** The harness depends on the exact print format of `printMemrefF32`
(`"data = \n" + Python-list-like literal`), which is undocumented/internal to
MLIR's runner-utils library and could change between LLVM versions. Tool paths
are discovered from `build/CMakeCache.txt`'s `MLIR_DIR` rather than hardcoded.

---

## Numerical validation: test scope excludes batch dimension and non-tile-divisible shapes

**Decision:** `test/numerical/validate.py`'s default suite uses small,
single-batch (`[seq, head_dim]`, no batch axis), tile-divisible shapes
(seq lengths 4–16, tile sizes 4–8), not the full Requirements.md §5.1 matrix
(seq lengths up to 4096, batch sizes up to 32).

**Why:** `attention.fused`/the fusion pattern operate on 2-D `memref<seq x
head_dim>` — there is no batch dimension in the IR today (would need an outer
loop or a 3-D memref convention, neither implemented). Separately, `TilingPass`
only supports shapes evenly divisible by `tile-size` (see "Tiling pass: static
shapes only" above) — a non-divisible `seq_q`/`seq_k` silently drops the
remainder rather than erroring, so `validate.py` raises instead of silently
testing something incorrect.

**Cost:** This harness proves Pass 1+2 correctness for the shapes it covers,
but does not yet exercise the full production-scale matrix (large sequence
lengths, batching, non-divisible remainders). Batching and remainder handling
remain open work items, not just testing gaps.

---

## CPU benchmark: naive baseline reimplements softmax without `linalg.softmax`

**Decision:** `test/numerical/bench_codegen.py`'s `emit_baseline_module` uses
an `@attention_baseline` function that expands softmax into its four explicit
steps (rowmax → `exp(x - rowmax)` → rowsum → divide) via `linalg.generic`,
rather than calling `linalg.softmax` the way `codegen.py`'s
`@attention_unfused` does.

**Why:** `--convert-linalg-to-loops` has no lowering for `linalg.softmax`
(verified empirically: it passes through unchanged and the run fails later
when `mlir-runner` can't parse the leftover op). `@attention_unfused` only
ever gets *fed into* `attention-opt --fusion-pass`, which consumes the
`linalg.softmax` before any loop-lowering happens, so this gap never mattered
for numerical validation. The CPU baseline, by contrast, must be directly
executable with no attention-opt pass in front of it, so it needs a softmax
expressed in ops that `--convert-linalg-to-loops` actually handles.

**Cost:** Two independent "unfused attention" implementations now exist in
the test tree (`codegen.py`'s `linalg.softmax` form, used as fusion-pass
input; `bench_codegen.py`'s expanded form, used as the benchmark baseline).
Verified separately against `reference.py` (both match within tolerance,
including with a causal mask) so this isn't a correctness gap, but a change
to one should prompt checking whether the other needs the same fix.

---

## CPU benchmark: bare `call`/`return` don't parse inside `scf.for` — use `func.call`

**Decision:** `bench_codegen.py`'s timing loop uses `func.call @foo(...)`
inside `scf.for` bodies, not the bare `call @foo(...)` spelling used
elsewhere in the test tree (e.g. `codegen.py`'s `@main`).

**Why:** Found empirically — `call @foo() : () -> ()` parses fine as the
direct body of a `func.func`, but fails inside a nested `scf.for` region with
a confusing `Dialect `` not found for custom op 'call'` error (the unprefixed
mnemonic only resolves to `func.call` in certain parse contexts, not inside
arbitrary nested regions). Also required an explicit `scf.yield` terminator in
each loop body — omitting it produces further confusing downstream parse
errors rather than a clear "missing terminator" diagnostic.

**Cost:** None — `func.call` is the fully-qualified spelling and works
everywhere; it's simply more verbose than the bare form.

---

## CPU benchmark: timing via in-process `rtclock()`, not subprocess wall-clock

**Decision:** `bench_codegen.py` generates an untimed warmup `scf.for` loop
followed by a timed `scf.for` loop bracketed by `rtclock()` calls (from
`mlir_c_runner_utils`), with the elapsed time printed via `printF64`. Python
(`benchmark.py`) times nothing itself — it runs several independent
subprocess trials and reads back the number each one printed.

**Why:** `mlir-runner` JIT-compiles the whole module before executing `main`,
so if Python timed the subprocess call from outside, the measurement would
include process startup and JIT compile time — both roughly constant
per-invocation costs that would swamp the actual per-call compute time being
measured, especially for small shapes. Bracketing many loop iterations with
`rtclock()` *inside* the already-JIT-compiled program measures only the
repeated calls themselves.

**Cost:** Requires generating a correct `scf.for` timing harness per module
(see the `func.call`/`scf.yield` gotchas above) rather than a one-line Python
`time.perf_counter()` wrap.

---

## CPU benchmark: `perf stat` (Requirements.md §5.2) unavailable — wall-clock speedup only

**Decision:** `benchmark.py` reports only wall-clock speedup (baseline
median / fused median, over independent trials), not the
`perf stat -e cycles,instructions,cache-misses` profiler counters Requirements
.md §5.2 asks for.

**Why:** `perf` is Linux-only; this dev environment is macOS, which has no
direct equivalent exposed via a stable CLI (Instruments.app exists but isn't
scriptable the same way). The actual §5.2 acceptance gate is the wall-clock
`>1.2x` speedup number, which this harness does measure and check.

**Cost:** No visibility into *why* the fused/tiled version is faster (cache
misses avoided, instruction count, etc.) — only that it is. Revisit if/when
benchmarking moves to a Linux machine or GPU host where `perf`/`ncu` are
available.
