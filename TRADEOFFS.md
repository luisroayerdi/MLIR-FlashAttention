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

## Vectorization pass: drives MLIR's built-in `linalg::vectorize`, not a hand-rolled VF/remainder scheme

**Decision:** Pass 3 walks the function for the `linalg.generic`/`linalg.fill`/
`memref.copy` ops `TilingPass` emits and vectorizes each one via
`mlir::linalg::vectorize` / `mlir::linalg::vectorizeCopy`, using each op's own
full static iteration-space shape as the vector shape (no `inputVectorSizes`
override). It does not implement the VF=8/remainder-loop scheme Requirements.md
§4.3 illustrates.

**Why:** `TilingPass` output has no raw scalar `affine.for`+`memref.load`/
`memref.store` loops to pattern-match — every tile-body computation is already
a `linalg.generic` (or `linalg.fill`/`memref.copy`) over a statically-shaped
tile, which is precisely the input MLIR's own linalg vectorizer targets.
Hand-rolling VF-chunked rewrites would duplicate logic MLIR already provides
and battle-tests, for no benefit at this static-shape granularity: since
`TilingPass` requires tile dimensions to be compile-time constants, there is
no "leftover" remainder to handle in the first place — the tile size itself
plays the role Requirements.md's VF does. Design.md §5.2 (original, April
2026) already anticipated this: "The pass uses `mlir::vectorize`
infrastructure where available."

**Cost:** The vector ops this pass emits are tile-shaped (e.g.
`vector<32x32xf32>`), not hardware-register-shaped (`vector<8xf32>`) —
decomposing to real SIMD width is deferred to downstream
`--convert-vector-to-scf`/`--convert-vector-to-llvm` passes, which are not yet
wired into `test/numerical/pipeline.py`'s lowering flags (see below).
Requirements.md's illustrative IR snippet for this pass is now acknowledged
as inaccurate for this codebase, matching the precedent set for Passes 1–2.

---

## Vectorization pass: `linalg::vectorize()` requires an explicit `replaceOp` — it doesn't erase the original op

**Decision:** After a successful `linalg::vectorize(rewriter, op)` call, Pass 3
explicitly calls `rewriter.replaceOp(op, result.replacements)`.

**Why (non-obvious MLIR behavior, found by direct investigation of
`mlir/lib/Dialect/Linalg/Transforms/Vectorization.cpp`):** `linalg::vectorize`
builds the `vector.transfer_read`/arithmetic/`vector.transfer_write`
replacement sequence but leaves erasing/replacing the original op to the
caller. For a buffer (memref) DPS op — every op in `TilingPass`'s output —
the op has zero SSA results, so `result.replacements` comes back empty and
`replaceOp` degrades to a plain erase. Omitting this call was the first
implementation attempt here: it left the original scalar op in the IR
*after* the new vector code, silently overwriting the vectorized write with
an identical scalar recomputation on the very next op. FileCheck for
`vector.transfer_write` still passed and the numbers would still have been
correct — the bug was purely "no actual vectorization is happening" — which
is exactly the kind of thing that would NOT be caught by numerical
validation alone. Confirmed correct usage against upstream:
`transform::VectorizeOp` in `LinalgTransformOps.cpp` follows the identical
`vectorize()` → `rewriter.replaceOp()` pattern.

**Cost:** None — this is the correct/required usage, not a workaround.
Recorded because the failure mode (silently doing nothing while looking
correct) is easy to reintroduce if this pass is refactored later.

---

## Vectorization pass: wired into the numerical/benchmark harness via an opt-in `vectorize` flag

**Decision:** `test/numerical/pipeline.py`'s `_LOWER_FLAGS` now always include
the vector-dialect lowering passes (`--convert-vector-to-scf`,
`--lower-vector-multi-reduction`, `--convert-vector-to-llvm`,
`--convert-ub-to-llvm`); `run_module`/`run_fused_timed` take a `vectorize:
bool` that appends `--vectorization-pass` to the `attention-opt` invocation.
`validate.py`/`benchmark.py` expose this as a `--vectorize` CLI flag; the
default suites still run scalar-only by default (matching Pass 1–2's
existing default behavior), with `--vectorize` re-running the same suite
through Pass 3 as well.

**Why:** Requirements.md §4.3 requires "Numerical: vectorized matches scalar"
for this pass, the same bar Pass 1–2 already cleared. Making it opt-in rather
than replacing the default keeps the existing (already-validated) Pass 1–2
suite runs untouched while adding equivalent coverage for Pass 3.

**Found empirically extending the lowering pipeline (§5.2's original "not yet
wired in" note undersold the work involved):** getting vectorized IR to
JIT-execute at all required three additions beyond what Pass 1–2 needed:
1. `--convert-vector-to-scf` to break multi-dimensional `vector.transfer_*`
   ops (e.g. `vector<32x32xf32>`) into loops of hardware-width ones.
2. `--lower-vector-multi-reduction` — `--convert-vector-to-llvm` alone does
   **not** lower `vector.multi_reduction` (used for the QK^T/PV
   matmul-style ops and the row max/sum reductions); it must be lowered to
   simpler vector ops first, in a separate pass, before `--convert-vector-to-llvm` runs.
3. `--convert-ub-to-llvm` — `linalg::vectorize()` uses `ub.poison` as the
   padding value for `vector.transfer_read`. This is legitimate, *heavily
   used* IR (it seeds `llvm.insertvalue` chains that build up
   `!llvm.array<N x vectorMxf32>` results — not dead code, despite initially
   looking like it from a stale intermediate inspection). `mlir-runner`'s
   registry only calls `registerAllToLLVMIRTranslations`, not full dialect
   registration, so it cannot even *parse* a leftover `ub.poison`; the `ub`
   dialect must be fully converted to `llvm.mlir.poison` before handoff.

All three were found by iteratively lowering a concrete vectorized test
module with `mlir-opt` and reading the resulting parse/verification errors,
the same empirical methodology TRADEOFFS.md's original 11-flag `_LOWER_FLAGS`
entry describes.

**Cost:** The lowering pipeline is now 15 flags instead of 11, all always
applied (verified as no-ops on Pass 1–2's non-vectorized output — see next
entry). No new cost beyond that; Pass 1–2's existing numerical/CPU-benchmark
results are unaffected (identical output on the same reference case).

---

## Vectorization pass: `memref<...xi1>` (mask) operands are never vectorized

**Decision:** `VectorizationPass` skips (leaves scalar) any `linalg.generic`/
`linalg.fill`/`memref.copy` op with an `i1`-element-type memref operand —
concretely, only the mask-select generic (`arith.select` over the causal
mask tile).

**Why (found empirically, via the masked-attention numerical suite failing
with ~1.0 max error while unmasked cases passed at ~1e-6):** `memref<...xi1>`
stores each boolean as a full byte, but MLIR's `vector<...xi1>` type lowers
(via `--convert-vector-to-llvm`) to an LLVM `llvm.load : vector<Nxi1>`, whose
in-memory representation is bit-packed (LLVM's native i1-vector ABI), not
byte-per-element. Vectorizing the mask-select op reads the mask tile through
that mismatched layout and produces garbage — this is a known general
MLIR/LLVM limitation around boolean vectors, not specific to this pass.

**Cost:** The mask-select op remains scalar `linalg.generic` after Pass 3 —
confirmed correct (5/5 masked+unmasked configs pass numerically, matching
Pass 1–2's error magnitudes exactly) but not "fully vectorized" in the sense
Pass 3's summary implies. `test/Attention/vectorization.mlir`'s MASK check
prefix deliberately does not assert zero remaining `linalg.generic` ops, for
this reason.

---

## Vectorization pass: full-tile vectorization does not scale to CPU JIT compilation at production tile sizes

**Decision:** `benchmark.py --vectorize --suite` uses a new, separate
`VECTORIZED_SUITE` (small shapes: `tile-size` 8–16, `head-dim` 16) instead of
`DEFAULT_SUITE` (`tile-size=32`, `head-dim=64`, the production-scale A100
config Pass 1–2 already validate against).

**Why (found empirically running the vectorized suite against
`DEFAULT_SUITE`):** `VectorizationPass` vectorizes each tile op to its own
full static shape with no decomposition to hardware-width vectors (Design.md
§5.2's "no manual VF/remainder loop" design decision). The QK^T/PV reduction
ops lower to an LLVM `!llvm.array<... x vector<...>>` whose flattened element
count is `tile_size² × head_dim`. Measured JIT behavior against that count:

| tile × tile × head_dim | element count | result |
|---|---|---|
| 8×8×16 | 1,024 | fast (seconds), 5.1x speedup |
| 16×16×16 | 4,096 | fast (seconds), 6.5-7.0x speedup |
| 16×16×32 | 8,192 | hung — multi-minute, multi-GB RSS, killed rather than waited out |
| 32×32×64 (`DEFAULT_SUITE`'s tile/head-dim) | 65,536 | hung — same |

The cliff is sharp (between 4,096 and 8,192), consistent with a
combinatorial (not linear) blowup somewhere in LLVM's handling of large
flattened array-of-vector aggregates (`insertvalue`/`extractvalue`/
`shufflevector` chains) during instruction selection/JIT compilation — this
was not root-caused further (out of scope for this session); the workaround
was to characterize the safe range empirically rather than fix the
underlying scaling.

**Cost:** Pass 3's CPU-benchmark evidence (4.8x-6.5x speedup, comfortably
past the §4.3 "1.5-2x vs scalar" target) is only representative at small
scale, not at the `tile=32`/`head_dim=64` production scale Pass 1-2's own
benchmark suite already validates. Closing this gap needs `linalg::vectorize`'s
`inputVectorSizes` parameter (explicit hardware-width chunking, e.g.
`vector<8xf32>`, with the remainder handled by the surrounding
`--convert-vector-to-scf` loop) rather than the current full-tile-shape
default — deferred as future work, not blocking for Pass 4.

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

---

## Mask specialization pass: inline `affine.if` + block cloning, not outlined kernel functions

**Decision:** `MaskSpecializationPass` builds the FULL/MASKED/BOUNDARY
dispatch as `affine.if`/`affine.if...else` directly around cloned/moved
copies of the K-loop body's existing ops, rather than the outlined
`@inner_full`/`@inner_boundary` functions Design.md §6.3's original
pseudocode showed (with inlining left to "a subsequent canonicalization
step").

**Why:** Outlining would require synthesizing new `func.func` signatures for
each variant (which operands does each body actually capture? — the K-loop
body references values from three enclosing scopes: function arguments,
the Q-loop's tile-local accumulators, and its own induction variables), then
relying on inlining actually happening downstream to avoid real call
overhead — none of which this project's pipeline (`test/numerical/
pipeline.py`'s `_LOWER_FLAGS`) currently guarantees. Building the control
flow directly with `IRMapping`-based cloning (for FULL) and `Operation::
moveBefore` (for BOUNDARY, reusing the original ops rather than re-cloning
them) sidesteps both problems and is less code.

**Cost:** Duplicates the K-loop body's IR (once for the FULL branch, via
clone) rather than sharing a single outlined definition — for a large tile
body this means more IR to carry through the rest of the pipeline. Not
measured as a problem at the tile sizes this project benchmarks (`tile=32`),
but would be worth revisiting (e.g. `--inline`-friendly outlined functions
after all) if tile-body size grows significantly (more passes stacking
per-tile logic) or if binary/compile-time size becomes a concern.

---

## Mask specialization pass: reuses Pass 3's i1-memref identification signature

**Decision:** `MaskSpecializationPass` finds the mask-select op the same way
`VectorizationPass` finds the op to *exclude* from vectorization: a
`linalg.generic` with an `i1`-element-type memref among its operands (see
TRADEOFFS.md "Vectorization pass: `memref<...xi1>` (mask) operands are never
vectorized"). This is not a shared helper — each pass has its own small
static function with the identical check.

**Why:** Both passes need to identify the exact same op (the one TilingPass
emits for "3. Optional mask" — see TilingPass.cpp) for unrelated reasons
(Pass 3: don't vectorize it; Pass 4: specialize around it), and the check
is a two-line predicate. Not worth a shared header for this project's size.

**Cost:** If the mask-select op's identifying shape ever changes (e.g.
TilingPass starts using a different mask representation), both copies need
updating. Low risk given how small and stable each copy is.

---

## Mask specialization pass: correctness depends entirely on the mask being causal, unverified

**Decision:** `MaskSpecializationPass` classifies every K/V tile as
FULL/MASKED/BOUNDARY purely from the Q-loop/K-loop induction variables and
the tile size — it never inspects the mask memref's actual runtime contents.

**Why:** This is the whole point of the optimization (skip work *without*
reading the mask for FULL/MASKED tiles) — reading the mask to decide whether
to skip reading the mask would defeat it. Design.md §6.4 already scoped this
pass to "square causal masks" for exactly this reason; this entry makes the
consequence explicit.

**Cost:** This is a real, silent correctness risk if the pass is ever applied
outside this project's own test scope: passing a non-causal boolean mask
(e.g. a padding mask, a sliding-window mask, an arbitrary sparse pattern) to
`attention.fused` and then running `--mask-specialization-pass` produces
wrong results with no diagnostic — the FULL branch would skip masking on
tiles the actual mask marks as (partially) masked, and the MASKED branch
would zero out tiles the actual mask allows. Verified safe for every config
this project's numerical suite covers (all use `np.triu(..., k=1)`, the
same causal construction `attention_reference` and every test module use).
Not verified, and not easily verifiable without adding a runtime mask-shape
check the pass currently has no mechanism for.

**Measurement:** `validate.py --suite --mask-specialize` (5/5, error
magnitudes identical to the non-specialized masked baseline) and two ad hoc
larger tile grids (4x4 and 5x5 tiles, `--seq-q 16/20 --tile-size 4`) chosen
specifically to exercise all three tile classifications together, not just
the 2x2-grid case in `DEFAULT_SUITE`. `benchmark.py --mask-specialize --suite`
measures 1.77x-1.87x speedup vs. Pass 1-2's generic per-element masking —
comfortably past Requirements.md §4.4's own "1.15-1.3x vs generic masking"
target.

---

## Phase 2 integration: `--full-pipeline` reuses `VECTORIZED_SUITE`'s scale, gated at the §5.4 Go/No-Go threshold

**Decision:** `benchmark.py --full-pipeline` — the Requirements.md §9.2
Phase 2 / §5.4 Go/No-Go CPU benchmark for all four passes together — is a
third, separate comparison mode from `--vectorize` and `--mask-specialize`
(it overrides both), using `bench_case`'s existing unfused-vs-fused
comparison but with `vectorize=True, mask_specialize=True` forced on and
the pass/fail bar raised from `SPEEDUP_THRESHOLD` (1.2x, §5.2) to
`GO_NO_GO_THRESHOLD` (1.5x, §5.4). Its suite is `VECTORIZED_SUITE`
(`tile-size` 8-16, `head-dim` 16), not `DEFAULT_SUITE`.

**Why:** Requirements.md §5.4's Go/No-Go criteria ("PROCEED if... Performance:
>1.5x speedup vs unfused") is the literal checkpoint gating whether Phase 3
(GPU Lowering) can begin — it's a different, stricter bar than §5.2's CPU
Validation `>1.2x`, and it's specifically about the *complete* pipeline, not
any single pass. Since Pass 3 is part of "complete," Pass 3's own JIT-scale
ceiling (`tile_size^2 * head_dim <~4096`, see the vectorization-pass
scalability entry above) applies here too — there's no way to benchmark the
full four-pass pipeline at `DEFAULT_SUITE`'s production scale
(`tile=32`/`head_dim=64`) without hitting the same multi-minute JIT hang.

**Cost:** Same caveat as Pass 3's own benchmark: the Go/No-Go result
(5.0x-7.6x speedup, comfortably past 1.5x) is only representative at small
scale. This is now the *second* place (after `VECTORIZED_SUITE` itself) this
limitation blocks production-scale measurement — closing Pass 3's scaling
gap (deferred future work, see above) would let this checkpoint run at the
scale that actually matters for the eventual GPU-lowering decision.
