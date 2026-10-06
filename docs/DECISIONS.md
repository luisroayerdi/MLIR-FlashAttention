# Decisions

Why the code is the way it is. A PR that contradicts a decision must say why
and update its row. "Revisit when" marks decisions expected to change, with
the `SPECS.md` feature that will change them.

## Dialect

| Decision | Why | Revisit when | Ref |
|---|---|---|---|
| `attention.fused` includes V (Q, K, V, scale, mask, output) | Without P·V inside the op, the N×N weight matrix must be materialized between softmax and the second matmul; tiling can't avoid it | — | Phase 1 |
| `scale` is an SSA f32 operand, not an attribute | `1/sqrt(d)` may be a runtime value; an attribute would force head_dim to be constant-folded before fusion | — | Phase 1 |
| Mask is optional (`AttrSizedOperandSegments`) | Unmasked attention shouldn't allocate a dummy mask | §3.1 affine masks | Phase 1 |
| One head per op: Q, K, V are `[N, d]`, no batch or head dimension | Smallest shape that exercises every pass | §3.1 batched multi-head shapes | Phase 1 |

## Passes

| Decision | Why | Revisit when | Ref |
|---|---|---|---|
| Fusion matches on memrefs by tracing DPS inits, not on tensors | The pipeline uses memrefs; tensors would require bufferization interfaces for every custom op | §3.2 recognition of torch-mlir forms (tensor input) | Phase 1 |
| Fusion declines non-f32 input and intermediates with users outside the chain (other than `memref.dealloc`); it assumes an `i1` select mask | `attention.fused` accepts only f32, and fusion stops writing the intermediate buffers, so another reader would see uninitialized memory. The `linalg.fill` zeroing the QKᵀ accumulator is allowed | §3.1 f16/bf16; §3.2 recognition of torch-mlir forms | Phase 1 |
| Q·Kᵀ matched as `linalg.generic` (parallel, parallel, reduction), not `linalg.matmul` | Real unfused IR transposes K through indexing maps; `linalg.matmul` would need K pre-transposed | §3.2 tensor cores (the `mma.sync` rewrite needs `linalg.matmul`) | Phase 1 |
| Tiling fully lowers `attention.fused` to affine/linalg/memref | Downstream lowering and `mlir-runner` can't handle the custom op | — | Phase 1 |
| Online softmax is introduced by tiling, not fusion | It only exists for tiled execution; carrying running max/sum in the fused op complicates its definition | — | Phase 1 |
| Tiling loop order: Q tiles outer, K/V tiles inner | Q tiles write disjoint output rows, so the K/V loop needs no synchronization; K/V stream through, as in FlashAttention-1 | — | Phase 1 |
| Static shapes only; N divisible by tile size | `affine.for` needs constant bounds; dynamic shapes need `scf.for` plus runtime checks | §3.1 decode shapes | Phase 1 |
| Tile buffers use `memref.alloca` | No deallocs in loop bodies; stack grows with tile size | §3.2 shared-memory staging | Phase 1 |
| Default tile size 128 | 128×128 f32 tiles fit A100 shared memory; use 32 on CPU | §3.1 f16/bf16 | Phase 1 |
| Vectorization uses upstream `linalg::vectorize` at full tile shape, followed by an explicit `replaceOp` | Reuses upstream; `vectorize` doesn't erase the original op | §3.2 production scale (full-tile vectors blow up JIT past tile²·d ≈ 4096) | Phase 1 |
| Ops with i1 mask memrefs are never vectorized | Vectorizing i1 memrefs corrupted masked results (~1.0 max error) | §3.1 affine masks (no mask memref) | Phase 1 |
| Vectorization is best-effort: ops `linalg::vectorize` rejects stay scalar | A dynamic shape or unsupported op shouldn't fail the pipeline | — | Phase 1 |
| Mask specialization uses inline `affine.if` plus block cloning, not outlined functions | Outlining needs synthesized signatures and a later inlining step | — | Phase 1 |
| Mask specialization runs after tiling and assumes a top-left-aligned causal mask (`mask[i,j]` true iff `j > i`); seq_q ≠ seq_k is allowed | It classifies tiles of the `affine.for` nest tiling creates by `i`/`j` position alone. Not verified: a causal mask aligned differently (e.g. bottom-right when seq_q ≠ seq_k) is misclassified | §3.2 affine mask families | Phase 1 |
| Mask classification uses loop indices only; causal precondition not verified | Reading the mask to decide whether to skip reading it defeats the purpose | §3.2 precondition verification | Phase 1 |
| Passes declare `getDependentDialects` | MLIR loads dialects only when parsed; created ops would otherwise fail | — | Phase 1 |

## GPU

| Decision | Why | Revisit when | Ref |
|---|---|---|---|
| GPU launch via `convertAffineLoopNestToGPULaunch` (one block per query tile), not `affine-parallelize` | `affine-parallelize` treats loops containing linalg/memref ops as not parallel | §3.2 warp-cooperative launch | `c8814c9` |
| One thread per GPU block | Simplest mapping that is correct by construction. Precondition, not verified: each Q-tile iteration owns its accumulators and writes a disjoint output subview | §3.2 warp-cooperative launch | Phase 1 |
| Linalg/affine/scf/memref lowered before `-gpu-lower-to-nvvm-pipeline` | The NVVM pipeline has no lowering for these ops in pass-generated kernels | — | `4d8415f` |
| Every memref a kernel touches is `gpu.host_register`ed before the launch | Registering after the launch, or only the output, silently breaks on hardware | §3.4 GPU timing harness | Phase 1 |
| Tensor cores measured as a standalone matmul, not in the fused kernel | `transform.nvgpu.rewrite_matmul_as_mma_sync` needs a literal `linalg.matmul` (QKᵀ reads K transposed through a `linalg.generic`), supports only 16×8×4 tf32 and 16×8×16 f16, and assumes one 32-thread warp; Stage A launches one thread per block | §3.2 tensor cores | `5812926` |
| `attention-opt` registers all dialects and all extensions | Transform-dialect extension ops (nvgpu) aren't registered by `registerAllDialects` | — | Phase 1 |
| GPU execution needs LLVM built with `-DLLVM_TARGETS_TO_BUILD="Native;NVPTX"` and `-DMLIR_ENABLE_CUDA_RUNNER=ON` | NVPTX codegen and the CUDA runtime library; FileCheck tests of the GPU passes don't need either | — | Phase 1 |
| Default GPU target `sm_89` (RTX 4090) | Phase 1 hardware; tf32 `mma.sync` from sm_80 runs unchanged on Ada | §3.2 tensor cores (sm_80/sm_90) | Phase 1 |

## Harness

| Decision | Why | Revisit when | Ref |
|---|---|---|---|
| Reference is NumPy, not PyTorch | Attention is a fixed formula; avoids a heavy dependency | §3.3 fp64 reference | Phase 1 |
| Execution through CLI `mlir-runner` and parsed stdout | MLIR Python bindings are off in this LLVM build | — | Phase 1 |
| Timing inside the compiled program (`rtclock`), warmup excluded | Keeps JIT compile and process startup out of measurements | §3.4 CUDA-event timing | Phase 1 |
| Vectorized benchmarks use a smaller suite (`VECTORIZED_SUITE`: tile 8–16, head_dim 16) | Full-tile vectors JIT-compile in seconds up to tile²·d ≈ 4096 elements and hang at 8192+ | §3.2 production scale | Phase 1 |
| Unfused baseline expands softmax into four generics | `convert-linalg-to-loops` can't lower `linalg.softmax` | §3.5 baselines | Phase 1 |
