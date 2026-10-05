# Literature

What already exists, and how it relates to this project. Check here before
building anything (see `CONTRIBUTING.md`). Dates are when an entry was last
checked against its sources; `unverified` entries must be checked before
anyone relies on them. Recheck entries older than 60 days.

## Systems

| System | How it handles attention | Relation | Sources | Checked |
|---|---|---|---|---|
| IREE | Named op `iree_linalg_ext.attention` (Q, K, V, scale, optional mask); `online_attention` adds running max/sum and tiles along the softmax reduction. Mask applied element-wise (`QK += M`). Open PRs add `is_causal` computed from indices, and FlexAttention mask lowering to a tensor; no tile skipping found. | Closest prior work; `attention.fused` plays the same role. Baseline. Recognition of unfused input: open question. | [docs](https://iree.dev/reference/mlir-dialects/LinalgExt/), [#17536](https://github.com/iree-org/iree/pull/17536), [#18525](https://github.com/iree-org/iree/pull/18525), [#23999](https://github.com/iree-org/iree/pull/23999), [#24056](https://github.com/iree-org/iree/pull/24056), [#24426](https://github.com/iree-org/iree/pull/24426), [#12084](https://github.com/iree-org/iree/pull/12084) | 2026-10-05 |
| Triton | Kernel author writes tiling and online softmax in Python; compiler handles layouts, shared memory, tensor cores, pipelining. Own MLIR dialects. | Baseline (tutorial fused-attention kernel). | [anatomy](https://arxiv.org/abs/2511.11581), [tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | unverified |
| Neptune | Tensor compiler that fuses reduction chains by breaking loop-carried dependencies and adding algebraic correction terms; from plain attention plus a scheduling template it generates FlashAttention- and FlashDecoding-equivalent kernels. Reports 1.35× average over the next best of Triton, TVM, FlexAttention on four NVIDIA/AMD GPUs. | Derives FlashAttention from plain attention automatically, outside MLIR. Cited in the `linalg.attention` RFC as the model for representation design. Baseline candidate. | [arXiv](https://arxiv.org/abs/2510.08726), [talk](https://drive.google.com/file/d/1Q90dhcSlOyrsl0nWIVp5zXySSuGGdQQB/view) | 2026-10-05 (abstract) |
| FlexAttention | User supplies score/mask functions; generates Triton kernels; skips blocks at runtime via block-sparse masks. MLSys 2025. | Closest work on mask structure; the RFC's suggested model for a parametrized attention op. | [arXiv](https://arxiv.org/abs/2412.05496), [blog](https://pytorch.org/blog/flexattention/), [inference](https://pytorch.org/blog/flexattention-for-inference/), [FA4 backend](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/), [attention-gym](https://github.com/meta-pytorch/attention-gym), [talk](https://www.youtube.com/watch?v=ju-KlcuWlbk) | unverified |
| TVM | Own IR (Relax, TensorIR), schedule search; LLM stack dispatches attention to libraries (FlashInfer). | Baseline; library dispatch reported. | [arXiv](https://arxiv.org/abs/1802.04799) | unverified |
| torch.compile | Inductor replaces attention patterns with SDPA, which dispatches to FlashAttention or cuDNN. | Same baseline as SDPA. | [fuse_attention.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/fuse_attention.py) | unverified |
| FlashAttention 1/2/3 | Hand-written CUDA (CUTLASS/CuTe): tiling, online softmax, causal block skipping, work partitioning (2), Hopper asynchrony (3). | Performance reference. | [FA1](https://arxiv.org/abs/2205.14135), [FA2](https://arxiv.org/abs/2307.08691), [FA3](https://arxiv.org/abs/2407.08608) | 2026-10-05 |

## Upstream MLIR

| Item | Status | Relation | Sources | Checked |
|---|---|---|---|---|
| `linalg.attention` RFC (GSoC) | Mar 2026; proposed adapting IREE's op signature. Stalled: upstream can't transform `linalg.softmax` (irregular, IREE-shaped decomposition; unclear what a tile of a composite op is); attention variants unaddressed; suggested co-designing the op with the optimizations it must enable (FlexAttention, Neptune). | Evidence for the representation question. | [Discourse](https://discourse.llvm.org/t/rfc-gsoc-adding-support-for-attention-in-linalg/90166) | 2026-10-05 |
| Linalg Forms RFC | Aug 2025; Linalg moving to an operation tree (generic → category → named). Open question: where composite ops like softmax fit. | The upstream design question attention depends on. | [Discourse](https://discourse.llvm.org/t/rfc-linalg-forms/87994) | 2026-10-05 |
| Softmax→matmul fusion (Intel XeGPU) | Rewrites `linalg.softmax` → `linalg.matmul` into tiled online softmax via generic ops and tile-and-fuse. PR closed unmerged 24 s after opening (stacked-PR branch); status elsewhere unknown. | Same transformation as our fusion + tiling, from generic ops. | [#204961](https://github.com/llvm/llvm-project/pull/204961), [thread](https://lists.llvm.org/pipermail/mlir-commits/2026-June/171996.html) | 2026-10-05 |
| `transform.nvgpu.rewrite_matmul_as_mma_sync` | Needs literal non-transposed `linalg.matmul`, two fragment shapes, single warp. | Used in Phase 1 tensor-core microbenchmark. | [nvgpu docs](https://mlir.llvm.org/docs/Dialects/NVGPU/) | unverified |

## Papers

| Paper | Relevance | Source |
|---|---|---|
| Online normalizer calculation for softmax (Milakov, Gimelshein) | The online-softmax algorithm | [arXiv](https://arxiv.org/abs/1805.02867) |
| Self-attention Does Not Need O(n²) Memory (Rabe, Staats) | Memory-efficient attention, cited in the RFC | [arXiv](https://arxiv.org/abs/2112.05682) |
| Composable and Modular Code Generation in MLIR (Vasilache et al.) | Linalg structured codegen | [arXiv](https://arxiv.org/abs/2202.03293) |
| The MLIR Transform Dialect (Lücke et al.) | Transform-dialect scheduling | [arXiv](https://arxiv.org/abs/2409.03864) |

## Upstream gaps

Gaps observed in this project. Each: failure, reproducer, upstream status.

| Gap | Observed | Upstream status |
|---|---|---|
| `-gpu-lower-to-nvvm-pipeline` lacks linalg/memref lowering for pass-generated kernels | Phase 1 GPU run (`4d8415f`) | not reported |
| `rewrite_matmul_as_mma_sync` rejects transposed K and `linalg.generic` matmuls | Phase 1 Stage B design | not reported |
| `affine-parallelize` treats loops containing linalg/memref ops as not parallel | Phase 1 Stage A design | not reported |
