# Fused Attention via MLIR

### Motivation

FlashAttention demonstrated that fusing the attention computation (MatMul → Scale → Mask → Softmax) into a single kernel dramatically reduces memory bandwidth pressure. However, FlashAttention achieves this through hand-written CUDA. A solution that is hardware-specific, not reusable, and opaque to compiler analysis.

This project asks a different question: **can a general-purpose compiler pipeline express and optimize the same fusion automatically?** The goal is not to beat FlashAttention on raw throughput, but to build a clean, portable, and extensible MLIR lowering pipeline that produces a fused kernel from high-level op definitions.

### Contribution

The project is centered on the design of a multi-stage MLIR lowering pipeline:

1. **Fusion pass** 
	Identify and merge the four ops at the `linalg` level before any lowering occurs.
2. **Hardware-aware tiling** 
	A parameterized tiling pass where tile shapes are derived from target tensor core constraints (e.g. 16×16×16 on Ampere).
3. **Dialect lowering chain** 
	A clean `linalg` → `vector` → `nvgpu`/`rocdl` lowering that demonstrates backend portability across NVIDIA and AMD targets
4. **Mask handling** 
	Causal masking creates three distinct tile cases along the attention diagonal. Fully visible, fully masked, and boundary tiles that straddle it. The pipeline handles each explicitly rather than falling back to scalar code that would break fusion.

The novel angle is the pipeline architecture itself: a reusable, inspectable set of passes that encodes expert fusion knowledge in the compiler rather than in hand-written kernels.
### Baselines & Evaluation

The goal of evaluation is to characterize the compiler pipeline, not to win a benchmark:

- **Baseline 1** unfused `torch.compile` output, to measure the cost of not fusing
- **Baseline 2** FlashAttention-2 via Triton, as the reference for what expert hand-optimization achieves
- **Measurement axes** kernel launch overhead, memory traffic (via `ncu`), and compile-time pass cost; _not_ just wall-clock throughput

The expectation is that the MLIR pipeline won't match FA2 on raw speed. The interesting result is **how close it gets and why**, and whether the same pipeline compiles cleanly for a second backend.

### Sources
FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.
MLIR-based code generation for GPU tensor cores

