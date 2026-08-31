# MLIR Attention Pipeline - Requirements Specification

**Version:** 1.0  
**Date:** April 2026  
**Language:** C++ (MLIR passes), Python (testing/benchmarking)

---

## 1. Problem Statement

Modern attention kernels (FlashAttention) achieve 2-10x speedups through hand-written CUDA. This approach is:

- Hardware-specific (NVIDIA-only)
- Opaque (low-level code)
- Not reusable
- Not portable

**Research Question:** Can we express FA1/FA2 optimizations as MLIR compiler passes? How much performance can we recover?

**Goals:**

- Implement FA1/FA2 as modular MLIR passes
- Measure compiler-achievable performance
- Identify compiler limitations
- Document tradeoffs

**Non-Goals:**

- Beat FlashAttention-2 performance
- Production system
- New algorithms

---

## 2. Success Criteria

### Minimum Viable

- Fusion pass (correctness tested)
- Tiling pass (correctness tested)
- CPU executable code
- Speedup over unfused baseline (>20%)
- Numerical correctness vs PyTorch (error < 1e-5)

### Target

- All minimum criteria
- GPU lowering (nvgpu dialect)
- Tensor cores utilized (profiler verified)
- Performance comparison table (all baselines)
- Gap analysis documented

### Stretch

- Match torch.compile performance
- Multi-backend (NVIDIA + AMD)
- One FA2 optimization working

---

## 3. FlashAttention Techniques

### 3.1 FlashAttention 1 (Dao et al., 2022)

**Paper:** https://arxiv.org/abs/2205.14135

|Technique|Compiler Feasibility|Priority|
|---|---|---|
|Op Fusion|High|REQUIRED|
|Tiling|High|REQUIRED|
|Online Softmax|Medium|Implicit in fusion|
|Recomputation|Medium|Out of scope (forward only)|

**Rationale:** Core techniques are compiler-friendly, hardware-agnostic, sufficient for complete study.

### 3.2 FlashAttention 2 (Dao, 2023)

**Paper:** https://arxiv.org/abs/2307.08691

|Technique|Compiler Feasibility|Priority|
|---|---|---|
|Tensor Core Utilization|High|REQUIRED|
|Reduced Non-MatMul FLOPs|Medium|SHOULD HAVE|
|Better Parallelization|Medium|SHOULD HAVE|
|Work Partitioning|Medium-Low|STRETCH|
|Register Optimization|Low|Out of scope|

**Critical:** Attempt ALL plausibly compiler-expressible optimizations. Failures are research data.

**Protocol:** Before implementing any FA2 technique:

1. Claude Code reads FA2 paper section
2. Extracts technique
3. Asks human: "I interpret this as [MLIR approach]. Does this match literature?"
4. Human confirms
5. Updates DESIGN.md
6. Implements

### 3.3 Out of Scope

FA3/FA4 techniques are reference only:

- Warp specialization (FA3) - low-level control unavailable
- Async memory ops (FA3) - limited MLIR support
- Dynamic tiling (FA4) - requires runtime compilation
- Blackwell optimizations (FA4) - too hardware-specific, too recent

Use in: Limitations section, Future Work, gap analysis.

---

## 4. MLIR Pass Specifications

### 4.1 Pass 1: Operation Fusion

**Goal:** Merge linalg.matmul → scale → mask → softmax into single operation.

**Input:**

```mlir
%qk = linalg.matmul ins(%Q, %K)
%scaled = linalg.generic { arith.divf %qk, %scale }
%masked = linalg.generic { arith.select %mask, %scaled, %neg_inf }
%probs = linalg.softmax ins(%masked)
```

**Output:**

```mlir
%probs = attention.fused ins(%Q, %K, %scale, %mask)
```

**Algorithm:**

1. Pattern match 4-op sequence
2. Verify data dependencies
3. Create fused operation
4. Replace sequence

**Tests:**

- Numerical: output matches unfused (error < 1e-5)
- Edge cases: empty sequences, all-masked, numerical stability
- Compilation: fused op verifies

**Performance Target:**

- Memory traffic: -40 to -60%
- Speedup: 1.4-1.8x vs unfused

**Justification:** FA1 core technique, standard compiler optimization, enables downstream passes.

**Commands:**

```bash
# Test pass in isolation
./build/bin/attention-opt test/fusion.mlir --fusion-pass | FileCheck test/fusion.mlir

# Run correctness tests
cd test && lit -v unit/fusion_pass.mlir
```

---

### 4.2 Pass 2: Memory-Aware Tiling

**Goal:** Insert tiling loops to fit tiles in GPU SRAM.

**Input:**

```mlir
attention.fused ins(%Q, %K : memref<1024x64xf32>, ...)
```

**Output:**

```mlir
for %tile_i = 0 to 8 {
  for %tile_j = 0 to 8 {
    %Q_tile = memref.subview %Q[%tile_i*128, 0][128, 64]
    %K_tile = memref.subview %K[%tile_j*128, 0][128, 64]
    attention.fused ins(%Q_tile, %K_tile)
  }
}
```

**Algorithm:**

1. Query target SRAM size (192KB for A100)
2. Calculate tile size: sqrt(SRAM / (3 * sizeof(float)))
3. Round to tensor core alignment (16x16)
4. Insert affine.for loops

**Tests:**

- Numerical: tiled matches untiled (error < 1e-5)
- Memory: tile fits in SRAM (analytical)
- Coverage: all elements processed once

**Performance Target:**

- SRAM utilization: >80%
- Speedup: 1.8-2.5x vs unfused+untiled

**Justification:** FA1 core technique, fundamental GPU optimization, enables efficient memory access.

**Commands:**

```bash
# Test tiling pass
./build/bin/attention-opt test/tiling.mlir --tiling-pass | FileCheck test/tiling.mlir

# Verify tile size calculation
python3 test/verify_tile_size.py --sram-kb=192
```

---

### 4.3 Pass 3: Vectorization

**Goal:** Convert scalar ops to vector ops (SIMD).

**Input:**

```mlir
for %i in range(1024) {
  %val = memref.load %input[%i]
  %result = arith.addf %val, %const
  memref.store %result, %output[%i]
}
```

**Output:**

```mlir
for %i in range(128) {
  %vec = vector.load %input[%i*8 : %i*8+8]
  %result = vector.addf %vec, %const_vec
  vector.store %result, %output[%i*8 : %i*8+8]
}
```

**Algorithm:**

1. Detect vectorizable loops (no dependencies)
2. Determine vector width (8 for f32)
3. Insert vector.load/vector.store
4. Generate remainder loop

**Tests:**

- Numerical: vectorized matches scalar
- Remainder: non-divisible sizes handled

**Performance Target:**

- Throughput: 1.5-2x vs scalar
- Bandwidth: approach peak

**Justification:** Standard optimization, necessary for competitive performance, demonstrates MLIR vector dialect.

**Commands:**

```bash
# Test vectorization
./build/bin/attention-opt test/vectorization.mlir --vectorization-pass | FileCheck test/vectorization.mlir

# Assembly check
./build/bin/attention-opt test/vectorization.mlir --vectorization-pass | mlir-translate --mlir-to-llvmir | llc -o - | grep vector
```

---

### 4.4 Pass 4: Causal Mask Specialization

**Goal:** Generate specialized kernels for different tile types in causal masking.

**Algorithm:**

1. Classify tiles:
    - Full tiles: all valid (below diagonal)
    - Masked tiles: all invalid (above diagonal)
    - Boundary tiles: partial (straddles diagonal)
2. Generate three kernel variants:
    - Full: no mask checks
    - Masked: skip computation
    - Boundary: check edges only
3. Insert dispatch logic

**Tests:**

- Numerical: matches generic masked attention
- Classification: all tiles correctly categorized
- Edge cases: square, non-square, small sizes

**Performance Target:**

- Speedup vs generic masking: 1.15-1.3x
- Reduced branch divergence (profiler verified)

**Justification:** Domain-specific optimization, demonstrates compiler can encode expert knowledge, branching kills GPU performance.

**Commands:**

```bash
# Test mask specialization
./build/bin/attention-opt test/mask.mlir --mask-specialization-pass | FileCheck test/mask.mlir

# Verify tile classification
python3 test/verify_mask_tiles.py --seq-len=1024 --tile-size=128
```

---

### 4.5 Pass 5: GPU Backend Lowering

**Goal:** Execute the compiler-built pipeline (Passes 1-4 output) on real GPU hardware. First, general-purpose GPU parallelization with no tensor cores (Stage A), to get a correct, timed baseline. Then, drive MLIR's existing Transform-dialect `linalg.matmul` -> `nvgpu.mma.sync` lowering recipe on top (Stage B), rather than hand-authoring tensor-core intrinsic insertion from scratch -- consistent with this project's own thesis of reusing compiler infrastructure instead of hand-writing kernels.

**Input:**

```mlir
// Passes 1-4 output: fused, tiled, vectorized, mask-specialized
// affine/linalg/memref IR (see Design.md 8.2.1)
```

**Output, Stage A:**

```mlir
gpu.launch blocks(...) threads(...) {
  // tiled loop nest, thread/block parallel
}
```

**Output, Stage B** (applied on top of Stage A):

```mlir
%A_frag = nvgpu.ldmatrix %A_shared_tile
%B_frag = nvgpu.ldmatrix %B_shared_tile
%C_frag = nvgpu.mma.sync %A_frag, %B_frag, %C_acc
nvgpu.stmatrix %C_frag, %C_shared_tile
```

**Algorithm:**

1. Stage A: `gpu-kernel-outlining` + `gpu.launch` wrapping of the tiled loop nest, using MLIR's existing generic GPU lowering pipeline -- no custom intrinsic-insertion logic.
2. Stage B: apply MLIR's existing Transform-dialect matmul-to-`mma.sync` recipe to the QK^T/PV `linalg.matmul` ops inside the Stage A kernel, then `-convert-nvgpu-to-nvvm` for final lowering.

**Tests:**

- Numerical: GPU execution matches the CPU/numpy reference at the same tolerance as §5.1.
- PTX: `ptxas` succeeds on the emitted PTX (both stages).
- Stage B specifically: `nvgpu.mma.sync`/`ldmatrix`/`stmatrix` appear where Stage A's output has plain `linalg.matmul`.

**Performance Target:** any measured GPU speedup vs. the CPU baseline demonstrates the pipeline works end-to-end (minimum bar). Beyond that, report the measured wall-clock speedup for Stage A and Stage B, and -- if profiler validation (§5.3 Stage 3) is pursued -- measured tensor-core utilization and occupancy. No fixed target number is required going in; consistent with this project's non-goal (§1) of beating FlashAttention-2, the measured number and the gap it reveals are the result, not a bar to clear.

**Justification:** FA2 technique, shows backend-specific optimization, demonstrates the gap between high-level fusion and hardware exploitation -- and, per the project's own thesis, tests whether that gap can be closed by driving existing compiler infrastructure rather than hand-writing it.

**Commands:** (illustrative -- exact flags depend on implementation)

```bash
# Stage A: lower and run on GPU
./build/bin/attention-opt test/Attention/gpu_lowering.mlir --gpu-lowering-pass="stage=a" \
  | mlir-translate --mlir-to-nvvmir | ptxas --gpu-name=sm_89 -o kernel.ptx

# Stage B: same, with tensor-core lowering applied
./build/bin/attention-opt test/Attention/gpu_lowering.mlir --gpu-lowering-pass="stage=b" \
  | mlir-translate --mlir-to-nvvmir | ptxas --gpu-name=sm_89 -o kernel.ptx
grep "mma.sync" kernel.ptx
```

---

### 4.6 Pass 6: Work Distribution (FA2)

**Goal:** Improve thread block dimensions and work partitioning.

**Algorithm:**

1. Analyze workload (sequence length, head dim)
2. Compute optimal block dimensions:
    - SM count
    - Register pressure
    - SRAM capacity
3. Generate backend hints

**Tests:**

- Numerical: output unchanged
- Resources: no register spills (profiler)

**Performance Target:**

- GPU occupancy: >75%
- Speedup: 1.1-1.2x vs naive

**Priority:** STRETCH GOAL

**Justification:** FA2 technique, high risk (may not be expressible at MLIR level), attempt if core passes succeed.

**Commands:**

```bash
# Test work distribution
./build/bin/attention-opt test/work_dist.mlir --work-distribution-pass | FileCheck test/work_dist.mlir

# Profile occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./kernel
```

---

## 5. Testing Protocol

### 5.1 Correctness Validation

**Numerical Validation:**

```python
def validate(mlir_output, pytorch_output):
    max_error = np.max(np.abs(mlir_output - pytorch_output))
    mean_error = np.mean(np.abs(mlir_output - pytorch_output))
    within_tol = np.sum(np.abs(mlir_output - pytorch_output) < 1e-5) / mlir_output.size
    
    assert max_error < 1e-5
    assert mean_error < 1e-6
    assert within_tol > 0.999
```

**Test Cases:**

- Sequence lengths: [128, 256, 512, 1024, 2048, 4096]
- Batch sizes: [1, 4, 8, 16, 32]
- Head dimensions: [64, 128]
- Edge cases: empty, all-masked, large values

**Commands:**

```bash
# Run all correctness tests
cd test && lit -v numerical/

# Run specific configuration
python3 test/numerical/validate.py --seq-len=1024 --batch=16 --head-dim=64
```

### 5.2 CPU Validation (Pre-Hardware Checkpoint)

**Build and run:**

```bash
./build/bin/attention-opt test.mlir --fusion-pass --tiling-pass | \
  mlir-cpu-runner -e main -entry-point-result=void
```

**Profile:**

```bash
perf stat -e cycles,instructions,cache-misses ./cpu_executable
```

**Acceptance Criteria:**

- Executes without errors
- Numerical correctness validated
- Speedup vs unfused: >1.2x

**If fails:** STOP. Do not proceed to GPU. Debug CPU first.

### 5.3 GPU Hardware Testing

**Hardware:** RTX 4090 (primary -- correctness and wall-clock validation, Stage 2), A100 (secondary -- profiler-verified metrics only, Stage 3, optional). Split across two providers because profiler access (`ncu`) requires a full VM/bare-metal host; containerized GPU rental platforms structurally block it regardless of in-container root access (`ERR_NVGPUCTRPERM`). Wall-clock timing, which answers this project's actual performance-recovery question (§1), has no such restriction and runs on either.

**Measurement:** in-process timing via `rtclock()`/`printF64()` inside the compiled kernel, matching §5.2's CPU methodology -- bracketing repeated calls after an untimed warmup loop, so process startup and JIT-compile time are excluded.

**Statistical Requirements:**

- Report median (not mean)
- Report standard deviation
- Flag variance >5% as measurement issue

**Provenance:** each result is recorded with `nvidia-smi` output (GPU model, driver version), the exact commit hash, and the raw execution log alongside the extracted number -- necessary for a result collected on rented, ephemeral hardware to be checkable rather than merely asserted.

**Commands:**

```bash
# Stage 2: collect wall-clock results on the GPU instance
python3 benchmarks/analyze_ablation.py --collect --hardware "RTX 4090 (RunPod)"

# pull results down, then plot locally (no GPU needed)
python3 benchmarks/analyze_ablation.py --plot

# Stage 3 (optional): profiler validation, full VM host only
ncu --set full --export profile.ncu-rep <kernel invocation>
ncu --import profile.ncu-rep --page details
```

### 5.4 Go/No-Go Criteria

**Stage 2 (required) -- PROCEED if:**

- Correctness: all tests pass (error < 1e-5)
- Functionality: no crashes/hangs
- A measured wall-clock speedup is obtained and recorded, in either direction -- this demonstrates the pipeline runs end-to-end on real hardware. Per this project's non-goal (§1) of beating FlashAttention-2, the number itself is the result, not a pass/fail bar.

**Stage 2 -- STOP if:**

- Correctness fails
- Execution crashes or hangs across all attempted shapes

**Stage 3 (optional) -- if pursued:**

- Profiler metrics (tensor core utilization, occupancy, memory bandwidth -- §7.2) are measured and recorded alongside Stage 2's wall-clock numbers. Reference points from production tensor-core kernels (>70% utilization, >75% occupancy) are useful context for the gap analysis, not a bar this pass must clear.

---

## 6. Baselines and Comparisons

### 6.1 Baseline 1: Unfused PyTorch

```python
def unfused_attention(Q, K, V, mask):
    qk = torch.matmul(Q, K.transpose(-2, -1))
    scaled = qk / math.sqrt(d_k)
    masked = scaled.masked_fill(mask, float('-inf'))
    probs = F.softmax(masked, dim=-1)
    return torch.matmul(probs, V)
```

**Purpose:** Worst-case lower bound

**Command:**

```bash
python3 benchmarks/baselines/unfused_pytorch.py --seq-len=1024 --batch=16
```

### 6.2 Baseline 2: torch.compile (Triton)

```python
@torch.compile
def compiled_attention(Q, K, V, mask):
    return unfused_attention(Q, K, V, mask)
```

**Purpose:** State-of-art automatic fusion

**Expected:** Our competitive target (within 20-30%)

**Command:**

```bash
python3 benchmarks/baselines/torch_compile.py --seq-len=1024 --batch=16
```

### 6.3 Baseline 3: FlashAttention-2

```python
from flash_attn import flash_attn_func
output = flash_attn_func(Q, K, V, causal=True)
```

**Purpose:** Hand-optimized upper bound

**Expected:** We won't beat this

**Command:**

```bash
python3 benchmarks/baselines/flash_attn2.py --seq-len=1024 --batch=16
```

### 6.4 Ablation Study

**What is Ablation?**

Ablation means testing each pass independently to measure its contribution. We progressively enable passes and measure the performance delta each adds.

**Configurations:**

|Config|Passes Enabled|Purpose|
|---|---|---|
|Unfused|None|Baseline|
|Fusion+Tiling|Fusion + Tiling|Smallest executable unit -- Fusion alone produces `attention.fused`, which Tiling is what expands into runnable IR (Design.md §4); they can't be measured independently|
|+ Vectorization|+ Vectorization|Measure vectorization's contribution|
|+ Mask Specialization|+ Mask Specialization|Measure mask specialization's contribution|
|+ GPU (Stage A)|+ GPU, no tensor cores (§4.5)|Measure the cost/benefit of moving to GPU execution|
|+ GPU (Stage B)|+ tensor cores (§4.5)|Measure tensor-core lowering's contribution|

**Why This Matters:**

This ablation study is the research contribution. It answers:

- Which passes contribut most to performance?
- What speedup is achievable with compilers?
- Where is the performance gap vs hand-tuned code?

**Current Measured Results** (CPU, local -- `benchmarks/analyze_ablation.py`, raw data in `results/ablation.csv`):

|Configuration|Speedup vs. Unfused|
|---|---|
|Fusion+Tiling|1.41x|
|+ Vectorization|4.85x|
|+ Mask Specialization|7.53x|
|+ GPU (Stage A)|pending Stage 2|
|+ GPU (Stage B)|pending Stage 2/3|

Measured at seq=32x32, head_dim=16, tile=8, causal mask -- the shape small enough that Vectorization's JIT-compile-time ceiling (Design.md §5.3) doesn't apply to any ladder step. Not directly comparable to Passes 1-2's own production-scale benchmark (Design.md §8.2.1, 1.36x-1.49x) since that uses a different shape; this table measures each pass's cumulative contribution at one fixed shape, not absolute production-scale performance.

**Analysis:** Vectorization contributes the largest single delta measured so far (3.44x, from 1.41x to 4.85x); Mask Specialization adds a further 1.55x on top. The GPU rows are what Stage 2/3 will fill in -- comparing the compiler-recovered GPU number against torch.compile and FlashAttention-2 (§6.2-6.3) is the actual gap analysis this project is aiming to produce.

**Commands:**

```bash
# Run full ablation study
python3 benchmarks/ablation.py --seq-len=1024 --batch=16 --all-configs

# Run specific configuration
python3 benchmarks/ablation.py --seq-len=1024 --batch=16 --config=fusion_only
python3 benchmarks/ablation.py --seq-len=1024 --batch=16 --config=fusion_tiling

# Generate comparison table
python3 benchmarks/analyze_ablation.py --output=results/ablation.csv
```

---

## 7. Performance Metrics

### 7.1 Primary Metrics

**Throughput (tokens/sec):**

```python
throughput = (batch_size * seq_len) / median_time
```

**Memory Bandwidth (GB/s):**

```python
bytes_moved = (Q.numel() + K.numel() + V.numel() + output.numel()) * 4
bandwidth = bytes_moved / median_time
```

**TFLOPS:**

```python
flops = 2 * batch * seq * seq * head_dim
tflops = flops / (median_time * 1e12)
```

**Speedup:**

```python
speedup = baseline_time / our_time
```

### 7.2 Profiler Metrics (Stage 3, optional -- see §5.3)

```bash
ncu --set full --export profile.ncu-rep ./mlir_attention
```

Requires a full VM/bare-metal host (§5.3) -- not available on containerized GPU rental platforms. If pursued:

**Analyze:**

- Memory bandwidth utilization (reference: >80% on production tensor-core kernels)
- Tensor core utilization (reference: >70%)
- SM occupancy (reference: >75%)
- L2 cache hit rate
- Register spills (reference: 0)

These are reference points from production kernels for context, not a bar this pass must clear (§5.4).

**Commands:**

```bash
# Profile memory bandwidth
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./kernel

# Profile tensor cores
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./kernel

# Profile occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./kernel
```

---

## 8. Deliverables

### Code Artifacts

- Working MLIR passes (fusion, tiling, vectorization, mask, GPU lowering)
- Test suite (>90% coverage)
- Benchmark harness
- Documentation

### Experimental Data

- Baseline measurements (PyTorch unfused, torch.compile, FA2)
- Ablation study results
- Profiler data (bandwidth, tensor cores, occupancy)

### Analysis

- Expressiveness taxonomy (what works, what doesn't)
- Performance gap quantification
- Design recommendations for MLIR
- Limitations and future work

### Documentation

- Research report (10-15 pages)
- TRADEOFFS.md (complete)
- Pass documentation (docs/passes/)
