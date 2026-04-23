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

### 4.5 Pass 5: GPU Backend Lowering (Tensor Cores)

**Goal:** Lower to nvgpu dialect for tensor core utilization.

**Input:**

```mlir
linalg.matmul ins(%A, %B : memref<16x16xf32>) outs(%C)
```

**Output:**

```mlir
%A_frag = nvgpu.ldmatrix %A
%B_frag = nvgpu.ldmatrix %B
%C_frag = nvgpu.mma %A_frag, %B_frag, %C_acc
nvgpu.stmatrix %C_frag, %C
```

**Algorithm:**

1. Detect matmul candidates (16x16 aligned)
2. Insert layout transformations
3. Replace linalg.matmul with nvgpu.mma
4. Insert load/store operations

**Tests:**

- Numerical: tensor core matches standard matmul
- Alignment: verify 16x16
- PTX: verify mma.sync instructions

**Performance Target:**

- Tensor core utilization: >70% (profiler)
- Speedup vs standard matmul: 8-12x
- TFLOPS: 70-80% of theoretical peak

**Justification:** FA2 technique, shows backend-specific optimization, demonstrates gap between high-level fusion and hardware exploitation.

**Commands:**

```bash
# Test GPU lowering
./build/bin/attention-opt test/gpu_lowering.mlir --gpu-lowering-pass | FileCheck test/gpu_lowering.mlir

# Verify PTX generation
./build/bin/attention-opt test/gpu_lowering.mlir --gpu-lowering-pass | mlir-translate --mlir-to-nvvmir | ptxas --gpu-name=sm_80 -o kernel.ptx
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

### 5.3 GPU Hardware Testing (MAIN MILESTONE)

**Hardware:** NVIDIA A100 (primary), H100 (secondary), RTX 4090 (fallback)

**Warmup:**

```python
for _ in range(10):
    run_kernel(Q, K, V)
torch.cuda.synchronize()
```

**Measurement:**

```python
times = []
for _ in range(100):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    run_kernel(Q, K, V)
    end.record()
    
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

median_time = np.median(times)
std_dev = np.std(times)
```

**Statistical Requirements:**

- Report median (not mean)
- Report standard deviation
- Flag variance >5% as measurement issue

**Commands:**

```bash
# Run GPU benchmark
python3 benchmarks/mlir/run_gpu.py --seq-len=1024 --batch=16 --iterations=100

# Profile with nsight
ncu --set full --export profile.ncu-rep python3 benchmarks/mlir/run_gpu.py

# Analyze profile
ncu --import profile.ncu-rep --page details
```

### 5.4 Go/No-Go Criteria

**PROCEED if:**

- Correctness: all tests pass (error < 1e-5)
- Functionality: no crashes/hangs
- Performance: >1.5x speedup vs unfused
- Profiler: tensor cores used (if Pass 5 enabled)

**STOP if:**

- Correctness fails
- Performance <1.2x vs unfused
- Profiler shows 0% tensor core usage, excessive memory traffic

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
|+Fusion|Fusion only|Measure fusion impact|
|+Tiling|Fusion + Tiling|Measure tiling impact|
|+Vector|Fusion + Tiling + Vectorization|Measure vectorization|
|+GPU|All + Tensor cores|Full pipeline|

**Why This Matters:**

This ablation study is the research contribution. It answers:

- Which passes contribute most to performance?
- What speedup is achievable with compilers?
- Where is the performance gap vs hand-tuned code?

**Example Results Table:**

|Configuration|Memory Traffic|Speedup|Delta|
|---|---|---|---|
|Unfused|100%|1.0x|-|
|+Fusion|60%|1.5x|+0.5x|
|+Tiling|45%|2.0x|+0.5x|
|+Vector|40%|2.5x|+0.5x|
|+GPU|35%|3.0x|+0.5x|

**Analysis from this table:**

- Fusion contributes 0.5x speedup (largest single contribution)
- Tiling contributes 0.5x speedup
- Vectorization contributes 0.5x speedup
- Tensor cores contribute 0.5x speedup
- Total compiler approach: 3.0x
- FA2 reference: 4.5x
- Gap: 1.5x remains unexplained

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

### 7.2 Profiler Metrics

```bash
ncu --set full --export profile.ncu-rep ./mlir_attention
```

**Analyze:**

- Memory bandwidth utilization (target: >80%)
- Tensor core utilization (target: >70%)
- SM occupancy (target: >75%)
- L2 cache hit rate
- Register spills (target: 0)

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

## 8. Documentation Structure

```
MLIR-Scheduling-Kernel/
├── README.md                # Quick start, build instructions
├── REQUIREMENTS.md          # This document
├── DESIGN.md                # Created by Claude Code BEFORE implementing
├── TRADEOFFS.md             # Updated continuously during development
├── docs/
│   ├── passes/
│   │   ├── fusion.md        # Pass 1 technical details
│   │   ├── tiling.md        # Pass 2 technical details
│   │   └── ...
│   ├── benchmarking.md      # How to run benchmarks
│   └── profiling.md         # Profiler guide
├── include/Attention/       # Headers
├── lib/Attention/           # Implementation
├── test/                    # Tests
├── benchmarks/              # Benchmark scripts
│   ├── baselines/           # PyTorch, torch.compile, FA2
│   ├── mlir/                # Our pipeline
│   └── analysis/            # Result plotting
└── results/                 # Experimental data (gitignored)
```

### 8.1 Documentation Ownership

**README.md** (User-facing):

- Quick start
- Build instructions
- How to run tests
- Links to docs

**DESIGN.md** (Created by Claude Code):

- CRITICAL: Create FIRST before implementing
- One section per pass
- Algorithm pseudocode
- IR transformation examples
- Rationale for design choices
- Known limitations
- Updated as design evolves

**TRADEOFFS.md** (Living document):

- Updated continuously during development
- Documents every performance/correctness/complexity tradeoff
- Format:
    
    ```markdown
    ## Tradeoff: Tile Size Selection**Decision:** Static 128×128 tiles**Alternatives:** Dynamic sizing, multiple fixed sizes**Chosen Because:**- Simplicity- Optimal for A100 SRAM (192KB)**Cost:**- Not optimal for H100- Future: configurable via pass option**Measurement:**- SRAM utilization: 85% on A100- Would be 60% on H100
    ```
    

**docs/passes/*.md** (Technical):

- Implementation details
- Edge cases
- Testing strategy
- Future improvements

---

## 9. Development Workflow

### 9.1 Process Flow

```
Requirements (this document)
    ↓
Design Document (Claude Code creates DESIGN.md)
    ↓ (human reviews and approves)
Implementation (pass by pass)
    ↓
    For each pass:
        1. Claude Code implements
        2. Human reviews code
        3. Claude Code writes tests
        4. Human runs tests, verifies
        5. Claude Code updates TRADEOFFS.md
        6. Human approves and merges
    ↓
Integration Testing
    ↓
CPU Validation
    ↓ (if pass → proceed, if fail → debug)
GPU Hardware Testing (MILESTONE)
    ↓
Benchmarking & Analysis
    ↓
Paper/Report Writing
```

### 9.2 Pass Implementation Order

**Phase 1: Core FA1 Passes**

- Pass 1: Fusion
- Pass 2: Tiling
- Pass 3: Vectorization
- Pass 4: Mask Specialization

**Phase 2: Integration & CPU Testing**

- Integration tests
- CPU benchmarks
- CHECKPOINT: CPU validation must pass

**Phase 3: GPU Lowering (FA2)**

- Pass 5: Tensor cores
- GPU profiler validation

**Phase 4: Optional Extensions**

- Pass 6: Work distribution (if time)

**Phase 5: Final Benchmarking**

- Full ablation study
- All baseline comparisons
- Gap analysis

### 9.3 Claude Code Protocol

**Before implementing any pass:**

1. Update DESIGN.md with proposed design
2. Wait for human approval
3. Implement the pass
4. Update TRADEOFFS.md with decisions made
5. Create docs/passes/{pass_name}.md

**Claude Code should ask:**

- "I'm implementing tiling with 128×128 tiles. Design doc updated. Approve?"
- "Found tradeoff: static vs dynamic tiling. Document now or after measurement?"

**After implementing any feature, Claude Code provides:**

- Exact commands to build
- Exact commands to test
- Exact commands to run
- Expected output

**Example:**

```
Claude Code: "I've implemented fusion pass. Here's how to test:

# Build
cd build && ninja

# Test pass in isolation
./bin/attention-opt ../test/fusion.mlir --fusion-pass | FileCheck ../test/fusion.mlir

# Run correctness tests
cd ../test && lit -v unit/fusion_pass.mlir

Expected output: All tests pass, no CHECK failures."
```

---

## 10. FA2 Implementation Protocol

Before implementing ANY FA2 optimization:

1. Claude Code reads FA2 paper section
2. Extracts key technique
3. Asks human: "FA2 describes {technique}. I interpret this as {MLIR approach}. Does this match literature?"
4. Human confirms or corrects
5. Claude Code updates DESIGN.md
6. Claude Code implements

**Example:**

"FA2 Section 3.1 describes 'reducing non-matmul FLOPs by avoiding redundant rescaling in softmax'. I interpret this as: compute exp(x) * scale where scale absorbs both max and sum.

Should I implement as: A) Pattern rewrite on linalg.softmax B) Part of fusion pass C) Separate optimization pass?"

---

## 11. Commands Reference

### Build

```bash
cd build
cmake .. -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir -G Ninja
ninja
```

### Test Single Pass

```bash
./build/bin/attention-opt test/{pass}.mlir --{pass-name}-pass | FileCheck test/{pass}.mlir
```

### Run All Tests

```bash
cd test && lit -v .
```

### CPU Execution

```bash
./build/bin/attention-opt test.mlir --fusion-pass --tiling-pass | \
  mlir-cpu-runner -e main -entry-point-result=void
```

### GPU Profiling

```bash
# Full profile
ncu --set full --export profile.ncu-rep python3 benchmarks/mlir/run_gpu.py

# Specific metrics
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./kernel
```

### Benchmarking

```bash
# Single baseline
python3 benchmarks/baselines/unfused_pytorch.py --seq-len=1024 --batch=16

# Full ablation
python3 benchmarks/ablation.py --seq-len=1024 --batch=16 --all-configs

# Generate results
python3 benchmarks/analyze_ablation.py --output=results/ablation.csv
```

---

## 12. Deliverables

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
