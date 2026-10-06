# MLIR-FlashAttention

MLIR-FlashAttention establishes, with evidence, what an MLIR compiler needs
to know about attention to generate FlashAttention-quality kernels.

[MLIR]-based compilers optimize attention in different ways. [IREE]
represents it as a named operator with an online-softmax decomposition,
[Triton] leaves the algorithm to the kernel author, and upstream MLIR has no
attention operator. This project builds [FlashAttention]'s optimizations
(fusion, online-softmax tiling, vectorization, mask specialization, GPU
mapping) as separate passes on upstream dialects. It uses them to measure
what each transformation is worth, what information it needs from the
representation, and how the result compares with IREE, Triton, [TVM], and
the FlashAttention and [cuDNN] kernels. Findings are reported to the
upstream MLIR community.

[MLIR]: https://mlir.llvm.org
[IREE]: https://iree.dev
[Triton]: https://triton-lang.org
[FlashAttention]: https://github.com/Dao-AILab/flash-attention
[TVM]: https://tvm.apache.org
[cuDNN]: https://developer.nvidia.com/cudnn

The goal, scope, and feature status are in [`SPECS.md`](SPECS.md).

## Build

Requires LLVM/MLIR built from source, CMake, and Ninja.

```bash
cmake -S . -B build -G Ninja -DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir
ninja -C build attention-opt
```

`$LLVM_BUILD` is the LLVM build directory. The driver is
`build/bin/attention-opt`.

## Test

FileCheck tests check the IR each pass produces. Each file in
`test/Attention/` lists its command in a `RUN:` line, for example:

```bash
build/bin/attention-opt test/Attention/fusion.mlir --fusion-pass \
  | $LLVM_BUILD/bin/FileCheck test/Attention/fusion.mlir
```

The numerical harness compiles the pipeline output, runs it with
`mlir-runner`, and compares the result with a NumPy reference:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r test/numerical/requirements.txt
python3 test/numerical/validate.py --suite
```

Tool paths are discovered from `build/CMakeCache.txt`.

## Repository layout

| Path | Contents |
|---|---|
| `include/Attention/`, `lib/Attention/` | Dialect and passes |
| `attention-opt/` | Compiler driver |
| `test/Attention/` | FileCheck tests |
| `test/numerical/` | Correctness and timing harness |
| `benchmarks/` | Analysis scripts |
| `scripts/` | Machine setup |
| `docs/` | Literature, design decisions, Phase 1 design |

## Documents

- [`SPECS.md`](SPECS.md): research questions, scope, and feature status.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): workflow, pull request format, and
  coding standards.
- [`docs/LITERATURE.md`](docs/LITERATURE.md): related systems and papers.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): design decisions and their
  reasons.
