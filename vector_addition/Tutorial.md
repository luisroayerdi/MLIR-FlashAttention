Vector addition is the first successful example of using an in-tree MLIR dialect and MLIR passes to perform a vector addition in parallel. We utilize the 'linalg' dialect to create the high-level IR and from there we used multiple MLIR passes to get an executable, including OpenMP to support shared memory parallelism for the vector addition.
## Basic Definitions
### MLIR
[MLIR](https://mlir.llvm.org/) is a framework for building reusable and extensible compiler infrastructure build from the [LLVM](https://llvm.org/). Unlike traditional compilers that use a single, fixed set of instructions, MLIR allows you to define your own "languages" at different levels of abstraction—from high-level math to low-level machine code all within the same system.
### MLIR Dialects
A [**Dialect**](https://mlir.llvm.org/docs/Dialects/) is a self-contained "vocabulary" in MLIR. It groups together related types, operations, and attributes.
### MLIR ops
**Ops** are the fundamental units of code in MLIR. Everything is an operation, from a simple addition to a complex neural network layer or even a function definition.
- Unlike instructions in other compilers, MLIR ops are **extensible**, you can define new ones whenever you need a specific semantic that doesn't exist yet.
### MLIR passes
A **Pass** is a specific transformation or analysis performed on the MLIR code.
- **Transformation Passes:** Convert code from one dialect to another (Lowering) or optimize code within a dialect.
- **Pipeline:** Passes are typically chained together to move code from high-level "intent" down to executable binary.
### Linalg Dialect and `linalg.generic`
The **Linalg** (Linear Algebra) dialect is designed for high-level data-parallel operations on tensors or buffers. Its genius lies in being "structure-aware", meaning it understands how data is accessed in loops without explicitly writing the loops yet.
#### `linalg.generic`
The most general operations within the `linalg` dialect. It uses **indexing maps** and **iterators** to describe _how_ to loop over data and _what_ math to perform, allowing you to represent almost any tensor operation in a way that the compiler can easily optimize for hardware.

## Build MLIR in IOS
Before we leverage MLIR to performa vector addition we need to make sure we have llvm installed and linked to out project folder

To install LLVM run:
``` bash
brew install llvm
```

To add the installation to our path run:
- `export PATH="$(brew --prefix llvm)/bin:$PATH"` Adds a temporary Path to your llvm installation
-  Verify installation: `mlir-opt --version` (Should print version number)

## High level IR `vec_add.mlir`
We are gonna leverage `linalg.generic` to generate the IR to represent our vector addition, this is the starting point of this example and its "generality" its what allow the compiler to perform optimizations on later passes.

```mlir
func.func @parallel_vector_add(
  %arg0: memref<1024xf32>, 
  %arg1: memref<1024xf32>, 
  %arg2: memref<1024xf32>) {
  // This is the "High Level" parallel intent
  linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>, // Access arg0[d0]
      affine_map<(d0) -> (d0)>, // Access arg1[d0]
      affine_map<(d0) -> (d0)>  // Access arg2[d0]
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : memref<1024xf32>, memref<1024xf32>)
    outs(%arg2 : memref<1024xf32>) {
    ^bb0(%in1: f32, %in2: f32, %out: f32):
      %0 = arith.addf %in1, %in2 : f32
      linalg.yield %0 : f32
  }
  return
}
```

This code defines a function that performs a **pointwise vector addition** ($C = A + B$) using the structural power of MLIR. Instead of writing a manual `for` loop, it describes the _intent_ of the operation.

### **The Components**

- **`func.func`**: Defines the function `parallel_vector_add`. It takes three `memref` arguments (pointers to memory buffers) of 1024 floating-point numbers (`f32`).
    
- **`linalg.generic`**: The core operation. It coordinates how the data in the buffers is accessed and processed.
    
- **`indexing_maps`**: These define the "shape" of the computation. `(d0) -> (d0)` means that for every index `d0`, we access the exact same index in all three buffers (a simple 1-to-1 mapping).
    
- **`iterator_types = ["parallel"]`**: Tells the compiler that every iteration is independent. This is a huge hint that the compiler can safely run this on multiple CPU cores or a GPU without worrying about data conflicts.
    
- **`ins` / `outs`**: Explicitly labels which buffers are inputs (`%arg0`, `%arg1`) and which is the output (`%arg2`).
    
- **The Region (`^bb0`)**: This is the "inner loop body." It pulls one element from each input (`%in1`, `%in2`), adds them together using `arith.addf`, and "yields" (returns) the result to be stored in the output buffer.
## Lower from HIGH Level IR to LLVM IR
The following command runs 8 MLIR passes to lower-down `vec_add.mlir` to an IR that is ready to get translated into LLVM IR, we name it: `vec_add_llvm.mlir`.

``` bash
mlir-opt vec_add.mlir \                    
  --convert-linalg-to-parallel-loops \
  --convert-scf-to-openmp \
  --convert-openmp-to-llvm \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts \
  -o vec_add_llvm.mlir
```

From here we run: 

```bash
mlir-translate --mlir-to-llvmir vec_add_llvm.mlir -o vec_add.ll
```

This generates the LLVM IR in `vec_add.ll`

## LLVM IR to Executable
`vec_add.ll` can now be compiled by the clang compiler, we just now need to create a C file to serve as our entry point and populate the arrays we want to perform addition on. We run the following command to compile:

```bash
clang -fopenmp main.c vec_add.ll -o vec_add
```

We run to perform the vector addition:
`./vec_add`

## Overview of MLIR Passes

| Pass                                | Dialect in -> Dialect out | Description                                                                                     |
| ----------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------- |
| `–convert-linalg-to-parallel-loops` | linalg->scf               | Converts abstract operations (linalg.generic)<br>into explicit parallel loops (scf.parallel).   |
| `-convert-scf-to-OpenMP`            | scf->omp                  | Converts abstract operations (linalg.generic)<br>into explicit parallel loops (scf.parallel).   |
| `–convert-openmp-to-llvm`           | omp->llvm                 | Converts parallel loops (scf.parallel)<br>into OpenMP constructs (omp.parallel,<br>omp.wsloop). |
| `-convert-scf-to-cf`                | scf->cf                   | Converts OpenMP ops into LLVM dialect ops<br>and runtime library calls.                         |
| `convert-arith-to-llvm`             | arith->llvm               | Converts structured control flow (for/while<br>loops) into basic blocks and branches.           |
| `-convert-func-to-llvm`             | func->llvm                | Converts arithmetic operations (arith.addf)<br>into LLVM arithmetic (llvm.fadd).                |
| `-finalize-meref-to-llvm`           | memref->llvm              | Converts function signatures and calls to<br>LLVM function format.                              |
| `-reconcile-unrealized-casts`       | (cleanup)                 | Removes temporary type conversion artifacts<br>from previous passes.                            |

