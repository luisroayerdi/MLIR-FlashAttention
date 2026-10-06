//===- AttentionOps.td - Attention dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "Attention/AttentionTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Fused multi-head attention operation.
// Computes: output = softmax(Q @ K^T * scale [+ mask]) @ V
// All operands are 2-D f32 memrefs (one head, one batch element).
// Writes to `output` in place; has no SSA result.
def Attention_FusedOp : Attention_Op<"fused", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Fused attention: output = softmax(Q @ K^T * scale [+ mask]) @ V";
    let description = [{
        Computes the complete attention operation as a single fused op.
        Q, K, V are 2-D memrefs [seq x head_dim]. The mask (optional) is a
        2-D boolean memref [seq_q x seq_k] where true means "mask this position
        to -inf".  Output is written in place to the `output` memref.

        Including V in the fused op is required for the tiling pass to introduce
        online softmax without materialising the full seq_q x seq_k attention
        weight matrix (see docs/DECISIONS.md, Dialect).
    }];

    let arguments = (ins
        MemRefOf<[F32]>:$Q,              // [seq_q, head_dim]
        MemRefOf<[F32]>:$K,              // [seq_k, head_dim]
        MemRefOf<[F32]>:$V,              // [seq_k, head_dim]
        F32:$scale,                      // 1/sqrt(head_dim), runtime value
        Optional<MemRefOf<[I1]>>:$mask,  // [seq_q, seq_k]; absent = no masking
        MemRefOf<[F32]>:$output          // [seq_q, head_dim]; written in place
    );
    // No SSA results — pure side-effecting write to output memref.

    let assemblyFormat = [{
        `ins` `(` $Q `,` $K `,` $V `:` type($Q) `,` type($K) `,` type($V) `)`
        `scale` `(` $scale `:` type($scale) `)`
        (`mask` `(` $mask^ `:` type($mask) `)`)?
        `outs` `(` $output `:` type($output) `)`
        attr-dict
    }];

    let hasVerifier = 1;
}

#endif // STANDALONE_OPS
