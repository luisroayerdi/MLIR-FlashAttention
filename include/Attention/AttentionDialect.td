//===- AttentionDialect.td - Attention dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_DIALECT
#define STANDALONE_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Attention dialect definition.
//===----------------------------------------------------------------------===//

def Attention_Dialect : Dialect {
    let name = "attention";
    let summary = "A attention out-of-tree MLIR dialect.";
    let description = [{
        Attention dialect: one op, attention.fused, marking a recognized attention computation
    }];
    let cppNamespace = "::mlir::attention";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base attention operation definition.
//===----------------------------------------------------------------------===//

class Attention_Op<string mnemonic, list<Trait> traits = []> :
        Op<Attention_Dialect, mnemonic, traits>;

#endif // STANDALONE_DIALECT
