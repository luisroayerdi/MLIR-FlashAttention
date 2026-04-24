//===- AttentionPasses.h - Attention passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef STANDALONE_STANDALONEPASSES_H
#define STANDALONE_STANDALONEPASSES_H

#include "Attention/AttentionDialect.h"
#include "Attention/AttentionOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace attention {
#define GEN_PASS_DECL
#include "Attention/AttentionPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Attention/AttentionPasses.h.inc"
} // namespace attention
} // namespace mlir

#endif
