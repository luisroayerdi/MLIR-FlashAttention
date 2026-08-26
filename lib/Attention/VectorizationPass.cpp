//===- VectorizationPass.cpp - Attention vectorization pass -----*- C++ -*-===//
//
// Pass 3: Vectorization
//
// Converts the scalar linalg.generic / linalg.fill / memref.copy ops that
// TilingPass emits inside each tile body into vector-dialect form, using
// MLIR's built-in linalg vectorizer (linalg::vectorize / linalg::vectorizeCopy).
// Each op is vectorized over its own full static iteration-space shape (no
// manual VF chunking / remainder loop) — decomposition to hardware-width
// vectors is left to downstream convert-vector-to-scf / convert-vector-to-llvm
// passes.
//
//===----------------------------------------------------------------------===//

#include "Attention/AttentionPasses.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::attention {
#define GEN_PASS_DEF_VECTORIZATIONPASS
#include "Attention/AttentionPasses.h.inc"

namespace {

using llvm::SmallVector;

// memref<...xi1> stores each bool as a full byte, but the vector dialect's
// i1 vector type lowers (via --convert-vector-to-llvm) to a *bit-packed*
// llvm.load/store. Vectorizing an op with an i1 memref operand therefore
// reads/writes the wrong bytes — found empirically via the masked-attention
// numerical validation suite (see TRADEOFFS.md "Vectorization pass: i1 (mask)
// memrefs are not vectorized"). Leave such ops scalar.
static bool hasI1MemRefOperand(Operation *op) {
  return llvm::any_of(op->getOperands(), [](Value v) {
    auto memrefTy = dyn_cast<MemRefType>(v.getType());
    return memrefTy && memrefTy.getElementType().isInteger(1);
  });
}

struct VectorizationPassImpl
    : public impl::VectorizationPassBase<VectorizationPassImpl> {
  using impl::VectorizationPassBase<
      VectorizationPassImpl>::VectorizationPassBase;

  // This pass creates vector-dialect ops; the vector dialect must be loaded
  // before the pass runs (see TRADEOFFS.md: "Pass implementations declare
  // getDependentDialects").
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());

    // Collect targets up front. Vectorizing one op replaces/erases only that
    // op (and creates new ops around it) — it never touches sibling ops
    // still queued here — so collecting first and mutating after is safe.
    SmallVector<Operation *> targets;
    func.walk([&](Operation *op) {
      if (isa<linalg::GenericOp, linalg::FillOp, memref::CopyOp>(op))
        targets.push_back(op);
    });

    for (Operation *op : targets) {
      if (hasI1MemRefOperand(op))
        continue;
      rewriter.setInsertionPoint(op);
      if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
        // Best-effort: leave the op alone if it can't be vectorized (e.g.
        // non-static shape).
        (void)linalg::vectorizeCopy(rewriter, copyOp);
        continue;
      }
      if (!linalg::hasVectorizationImpl(op))
        continue;
      // linalg::vectorize() builds the vector.transfer_read/...write
      // replacement but does NOT erase/replace the original op itself —
      // that is the caller's responsibility (mirrored from how the
      // transform-dialect VectorizeOp uses this same API). For a buffer
      // (memref) DPS op the op has no SSA results, so `replacements` is
      // empty and replaceOp degrades to a plain erase.
      FailureOr<linalg::VectorizationResult> result =
          linalg::vectorize(rewriter, op);
      if (succeeded(result))
        rewriter.replaceOp(op, result->replacements);
    }
  }
};

} // namespace
} // namespace mlir::attention
