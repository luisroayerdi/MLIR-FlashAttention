//===- FusionPass.cpp - Attention fusion pass --------------------*- C++ -*-===//
//
// Pass 1: Operation Fusion
//
// Matches the 5-op unfused attention sequence and replaces it with
// attention.fused.  The pattern is anchored on the final linalg.matmul
// (the PV accumulation) and traced backward through SSA/buffer use-def chains.
//
// Input pattern (memref-based, all allocations visible as SSA values):
//   linalg.generic  ins(%Q, %K)          outs(%qk)    // QK^T
//   linalg.generic  ins(%qk)             outs(%sc)    // scale
//   linalg.generic  ins(%sc, %mask)      outs(%mk)    // mask  (optional)
//   linalg.softmax  ins(%mk)             outs(%p)     // softmax
//   linalg.matmul   ins(%p, %V)          outs(%out)   // PV  ← anchor
//
// Output:
//   attention.fused ins(%Q,%K,%V) scale(%s) [mask(%mask)] outs(%out)
//
//===----------------------------------------------------------------------===//

#include "Attention/AttentionOps.h"
#include "Attention/AttentionPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::attention {
#define GEN_PASS_DEF_FUSIONPASS
#include "Attention/AttentionPasses.h.inc"

namespace {

// ── helpers ────────────────────────────────────────────────────────────────

// Return the unique linalg.softmax that has `buf` as its DPS init (output),
// or nullptr if none / ambiguous.
static linalg::SoftmaxOp findSoftmaxWriterOf(Value buf) {
  linalg::SoftmaxOp found;
  for (Operation *user : buf.getUsers()) {
    auto softmax = dyn_cast<linalg::SoftmaxOp>(user);
    if (!softmax)
      continue;
    for (Value init : softmax.getDpsInits())
      if (init == buf) {
        if (found)
          return nullptr; // two writers — bail out
        found = softmax;
      }
  }
  return found;
}

// Return the unique linalg.generic that has `buf` as its DPS init, or nullptr.
static linalg::GenericOp findGenericWriterOf(Value buf) {
  linalg::GenericOp found;
  for (Operation *user : buf.getUsers()) {
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (!generic)
      continue;
    for (Value init : generic.getDpsInits())
      if (init == buf) {
        if (found)
          return nullptr;
        found = generic;
      }
  }
  return found;
}

// Walk the body of a scale generic and return the outer SSA value being
// multiplied with the element (i.e., not a block argument).
// Looks for the first arith.mulf whose one operand is not a block arg.
static Value extractScaleFromBody(linalg::GenericOp scaleGeneric) {
  for (Operation &op : scaleGeneric.getBody()->getOperations()) {
    auto mulf = dyn_cast<arith::MulFOp>(&op);
    if (!mulf)
      continue;
    Value lhs = mulf->getOperand(0), rhs = mulf->getOperand(1);
    if (!isa<BlockArgument>(lhs))
      return lhs;
    if (!isa<BlockArgument>(rhs))
      return rhs;
  }
  return nullptr;
}

// ── pattern ────────────────────────────────────────────────────────────────

struct FuseAttentionPattern : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp pvMatmul,
                                PatternRewriter &rewriter) const override {
    // ── Step 1: PV matmul operands ─────────────────────────────────────────
    auto pvInputs = pvMatmul.getDpsInputs();
    auto pvInits  = pvMatmul.getDpsInits();
    llvm::errs() << "[DBG] matmul inputs=" << pvInputs.size()
                 << " inits=" << pvInits.size() << "\n";
    if (pvInputs.size() < 2 || pvInits.empty())
      return failure();

    Value probsBuf = pvInputs[0]; // written by softmax
    Value V        = pvInputs[1];
    Value outBuf   = pvInits[0];

    // ── Step 2: softmax ────────────────────────────────────────────────────
    linalg::SoftmaxOp softmax = findSoftmaxWriterOf(probsBuf);
    llvm::errs() << "[DBG] softmax=" << (softmax ? "found" : "null") << "\n";
    if (!softmax)
      return failure();

    Value softmaxInBuf = softmax.getDpsInputs()[0]; // input to softmax

    // ── Step 3: mask op (optional) or scale op ────────────────────────────
    linalg::GenericOp maskOp, scaleOp;
    Value maskBuf;

    linalg::GenericOp afterSoftmax = findGenericWriterOf(softmaxInBuf);
    llvm::errs() << "[DBG] afterSoftmax=" << (afterSoftmax ? "found" : "null") << "\n";
    if (!afterSoftmax)
      return failure();

    if (afterSoftmax.getDpsInputs().size() == 2) {
      // Two inputs → this is the mask generic: ins=[scores, mask_i1]
      maskOp  = afterSoftmax;
      maskBuf = afterSoftmax.getDpsInputs()[1];
      Value scaledBuf = afterSoftmax.getDpsInputs()[0];
      scaleOp = findGenericWriterOf(scaledBuf);
      if (!scaleOp)
        return failure();
    } else {
      // One input → this is the scale generic (no mask in the sequence)
      scaleOp = afterSoftmax;
    }

    // ── Step 4: QK^T generic ──────────────────────────────────────────────
    Value qkBuf = scaleOp.getDpsInputs()[0];
    linalg::GenericOp qkGeneric = findGenericWriterOf(qkBuf);
    if (!qkGeneric || qkGeneric.getDpsInputs().size() < 2)
      return failure();

    Value Q = qkGeneric.getDpsInputs()[0];
    Value K = qkGeneric.getDpsInputs()[1];

    // ── Step 5: scale value ────────────────────────────────────────────────
    Value scaleVal = extractScaleFromBody(scaleOp);
    if (!scaleVal)
      return failure();

    // ── Step 6: build attention.fused ─────────────────────────────────────
    rewriter.setInsertionPoint(pvMatmul);
    rewriter.create<FusedOp>(pvMatmul.getLoc(), Q, K, V, scaleVal, maskBuf,
                             outBuf);

    // Erase the 5 original ops (softmax + generics + PV matmul).
    // Erase in reverse order so uses disappear before defs are removed.
    rewriter.eraseOp(pvMatmul);
    rewriter.eraseOp(softmax);
    if (maskOp)
      rewriter.eraseOp(maskOp);
    rewriter.eraseOp(scaleOp);
    rewriter.eraseOp(qkGeneric);

    return success();
  }
};

// ── pass ───────────────────────────────────────────────────────────────────

struct FusionPassImpl : public impl::FusionPassBase<FusionPassImpl> {
  using impl::FusionPassBase<FusionPassImpl>::FusionPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseAttentionPattern>(&getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozen)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::attention
