//===- MaskSpecializationPass.cpp - Attention mask specialization -*- C++ -*-===//
//
// Pass 4: Causal Mask Specialization
//
// Must run after --tiling-pass. For each K/V inner tile loop TilingPass
// emits that applies a boolean mask, wraps its per-iteration body in a
// two-level affine.if dispatching on tile position relative to the causal
// diagonal (Design.md §6):
//
//   MASKED   (k_start > q_end):  skip the tile entirely
//   FULL     (q_start >= k_end): run the unmasked computation
//   BOUNDARY (neither):          the original per-element masked computation
//
// PRECONDITION (not verified against the mask's actual contents -- see
// TRADEOFFS.md): the mask must be causal (mask[i,j] == true iff j > i).
//
//===----------------------------------------------------------------------===//

#include "Attention/AttentionPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::attention {
#define GEN_PASS_DEF_MASKSPECIALIZATIONPASS
#include "Attention/AttentionPasses.h.inc"

namespace {

using llvm::SmallVector;

// The mask-select op TilingPass emits: a linalg.generic with an i1-element
// memref among its DPS inputs (see TilingPass.cpp step 3, "Optional mask").
static bool isI1MemRef(Value v) {
  auto memrefTy = dyn_cast<MemRefType>(v.getType());
  return memrefTy && memrefTy.getElementType().isInteger(1);
}

static linalg::GenericOp findMaskSelectOp(affine::AffineForOp loop) {
  for (auto genOp : loop.getBody()->getOps<linalg::GenericOp>())
    if (llvm::any_of(genOp->getOperands(), isI1MemRef))
      return genOp;
  return nullptr;
}

struct MaskSpecializationPattern
    : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp kLoop,
                                PatternRewriter &rewriter) const override {
    // kLoop must be the K/V inner loop: directly nested in another
    // affine.for (the Q outer loop) -- see TilingPass.cpp's loop nest.
    auto qLoop = dyn_cast_or_null<affine::AffineForOp>(kLoop->getParentOp());
    if (!qLoop)
      return failure();

    linalg::GenericOp maskSelectOp = findMaskSelectOp(kLoop);
    if (!maskSelectOp)
      return failure(); // unmasked K-loop: nothing to specialize

    if (!kLoop.hasConstantLowerBound() || !kLoop.hasConstantUpperBound())
      return failure();
    int64_t tileSize = kLoop.getStepAsInt();
    if (qLoop.getStepAsInt() != tileSize)
      return failure(); // classification assumes one shared tile size

    // maskSelectOp's DPS inputs are (mask_tile : i1, S_tile : f32) and its
    // sole DPS init is S_masked (see TilingPass.cpp generic2DParallel call
    // in "3. Optional mask").
    auto dpsInputs = maskSelectOp.getDpsInputOperands();
    if (dpsInputs.size() != 2)
      return failure();
    Value maskTile = dpsInputs[0]->get();
    Value sTileOriginal = dpsInputs[1]->get();
    if (!isI1MemRef(maskTile))
      std::swap(maskTile, sTileOriginal);
    if (!isI1MemRef(maskTile))
      return failure();
    Value sMaskedOriginal = maskSelectOp.getDpsInitOperand(0)->get();

    Operation *maskSubviewOp = maskTile.getDefiningOp();

    Value iVar = qLoop.getInductionVar();
    Value jVar = kLoop.getInductionVar();

    Block *body = kLoop.getBody();
    SmallVector<Operation *> origOps;
    for (Operation &op : body->without_terminator())
      origOps.push_back(&op);
    if (origOps.empty())
      return failure();

    Location loc = kLoop.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    AffineExpr d0 = getAffineDimExpr(0, ctx); // i (Q-tile start)
    AffineExpr d1 = getAffineDimExpr(1, ctx); // j (K-tile start)
    AffineExpr t = getAffineConstantExpr(tileSize, ctx);

    // MASKED: k_start > q_end  <=>  j > i + T - 1  <=>  j - i - T >= 0
    IntegerSet maskedSet =
        IntegerSet::get(2, 0, {d1 - d0 - t}, {false});
    // FULL: q_start >= k_end  <=>  i >= j + T - 1  <=>  i - j - T + 1 >= 0
    IntegerSet fullSet =
        IntegerSet::get(2, 0, {d0 - d1 - t + 1}, {false});

    rewriter.setInsertionPoint(origOps.front());
    auto outerIf = rewriter.create<affine::AffineIfOp>(
        loc, maskedSet, ValueRange{iVar, jVar}, /*withElseRegion=*/true);
    // Then-block (MASKED): left empty -- skip the tile entirely.

    rewriter.setInsertionPoint(outerIf.getElseBlock()->getTerminator());
    auto innerIf = rewriter.create<affine::AffineIfOp>(
        loc, fullSet, ValueRange{iVar, jVar}, /*withElseRegion=*/true);

    // Then-block (FULL): clone origOps, dropping the mask subview + select,
    // redirecting consumers of S_masked to the cloned S_tile instead.
    rewriter.setInsertionPoint(innerIf.getThenBlock()->getTerminator());
    IRMapping mapping;
    for (Operation *op : origOps) {
      if (op == maskSubviewOp)
        continue;
      if (op == maskSelectOp.getOperation()) {
        mapping.map(sMaskedOriginal, mapping.lookupOrDefault(sTileOriginal));
        continue;
      }
      rewriter.clone(*op, mapping);
    }

    // Else-block (BOUNDARY): move the original ops, unchanged.
    Operation *boundaryTerminator = innerIf.getElseBlock()->getTerminator();
    for (Operation *op : origOps)
      op->moveBefore(boundaryTerminator);

    return success();
  }
};

struct MaskSpecializationPassImpl
    : public impl::MaskSpecializationPassBase<MaskSpecializationPassImpl> {
  using impl::MaskSpecializationPassBase<
      MaskSpecializationPassImpl>::MaskSpecializationPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MaskSpecializationPattern>(&getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozen)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::attention
