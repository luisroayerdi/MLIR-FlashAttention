//===- TilingPass.cpp - Attention tiling pass --------------------*- C++ -*-===//
//
// Pass 2: Memory-Aware Tiling with Online Softmax
//
// Expands attention.fused into nested affine.for loops implementing the
// online softmax algorithm (Milakov & Gimelshein 2018) so no N×N attention
// weight matrix needs to be materialised.  After this pass, no attention.fused
// ops remain; all IR is standard affine + linalg + memref.
//
// Generated loop structure (pseudocode):
//   for i in [0, seq_q, T]:
//     O_acc[T,D]  = 0,  m_acc[T] = -inf,  l_acc[T] = 0   (accumulators)
//     for j in [0, seq_k, T]:
//       S[T,T]   = Q[i:,:]  @ K[j:,:]^T  * scale    (QK^T + scale)
//       [optionally] S[r,s] = mask ? -inf : S[r,s]   (causal mask)
//       m_new[r] = max(m_acc[r], max_s S[r,s])
//       P[T,T]   = exp(S - m_new)                     (unnormalised probs)
//       alpha[r] = exp(m_acc - m_new)                 (rescale factor)
//       l_new[r] = alpha*l_acc + rowsum(P)
//       O_acc   = alpha*O_acc + P @ V[j:,:]           (accumulate output)
//       m_acc, l_acc = m_new, l_new
//     output[i:,:] = O_acc / l_acc                   (final rescale)
//
//===----------------------------------------------------------------------===//

#include "Attention/AttentionDialect.h"
#include "Attention/AttentionOps.h"
#include "Attention/AttentionPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <limits>

namespace mlir::attention {
#define GEN_PASS_DEF_TILINGPASS
#include "Attention/AttentionPasses.h.inc"

namespace {

using llvm::SmallVector;

// ── affine map helpers ──────────────────────────────────────────────────────

// Build a 2-D identity map: (d0,d1) -> (d0,d1)
static AffineMap id2D(MLIRContext *ctx) {
  auto d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx);
  return AffineMap::get(2, 0, {d0, d1}, ctx);
}

// Build a 1-D projection from 2-D: (d0,d1) -> (d0) — row projection
static AffineMap row2D(MLIRContext *ctx) {
  return AffineMap::get(2, 0, {getAffineDimExpr(0, ctx)}, ctx);
}

// Build a 1-D identity: (d0) -> (d0)
static AffineMap id1D(MLIRContext *ctx) {
  return AffineMap::get(1, 0, {getAffineDimExpr(0, ctx)}, ctx);
}

// ── subview helper ─────────────────────────────────────────────────────────

// Create memref.subview source[dynOffset, 0][staticRows, staticCols][1, 1]
static Value subview2D(OpBuilder &b, Location loc, Value source, Value dynOffset,
                       int64_t rows, int64_t cols) {
  SmallVector<OpFoldResult> offsets = {dynOffset,
                                       b.getIndexAttr(0)};
  SmallVector<OpFoldResult> sizes   = {b.getIndexAttr(rows),
                                       b.getIndexAttr(cols)};
  SmallVector<OpFoldResult> strides = {b.getIndexAttr(1),
                                       b.getIndexAttr(1)};
  return b.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);
}

// ── linalg.generic helpers ─────────────────────────────────────────────────

// Element-wise parallel-parallel generic over 2-D memrefs with same shape.
// ins and outs share the same 2-D identity indexing map.
// bodyFn receives block args (ins..., out) and must yield one value.
static void generic2DParallel(OpBuilder &b, Location loc,
                              ValueRange ins, Value out,
                              function_ref<Value(OpBuilder &, Location,
                                                 ValueRange)>
                                  bodyFn) {
  MLIRContext *ctx = b.getContext();
  SmallVector<AffineMap> maps(ins.size() + 1, id2D(ctx));
  SmallVector<utils::IteratorType> iters(2, utils::IteratorType::parallel);
  b.create<linalg::GenericOp>(
      loc, TypeRange{}, ins, ValueRange{out}, maps, iters,
      [&](OpBuilder &nb, Location nl, ValueRange args) {
        nb.create<linalg::YieldOp>(nl, bodyFn(nb, nl, args));
      });
}

// Row-reduction: 2-D in, 1-D out.  Maps: in=(d0,d1)->id2D, out=(d0,d1)->d0.
static void rowReduce(OpBuilder &b, Location loc, Value in, Value out,
                      function_ref<Value(OpBuilder &, Location, Value elem,
                                         Value acc)>
                          reduceFn) {
  MLIRContext *ctx = b.getContext();
  SmallVector<AffineMap> maps = {id2D(ctx), row2D(ctx)};
  SmallVector<utils::IteratorType> iters = {utils::IteratorType::parallel,
                                            utils::IteratorType::reduction};
  b.create<linalg::GenericOp>(
      loc, TypeRange{}, ValueRange{in}, ValueRange{out}, maps, iters,
      [&](OpBuilder &nb, Location nl, ValueRange args) {
        nb.create<linalg::YieldOp>(nl, reduceFn(nb, nl, args[0], args[1]));
      });
}

// Element-wise parallel over 1-D memrefs (single-dim identity maps).
static void generic1DParallel(OpBuilder &b, Location loc, ValueRange ins,
                              Value out,
                              function_ref<Value(OpBuilder &, Location,
                                                 ValueRange)>
                                  bodyFn) {
  MLIRContext *ctx = b.getContext();
  SmallVector<AffineMap> maps(ins.size() + 1, id1D(ctx));
  SmallVector<utils::IteratorType> iters(1, utils::IteratorType::parallel);
  b.create<linalg::GenericOp>(
      loc, TypeRange{}, ins, ValueRange{out}, maps, iters,
      [&](OpBuilder &nb, Location nl, ValueRange args) {
        nb.create<linalg::YieldOp>(nl, bodyFn(nb, nl, args));
      });
}

// Row-broadcast: 1-D in, 2-D in-place out; maps (d0,d1)->d0 and id2D.
// Used for "O_acc[r,:] *= alpha[r]" style updates.
static void rowBroadcast2D(OpBuilder &b, Location loc, Value rowVec,
                           Value matrix2D,
                           function_ref<Value(OpBuilder &, Location,
                                              Value rowElem, Value matElem)>
                               bodyFn) {
  MLIRContext *ctx = b.getContext();
  SmallVector<AffineMap> maps = {row2D(ctx), id2D(ctx)};
  SmallVector<utils::IteratorType> iters = {utils::IteratorType::parallel,
                                            utils::IteratorType::parallel};
  b.create<linalg::GenericOp>(
      loc, TypeRange{}, ValueRange{rowVec}, ValueRange{matrix2D}, maps, iters,
      [&](OpBuilder &nb, Location nl, ValueRange args) {
        nb.create<linalg::YieldOp>(nl, bodyFn(nb, nl, args[0], args[1]));
      });
}

// ── main pattern ───────────────────────────────────────────────────────────

struct TilingPattern : public OpRewritePattern<FusedOp> {
  TilingPattern(MLIRContext *ctx, int64_t tile)
      : OpRewritePattern(ctx), tileSize(tile) {}

  LogicalResult matchAndRewrite(FusedOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // ── Extract static shapes ───────────────────────────────────────────────
    auto Qtype = cast<MemRefType>(op.getQ().getType());
    auto Ktype = cast<MemRefType>(op.getK().getType());
    int64_t seqQ    = Qtype.getShape()[0];
    int64_t headDim = Qtype.getShape()[1];
    int64_t seqK    = Ktype.getShape()[0];
    int64_t T       = tileSize;

    if (seqQ == ShapedType::kDynamic || headDim == ShapedType::kDynamic ||
        seqK == ShapedType::kDynamic)
      return op.emitOpError(
          "tiling pass requires fully static shapes; dynamic shapes are not "
          "yet supported");

    auto f32  = rewriter.getF32Type();
    auto mT_D  = MemRefType::get({T, headDim}, f32); // [T x D]
    auto mT_T  = MemRefType::get({T, T},       f32); // [T x T]
    auto mT    = MemRefType::get({T},           f32); // [T]

    // ── Common scalar constants (created before the loop) ──────────────────
    rewriter.setInsertionPoint(op);
    Value zero   = rewriter.create<arith::ConstantFloatOp>(
                       loc, f32, APFloat(0.0f));
    Value negInf = rewriter.create<arith::ConstantFloatOp>(
                       loc, f32,
                       APFloat(-std::numeric_limits<float>::infinity()));

    // ── Outer Q-tile loop ──────────────────────────────────────────────────
    auto qLoop = rewriter.create<affine::AffineForOp>(loc, 0, seqQ, T);
    rewriter.setInsertionPoint(qLoop.getBody()->getTerminator());
    Value iVar = qLoop.getInductionVar();

    // Tile-local accumulators (live across the K-tile inner loop).
    Value O_acc = rewriter.create<memref::AllocaOp>(loc, mT_D);
    Value m_acc = rewriter.create<memref::AllocaOp>(loc, mT);
    Value l_acc = rewriter.create<memref::AllocaOp>(loc, mT);

    rewriter.create<linalg::FillOp>(loc, TypeRange{}, ValueRange{zero},
                                    ValueRange{O_acc});
    rewriter.create<linalg::FillOp>(loc, TypeRange{}, ValueRange{negInf},
                                    ValueRange{m_acc});
    rewriter.create<linalg::FillOp>(loc, TypeRange{}, ValueRange{zero},
                                    ValueRange{l_acc});

    // Tile subview for this Q tile.
    Value Q_tile = subview2D(rewriter, loc, op.getQ(), iVar, T, headDim);

    // ── Inner K/V-tile loop ────────────────────────────────────────────────
    auto kLoop = rewriter.create<affine::AffineForOp>(loc, 0, seqK, T);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(kLoop.getBody()->getTerminator());
      Value jVar = kLoop.getInductionVar();

      Value K_tile = subview2D(rewriter, loc, op.getK(), jVar, T, headDim);
      Value V_tile = subview2D(rewriter, loc, op.getV(), jVar, T, headDim);

      // ── 1. S_tile = Q_tile @ K_tile^T ──────────────────────────────────
      // Indexing: Q[d0,d2] * K[d1,d2] -> S[d0,d1]  (K transposed read)
      Value S_tile = rewriter.create<memref::AllocaOp>(loc, mT_T);
      rewriter.create<linalg::FillOp>(loc, TypeRange{}, ValueRange{zero},
                                      ValueRange{S_tile});
      {
        auto d0 = getAffineDimExpr(0, ctx);
        auto d1 = getAffineDimExpr(1, ctx);
        auto d2 = getAffineDimExpr(2, ctx);
        SmallVector<AffineMap> maps = {
            AffineMap::get(3, 0, {d0, d2}, ctx), // Q_tile [T,D]
            AffineMap::get(3, 0, {d1, d2}, ctx), // K_tile [T,D] transposed
            AffineMap::get(3, 0, {d0, d1}, ctx)  // S_tile [T,T]
        };
        SmallVector<utils::IteratorType> iters = {
            utils::IteratorType::parallel, utils::IteratorType::parallel,
            utils::IteratorType::reduction};
        rewriter.create<linalg::GenericOp>(
            loc, TypeRange{}, ValueRange{Q_tile, K_tile}, ValueRange{S_tile},
            maps, iters, [](OpBuilder &b, Location l, ValueRange args) {
              auto prod = b.create<arith::MulFOp>(l, args[0], args[1]);
              auto sum  = b.create<arith::AddFOp>(l, args[2], prod);
              b.create<linalg::YieldOp>(l, ValueRange{sum});
            });
      }

      // ── 2. Scale: S_tile *= scale ───────────────────────────────────────
      Value scaleVal = op.getScale();
      generic2DParallel(
          rewriter, loc, ValueRange{S_tile}, S_tile,
          [scaleVal](OpBuilder &b, Location l, ValueRange args) -> Value {
            return b.create<arith::MulFOp>(l, args[0], scaleVal);
          });

      // ── 3. Optional mask ────────────────────────────────────────────────
      // If a mask is present, create a separate S_masked buffer rather than
      // aliasing S_tile, to avoid a linalg in+out aliasing issue.
      Value scoresBuf = S_tile; // default: no mask
      if (Value maskMem = op.getMask()) {
        SmallVector<OpFoldResult> mOff = {iVar, jVar};
        SmallVector<OpFoldResult> mSz  = {rewriter.getIndexAttr(T),
                                          rewriter.getIndexAttr(T)};
        SmallVector<OpFoldResult> mStr = {rewriter.getIndexAttr(1),
                                          rewriter.getIndexAttr(1)};
        Value mask_tile = rewriter.create<memref::SubViewOp>(
            loc, maskMem, mOff, mSz, mStr);

        Value S_masked = rewriter.create<memref::AllocaOp>(loc, mT_T);
        // mask_tile[r,s]==true → -inf, else keep score
        MLIRContext *mctx = ctx;
        generic2DParallel(
            rewriter, loc, ValueRange{mask_tile, S_tile}, S_masked,
            [mctx](OpBuilder &b, Location l, ValueRange args) -> Value {
              auto ni = b.create<arith::ConstantFloatOp>(
                  l, b.getF32Type(),
                  APFloat(-std::numeric_limits<float>::infinity()));
              return b.create<arith::SelectOp>(l, args[0], ni.getResult(),
                                               args[1]);
            });
        scoresBuf = S_masked;
      }

      // ── 4. Online softmax update ────────────────────────────────────────
      //
      // m_tile[r] = max_s scoresBuf[r,s]
      Value m_tile = rewriter.create<memref::AllocaOp>(loc, mT);
      rewriter.create<linalg::FillOp>(loc, TypeRange{}, ValueRange{negInf},
                                      ValueRange{m_tile});
      rowReduce(rewriter, loc, scoresBuf, m_tile,
                [](OpBuilder &b, Location l, Value e, Value a) -> Value {
                  return b.create<arith::MaximumFOp>(l, e, a);
                });

      // m_new[r] = max(m_acc[r], m_tile[r])
      Value m_new = rewriter.create<memref::AllocaOp>(loc, mT);
      generic1DParallel(
          rewriter, loc, ValueRange{m_acc, m_tile}, m_new,
          [](OpBuilder &b, Location l, ValueRange args) -> Value {
            return b.create<arith::MaximumFOp>(l, args[0], args[1]);
          });

      // alpha[r] = exp(m_acc[r] - m_new[r])
      Value alpha = rewriter.create<memref::AllocaOp>(loc, mT);
      generic1DParallel(
          rewriter, loc, ValueRange{m_acc, m_new}, alpha,
          [](OpBuilder &b, Location l, ValueRange args) -> Value {
            auto diff = b.create<arith::SubFOp>(l, args[0], args[1]);
            return b.create<math::ExpOp>(l, diff);
          });

      // P_tile[r,s] = exp(scoresBuf[r,s] - m_new[r])
      Value P_tile = rewriter.create<memref::AllocaOp>(loc, mT_T);
      {
        MLIRContext *pctx = ctx;
        // Maps: scores=(d0,d1)->id2D, m_new=(d0,d1)->d0, P=(d0,d1)->id2D
        SmallVector<AffineMap> pMaps = {id2D(pctx), row2D(pctx), id2D(pctx)};
        SmallVector<utils::IteratorType> pIters(2,
                                               utils::IteratorType::parallel);
        rewriter.create<linalg::GenericOp>(
            loc, TypeRange{}, ValueRange{scoresBuf, m_new},
            ValueRange{P_tile}, pMaps, pIters,
            [](OpBuilder &b, Location l, ValueRange args) {
              auto diff = b.create<arith::SubFOp>(l, args[0], args[1]);
              auto ex   = b.create<math::ExpOp>(l, diff);
              b.create<linalg::YieldOp>(l, ValueRange{ex});
            });
      }

      // P_sum[r] = sum_s P_tile[r,s]
      Value P_sum = rewriter.create<memref::AllocaOp>(loc, mT);
      rewriter.create<linalg::FillOp>(loc, TypeRange{}, ValueRange{zero},
                                      ValueRange{P_sum});
      rowReduce(rewriter, loc, P_tile, P_sum,
                [](OpBuilder &b, Location l, Value e, Value a) -> Value {
                  return b.create<arith::AddFOp>(l, a, e);
                });

      // l_new[r] = alpha[r] * l_acc[r] + P_sum[r]
      Value l_new = rewriter.create<memref::AllocaOp>(loc, mT);
      generic1DParallel(
          rewriter, loc, ValueRange{alpha, l_acc, P_sum}, l_new,
          [](OpBuilder &b, Location l, ValueRange args) -> Value {
            auto al = b.create<arith::MulFOp>(l, args[0], args[1]);
            return b.create<arith::AddFOp>(l, al, args[2]);
          });

      // O_acc[r,:] = alpha[r] * O_acc[r,:]  (rescale previous output)
      rowBroadcast2D(rewriter, loc, alpha, O_acc,
                     [](OpBuilder &b, Location l, Value a, Value o) -> Value {
                       return b.create<arith::MulFOp>(l, a, o);
                     });

      // O_acc[r,:] += P_tile[r,s] * V_tile[s,:]
      // Indexing: P[d0,d2] * V[d2,d1] -> O_acc[d0,d1]
      {
        auto d0 = getAffineDimExpr(0, ctx);
        auto d1 = getAffineDimExpr(1, ctx);
        auto d2 = getAffineDimExpr(2, ctx);
        SmallVector<AffineMap> maps = {
            AffineMap::get(3, 0, {d0, d2}, ctx), // P_tile [T,T]
            AffineMap::get(3, 0, {d2, d1}, ctx), // V_tile [T,D]
            AffineMap::get(3, 0, {d0, d1}, ctx)  // O_acc  [T,D]
        };
        SmallVector<utils::IteratorType> iters = {
            utils::IteratorType::parallel, utils::IteratorType::parallel,
            utils::IteratorType::reduction};
        rewriter.create<linalg::GenericOp>(
            loc, TypeRange{}, ValueRange{P_tile, V_tile}, ValueRange{O_acc},
            maps, iters, [](OpBuilder &b, Location l, ValueRange args) {
              auto pv  = b.create<arith::MulFOp>(l, args[0], args[1]);
              auto sum = b.create<arith::AddFOp>(l, args[2], pv);
              b.create<linalg::YieldOp>(l, ValueRange{sum});
            });
      }

      // Update running state: m_acc = m_new, l_acc = l_new
      rewriter.create<memref::CopyOp>(loc, m_new, m_acc);
      rewriter.create<memref::CopyOp>(loc, l_new, l_acc);
    } // end inner loop body

    // ── After inner loop: final rescale O_acc / l_acc ─────────────────────
    rewriter.setInsertionPointAfter(kLoop);
    rowBroadcast2D(rewriter, loc, l_acc, O_acc,
                   [](OpBuilder &b, Location l, Value lv, Value ov) -> Value {
                     return b.create<arith::DivFOp>(l, ov, lv);
                   });

    // Copy O_acc to output tile.
    Value out_tile = subview2D(rewriter, loc, op.getOutput(), iVar, T, headDim);
    rewriter.create<memref::CopyOp>(loc, O_acc, out_tile);

    // ── Erase the original attention.fused op ─────────────────────────────
    rewriter.setInsertionPointAfter(qLoop);
    rewriter.eraseOp(op);
    return success();
  }

  int64_t tileSize;
};

// ── pass ───────────────────────────────────────────────────────────────────

struct TilingPassImpl : public impl::TilingPassBase<TilingPassImpl> {
  using impl::TilingPassBase<TilingPassImpl>::TilingPassBase;

  // Declare dialects created by this pass so they are loaded before it runs.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AttentionDialect,
                    affine::AffineDialect,
                    arith::ArithDialect,
                    linalg::LinalgDialect,
                    math::MathDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TilingPattern>(&getContext(), tileSize);
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozen)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::attention
