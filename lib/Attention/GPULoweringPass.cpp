//===- GPULoweringPass.cpp - Attention GPU lowering pass ---------*- C++ -*-===//
//
// Pass 5, Stage A: GPU Backend Lowering (Design.md §7)
//
// Wraps TilingPass's top-level Q-tile loop in a gpu.launch -- one GPU block
// per Q tile, one thread per block, so the inner K/V loop stays a sequential
// affine.for inside the kernel -- and outlines it into a gpu.func, entirely
// by driving two pieces of existing MLIR infrastructure:
// mlir::convertAffineLoopNestToGPULaunch and the gpu-kernel-outlining pass.
// No tensor cores, no shared-memory promotion, no dependence analysis of our
// own: that is either Stage B (not yet implemented) or a documented
// precondition on TilingPass's output (see AttentionPasses.td).
//
//===----------------------------------------------------------------------===//

#include "Attention/AttentionPasses.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::attention {
#define GEN_PASS_DEF_GPULOWERINGPASS
#include "Attention/AttentionPasses.h.inc"

namespace {

struct GPULoweringPassImpl
    : public impl::GPULoweringPassBase<GPULoweringPassImpl> {
  using impl::GPULoweringPassBase<GPULoweringPassImpl>::GPULoweringPassBase;

  // This pass creates gpu-dialect ops directly (LaunchOp, then whatever
  // gpu-kernel-outlining creates), so those dialects must be loaded before
  // it runs.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, DLTIDialect, cf::ControlFlowDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Wrap each function's top-level Q-tile loop in a gpu.launch. Only
    // top-level affine.for ops are considered here -- the inner K/V loop is
    // nested inside one and is deliberately left untouched; see this pass's
    // TableGen description for why that is the intended structure.
    for (auto func : module.getOps<func::FuncOp>()) {
      for (Operation &op :
           llvm::make_early_inc_range(func.getFunctionBody().getOps())) {
        auto forOp = dyn_cast<affine::AffineForOp>(&op);
        if (!forOp)
          continue;
        if (failed(convertAffineLoopNestToGPULaunch(forOp,
                                                     /*numBlockDims=*/1,
                                                     /*numThreadDims=*/0)))
          return signalPassFailure();
      }
    }

    // Outline every gpu.launch created above into a gpu.func, reusing
    // MLIR's existing pass rather than hand-rolling outlining.
    OpPassManager pm(ModuleOp::getOperationName());
    pm.addPass(createGpuKernelOutliningPass());
    if (failed(runPipeline(pm, module)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::attention
