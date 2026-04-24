add_mlir_dialect_library(MLIRAttention
        AttentionDialect.cpp
        AttentionOps.cpp
        AttentionTypes.cpp
        AttentionPasses.cpp
        FusionPass.cpp
        TilingPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Attention

        DEPENDS
        MLIRAttentionOpsIncGen
        MLIRAttentionPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRMemRefDialect
        MLIRAffineDialect
        MLIRAffineUtils
        MLIRArithDialect
        MLIRMathDialect
        MLIRTransformUtils
        MLIRRewrite
        )
