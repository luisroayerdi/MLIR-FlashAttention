add_mlir_dialect_library(MLIRMyDialect
  MyDialect.cpp
  
  DEPENDS
  MyDialectIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
)

target_include_directories(obj.MLIRMyDialect PRIVATE
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_BINARY_DIR}/include/MyDialect
)