set(LLVM_TARGET_DEFINITIONS MyDialect.td)
mlir_tablegen(MyDialect.h.inc -gen-op-decls --op-exclude-regex=BytecodeOpInterface)
mlir_tablegen(MyDialect.cpp.inc -gen-op-defs --op-exclude-regex=BytecodeOpInterface)
mlir_tablegen(MyDialectDialect.h.inc -gen-dialect-decls -dialect=my)
mlir_tablegen(MyDialectDialect.cpp.inc -gen-dialect-defs -dialect=my)
add_custom_target(MyDialectIncGen DEPENDS 
  ${CMAKE_CURRENT_BINARY_DIR}/MyDialect.h.inc
  ${CMAKE_CURRENT_BINARY_DIR}/MyDialect.cpp.inc
  ${CMAKE_CURRENT_BINARY_DIR}/MyDialectDialect.h.inc
  ${CMAKE_CURRENT_BINARY_DIR}/MyDialectDialect.cpp.inc
)