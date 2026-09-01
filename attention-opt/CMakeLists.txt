set(LIBS
    MLIRArithDialect
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRRegisterAllPasses
    MLIRAttention
  )
add_llvm_executable(attention-opt attention-opt.cpp)

llvm_update_compile_flags(attention-opt)
target_link_libraries(attention-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(attention-opt)
