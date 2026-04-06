add_llvm_executable(my-opt
  my-opt.cpp
)

target_link_libraries(my-opt PRIVATE
  MLIRMyDialect
  MLIRIR
  MLIRParser
  MLIROptLib
)

llvm_update_compile_flags(my-opt)