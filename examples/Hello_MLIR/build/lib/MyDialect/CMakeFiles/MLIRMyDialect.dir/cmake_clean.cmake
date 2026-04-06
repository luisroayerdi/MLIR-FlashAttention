file(REMOVE_RECURSE
  "libMLIRMyDialect.a"
  "libMLIRMyDialect.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRMyDialect.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
