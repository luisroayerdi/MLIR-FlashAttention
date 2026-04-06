#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "MyDialect/MyDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  
  // Register only your custom dialect
  registry.insert<mlir::my::MyDialect>();
  
  // Run the MLIR optimizer driver
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "My Dialect Optimizer\n", registry));
}
