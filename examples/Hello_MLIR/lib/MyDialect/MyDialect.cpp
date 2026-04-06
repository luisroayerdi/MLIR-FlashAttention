#include "MyDialect/MyDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::my;

#include "MyDialect/MyDialectDialect.cpp.inc"

void MyDialect::initialize() {
  addOperations<AddOp>();
}

#define GET_OP_CLASSES
#include "MyDialect/MyDialect.cpp.inc"