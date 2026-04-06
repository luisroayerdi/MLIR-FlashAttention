#ifndef MY_DIALECT_H
#define MY_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "MyDialect/MyDialectDialect.h.inc"

#define GET_OP_CLASSES
#include "MyDialect/MyDialect.h.inc"

#endif