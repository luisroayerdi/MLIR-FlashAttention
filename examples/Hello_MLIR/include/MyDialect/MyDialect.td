include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MyDialect : Dialect {
  let name = "my";
  let summary = "Minimal example dialect";
  let cppNamespace = "::mlir::my";
}

class MyDialect_Op<string mnemonic, list<Trait> traits = []> :
    Op<MyDialect, mnemonic, traits>;

def AddOp : MyDialect_Op<"add", [Pure]> {
  let summary = "Vector addition with block size";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, I32Attr:$blockSize);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`(` $lhs `,` $rhs `)` `block` $blockSize attr-dict `:` functional-type(operands, results)";
}