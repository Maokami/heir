#ifndef LIB_DIALECT_TFHERUST_IR_TFHERUSTDIALECT_TD_
#define LIB_DIALECT_TFHERUST_IR_TFHERUSTDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def TfheRust_Dialect : Dialect {
  let name = "tfhe_rust";

  let description = [{
    The `thfe_rust` dialect is an exit dialect for generating rust code against the tfhe-rs library API,
    using the shortint parameters and encoding scheme.

    See https://github.com/zama-ai/tfhe-rs
  }];

  let cppNamespace = "::mlir::heir::tfhe_rust";

  let useDefaultTypePrinterParser = 1;
}

#endif  // LIB_DIALECT_TFHERUST_IR_TFHERUSTDIALECT_TD_
