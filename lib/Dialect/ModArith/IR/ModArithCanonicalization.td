#ifndef LIB_DIALECT_MODARITH_IR_MODARITHCANONICALIZATION_TD_
#define LIB_DIALECT_MODARITH_IR_MODARITHCANONICALIZATION_TD_

include "lib/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

def Equal : Constraint<CPred<"$0 == $1">>;

def IsZero : Constraint<CPred<"$0.getValue().getInt() == 0">>;

//// add %x, 0 -> %x
//def AddZero : Pat<
//  (ModArith_AddOp $lhs, (ModArith_ConstantOp $cst)),
//  (replaceWithValue $lhs),
//  [
//    (IsZero $cst),
//  ]
//>;
//
//// sub %x, 0 -> %x
//def SubZero : Pat<
//  (ModArith_SubOp $lhs, (ModArith_ConstantOp $cst)),
//  (replaceWithValue $lhs),
//  [
//    (IsZero $cst),
//  ]
//>;
//
//// add (add %x, %c0), %c1 -> add %x, reduce (add %c0, %c1)
//def AddAddConstant : Pat<
//  (ModArith_AddOp
//    (ModArith_AddOp 
//      $lhs, 
//      (ModArith_ConstantOp $c0)),
//    (ModArith_ConstantOp $c1)),
//  (ModArith_AddOp
//    $lhs,
//    (ModArith_ReduceOp
//      (ModArith_AddOp
//        $rhs,
//        $cst))),
//  []
//>;

#endif  // LIB_DIALECT_MODARITH_IR_MODARITHCANONICALIZATION_TD_
