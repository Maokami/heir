#ifndef LIB_DIALECT_MODARITH_IR_MODARITHCANONICALIZATION_TD_
#define LIB_DIALECT_MODARITH_IR_MODARITHCANONICALIZATION_TD_

include "lib/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

def Equal : Constraint<CPred<"$0 == $1">>;

def IsZero : Constraint<CPred<"$0.getValue().getInt() == 0">>;
def IsOne : Constraint<CPred<"$0.getValue().getInt() == 1">>;
def IsMinusOne : Constraint<CPred<"$0.getValue().getInt() == $0.getType().getModulus().getValue().getSExtValue() - 1">>;

def CreateZeroConstant : NativeCodeCall<
   "$_builder.create<::mlir::heir::mod_arith::ConstantOp>($0.getLoc(), dyn_cast<::mlir::heir::mod_arith::ModArithType>($0.getType()), 0);">
  ;

// add x, 0 -> x
def AddZero : Pat<
  (ModArith_AddOp $lhs, (ModArith_ConstantOp $cst)),
  (replaceWithValue $lhs),
  [
    (IsZero $cst),
  ]
>;

// sub x, 0 -> x
def SubZero : Pat<
  (ModArith_SubOp $lhs, (ModArith_ConstantOp $cst)),
  (replaceWithValue $lhs),
  [
    (IsZero $cst),
  ]
>;

// mul x, 0 -> 0
def MulZero : Pat<
  (ModArith_MulOp $lhs, (ModArith_ConstantOp:$rhs $cst)),
  (replaceWithValue $rhs),
  [
    (IsZero $cst),
  ]
>;

// mul x, 1 -> x
def MulOne : Pat<
  (ModArith_MulOp $lhs, (ModArith_ConstantOp $cst)),
  (replaceWithValue $lhs),
  [
    (IsOne $cst),
  ]
>;

// add(add(x, c0), c1) -> add(x, c0 + c1)
def AddAddConstant : Pat<
  (ModArith_AddOp
    (ModArith_AddOp 
      $x, 
      (ModArith_ConstantOp:$c0 $a)),
    (ModArith_ConstantOp:$c1 $b)),
  (ModArith_AddOp
    $x,
    (ModArith_AddOp
      $c0,
      $c1)),
  []
>;

// add (sub (x, c0), c1) -> add (x, c1 - c0)
def AddSubConstantRHS : Pat<
  (ModArith_AddOp
    (ModArith_SubOp 
      $x, 
      (ModArith_ConstantOp:$c0 $a)),
    (ModArith_ConstantOp:$c1 $b)),
  (ModArith_AddOp
    $x,
    (ModArith_SubOp
      $c1,
      $c0)),
  []
>;

// add(sub(c0, x), c1) -> sub(c0 + c1, x)
def AddSubConstantLHS : Pat<
  (ModArith_AddOp
    (ModArith_SubOp 
      (ModArith_ConstantOp:$c0 $a), 
      $x),
    (ModArith_ConstantOp:$c1 $b)),
  (ModArith_SubOp
    (ModArith_AddOp
      $c0,
      $c1),
    $x),
  []
>;

// add(x, mul(y, (modulus - 1))) -> sub(x, y)
def AddMulNegativeOneRhs : Pat<
  (ModArith_AddOp
    $x,
    (ModArith_MulOp
      $y,
      (ModArith_ConstantOp $cst))),
  (ModArith_SubOp
    $x,
    $y),
  [
    (IsMinusOne $cst),
  ]
>;

// add(mul(x, modulus - 1), y) -> sub(y, x)
def AddMulNegativeOneLhs : Pat<
  (ModArith_AddOp
    (ModArith_MulOp
      $x,
      (ModArith_ConstantOp $cst)),
    $y),
  (ModArith_SubOp
    $y,
    $x),
  [
    (IsMinusOne $cst),
  ]
>;

// mul(mul(x, c0), c1) -> mul(x, c0 * c1)
def MulMulConstant : Pat<
  (ModArith_MulOp
    (ModArith_MulOp 
      $x, 
      (ModArith_ConstantOp:$c0 $a)),
    (ModArith_ConstantOp:$c1 $b)),
  (ModArith_MulOp
    $x,
    (ModArith_MulOp
      $c0,
      $c1)),
  []
>;

// sub(add(x, c0), c1) -> add(x, c0 - c1)
def SubRHSAddConstant : Pat<
  (ModArith_SubOp
    (ModArith_AddOp 
      $x, 
      (ModArith_ConstantOp:$c0 $a)),
    (ModArith_ConstantOp:$c1 $b)),
  (ModArith_AddOp
    $x,
    (ModArith_SubOp
      $c0,
      $c1)),
  []
>;

// sub(c1, add(x, c0)) -> sub(c1 - c0, x)
def SubLHSAddConstant : Pat<
  (ModArith_SubOp
    (ModArith_ConstantOp:$c1 $a),
    (ModArith_AddOp 
      $x, 
      (ModArith_ConstantOp:$c0 $b))),
  (ModArith_SubOp
    (ModArith_SubOp
      $c1,
      $c0),
    $x),
  []
>;

// sub(sub(x, c0), c1) -> sub(x, c0 + c1)
def SubRHSSubConstantRHS : Pat<
  (ModArith_SubOp
    (ModArith_SubOp 
      $x, 
      (ModArith_ConstantOp:$c0 $a)),
    (ModArith_ConstantOp:$c1 $b)),
  (ModArith_SubOp
    $x,
    (ModArith_AddOp
      $c0,
      $c1)),
  []
>;

// sub(sub(c0, x), c1) -> sub(c0 - c1, x)
def SubRHSSubConstantLHS : Pat<
  (ModArith_SubOp
    (ModArith_SubOp 
      (ModArith_ConstantOp:$c0 $a), 
      $x),
    (ModArith_ConstantOp:$c1 $b)),
  (ModArith_SubOp
    (ModArith_SubOp
      $c0,
      $c1),
    $x),
  []
>;

// sub(c1, sub(x, c0)) -> sub(c0 + c1, x)
def SubLHSSubConstantRHS : Pat<
  (ModArith_SubOp
    (ModArith_ConstantOp:$c1 $a),
    (ModArith_SubOp 
      $x, 
      (ModArith_ConstantOp:$c0 $b))),
  (ModArith_SubOp
    (ModArith_AddOp
      $c0,
      $c1),
    $x),
  []
>;

// sub(c1, sub(c0, x)) -> add(x, c1 - c0)
def SubLHSSubConstantLHS : Pat<
  (ModArith_SubOp
    (ModArith_ConstantOp:$c1 $a),
    (ModArith_SubOp 
      (ModArith_ConstantOp:$c0 $b), 
      $x)),
  (ModArith_AddOp
    $x,
    (ModArith_SubOp
      $c1,
      $c0)),
  []
>;

// sub(sub(a, b), a) -> sub(0, b)
def SubSubLHSRHSLHS : Pat<
  (ModArith_SubOp
    (ModArith_SubOp 
      $a, 
      $b),
    $a),
  (ModArith_SubOp
    (CreateZeroConstant $a),
    $b),
  []
>;

#endif  // LIB_DIALECT_MODARITH_IR_MODARITHCANONICALIZATION_TD_
