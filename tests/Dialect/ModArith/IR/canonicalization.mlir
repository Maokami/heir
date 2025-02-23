// RUN: heir-opt -canonicalize %s | FileCheck %s

!Zp = !mod_arith.int<8 : i64>

// CHECK-LABEL: @test_add_zero
func.func @test_add_zero() -> !Zp {
  // CHECK: mod_arith.constant 42 : !Z8_i64_
  %zero = mod_arith.constant 0 : !Zp
  %e1 = mod_arith.constant 42 : !Zp
  %add = mod_arith.add %e1, %zero : !Zp
  return %add : !Zp
}

// CHECK-LABEL: @test_sub_zero
func.func @test_sub_zero() -> !Zp {
  // CHECK: mod_arith.constant 42 : !Z8_i64_
  %zero = mod_arith.constant 0 : !Zp
  %e1 = mod_arith.constant 42 : !Zp
  %sub = mod_arith.sub %e1, %zero : !Zp
  return %sub : !Zp
}