// RUN: heir-opt --polynomial-to-standard %s | FileCheck %s

#cycl_2048 = #_polynomial.polynomial<1 + x**1024>
#ring = #_polynomial.ring<cmod=4294967296, ideal=#cycl_2048>
#ring_prime = #_polynomial.ring<cmod=4294967291, ideal=#cycl_2048>


// CHECK-LABEL: @test_lower_add_power_of_two_cmod
func.func @test_lower_add_power_of_two_cmod() -> !_polynomial.polynomial<#ring> {
  // 2 + 2x + 2x^2 + ... + 2x^{1023}
  // CHECK: [[X:%.+]] = arith.constant dense<2> : [[T:tensor<1024xi32>]]
  %coeffs1 = arith.constant dense<2> : tensor<1024xi32>
  // CHECK: [[Y:%.+]] = arith.constant dense<3> : [[T]]
  %coeffs2 = arith.constant dense<3> : tensor<1024xi32>
  // CHECK-NOT: _polynomial.from_tensor
  %poly0 = _polynomial.from_tensor %coeffs1 : tensor<1024xi32> -> !_polynomial.polynomial<#ring>
  %poly1 = _polynomial.from_tensor %coeffs2 : tensor<1024xi32> -> !_polynomial.polynomial<#ring>
  // CHECK-NEXT: [[ADD:%.+]] = arith.addi [[X]], [[Y]]
  %poly2 = _polynomial.add(%poly0, %poly1) {ring = #ring} : !_polynomial.polynomial<#ring>
  // CHECK: return  [[ADD]] : [[T]]
  return %poly2 : !_polynomial.polynomial<#ring>
}

// CHECK-LABEL: @test_lower_add_prime_cmod
func.func @test_lower_add_prime_cmod() -> !_polynomial.polynomial<#ring_prime> {
  // CHECK: [[X:%.+]] = arith.constant dense<2> : [[TCOEFF:tensor<1024xi31>]]
  %coeffs1 = arith.constant dense<2> : tensor<1024xi31>
  // CHECK: [[Y:%.+]] = arith.constant dense<3> : [[TCOEFF]]
  %coeffs2 = arith.constant dense<3> : tensor<1024xi31>
  // CHECK-NOT: _polynomial.from_tensor
  // CHECK: [[XEXT:%.+]] = arith.extsi [[X]] : [[TCOEFF]] to [[T:tensor<1024xi32>]]
  // CHECK: [[YEXT:%.+]] = arith.extsi [[Y]] : [[TCOEFF]] to [[T:tensor<1024xi32>]]
  %poly0 = _polynomial.from_tensor %coeffs1 : tensor<1024xi31> -> !_polynomial.polynomial<#ring_prime>
  %poly1 = _polynomial.from_tensor %coeffs2 : tensor<1024xi31> -> !_polynomial.polynomial<#ring_prime>

  // CHECK: [[MOD:%.+]] = arith.constant dense<4294967291> : [[T2:tensor<1024xi33>]]
  // CHECK: [[XEXT2:%.+]] = arith.extsi [[XEXT]] : [[T]] to [[T2]]
  // CHECK: [[YEXT2:%.+]] = arith.extsi [[YEXT]] : [[T]] to [[T2]]
  // CHECK: [[ADD_RESULT:%.+]] = arith.addi [[XEXT2]], [[YEXT2]]
  // CHECK: [[REM_RESULT:%.+]] = arith.remsi [[ADD_RESULT]], [[MOD]]
  // CHECK: [[TRUNC_RESULT:%.+]] = arith.trunci [[REM_RESULT]] : [[T2]] to [[T]]
  %poly2 = _polynomial.add(%poly0, %poly1) {ring = #ring_prime} : !_polynomial.polynomial<#ring_prime>

  // CHECK: return  [[TRUNC_RESULT]] : [[T]]
  return %poly2 : !_polynomial.polynomial<#ring_prime>
}

// CHECK-LABEL: @test_lower_add_tensor
func.func @test_lower_add_tensor() -> tensor<2x!_polynomial.polynomial<#ring>> {
  // 2 + 2x + 2x^2 + ... + 2x^{1023}
  // CHECK-DAG: [[A:%.+]] = arith.constant dense<2> : [[T:tensor<1024xi32>]]
  %coeffsA = arith.constant dense<2> : tensor<1024xi32>
  // CHECK-DAG: [[B:%.+]] = arith.constant dense<3> : [[T]]
  %coeffsB = arith.constant dense<3> : tensor<1024xi32>
  // CHECK-DAG: [[C:%.+]] = arith.constant dense<4> : [[T]]
  %coeffsC = arith.constant dense<4> : tensor<1024xi32>
  // CHECK-DAG: [[D:%.+]] = arith.constant dense<5> : [[T]]
  %coeffsD = arith.constant dense<5> : tensor<1024xi32>
  %polyA = _polynomial.from_tensor %coeffsA : tensor<1024xi32> -> !_polynomial.polynomial<#ring>
  %polyB = _polynomial.from_tensor %coeffsB : tensor<1024xi32> -> !_polynomial.polynomial<#ring>
  %polyC = _polynomial.from_tensor %coeffsC : tensor<1024xi32> -> !_polynomial.polynomial<#ring>
  %polyD = _polynomial.from_tensor %coeffsD : tensor<1024xi32> -> !_polynomial.polynomial<#ring>
  %tensor1 = tensor.from_elements %polyA, %polyB : tensor<2x!_polynomial.polynomial<#ring>>
  %tensor2 = tensor.from_elements %polyC, %polyD : tensor<2x!_polynomial.polynomial<#ring>>
  // CHECK: [[S1:%.+]] = arith.constant dense<[1, 1024]> : [[TI:tensor<2xindex>]]
  // CHECK: [[T1:%.+]] = tensor.reshape [[A]]([[S1]]) : ([[T]], [[TI]]) -> [[TEX:tensor<1x1024xi32>]]
  // CHECK: [[S2:%.+]] = arith.constant dense<[1, 1024]> : [[TI]]
  // CHECK: [[T2:%.+]] = tensor.reshape [[B]]([[S2]]) : ([[T]], [[TI]]) -> [[TEX]]
  // CHECK: [[C1:%.+]] = tensor.concat dim(0) [[T1]], [[T2]] : ([[TEX]], [[TEX]]) -> [[TT:tensor<2x1024xi32>]]
  // CHECK: [[S3:%.+]] = arith.constant dense<[1, 1024]> : [[TI]]
  // CHECK: [[T3:%.+]] = tensor.reshape [[C]]([[S3]]) : ([[T]], [[TI]]) -> [[TEX]]
  // CHECK: [[S4:%.+]] = arith.constant dense<[1, 1024]> : [[TI]]
  // CHECK: [[T4:%.+]] = tensor.reshape [[D]]([[S4]]) : ([[T]], [[TI]]) -> [[TEX]]
  // CHECK: [[C2:%.+]] = tensor.concat dim(0) [[T3]], [[T4]] : ([[TEX]], [[TEX]]) -> [[TT:tensor<2x1024xi32>]]
  // CHECK-NOT: _polynomial.from_tensor
  // CHECK-NOT: tensor.from_elements
  %tensor3 = affine.for %i = 0 to 2 iter_args(%t0 = %tensor1) ->  tensor<2x!_polynomial.polynomial<#ring>> {
      // CHECK: [[FOR:%.]] = affine.for [[I:%.+]] = 0 to 2 iter_args([[T0:%.+]] = [[C1]]) -> ([[TT]]) {
      %a = tensor.extract %tensor1[%i] :  tensor<2x!_polynomial.polynomial<#ring>>
      %b = tensor.extract %tensor2[%i] :  tensor<2x!_polynomial.polynomial<#ring>>
      // CHECK: [[AA:%.+]] = tensor.extract_slice [[C1]][[[I]], 0] [1, 1024] [1, 1] : [[TT]]
      // CHECK: [[BB:%.+]] = tensor.extract_slice [[C2]][[[I]], 0] [1, 1024] [1, 1] : [[TT]]
      // CHECK-NOT: tensor.extract %
      %s = _polynomial.add(%a, %b) : !_polynomial.polynomial<#ring>
      // CHECK: [[SUM:%.+]] = arith.addi [[AA]], [[BB]] : [[T]]
      // CHECK-NOT: _polynomial.add
      %t = tensor.insert %s into %t0[%i] :  tensor<2x!_polynomial.polynomial<#ring>>
      // CHECK: [[INS:%.+]] = tensor.insert_slice [[SUM]] into [[T0]][[[I]], 0] [1, 1024] [1, 1] : [[T]] into [[TT]]
      // CHECK-NOT: tensor.insert %
      affine.yield %t :  tensor<2x!_polynomial.polynomial<#ring>>
      // CHECK: affine.yield [[INS]] : [[TT]]
    }
  return %tensor3 :  tensor<2x!_polynomial.polynomial<#ring>>
  // CHECK: return [[FOR]] : [[TT]]
}
