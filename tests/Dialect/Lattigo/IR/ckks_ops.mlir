// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

!pk = !lattigo.rlwe.public_key
!sk = !lattigo.rlwe.secret_key
!rk = !lattigo.rlwe.relinearization_key
!gk5 = !lattigo.rlwe.galois_key<galoisElement = 5>
!eval_key_set = !lattigo.rlwe.evaluation_key_set

!ct = !lattigo.rlwe.ciphertext
!pt = !lattigo.rlwe.plaintext

!encryptor = !lattigo.rlwe.encryptor<publicKey = true>
!decryptor = !lattigo.rlwe.decryptor
!key_generator = !lattigo.rlwe.key_generator

!evaluator = !lattigo.ckks.evaluator
!encoder = !lattigo.ckks.encoder
!params = !lattigo.ckks.parameter

!value = tensor<8xf32>
!value_complex = tensor<8xcomplex<f32>>

#paramsLiteral = #lattigo.ckks.parameters_literal<
    logN = 14,
    logQ = [56, 55, 55],
    logP = [55],
    logDefaultScale = 55
>

#paramsLiteral2 = #lattigo.ckks.parameters_literal<
    logN = 14,
    Q = [65537, 17, 23],
    P = [29],
    logDefaultScale = 10
>

module {
  // CHECK-LABEL: func @test_new_params_from_literal
  func.func @test_new_params_from_literal() {
    // CHECK: %[[v1:.*]] = lattigo.ckks.new_parameters_from_literal
    %params = lattigo.ckks.new_parameters_from_literal {paramsLiteral = #paramsLiteral} : () -> !params
    // CHECK: %[[v2:.*]] = lattigo.ckks.new_parameters_from_literal
    %params2 = lattigo.ckks.new_parameters_from_literal {paramsLiteral = #paramsLiteral2} : () -> !params
    return
  }

  // CHECK-LABEL: func @test_ckks_new_encoder
  func.func @test_ckks_new_encoder(%params: !params) {
    // CHECK: %[[v1:.*]] = lattigo.ckks.new_encoder
    %encoder = lattigo.ckks.new_encoder %params : (!params) -> !encoder
    return
  }

  // CHECK-LABEL: func @test_ckks_new_evaluator_no_key_set
  func.func @test_ckks_new_evaluator_no_key_set(%params: !params) {
    // CHECK: %[[v1:.*]] = lattigo.ckks.new_evaluator
    %evaluator = lattigo.ckks.new_evaluator %params : (!params) -> !evaluator
    return
  }

  // CHECK-LABEL: func @test_ckks_new_evaluator
  func.func @test_ckks_new_evaluator(%params: !params, %eval_key_set: !eval_key_set) {
    // CHECK: %[[v1:.*]] = lattigo.ckks.new_evaluator
    %evaluator = lattigo.ckks.new_evaluator %params, %eval_key_set : (!params, !eval_key_set) -> !evaluator
    return
  }

  // CHECK-LABEL: func @test_ckks_new_plaintext
  func.func @test_ckks_new_plaintext(%params: !params) {
    // CHECK: %[[v1:.*]] = lattigo.ckks.new_plaintext
    %pt = lattigo.ckks.new_plaintext %params : (!params) -> !pt
    return
  }

  // CHECK-LABEL: func @test_ckks_encode
  func.func @test_ckks_encode(%encoder: !encoder, %value : !value, %pt: !pt) {
    // CHECK: %[[v1:.*]] = lattigo.ckks.encode
    %encoded = lattigo.ckks.encode %encoder, %value, %pt : (!encoder, !value, !pt) -> !pt
    return
  }

  // CHECK-LABEL: func @test_ckks_encode_complex
  func.func @test_ckks_encode_complex(%encoder: !encoder, %value : !value_complex, %pt: !pt) {
    // CHECK: %[[v1:.*]] = lattigo.ckks.encode
    %encoded = lattigo.ckks.encode %encoder, %value, %pt : (!encoder, !value_complex, !pt) -> !pt
    return
  }

  // CHECK-LABEL: func @test_ckks_add
  func.func @test_ckks_add(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.ckks.add
    %output = lattigo.ckks.add %evaluator, %lhs, %rhs : (!evaluator, !ct, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_ckks_sub
  func.func @test_ckks_sub(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.ckks.sub
    %output = lattigo.ckks.sub %evaluator, %lhs, %rhs : (!evaluator, !ct, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_ckks_mul
  func.func @test_ckks_mul(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.ckks.mul
    %output = lattigo.ckks.mul %evaluator, %lhs, %rhs : (!evaluator, !ct, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_ckks_decode
  func.func @test_ckks_decode(%encoder: !encoder, %value : !value, %pt: !pt) {
    // CHECK: %[[v1:.*]] = lattigo.ckks.decode
    %decoded = lattigo.ckks.decode %encoder, %pt, %value : (!encoder, !pt, !value) -> !value
    return
  }

  // CHECK-LABEL: func @test_ckks_relinearize
  func.func @test_ckks_relinearize(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.ckks.relinearize
    %output = lattigo.ckks.relinearize %evaluator, %ct : (!evaluator, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_ckks_rescale
  func.func @test_ckks_rescale(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.ckks.rescale
    %output = lattigo.ckks.rescale %evaluator, %ct : (!evaluator, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_ckks_rotate
  func.func @test_ckks_rotate(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.ckks.rotate
    %output = lattigo.ckks.rotate %evaluator, %ct {offset = 1} : (!evaluator, !ct) -> !ct
    return
  }
}
