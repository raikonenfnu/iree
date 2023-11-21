func.func @main() {
  %c64 = arith.constant 64 : index
  %c8192 = arith.constant 8192 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %3 = util.unfoldable_constant dense<1.0> : tensor<1x4096xf16>
  %4 = util.unfoldable_constant dense<1.0> : tensor<32000x4096xf16>
  %5 = tensor.empty() : tensor<1x32000xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
  %7 = linalg.matmul_transpose_b ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
  check.expect_eq_const(%7, dense<4096.0> : tensor<1x32000xf16>) : tensor<1x32000xf16>
  return
}

