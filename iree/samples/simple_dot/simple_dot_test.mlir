func @simple_dot(%arg0: tensor<4x3xf32>, %arg1: tensor<3x6xf32>) -> tensor<4x6xf32>
{
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "ab,bc->ac"} : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
  return %0 : tensor<4x6xf32>
}
