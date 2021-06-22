func @simple_bmm(%arg0: tensor<2x4x3xf32>, %arg1: tensor<2x3x6xf32>) -> tensor<2x4x6xf32>
{
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "bac,bcd->bad"} : (tensor<2x4x3xf32>, tensor<2x3x6xf32>) -> tensor<2x4x6xf32>
  return %0 : tensor<2x4x6xf32>
}
