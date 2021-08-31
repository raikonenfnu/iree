func @simple_dot() -> tensor<2x1xf32>
{
  %arg0 = constant dense <[[1.0,2.0,3.0,4.0],[1.0,2.0,3.0,4.0]]> : tensor<2x4xf32>
  %arg1 = constant dense <[[1.0],[2.0],[3.0],[4.0]]> : tensor<4x1xf32>
  %0 = "mhlo.dot"(%arg0, %arg1) {name = "dot0"} : (tensor<2x4xf32>, tensor<4x1xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}
