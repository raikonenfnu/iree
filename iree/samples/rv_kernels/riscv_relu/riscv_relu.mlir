#map = affine_map<(d0) -> (d0)>
module  {
  func @forward(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {iree.abi = "{\22a\22:[[\22ndarray\22,\22f32\22,1,4]],\22r\22:[[\22ndarray\22,\22f32\22,1,4]],\22v\22:1}"} {
    %cst = constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [4] : tensor<4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %2 = cmpf ugt, %arg1, %cst : f32
      %3 = select %2, %arg1, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
