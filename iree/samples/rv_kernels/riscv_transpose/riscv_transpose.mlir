#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
module  {
  func @forward(%arg0: tensor<2x4xf32>) -> tensor<4x2xf32> attributes {iree.abi = "{\22a\22:[[\22ndarray\22,\22f32\22,2,2,4]],\22r\22:[[\22ndarray\22,\22f32\22,2,4,2]],\22v\22:1}"} {
    %c0_i64 = constant 0 : i64
    %c1_i64 = constant 1 : i64
    %0 = linalg.init_tensor [4, 2] : tensor<4x2xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<2x4xf32>) outs(%0 : tensor<4x2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      linalg.yield %arg1 : f32
    } -> tensor<4x2xf32>
    return %1 : tensor<4x2xf32>
  }
}
