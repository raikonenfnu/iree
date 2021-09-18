#map = affine_map<(d0) -> (d0)>
module  {
  func @forward(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<?xf32> attributes {iree.abi = "{\22a\22:[[\22ndarray\22,\22f32\22,1,4],[\22ndarray\22,\22f32\22,1,4]],\22r\22:[[\22ndarray\22,\22f32\22,1,null]],\22v\22:1}"} {
    %0 = linalg.init_tensor [4] : tensor<4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%0 : tensor<4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %5 = addf %arg2, %arg3 : f32
      linalg.yield %5 : f32
    } -> tensor<4xf32>
    %2 = tensor.cast %1 : tensor<4xf32> to tensor<?xf32>
    return %2 : tensor<?xf32>
  }
}
