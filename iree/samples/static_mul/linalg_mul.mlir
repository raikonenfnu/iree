#map = affine_map<(d0) -> (d0)>
module  {
  func @forward(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> () attributes {iree.abi = "{\22a\22:[[\22ndarray\22,\22f32\22,1,4],[\22ndarray\22,\22f32\22,1,4]],\22r\22:[[\22ndarray\22,\22f32\22,1,null]],\22v\22:1}"} {
    %0 = linalg.init_tensor [4] : tensor<4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%arg2 : tensor<4xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      // %0 = "arith.mulf"(%arg0, %arg1) {name = "mul.1"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %3 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %3 : f32
    } -> tensor<4xf32>
    return
  }
}