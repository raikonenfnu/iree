#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module  {
  func @forward(%arg0: tensor<4xf32>) -> tensor<f32> attributes {iree.abi = "{\22a\22:[[\22ndarray\22,\22f32\22,1,4]],\22r\22:[[\22ndarray\22,\22f32\22,0]],\22v\22:1}"} {
    %cst = constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [] : tensor<f32>
    %1 = linalg.fill(%cst, %0) : f32, tensor<f32> -> tensor<f32> 
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%arg0 : tensor<4xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %3 = addf %arg1, %arg2 : f32
      linalg.yield %3 : f32
    } -> tensor<f32>
    return %2 : tensor<f32>
  }
}
