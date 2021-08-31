#map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
module  {
  func @forward(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi = "{\22a\22:[[\22ndarray\22,\22i32\22,2,5,4],[\22ndarray\22,\22f32\22,2,10,3]],\22r\22:[[\22ndarray\22,\22f32\22,2,5,3]],\22v\22:1}"} {
    %cst = constant 0.000000e+00 : f32
    %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<5x4xi32>
    %1 = hal.tensor.cast %arg1 : !hal.buffer_view -> tensor<10x3xf32>
    %2 = linalg.init_tensor [5, 4, 3] : tensor<5x4x3xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<5x4xi32>) outs(%2 : tensor<5x4x3xf32>) {
    ^bb0(%arg2: i32, %arg3: f32):  // no predecessors
      %8 = index_cast %arg2 : i32 to index
      %9 = linalg.index 2 : index
      %10 = tensor.extract %1[%8, %9] : tensor<10x3xf32>
      linalg.yield %10 : f32
    } -> tensor<5x4x3xf32>
    %4 = linalg.init_tensor [5, 3] : tensor<5x3xf32>
    %5 = linalg.fill(%cst, %4) : f32, tensor<5x3xf32> -> tensor<5x3xf32> 
    %6 = linalg.generic {indexing_maps = [#map2, #map0], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3 : tensor<5x4x3xf32>) outs(%5 : tensor<5x3xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      %8 = addf %arg2, %arg3 : f32
      linalg.yield %8 : f32
    } -> tensor<5x3xf32>
    %7 = hal.tensor.cast %6 : tensor<5x3xf32> -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}

