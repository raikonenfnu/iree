#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5, d4)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
module  {
  func @einsum_basic(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
    %0 = linalg.init_tensor [3, 4, 6] : tensor<3x4x6xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%arg0, %arg1 : tensor<3x4x5xf32>, tensor<3x5x6xf32>) outs(%0 : tensor<3x4x6xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %2 = mulf %arg2, %arg3 : f32
      %3 = addf %arg4, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<3x4x6xf32>
    return %1 : tensor<3x4x6xf32>
  }
  func @einsum_pointwisemul(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
    %0 = linalg.init_tensor [3, 4, 5] : tensor<3x4x5xf32>
    %1 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<3x4x5xf32>, tensor<3x4x5xf32>) outs(%0 : tensor<3x4x5xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %2 = mulf %arg2, %arg3 : f32
      linalg.yield %2 : f32
    } -> tensor<3x4x5xf32>
    return %1 : tensor<3x4x5xf32>
  }
  func @einsum_matmul(%arg0: tensor<7x9xf32>, %arg1: tensor<9x5xf32>) -> tensor<7x5xf32> {
    %0 = linalg.init_tensor [7, 5] : tensor<7x5xf32>
    %1 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<7x9xf32>, tensor<9x5xf32>) outs(%0 : tensor<7x5xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %2 = mulf %arg2, %arg3 : f32
      %3 = addf %arg4, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<7x5xf32>
    return %1 : tensor<7x5xf32>
  }
  func @einsum_broadcast4(%arg0: tensor<3x4x5x6x7xf32>, %arg1: tensor<7x8xf32>) -> tensor<3x4x5x6x8xf32> {
    %0 = linalg.init_tensor [3, 4, 5, 6, 8] : tensor<3x4x5x6x8xf32>
    %1 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x4x5x6x7xf32>, tensor<7x8xf32>) outs(%0 : tensor<3x4x5x6x8xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %2 = mulf %arg2, %arg3 : f32
      %3 = addf %arg4, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<3x4x5x6x8xf32>
    return %1 : tensor<3x4x5x6x8xf32>
  }
  func @einsum_ellipsis(%arg0: tensor<1x512x128xf32>, %arg1: tensor<128x256xf32>) -> tensor<1x512x256xf32> {
    %0 = linalg.init_tensor [1, 512, 256] : tensor<1x512x256xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map10, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%arg0, %arg1 : tensor<1x512x128xf32>, tensor<128x256xf32>) outs(%0 : tensor<1x512x256xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %2 = mulf %arg2, %arg3 : f32
      %3 = addf %arg4, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<1x512x256xf32>
    return %1 : tensor<1x512x256xf32>
  }
}

