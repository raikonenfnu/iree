// RUN: iree-opt -split-input-file --iree-codegen-convert-contraction-to-matmul -canonicalize -cse %s | IreeFileCheck %s

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d2)>

module  {
  func @einsum_bmm(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
    %0 = linalg.init_tensor [3, 4, 6] : tensor<3x4x6xf32>
    %cst = constant 0.000000e+00 : f32
    %1 = linalg.fill(%cst, %0) : f32, tensor<3x4x6xf32> -> tensor<3x4x6xf32> 
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%arg0, %arg1 : tensor<3x4x5xf32>, tensor<3x5x6xf32>) outs(%1 : tensor<3x4x6xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %3 = mulf %arg2, %arg3 : f32
      %4 = addf %arg4, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<3x4x6xf32>
    return %2 : tensor<3x4x6xf32>
  }

  func @einsum_bmm_dynamic(%arg0: tensor<?x?x3xf32>, %arg1: tensor<?x3x?xf32>) -> tensor<?x?x?xf32> {
    %c0 = constant 0 : index
    %0 = tensor.dim %arg0, %c0 : tensor<?x?x3xf32>
    %c1 = constant 1 : index
    %1 = tensor.dim %arg0, %c1 : tensor<?x?x3xf32>
    %c2 = constant 2 : index
    %2 = tensor.dim %arg1, %c2 : tensor<?x3x?xf32>
    %3 = linalg.init_tensor [%0, %1, %2] : tensor<?x?x?xf32>
    %cst = constant 0.000000e+00 : f32
    %4 = linalg.fill(%cst, %3) : f32, tensor<?x?x?xf32> -> tensor<?x?x?xf32> 
    %5 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%arg0, %arg1 : tensor<?x?x3xf32>, tensor<?x3x?xf32>) outs(%4 : tensor<?x?x?xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %6 = mulf %arg2, %arg3 : f32
      %7 = addf %arg4, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<?x?x?xf32>
    return %5 : tensor<?x?x?xf32>
  }

  func @einsum_bmm_trans(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x6x5xf32>) -> tensor<3x4x6xf32> {
    %0 = linalg.init_tensor [3, 4, 6] : tensor<3x4x6xf32>
    %cst = constant 0.000000e+00 : f32
    %1 = linalg.fill(%cst, %0) : f32, tensor<3x4x6xf32> -> tensor<3x4x6xf32> 
    %2 = linalg.generic {indexing_maps = [#map0, #map3, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%arg0, %arg1 : tensor<3x4x5xf32>, tensor<3x6x5xf32>) outs(%1 : tensor<3x4x6xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %3 = mulf %arg2, %arg3 : f32
      %4 = addf %arg4, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<3x4x6xf32>
    return %2 : tensor<3x4x6xf32>
  }

  func @einsum_matmul(%arg0: tensor<7x9xf32>, %arg1: tensor<9x5xf32>) -> tensor<7x5xf32> {
    %0 = linalg.init_tensor [7, 5] : tensor<7x5xf32>
    %cst = constant 0.000000e+00 : f32
    %1 = linalg.fill(%cst, %0) : f32, tensor<7x5xf32> -> tensor<7x5xf32> 
    %2 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg0, %arg1 : tensor<7x9xf32>, tensor<9x5xf32>) outs(%1 : tensor<7x5xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %3 = mulf %arg2, %arg3 : f32
      %4 = addf %arg4, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<7x5xf32>
    return %2 : tensor<7x5xf32>
  }
}