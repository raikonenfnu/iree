// RUN: iree-opt -split-input-file -iree-linalg-ext-to-loops %s | IreeFileCheck %s
#map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> ()>
#map4 = affine_map<(d0, d1) -> ()>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map7 = affine_map<(d0, d1, d2) -> ()>
#map8 = affine_map<(d0, d1, d2) -> (d2)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, d3)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>


func @sort_1d(%arg0: tensor<1x512x384xf32>, %arg1: tensor<384x384xf32>) -> tensor<1x512x384xf32> {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<384xf32>
  %1 = linalg.init_tensor [512, 384] : tensor<512x384xf32>
  %2 = linalg.fill(%cst_0, %1) : f32, tensor<512x384xf32> -> tensor<512x384xf32>
  %3 = linalg.tensor_collapse_shape %arg0 [[0, 1], [2]] : tensor<1x512x384xf32> into tensor<512x384xf32>
  %4 = linalg.matmul ins(%3, %arg1 : tensor<512x384xf32>, tensor<384x384xf32>) outs(%2 : tensor<512x384xf32>) -> tensor<512x384xf32>
  %5 = linalg.tensor_expand_shape %4 [[0, 1], [2]] : tensor<512x384xf32> into tensor<1x512x384xf32>
  %6 = linalg.init_tensor [1, 512, 384] : tensor<1x512x384xf32>
  %7 = linalg.generic {indexing_maps = [#map8, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_1 : tensor<384xf32>) outs(%6 : tensor<1x512x384xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
    linalg.yield %arg3 : f32
  } -> tensor<1x512x384xf32>
  %8 = linalg.init_tensor [1, 512, 384] : tensor<1x512x384xf32>
  %9 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %7 : tensor<1x512x384xf32>, tensor<1x512x384xf32>) outs(%8 : tensor<1x512x384xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %3128 = arith.addf %arg3, %arg4 : f32
    linalg.yield %3128 : f32
  } -> tensor<1x512x384xf32>
  return %5 : tensor<1x512x384xf32>
}

    // %190 = linalg.tensor_collapse_shape %266 [[0, 1], [2]] : tensor<1x512x1536xf32> into tensor<512x1536xf32>
    // %191 = "nod.matmul_tensor"(%267, %cst_22) : (tensor<512x1536xf32>, tensor<1536x384xf32>) -> tensor<512x384xf32>
    // %192 = linalg.tensor_expand_shape %268 [[0, 1], [2]] : tensor<512x384xf32> into tensor<1x512x384xf32>
    // %194 = linalg.generic {indexing_maps = [#map8, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_19 : tensor<1536xf32>) outs(%193 : tensor<1x512x1536xf32>) {
    // ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
    //   linalg.yield %arg3 : f32
    // } -> tensor<1x512x1536xf32>
    // %195 = linalg.init_tensor [1, 512, 1536] : tensor<1x512x1536xf32>
    // %196 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%192, %194 : tensor<1x512x1536xf32>, tensor<1x512x1536xf32>) outs(%195 : tensor<1x512x1536xf32>) {
    // ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    //   %3128 = arith.addf %arg3, %arg4 : f32
    //   linalg.yield %3128 : f32
    // } -> tensor<1x512x1536xf32>