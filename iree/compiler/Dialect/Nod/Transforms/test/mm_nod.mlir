module  {
  func @sort_1d(%arg0: tensor<1x512x384xf32>, %arg1: tensor<384x384xf32>) -> tensor<1x512x384xf32> {
    %cst = constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [512, 384] : tensor<512x384xf32>
    %1 = linalg.fill(%cst, %0) : f32, tensor<512x384xf32> -> tensor<512x384xf32> 
    %2 = linalg.tensor_collapse_shape %arg0 [[0, 1], [2]] : tensor<1x512x384xf32> into tensor<512x384xf32>
    %3 = "nod.matmul_tensor"(%2, %arg1, %1) : (tensor<512x384xf32>, tensor<384x384xf32>, tensor<512x384xf32>) -> tensor<512x384xf32>
    %4 = linalg.tensor_expand_shape %3 [[0, 1], [2]] : tensor<512x384xf32> into tensor<1x512x384xf32>
    return %4 : tensor<1x512x384xf32>
  }
}

