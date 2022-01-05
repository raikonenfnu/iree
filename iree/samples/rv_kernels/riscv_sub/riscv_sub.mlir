#map = affine_map<(d0) -> (d0)>
module  {
  func @forward(%arg0: tensor<?xf16>, %arg1: tensor<?xf16>) -> tensor<?xf16> attributes {} {
    %c0 = arith.constant 0 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?xf16>
    %0 = linalg.init_tensor [%dim0] : tensor<?xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<?xf16>, tensor<?xf16>) outs(%0 : tensor<?xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):  // no predecessors
      %5 = arith.subf %arg2, %arg3 : f16
      linalg.yield %5 : f16
    } -> tensor<?xf16>
    //%2 = tensor.cast %1 : tensor<?xf16> to tensor<?xf16>
    return %1 : tensor<?xf16>
  }
}
