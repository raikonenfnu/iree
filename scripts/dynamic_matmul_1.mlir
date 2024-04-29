func.func @main(%30 : tensor<?x64x128xf16>) -> tensor<?x28672xf16> {
    %c0 = arith.constant 0 : index
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0.000000e+00 : f16
    %dim = tensor.dim %30, %c0 : tensor<?x64x128xf16>
    %27 = util.unfoldable_constant dense<2> : tensor<28672x64x128xi4>
    %28 = util.unfoldable_constant dense<1.4> : tensor<28672x64xf16>
    %29 = util.unfoldable_constant dense<2.4> : tensor<28672x64xf16>
    %31 = tensor.empty(%dim) : tensor<?x28672xf16>
    %32 = tensor.empty() : tensor<28672x64x128xf16>
    %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %28, %29 : tensor<28672x64x128xi4>, tensor<28672x64xf16>, tensor<28672x64xf16>) outs(%32 : tensor<28672x64x128xf16>) {
    ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
        %36 = arith.extui %in : i4 to i32
        %37 = arith.uitofp %36 : i32 to f16
        %38 = arith.subf %37, %in_1 : f16
        %39 = arith.mulf %38, %in_0 : f16
        linalg.yield %39 : f16
    } -> tensor<28672x64x128xf16>
    %34 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x28672xf16>) -> tensor<?x28672xf16>
    %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%30, %33 : tensor<?x64x128xf16>, tensor<28672x64x128xf16>) outs(%34 : tensor<?x28672xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
        %36 = arith.mulf %in, %in_0 : f16
        %37 = arith.addf %36, %out : f16
        linalg.yield %37 : f16
    } -> tensor<?x28672xf16>
    return %35 : tensor<?x28672xf16>
}
