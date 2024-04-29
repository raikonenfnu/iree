hal.executable public @run_cached_initialize$async_dispatch_167 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>], target_arch = "gfx1100", ukernels = "none"}>) {
    hal.executable.export public @run_cached_initialize$async_dispatch_167_matmul_like_Dx8192x64x128_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 6, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>, #hal.interface.binding<0, 4>]} {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @run_cached_initialize$async_dispatch_167_matmul_like_Dx8192x64x128_f16() {
        %c0 = arith.constant 0 : index
        %c32_i64 = arith.constant 32 : i64
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = hal.interface.constant.load[4] : i32
        %5 = hal.interface.constant.load[5] : i32
        %6 = arith.extui %0 : i32 to i64
        %7 = arith.extui %1 : i32 to i64
        %8 = arith.shli %7, %c32_i64 : i64
        %9 = arith.ori %6, %8 : i64
        %10 = arith.index_castui %9 : i64 to index
        %11 = arith.extui %2 : i32 to i64
        %12 = arith.extui %3 : i32 to i64
        %13 = arith.shli %12, %c32_i64 : i64
        %14 = arith.ori %11, %13 : i64
        %15 = arith.index_castui %14 : i64 to index
        %16 = arith.extui %4 : i32 to i64
        %17 = arith.extui %5 : i32 to i64
        %18 = arith.shli %17, %c32_i64 : i64
        %19 = arith.ori %16, %18 : i64
        %20 = arith.index_castui %19 : i64 to index
        %21 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x64x128xi4>>
        %22 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x64xf16>>
        %23 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x64xf16>>
        %24 = flow.dispatch.workload.ordinal %20, 0 : index
        %25 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x64x128xf16>>{%24}
        %26 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%15) : !flow.dispatch.tensor<writeonly:tensor<?x8192xf16>>{%24}
        %27 = flow.dispatch.tensor.load %21, offsets = [0, 0, 0], sizes = [8192, 64, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x64x128xi4>> -> tensor<8192x64x128xi4>
        %28 = flow.dispatch.tensor.load %22, offsets = [0, 0], sizes = [8192, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x64xf16>> -> tensor<8192x64xf16>
        %29 = flow.dispatch.tensor.load %23, offsets = [0, 0], sizes = [8192, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x64xf16>> -> tensor<8192x64xf16>
        %30 = flow.dispatch.tensor.load %25, offsets = [0, 0, 0], sizes = [%24, 64, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x64x128xf16>>{%24} -> tensor<?x64x128xf16>
        %31 = tensor.empty(%24) : tensor<?x8192xf16>
        %32 = tensor.empty() : tensor<8192x64x128xf16>
        %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %28, %29 : tensor<8192x64x128xi4>, tensor<8192x64xf16>, tensor<8192x64xf16>) outs(%32 : tensor<8192x64x128xf16>) {
        ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
          %36 = arith.extui %in : i4 to i32
          %37 = arith.uitofp %36 : i32 to f16
          %38 = arith.subf %37, %in_1 : f16
          %39 = arith.mulf %38, %in_0 : f16
          linalg.yield %39 : f16
        } -> tensor<8192x64x128xf16>
        %34 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x8192xf16>) -> tensor<?x8192xf16>
        %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%30, %33 : tensor<?x64x128xf16>, tensor<8192x64x128xf16>) outs(%34 : tensor<?x8192xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %36 = arith.mulf %in, %in_0 : f16
          %37 = arith.addf %36, %out : f16
          linalg.yield %37 : f16
        } -> tensor<?x8192xf16>
        flow.dispatch.tensor.store %35, %26, offsets = [0, 0], sizes = [%24, 8192], strides = [1, 1] : tensor<?x8192xf16> -> !flow.dispatch.tensor<writeonly:tensor<?x8192xf16>>{%24}
        return
      }
    }
  }
}
