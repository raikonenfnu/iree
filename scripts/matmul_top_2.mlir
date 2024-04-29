hal.executable public @run_cached_initialize$async_dispatch_191 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>], target_arch = "gfx1100", ukernels = "none"}>) {
    hal.executable.export public @run_cached_initialize$async_dispatch_191_elementwise_Dx8192_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 8, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>, #hal.interface.binding<0, 4>]} {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @run_cached_initialize$async_dispatch_191_elementwise_Dx8192_f16() {
        %c0 = arith.constant 0 : index
        %c32_i64 = arith.constant 32 : i64
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = hal.interface.constant.load[4] : i32
        %5 = hal.interface.constant.load[5] : i32
        %6 = hal.interface.constant.load[6] : i32
        %7 = hal.interface.constant.load[7] : i32
        %8 = arith.extui %0 : i32 to i64
        %9 = arith.extui %1 : i32 to i64
        %10 = arith.shli %9, %c32_i64 : i64
        %11 = arith.ori %8, %10 : i64
        %12 = arith.index_castui %11 : i64 to index
        %13 = arith.extui %2 : i32 to i64
        %14 = arith.extui %3 : i32 to i64
        %15 = arith.shli %14, %c32_i64 : i64
        %16 = arith.ori %13, %15 : i64
        %17 = arith.index_castui %16 : i64 to index
        %18 = arith.extui %4 : i32 to i64
        %19 = arith.extui %5 : i32 to i64
        %20 = arith.shli %19, %c32_i64 : i64
        %21 = arith.ori %18, %20 : i64
        %22 = arith.index_castui %21 : i64 to index
        %23 = arith.extui %6 : i32 to i64
        %24 = arith.extui %7 : i32 to i64
        %25 = arith.shli %24, %c32_i64 : i64
        %26 = arith.ori %23, %25 : i64
        %27 = arith.index_castui %26 : i64 to index
        %28 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x224x128xi4>>
        %29 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x224xf16>>
        %30 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x224xf16>>
        %31 = flow.dispatch.workload.ordinal %27, 0 : index
        %32 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%12) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x224x128xf16>>{%31}
        %33 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%17) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x8192xf16>>{%31}
        %34 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%22) : !flow.dispatch.tensor<writeonly:tensor<?x8192xf16>>{%31}
        %35 = flow.dispatch.tensor.load %28, offsets = [0, 0, 0], sizes = [8192, 224, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x224x128xi4>> -> tensor<8192x224x128xi4>
        %36 = flow.dispatch.tensor.load %29, offsets = [0, 0], sizes = [8192, 224], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x224xf16>> -> tensor<8192x224xf16>
        %37 = flow.dispatch.tensor.load %30, offsets = [0, 0], sizes = [8192, 224], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x224xf16>> -> tensor<8192x224xf16>
        %38 = flow.dispatch.tensor.load %32, offsets = [0, 0, 0], sizes = [%31, 224, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x224x128xf16>>{%31} -> tensor<?x224x128xf16>
        %39 = flow.dispatch.tensor.load %33, offsets = [0, 0], sizes = [%31, 8192], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x8192xf16>>{%31} -> tensor<?x8192xf16>
        %40 = tensor.empty(%31) : tensor<?x8192xf16>
        %41 = tensor.empty() : tensor<8192x224x128xf16>
        %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %36, %37 : tensor<8192x224x128xi4>, tensor<8192x224xf16>, tensor<8192x224xf16>) outs(%41 : tensor<8192x224x128xf16>) {
        ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
          %46 = arith.extui %in : i4 to i32
          %47 = arith.uitofp %46 : i32 to f16
          %48 = arith.subf %47, %in_1 : f16
          %49 = arith.mulf %48, %in_0 : f16
          linalg.yield %49 : f16
        } -> tensor<8192x224x128xf16>
        %43 = linalg.fill ins(%cst : f16) outs(%40 : tensor<?x8192xf16>) -> tensor<?x8192xf16>
        %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%38, %42 : tensor<?x224x128xf16>, tensor<8192x224x128xf16>) outs(%43 : tensor<?x8192xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %46 = arith.mulf %in, %in_0 : f16
          %47 = arith.addf %46, %out : f16
          linalg.yield %47 : f16
        } -> tensor<?x8192xf16>
        %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%39, %44 : tensor<?x8192xf16>, tensor<?x8192xf16>) outs(%40 : tensor<?x8192xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %46 = arith.addf %in, %in_0 : f16
          linalg.yield %46 : f16
        } -> tensor<?x8192xf16>
        flow.dispatch.tensor.store %45, %34, offsets = [0, 0], sizes = [%31, 8192], strides = [1, 1] : tensor<?x8192xf16> -> !flow.dispatch.tensor<writeonly:tensor<?x8192xf16>>{%31}
        return
      }
    }
  }
}