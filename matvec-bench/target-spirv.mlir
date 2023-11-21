spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader, StorageBuffer16BitAccess, Float16, GroupNonUniformArithmetic], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_16bit_storage]> {
  spirv.GlobalVariable @__builtin__WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @__builtin__LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @__resource_var_0_0_ bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spirv.GlobalVariable @__resource_var_0_1_ bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spirv.GlobalVariable @__resource_var_0_2_ bind(0, 2) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f16, stride=2> [0])>, StorageBuffer>
  spirv.func @_main_dispatch_0_matmul_transpose_b_1x32000x4096_f16() "None" {
    %cst64_i32 = spirv.Constant 64 : i32
    %cst128_i32 = spirv.Constant 128 : i32
    %cst192_i32 = spirv.Constant 192 : i32
    %cst256_i32 = spirv.Constant 256 : i32
    %cst320_i32 = spirv.Constant 320 : i32
    %cst384_i32 = spirv.Constant 384 : i32
    %cst448_i32 = spirv.Constant 448 : i32
    %cst512_i32 = spirv.Constant 512 : i32
    %cst_vec_4xf16 = spirv.Constant dense<0.000000e+00> : vector<4xf16>
    %cst0_i32 = spirv.Constant 0 : i32
    %cst_f16 = spirv.Constant 0.000000e+00 : f16
    %__builtin__LocalInvocationId___addr = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
    %0 = spirv.Load "Input" %__builtin__LocalInvocationId___addr : vector<3xi32>
    %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi32>
    %__resource_var_0_0__addr = spirv.mlir.addressof @__resource_var_0_0_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %__resource_var_0_1__addr = spirv.mlir.addressof @__resource_var_0_1_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %__resource_var_0_2__addr = spirv.mlir.addressof @__resource_var_0_2_ : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f16, stride=2> [0])>, StorageBuffer>
    %__builtin__WorkgroupId___addr = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
    %2 = spirv.Load "Input" %__builtin__WorkgroupId___addr : vector<3xi32>
    %3 = spirv.CompositeExtract %2[0 : i32] : vector<3xi32>
    %4 = spirv.IMul %3, %cst512_i32 : i32
    %5 = spirv.IAdd %1, %4 : i32
    %6 = spirv.IAdd %5, %cst448_i32 : i32
    %7 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %6] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %8 = spirv.Load "StorageBuffer" %7 : vector<4xf32>
    %9 = spirv.IAdd %1, %cst448_i32 : i32
    %10 = spirv.AccessChain %__resource_var_0_0__addr[%cst0_i32, %9] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %11 = spirv.Load "StorageBuffer" %10 : vector<4xf32>
    %12 = spirv.IAdd %5, %cst384_i32 : i32
    %13 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %12] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %14 = spirv.Load "StorageBuffer" %13 : vector<4xf32>
    %15 = spirv.IAdd %1, %cst384_i32 : i32
    %16 = spirv.AccessChain %__resource_var_0_0__addr[%cst0_i32, %15] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %17 = spirv.Load "StorageBuffer" %16 : vector<4xf32>
    %18 = spirv.IAdd %5, %cst320_i32 : i32
    %19 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %18] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %20 = spirv.Load "StorageBuffer" %19 : vector<4xf32>
    %21 = spirv.IAdd %1, %cst320_i32 : i32
    %22 = spirv.AccessChain %__resource_var_0_0__addr[%cst0_i32, %21] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %23 = spirv.Load "StorageBuffer" %22 : vector<4xf32>
    %24 = spirv.IAdd %5, %cst256_i32 : i32
    %25 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %24] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %26 = spirv.Load "StorageBuffer" %25 : vector<4xf32>
    %27 = spirv.IAdd %1, %cst256_i32 : i32
    %28 = spirv.AccessChain %__resource_var_0_0__addr[%cst0_i32, %27] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %29 = spirv.Load "StorageBuffer" %28 : vector<4xf32>
    %30 = spirv.IAdd %5, %cst192_i32 : i32
    %31 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %30] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %32 = spirv.Load "StorageBuffer" %31 : vector<4xf32>
    %33 = spirv.IAdd %1, %cst192_i32 : i32
    %34 = spirv.AccessChain %__resource_var_0_0__addr[%cst0_i32, %33] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %35 = spirv.Load "StorageBuffer" %34 : vector<4xf32>
    %36 = spirv.IAdd %5, %cst128_i32 : i32
    %37 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %36] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %38 = spirv.Load "StorageBuffer" %37 : vector<4xf32>
    %39 = spirv.IAdd %1, %cst128_i32 : i32
    %40 = spirv.AccessChain %__resource_var_0_0__addr[%cst0_i32, %39] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %41 = spirv.Load "StorageBuffer" %40 : vector<4xf32>
    %42 = spirv.IAdd %5, %cst64_i32 : i32
    %43 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %42] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %44 = spirv.Load "StorageBuffer" %43 : vector<4xf32>
    %45 = spirv.IAdd %1, %cst64_i32 : i32
    %46 = spirv.AccessChain %__resource_var_0_0__addr[%cst0_i32, %45] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %47 = spirv.Load "StorageBuffer" %46 : vector<4xf32>
    %48 = spirv.IAdd %4, %1 : i32
    %49 = spirv.AccessChain %__resource_var_0_1__addr[%cst0_i32, %48] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %50 = spirv.Load "StorageBuffer" %49 : vector<4xf32>
    %51 = spirv.AccessChain %__resource_var_0_0__addr[%cst0_i32, %1] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %52 = spirv.Load "StorageBuffer" %51 : vector<4xf32>
    %53 = spirv.VectorShuffle [0 : i32, 1 : i32] %11 : vector<4xf32>, %11 : vector<4xf32> -> vector<2xf32>
    %54 = spirv.Bitcast %53 : vector<2xf32> to vector<4xf16>
    %55 = spirv.VectorShuffle [0 : i32, 1 : i32] %8 : vector<4xf32>, %8 : vector<4xf32> -> vector<2xf32>
    %56 = spirv.Bitcast %55 : vector<2xf32> to vector<4xf16>
    %57 = spirv.FMul %54, %56 : vector<4xf16>
    %58 = spirv.VectorShuffle [2 : i32, 3 : i32] %11 : vector<4xf32>, %11 : vector<4xf32> -> vector<2xf32>
    %59 = spirv.Bitcast %58 : vector<2xf32> to vector<4xf16>
    %60 = spirv.VectorShuffle [2 : i32, 3 : i32] %8 : vector<4xf32>, %8 : vector<4xf32> -> vector<2xf32>
    %61 = spirv.Bitcast %60 : vector<2xf32> to vector<4xf16>
    %62 = spirv.FMul %59, %61 : vector<4xf16>
    %63 = spirv.FAdd %57, %cst_vec_4xf16 : vector<4xf16>
    %64 = spirv.FAdd %62, %cst_vec_4xf16 : vector<4xf16>
    %65 = spirv.VectorShuffle [0 : i32, 1 : i32] %17 : vector<4xf32>, %17 : vector<4xf32> -> vector<2xf32>
    %66 = spirv.Bitcast %65 : vector<2xf32> to vector<4xf16>
    %67 = spirv.VectorShuffle [0 : i32, 1 : i32] %14 : vector<4xf32>, %14 : vector<4xf32> -> vector<2xf32>
    %68 = spirv.Bitcast %67 : vector<2xf32> to vector<4xf16>
    %69 = spirv.FMul %66, %68 : vector<4xf16>
    %70 = spirv.VectorShuffle [2 : i32, 3 : i32] %17 : vector<4xf32>, %17 : vector<4xf32> -> vector<2xf32>
    %71 = spirv.Bitcast %70 : vector<2xf32> to vector<4xf16>
    %72 = spirv.VectorShuffle [2 : i32, 3 : i32] %14 : vector<4xf32>, %14 : vector<4xf32> -> vector<2xf32>
    %73 = spirv.Bitcast %72 : vector<2xf32> to vector<4xf16>
    %74 = spirv.FMul %71, %73 : vector<4xf16>
    %75 = spirv.FAdd %69, %63 : vector<4xf16>
    %76 = spirv.FAdd %74, %64 : vector<4xf16>
    %77 = spirv.VectorShuffle [0 : i32, 1 : i32] %23 : vector<4xf32>, %23 : vector<4xf32> -> vector<2xf32>
    %78 = spirv.Bitcast %77 : vector<2xf32> to vector<4xf16>
    %79 = spirv.VectorShuffle [0 : i32, 1 : i32] %20 : vector<4xf32>, %20 : vector<4xf32> -> vector<2xf32>
    %80 = spirv.Bitcast %79 : vector<2xf32> to vector<4xf16>
    %81 = spirv.FMul %78, %80 : vector<4xf16>
    %82 = spirv.VectorShuffle [2 : i32, 3 : i32] %23 : vector<4xf32>, %23 : vector<4xf32> -> vector<2xf32>
    %83 = spirv.Bitcast %82 : vector<2xf32> to vector<4xf16>
    %84 = spirv.VectorShuffle [2 : i32, 3 : i32] %20 : vector<4xf32>, %20 : vector<4xf32> -> vector<2xf32>
    %85 = spirv.Bitcast %84 : vector<2xf32> to vector<4xf16>
    %86 = spirv.FMul %83, %85 : vector<4xf16>
    %87 = spirv.FAdd %81, %75 : vector<4xf16>
    %88 = spirv.FAdd %86, %76 : vector<4xf16>
    %89 = spirv.VectorShuffle [0 : i32, 1 : i32] %29 : vector<4xf32>, %29 : vector<4xf32> -> vector<2xf32>
    %90 = spirv.Bitcast %89 : vector<2xf32> to vector<4xf16>
    %91 = spirv.VectorShuffle [0 : i32, 1 : i32] %26 : vector<4xf32>, %26 : vector<4xf32> -> vector<2xf32>
    %92 = spirv.Bitcast %91 : vector<2xf32> to vector<4xf16>
    %93 = spirv.FMul %90, %92 : vector<4xf16>
    %94 = spirv.VectorShuffle [2 : i32, 3 : i32] %29 : vector<4xf32>, %29 : vector<4xf32> -> vector<2xf32>
    %95 = spirv.Bitcast %94 : vector<2xf32> to vector<4xf16>
    %96 = spirv.VectorShuffle [2 : i32, 3 : i32] %26 : vector<4xf32>, %26 : vector<4xf32> -> vector<2xf32>
    %97 = spirv.Bitcast %96 : vector<2xf32> to vector<4xf16>
    %98 = spirv.FMul %95, %97 : vector<4xf16>
    %99 = spirv.FAdd %93, %87 : vector<4xf16>
    %100 = spirv.FAdd %98, %88 : vector<4xf16>
    %101 = spirv.VectorShuffle [0 : i32, 1 : i32] %35 : vector<4xf32>, %35 : vector<4xf32> -> vector<2xf32>
    %102 = spirv.Bitcast %101 : vector<2xf32> to vector<4xf16>
    %103 = spirv.VectorShuffle [0 : i32, 1 : i32] %32 : vector<4xf32>, %32 : vector<4xf32> -> vector<2xf32>
    %104 = spirv.Bitcast %103 : vector<2xf32> to vector<4xf16>
    %105 = spirv.FMul %102, %104 : vector<4xf16>
    %106 = spirv.VectorShuffle [2 : i32, 3 : i32] %35 : vector<4xf32>, %35 : vector<4xf32> -> vector<2xf32>
    %107 = spirv.Bitcast %106 : vector<2xf32> to vector<4xf16>
    %108 = spirv.VectorShuffle [2 : i32, 3 : i32] %32 : vector<4xf32>, %32 : vector<4xf32> -> vector<2xf32>
    %109 = spirv.Bitcast %108 : vector<2xf32> to vector<4xf16>
    %110 = spirv.FMul %107, %109 : vector<4xf16>
    %111 = spirv.FAdd %105, %99 : vector<4xf16>
    %112 = spirv.FAdd %110, %100 : vector<4xf16>
    %113 = spirv.VectorShuffle [0 : i32, 1 : i32] %41 : vector<4xf32>, %41 : vector<4xf32> -> vector<2xf32>
    %114 = spirv.Bitcast %113 : vector<2xf32> to vector<4xf16>
    %115 = spirv.VectorShuffle [0 : i32, 1 : i32] %38 : vector<4xf32>, %38 : vector<4xf32> -> vector<2xf32>
    %116 = spirv.Bitcast %115 : vector<2xf32> to vector<4xf16>
    %117 = spirv.FMul %114, %116 : vector<4xf16>
    %118 = spirv.VectorShuffle [2 : i32, 3 : i32] %41 : vector<4xf32>, %41 : vector<4xf32> -> vector<2xf32>
    %119 = spirv.Bitcast %118 : vector<2xf32> to vector<4xf16>
    %120 = spirv.VectorShuffle [2 : i32, 3 : i32] %38 : vector<4xf32>, %38 : vector<4xf32> -> vector<2xf32>
    %121 = spirv.Bitcast %120 : vector<2xf32> to vector<4xf16>
    %122 = spirv.FMul %119, %121 : vector<4xf16>
    %123 = spirv.FAdd %117, %111 : vector<4xf16>
    %124 = spirv.FAdd %122, %112 : vector<4xf16>
    %125 = spirv.VectorShuffle [0 : i32, 1 : i32] %47 : vector<4xf32>, %47 : vector<4xf32> -> vector<2xf32>
    %126 = spirv.Bitcast %125 : vector<2xf32> to vector<4xf16>
    %127 = spirv.VectorShuffle [0 : i32, 1 : i32] %44 : vector<4xf32>, %44 : vector<4xf32> -> vector<2xf32>
    %128 = spirv.Bitcast %127 : vector<2xf32> to vector<4xf16>
    %129 = spirv.FMul %126, %128 : vector<4xf16>
    %130 = spirv.VectorShuffle [2 : i32, 3 : i32] %47 : vector<4xf32>, %47 : vector<4xf32> -> vector<2xf32>
    %131 = spirv.Bitcast %130 : vector<2xf32> to vector<4xf16>
    %132 = spirv.VectorShuffle [2 : i32, 3 : i32] %44 : vector<4xf32>, %44 : vector<4xf32> -> vector<2xf32>
    %133 = spirv.Bitcast %132 : vector<2xf32> to vector<4xf16>
    %134 = spirv.FMul %131, %133 : vector<4xf16>
    %135 = spirv.FAdd %129, %123 : vector<4xf16>
    %136 = spirv.FAdd %134, %124 : vector<4xf16>
    %137 = spirv.VectorShuffle [0 : i32, 1 : i32] %52 : vector<4xf32>, %52 : vector<4xf32> -> vector<2xf32>
    %138 = spirv.Bitcast %137 : vector<2xf32> to vector<4xf16>
    %139 = spirv.VectorShuffle [0 : i32, 1 : i32] %50 : vector<4xf32>, %50 : vector<4xf32> -> vector<2xf32>
    %140 = spirv.Bitcast %139 : vector<2xf32> to vector<4xf16>
    %141 = spirv.FMul %138, %140 : vector<4xf16>
    %142 = spirv.VectorShuffle [2 : i32, 3 : i32] %52 : vector<4xf32>, %52 : vector<4xf32> -> vector<2xf32>
    %143 = spirv.Bitcast %142 : vector<2xf32> to vector<4xf16>
    %144 = spirv.VectorShuffle [2 : i32, 3 : i32] %50 : vector<4xf32>, %50 : vector<4xf32> -> vector<2xf32>
    %145 = spirv.Bitcast %144 : vector<2xf32> to vector<4xf16>
    %146 = spirv.FMul %143, %145 : vector<4xf16>
    %147 = spirv.FAdd %141, %135 : vector<4xf16>
    %148 = spirv.FAdd %146, %136 : vector<4xf16>
    %149 = spirv.CompositeExtract %147[0 : i32] : vector<4xf16>
    %150 = spirv.CompositeExtract %147[1 : i32] : vector<4xf16>
    %151 = spirv.CompositeExtract %147[2 : i32] : vector<4xf16>
    %152 = spirv.CompositeExtract %147[3 : i32] : vector<4xf16>
    %153 = spirv.FAdd %149, %150 : f16
    %154 = spirv.FAdd %153, %151 : f16
    %155 = spirv.FAdd %154, %152 : f16
    %156 = spirv.CompositeExtract %148[0 : i32] : vector<4xf16>
    %157 = spirv.CompositeExtract %148[1 : i32] : vector<4xf16>
    %158 = spirv.CompositeExtract %148[2 : i32] : vector<4xf16>
    %159 = spirv.CompositeExtract %148[3 : i32] : vector<4xf16>
    %160 = spirv.FAdd %156, %157 : f16
    %161 = spirv.FAdd %160, %158 : f16
    %162 = spirv.FAdd %161, %159 : f16
    %163 = spirv.FAdd %155, %162 : f16
    %164 = spirv.GroupNonUniformFAdd "Subgroup" "Reduce" %163 : f16
    %165 = spirv.FAdd %164, %cst_f16 : f16
    %166 = spirv.IEqual %1, %cst0_i32 : i32
    spirv.mlir.selection {
      spirv.BranchConditional %166, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %167 = spirv.AccessChain %__resource_var_0_2__addr[%cst0_i32, %3] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f16, stride=2> [0])>, StorageBuffer>, i32, i32
      spirv.Store "StorageBuffer" %167, %165 : f16
      spirv.Branch ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      spirv.mlir.merge
    }
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @_main_dispatch_0_matmul_transpose_b_1x32000x4096_f16, @__builtin__LocalInvocationId__, @__builtin__WorkgroupId__
  spirv.ExecutionMode @_main_dispatch_0_matmul_transpose_b_1x32000x4096_f16 "LocalSize", 64, 1, 1
}