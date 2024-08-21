// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize --cse %s | FileCheck %s

#layout = #iree_vector_ext.layout<<[VECTORY, LANEY], [4, 4]>, <[VECTORX, LANEX], [4, 4]>>

// CHECK-LABEL: @distribute_elementwise_f16
func.func @distribute_elementwise_f16(%a: vector<16x16xf16>, %b: vector<16x16xf16>, %denom: vector<16x16xf16>) -> vector<16x16xi1> {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.0 : f16
  // CHECK: %[[ROOT:.*]] = arith.constant dense<0.000000e+00> : vector<16xf16>
  %root = arith.constant dense<0.0> : vector<16x16xf16>
  %rootl = iree_vector_ext.to_layout %root to #layout : vector<16x16xf16>
  // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xf16> -> vector<16xf16>
  // CHECK-DAG: %[[C:.*]] = arith.mulf %[[B]], %[[ROOT]] {{.*}} : vector<16xf16>
  %c = arith.mulf %rootl, %b : vector<16x16xf16>
  // CHECK-DAG: %[[DENOM:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xf16> -> vector<16xf16>
  // CHECK-DAG: %[[DIVD:.*]] = arith.divf %[[C]], %[[DENOM]] {{.*}} : vector<16xf16>
  %divd = arith.divf %c, %denom : vector<16x16xf16>
  // CHECK-DAG: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xf16> -> vector<16xf16>
  // CHECK-DAG: %[[D:.*]] = arith.addf %[[DIVD]], %[[A]] fastmath<reassoc,nnan> {{.*}} : vector<16xf16>
  %d = arith.addf %divd, %a fastmath<reassoc,nnan> : vector<16x16xf16>
  // CHECK-DAG: %[[R:.*]] = arith.cmpf ult, %[[D]], %[[ROOT]] {{.*}} : vector<16xf16>
  %r = arith.cmpf ult, %d, %root : vector<16x16xf16>
  // CHECK: iree_vector_ext.to_simd %[[R]] : vector<16xi1> -> vector<16x16xi1>
  return %r : vector<16x16xi1>
}

// CHECK-LABEL: @distribute_elementwise_i32
func.func @distribute_elementwise_i32(%a: vector<16x16xi32>, %b: vector<16x16xi32>) -> vector<16x16xi32> {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0 : i32
  // CHECK: %[[ROOT:.*]] = arith.constant dense<2> : vector<16xi32>
  %root = arith.constant dense<2> : vector<16x16xi32>
  %rootl = iree_vector_ext.to_layout %root to #layout : vector<16x16xi32>
  // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xi32> -> vector<16xi32>
  // CHECK-DAG: %[[C:.*]] = arith.muli %[[B]], %[[ROOT]] {{.*}} : vector<16xi32>
  %c = arith.muli %rootl, %b : vector<16x16xi32>
  // CHECK-DAG: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xi32> -> vector<16xi32>
  // CHECK-DAG: %[[D:.*]] = arith.addi %[[C]], %[[A]] {{.*}} : vector<16xi32>
  %d = arith.addi %c, %a : vector<16x16xi32>
  // CHECK: iree_vector_ext.to_simd %[[D]] : vector<16xi32> -> vector<16x16xi32>
  return %d : vector<16x16xi32>
}

#nested = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [2, 1, 1],
  batches_per_subgroup    = [8, 2, 4],
  outers_per_batch        = [1, 4, 4],
  threads_per_outer       = [8, 2, 4],
  elements_per_thread     = [1, 8, 2],

  subgroup_strides        = [1, 1, 1],
  thread_strides          = [1, 8, 16]
>

// CHECK-LABEL: @distribute_elementwise_nested_layout_f16
func.func @distribute_elementwise_nested_layout_f16(%a: vector<128x128x128xf16>, %b: vector<128x128x128xf16>) -> vector<128x128x128xf16> {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.0 : f16
  // CHECK: %[[ROOT:.*]] = arith.constant dense<0.000000e+00> : vector<8x2x4x1x4x4x1x8x2xf16>
  %root = arith.constant dense<0.0> : vector<128x128x128xf16>
  %rootl = iree_vector_ext.to_layout %root to #nested : vector<128x128x128xf16>
  // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<128x128x128xf16> -> vector<8x2x4x1x4x4x1x8x2xf16>
  // CHECK-DAG: %[[C:.*]] = arith.mulf %[[B]], %[[ROOT]] {{.*}} : vector<8x2x4x1x4x4x1x8x2xf16>
  %c = arith.mulf %rootl, %b : vector<128x128x128xf16>
  // CHECK-DAG: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<128x128x128xf16> -> vector<8x2x4x1x4x4x1x8x2xf16>
  // CHECK-DAG: %[[D:.*]] = arith.addf %[[C]], %[[A]] fastmath<reassoc,nnan> {{.*}} : vector<8x2x4x1x4x4x1x8x2xf16>
  %d = arith.addf %c, %a fastmath<reassoc,nnan> : vector<128x128x128xf16>
  // CHECK: iree_vector_ext.to_simd %[[D]] : vector<8x2x4x1x4x4x1x8x2xf16> -> vector<128x128x128xf16>
  return %d : vector<128x128x128xf16>
}

// CHECK-LABEL: @distribute_scf_for
func.func @distribute_scf_for(%a: vector<16x16xi32>, %b: vector<16x16xi32>) -> vector<16x16xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst_0 = arith.constant 0 : i32
  // CHECK: %[[ROOT:.*]] = arith.constant dense<0> : vector<16xi32>
  %root = arith.constant dense<0> : vector<16x16xi32>
  %rootl = iree_vector_ext.to_layout %root to #layout : vector<16x16xi32>
  // CHECK: iter_args(%[[ARG0:.*]] = %[[ROOT]]) -> (vector<16xi32>)
  %out = scf.for %i = %c0 to %c128 step %c1 iter_args(%arg0 = %rootl) -> (vector<16x16xi32>) {
    // These should be ideally folded if canonicalization was ever ran.
    // Canonicalization currently breaks other tests. If canonicalization
    // is ever ran, this should be updated.
    // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xi32> -> vector<16xi32>
    // CHECK-DAG: %[[C:.*]] = arith.muli %[[ARG0]], %[[B]] {{.*}} : vector<16xi32>
    %c = arith.muli %arg0, %b : vector<16x16xi32>
    // CHECK-DAG: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xi32> -> vector<16xi32>
    // CHECK-DAG: %[[D:.*]] = arith.addi %[[C]], %[[A]] {{.*}} : vector<16xi32>
    %d = arith.addi %c, %a : vector<16x16xi32>
    // CHECK: scf.yield %[[D]] : vector<16xi32>
    scf.yield %d : vector<16x16xi32>
  }
  return %out : vector<16x16xi32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout_row_major = #iree_vector_ext.layout<<[BATCHX, LANEY], [2, 8]>, <[BATCHY, LANEX, VECTORX], [2, 1, 8]>>
#layout_col_major = #iree_vector_ext.layout<<[BATCHX, LANEY, VECTORX], [1, 4, 4]>, <[BATCHY, LANEX], [2, 8]>>

// CHECK-LABEL: @distribute_transfer_read_row_major
func.func @distribute_transfer_read_row_major(%alloc: memref<4x4xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %alloc[%c0, %c0], %cst
          {in_bounds = [false, false]}
                  : memref<4x4xf16>, vector<16x16xf16>
  %rootl = iree_vector_ext.to_layout %root to #layout_row_major : vector<16x16xf16>
  // CHECK-COUNT-4: vector.load {{.*}}, vector<8xf16>
  func.return %rootl : vector<16x16xf16>
}

// CHECK-LABEL: @distribute_transfer_read_col_major
func.func @distribute_transfer_read_col_major(%alloc: memref<32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %alloc[%c0, %c0], %cst
          {in_bounds = [true, true]}
                  : memref<32x32xf16>, vector<16x16xf16>
  %rootl = iree_vector_ext.to_layout %root to #layout_col_major : vector<16x16xf16>
  // CHECK-COUNT-8: vector.load {{.*}}, vector<1xf16>
  func.return %rootl : vector<16x16xf16>
}

// CHECK-LABEL: @distribute_transfer_read_row_major_with_broadcast
func.func @distribute_transfer_read_row_major_with_broadcast(%a: index, %b: index, %alloc: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %alloc[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  %rootl = iree_vector_ext.to_layout %root to #layout_row_major : vector<16x16xf16>
  // CHECK-COUNT-4: vector.load {{.*}}, vector<8xf16>
  func.return %rootl : vector<16x16xf16>
}

// CHECK-LABEL: @distribute_transfer_read_col_major_with_broadcast
func.func @distribute_transfer_read_col_major_with_broadcast(%a: index, %b: index, %alloc: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %alloc[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  %rootl = iree_vector_ext.to_layout %root to #layout_col_major : vector<16x16xf16>
  // CHECK-COUNT-8: vector.load {{.*}}, vector<1xf16>
  func.return %rootl : vector<16x16xf16>
}

// CHECK-LABEL: @distribute_transfer_read_row_major_transpose
func.func @distribute_transfer_read_row_major_transpose(%a: index, %b: index, %alloc: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %alloc[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  %rootl = iree_vector_ext.to_layout %root to #layout_row_major : vector<16x16xf16>
  // CHECK-COUNT-32: vector.load {{.*}}, vector<1xf16>
  func.return %rootl : vector<16x16xf16>
}

// CHECK-LABEL: @distribute_transfer_read_col_major_transpose
func.func @distribute_transfer_read_col_major_transpose(%a: index, %b: index, %alloc: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %alloc[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  %rootl = iree_vector_ext.to_layout %root to #layout_col_major : vector<16x16xf16>
  // CHECK-COUNT-2: vector.load {{.*}}, vector<4xf16>
  func.return %rootl : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout_row_major = #iree_vector_ext.layout<<[BATCHX, LANEY], [2, 8]>, <[BATCHY, LANEX, VECTORX], [2, 1, 8]>>
#layout_col_major = #iree_vector_ext.layout<<[BATCHX, LANEY, VECTORX], [1, 4, 4]>, <[BATCHY, LANEX], [2, 8]>>

// TODO: Use affine min tricks based on the grid size to elide the mod.
// Note that this IR is invalid if subgroup size != 8.

func.func @distribute_transfer_write_row_major(%root: vector<16x16xf16>, %alloc: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %rootl = iree_vector_ext.to_layout %root to #layout_row_major : vector<16x16xf16>
  vector.transfer_write %rootl, %alloc[%c0, %c0]
          {in_bounds = [true, true]}
                  : vector<16x16xf16>, memref<64x64xf16>
  func.return
}
// CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 mod 8)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 mod 8 + 8)>

// CHECK-LABEL: @distribute_transfer_write_row_major
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[LANEID:.+]] = gpu.thread_id  x
// CHECK: %[[VEC_LANE_Y:.+]] = affine.apply #[[$MAP0]]()[%[[LANEID]]]
// CHECK: %[[DIST_SRC_VEC:.+]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xf16> -> vector<2x2x8xf16>
// CHECK: %[[BATCH_0_0:.+]] = vector.extract %[[DIST_SRC_VEC]][0, 0] : vector<8xf16> from vector<2x2x8xf16>
// CHECK: vector.store %[[BATCH_0_0]], %{{.*}}[%[[VEC_LANE_Y]], %[[C0]]] : memref<64x64xf16>, vector<8xf16>

// CHECK: %[[NEXT_VEC_LANE_Y:.+]] = affine.apply #[[$MAP1]]()[%[[LANEID]]]
// CHECK: %[[BATCH_1_0:.+]] = vector.extract %[[DIST_SRC_VEC]][1, 0] : vector<8xf16> from vector<2x2x8xf16>
// CHECK: vector.store %[[BATCH_1_0]], %{{.*}}[%[[NEXT_VEC_LANE_Y]], %[[C0]]] : memref<64x64xf16>, vector<8xf16>

// CHECK: %[[BATCH_0_1:.+]] = vector.extract %[[DIST_SRC_VEC]][0, 1] : vector<8xf16> from vector<2x2x8xf16>
// CHECK: vector.store %[[BATCH_0_1]], %{{.*}}[%[[VEC_LANE_Y]], %[[C8]]] : memref<64x64xf16>, vector<8xf16>

// CHECK: %[[BATCH_1_1:.+]] = vector.extract %[[DIST_SRC_VEC]][1, 1] : vector<8xf16> from vector<2x2x8xf16>
// CHECK: vector.store %[[BATCH_1_1]], %{{.*}}[%[[NEXT_VEC_LANE_Y]], %[[C8]]] : memref<64x64xf16>, vector<8xf16>

func.func @distribute_transfer_write_col_major(%root: vector<16x16xf16>, %alloc: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %rootl = iree_vector_ext.to_layout %root to #layout_col_major : vector<16x16xf16>
  vector.transfer_write %rootl, %alloc[%c0, %c0]
          {in_bounds = [true, true]}
                  : vector<16x16xf16>, memref<64x64xf16>
  func.return
}
// CHECK-LABEL: @distribute_transfer_write_col_major
// CHECK-COUNT-8: vector.store {{.*}}, vector<1xf16>

func.func @distribute_transfer_write_row_major_with_broadcast(%root: vector<16x16xf16>, %a: index, %b: index, %alloc: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  %rootl = iree_vector_ext.to_layout %root to #layout_row_major : vector<16x16xf16>
  vector.transfer_write %rootl, %alloc[%c0, %c0, %a, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>
  func.return
}
// CHECK-LABEL: @distribute_transfer_write_row_major_with_broadcast
// CHECK-COUNT-4: vector.store {{.*}}, vector<8xf16>

func.func @distribute_transfer_write_col_major_with_broadcast(%root: vector<16x16xf16>, %a: index, %b: index, %alloc: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  %rootl = iree_vector_ext.to_layout %root to #layout_col_major : vector<16x16xf16>
  vector.transfer_write %rootl, %alloc[%c0, %c0, %a, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>
  func.return
}
// CHECK-LABEL: @distribute_transfer_write_col_major_with_broadcast
// CHECK-COUNT-8: vector.store {{.*}}, vector<1xf16>

func.func @distribute_transfer_write_row_major_transpose(%root: vector<16x16xf16>, %a: index, %b: index, %alloc: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  %rootl = iree_vector_ext.to_layout %root to #layout_row_major : vector<16x16xf16>
  vector.transfer_write %rootl, %alloc[%c0, %c0, %a, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>
  func.return
}
// CHECK-LABEL: @distribute_transfer_write_row_major_transpose
// CHECK-COUNT-32: vector.store {{.*}}, vector<1xf16>

func.func @distribute_transfer_write_col_major_transpose(%root: vector<16x16xf16>, %a: index, %b: index, %alloc: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  %rootl = iree_vector_ext.to_layout %root to #layout_col_major : vector<16x16xf16>
  vector.transfer_write %rootl, %alloc[%c0, %c0, %a, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>
  func.return
}
// CHECK-LABEL: @distribute_transfer_write_col_major_transpose
// CHECK-COUNT-2: vector.store {{.*}}, vector<4xf16>


func.func @distribute_transfer_write_with_non_contiguous_broadcast(%root: vector<16x16xf16>, %a: index, %b: index, %alloc: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  %rootl = iree_vector_ext.to_layout %root to #layout_row_major : vector<16x16xf16>
  vector.transfer_write %rootl, %alloc[%c0, %a, %c0, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d1, d3)>}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>
  func.return
}
// CHECK-LABEL: func.func @distribute_transfer_write_with_non_contiguous_broadcast
// CHECK-SAME: %[[ROOT:.+]]: vector<16x16xf16>, %[[A:.+]]: index, %[[B:.+]]: index, %[[ALLOC:.+]]: memref<32x32x32x32xf16>)
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK-COUNT-4: vector.store %{{.+}}, %[[ALLOC]][%[[C0]], {{.+}}, %[[C0]], %{{.+}}] : memref<32x32x32x32xf16>, vector<8xf16>

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#row_layout = #iree_vector_ext.per_dim_layout<[BATCHX, LANEY, VECTORX], [1, 4, 4]>
#col_layout = #iree_vector_ext.per_dim_layout<[BATCHY, LANEX], [1, 16]>
#layout2d = #iree_vector_ext.layout<#row_layout, #col_layout>
#layout1d = #iree_vector_ext.layout<#col_layout>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {}>
#translation_info = #iree_codegen.translation_info<None subgroup_size = 64>
module {
  func.func @distribute_reduction_f16(%source: vector<16x16xf16>, %init: vector<16xf16>) -> vector<16xf16>
  attributes {hal.executable.target = #executable_target_rocm_hsaco_fb, translation_info = #translation_info} {
    %sourcel = iree_vector_ext.to_layout %source to #layout2d : vector<16x16xf16>
    %result = vector.multi_reduction <maximumf>, %sourcel, %init [0]
                    : vector<16x16xf16> to vector<16xf16>
    func.return %result : vector<16xf16>
  }
}
//      CHECK: func.func @distribute_reduction_f16(%[[ARG0:[a-zA-Z0-9_]+]]: vector<16x16xf16>, %[[ARG1:[a-zA-Z0-9_]+]]: vector<16xf16>) -> vector<16xf16>
//  CHECK-DAG:    %[[C32_I32:.+]] = arith.constant 32 : i32
//  CHECK-DAG:    %[[C64_I32:.+]] = arith.constant 64 : i32
//  CHECK-DAG:    %[[C16_I32:.+]] = arith.constant 16 : i32
//  CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<2xf16>
//  CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<1xf16>
//      CHECK:   %[[D0:.+]] = iree_vector_ext.to_simt %[[ARG1]] : vector<16xf16> -> vector<1xf16>
//      CHECK:   %[[D1:.+]] = vector.extract %[[D0]][0] : f16 from vector<1xf16>
//      CHECK:   %[[D2:.+]] = iree_vector_ext.to_simt %[[ARG0]] : vector<16x16xf16> -> vector<1x1x4xf16>
//      CHECK:   %[[D3:.+]] = vector.extract %[[D2]][0, 0, 0] : f16 from vector<1x1x4xf16>
//      CHECK:   %[[D4:.+]] = vector.insert %[[D3]], %[[CST]] [0] : f16 into vector<2xf16>
//      CHECK:   %[[D5:.+]] = vector.extract %[[D2]][0, 0, 1] : f16 from vector<1x1x4xf16>
//      CHECK:   %[[D6:.+]] = vector.insert %[[D5]], %[[D4]] [1] : f16 into vector<2xf16>
//      CHECK:   %[[D7:.+]] = vector.extract %[[D2]][0, 0, 2] : f16 from vector<1x1x4xf16>
//      CHECK:   %[[D8:.+]] = vector.insert %[[D7]], %[[D6]] [0] : f16 into vector<2xf16>
//      CHECK:   %[[D9:.+]] = vector.extract %[[D2]][0, 0, 3] : f16 from vector<1x1x4xf16>
//      CHECK:   %[[D10:.+]] = vector.insert %[[D9]], %[[D8]] [1] : f16 into vector<2xf16>
//      CHECK:   %[[D11:.+]] = arith.maximumf %[[D6]], %[[D10]] : vector<2xf16>
//      CHECK:   %[[D12:.+]] = vector.bitcast %[[D11]] : vector<2xf16> to vector<1xi32>
//      CHECK:   %[[D13:.+]] = vector.extract %[[D12]][0] : i32 from vector<1xi32>
//      CHECK:   %[[SHUFFLERESULT:.+]], %[[VALID:.+]] = gpu.shuffle  xor %[[D13]], %[[C16_I32]], %[[C64_I32]] : i32
//      CHECK:   %[[D14:.+]] = vector.broadcast %[[SHUFFLERESULT]] : i32 to vector<1xi32>
//      CHECK:   %[[D15:.+]] = vector.bitcast %[[D14]] : vector<1xi32> to vector<2xf16>
//      CHECK:   %[[D16:.+]] = arith.maximumf %[[D15]], %[[D11]] : vector<2xf16>
//      CHECK:   %[[D17:.+]] = vector.bitcast %[[D16]] : vector<2xf16> to vector<1xi32>
//      CHECK:   %[[D18:.+]] = vector.extract %[[D17]][0] : i32 from vector<1xi32>
//      CHECK:   %[[SHUFFLERESULT_1:.+]], %[[VALID_2:.+]] = gpu.shuffle  xor %[[D18]], %[[C32_I32]], %[[C64_I32]] : i32
//      CHECK:   %[[D19:.+]] = vector.broadcast %[[SHUFFLERESULT_1]] : i32 to vector<1xi32>
//      CHECK:   %[[D20:.+]] = vector.bitcast %[[D19]] : vector<1xi32> to vector<2xf16>
//      CHECK:   %[[D21:.+]] = arith.maximumf %[[D20]], %[[D16]] : vector<2xf16>
//      CHECK:   %[[D22:.+]] = vector.extract %[[D21]][0] : f16 from vector<2xf16>
//      CHECK:   %[[D23:.+]] = vector.extract %[[D21]][1] : f16 from vector<2xf16>
//      CHECK:   %[[D24:.+]] = arith.maximumf %[[D22]], %[[D23]] : f16
//      CHECK:   %[[D25:.+]] = arith.maximumf %[[D24]], %[[D1]] : f16
//      CHECK:   %[[D26:.+]] = vector.insert %[[D25]], %[[CST_0]] [0] : f16 into vector<1xf16>
//      CHECK:   %[[D27:.+]] = iree_vector_ext.to_simd %[[D26]] : vector<1xf16> -> vector<16xf16>

#executable_target_rocm_hsaco_fb2 = #hal.executable.target<"rocm", "rocm-hsaco-fb", {}>
module {
  func.func @distribute_reduction_f32(%source: vector<16x16xf32>, %init: vector<16xf32>) -> vector<16xf32>
  attributes {hal.executable.target = #executable_target_rocm_hsaco_fb, translation_info = #translation_info} {
    %sourcel = iree_vector_ext.to_layout %source to #layout2d : vector<16x16xf32>
    %result = vector.multi_reduction <maximumf>, %sourcel, %init [0]
                : vector<16x16xf32> to vector<16xf32>
    func.return %result : vector<16xf32>
  }
}
//      CHECK: func.func @distribute_reduction_f32(%[[ARG0:[a-zA-Z0-9_]+]]: vector<16x16xf32>, %[[ARG1:[a-zA-Z0-9_]+]]: vector<16xf32>) -> vector<16xf32>
//  CHECK-DAG:    %[[C32_I32:.+]] = arith.constant 32 : i32
//  CHECK-DAG:    %[[C64_I32:.+]] = arith.constant 64 : i32
//  CHECK-DAG:    %[[C16_I32:.+]] = arith.constant 16 : i32
//  CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1xf32>
//      CHECK:   %[[D0:.+]] = iree_vector_ext.to_simt %[[ARG1]] : vector<16xf32> -> vector<1xf32>
//      CHECK:   %[[D1:.+]] = vector.extract %[[D0]][0] : f32 from vector<1xf32>
//      CHECK:   %[[D2:.+]] = iree_vector_ext.to_simt %[[ARG0]] : vector<16x16xf32> -> vector<1x1x4xf32>
//      CHECK:   %[[D3:.+]] = vector.extract %[[D2]][0, 0, 0] : f32 from vector<1x1x4xf32>
//      CHECK:   %[[D4:.+]] = vector.insert %[[D3]], %[[CST]] [0] : f32 into vector<1xf32>
//      CHECK:   %[[D5:.+]] = vector.extract %[[D2]][0, 0, 1] : f32 from vector<1x1x4xf32>
//      CHECK:   %[[D6:.+]] = vector.insert %[[D5]], %[[D4]] [0] : f32 into vector<1xf32>
//      CHECK:   %[[D7:.+]] = arith.maximumf %[[D4]], %[[D6]] : vector<1xf32>
//      CHECK:   %[[D8:.+]] = vector.extract %[[D2]][0, 0, 2] : f32 from vector<1x1x4xf32>
//      CHECK:   %[[D9:.+]] = vector.insert %[[D8]], %[[D6]] [0] : f32 into vector<1xf32>
//      CHECK:   %[[D10:.+]] = arith.maximumf %[[D7]], %[[D9]] : vector<1xf32>
//      CHECK:   %[[D11:.+]] = vector.extract %[[D2]][0, 0, 3] : f32 from vector<1x1x4xf32>
//      CHECK:   %[[D12:.+]] = vector.insert %[[D11]], %[[D9]] [0] : f32 into vector<1xf32>
//      CHECK:   %[[D13:.+]] = arith.maximumf %[[D10]], %[[D12]] : vector<1xf32>
//      CHECK:   %[[D14:.+]] = vector.bitcast %[[D13]] : vector<1xf32> to vector<1xi32>
//      CHECK:   %[[D15:.+]] = vector.extract %[[D14]][0] : i32 from vector<1xi32>
//      CHECK:   %[[SHUFFLERESULT:.+]], %[[VALID:.+]] = gpu.shuffle  xor %[[D15]], %[[C16_I32]], %[[C64_I32]] : i32
//      CHECK:   %[[D16:.+]] = vector.broadcast %[[SHUFFLERESULT]] : i32 to vector<1xi32>
//      CHECK:   %[[D17:.+]] = vector.bitcast %[[D16]] : vector<1xi32> to vector<1xf32>
//      CHECK:   %[[D18:.+]] = arith.maximumf %[[D17]], %[[D13]] : vector<1xf32>
//      CHECK:   %[[D19:.+]] = vector.bitcast %[[D18]] : vector<1xf32> to vector<1xi32>
//      CHECK:   %[[D20:.+]] = vector.extract %[[D19]][0] : i32 from vector<1xi32>
//      CHECK:   %[[SHUFFLERESULT_0:.+]], %[[VALID_1:.+]] = gpu.shuffle  xor %[[D20]], %[[C32_I32]], %[[C64_I32]] : i32
//      CHECK:   %[[D21:.+]] = vector.broadcast %[[SHUFFLERESULT_0]] : i32 to vector<1xi32>
//      CHECK:   %[[D22:.+]] = vector.bitcast %[[D21]] : vector<1xi32> to vector<1xf32>
//      CHECK:   %[[D23:.+]] = arith.maximumf %[[D22]], %[[D18]] : vector<1xf32>
//      CHECK:   %[[D24:.+]] = vector.extract %[[D23]][0] : f32 from vector<1xf32>
//      CHECK:   %[[D25:.+]] = arith.maximumf %[[D24]], %[[D1]] : f32
//      CHECK:   %[[D26:.+]] = vector.insert %[[D25]], %[[CST]] [0] : f32 into vector<1xf32>
//      CHECK:   %[[D27:.+]] = iree_vector_ext.to_simd %[[D26]] : vector<1xf32> -> vector<16xf32>

#transpose_test_layout = #iree_vector_ext.layout<<[LANEY], [32]>, <[LANEX, VECTORX], [4, 4]>>
func.func @distribute_transpose(%mem: memref<32x32xf16>, %mem1: memref<32x32xf16>) -> vector<32x16xf16> {
  // CHECK: func.func @distribute_transpose(%[[MEM:.*]]: memref<32x32xf16>, %[[MEM1:.*]]: memref<32x32xf16>
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  // CHECK-COUNT-1: vector.load %[[MEM]]
  // CHECK-COUNT-4: vector.load %[[MEM1]]
  %a = vector.transfer_read %mem[%c0, %c0], %cst : memref<32x32xf16>, vector<32x16xf16>
  %b = vector.transfer_read %mem1[%c0, %c0], %cst : memref<32x32xf16>, vector<16x32xf16>
  // CHECK-NOT: vector.transpose
  %b_t = vector.transpose %b, [1, 0] : vector<16x32xf16> to vector<32x16xf16>
  // CHECK: %[[ADD:.*]] = arith.addf %{{.*}}, %{{.*}} : vector<4xf16>
  %c = arith.addf %a, %b_t : vector<32x16xf16>
  %cl = iree_vector_ext.to_layout %c to #transpose_test_layout : vector<32x16xf16>
  // CHECK: iree_vector_ext.to_simd %[[ADD]] : vector<4xf16> -> vector<32x16xf16>
  func.return %cl : vector<32x16xf16>
}

#row_broadcast_layout = #iree_vector_ext.per_dim_layout<[BATCHX, LANEX], [2, 16]>
#col_broadcast_layout = #iree_vector_ext.per_dim_layout<[BATCHY, LANEY, VECTORX], [2, 4, 4]>
#layout_broadcast_1d = #iree_vector_ext.layout<#row_broadcast_layout>
#layout_broadcast_2d = #iree_vector_ext.layout<#row_broadcast_layout, #col_broadcast_layout>
#layout_broadcast_1d_t = #iree_vector_ext.layout<#col_broadcast_layout>
#layout_broadcast_2d_t = #iree_vector_ext.layout<#col_broadcast_layout, #row_broadcast_layout>

func.func @distribute_broadcast_row_col(%source: vector<32xf32>) -> vector<32x32xf32> {
  %result = vector.broadcast %source : vector<32xf32> to vector<32x32xf32>
  %resultl = iree_vector_ext.to_layout %result to #layout_broadcast_2d : vector<32x32xf32>
  // CHECK-DAG: %[[S00:.*]] = vector.extract %[[SOURCE:.*]][0, 0]
  // CHECK-DAG: vector.insert %[[S00]], %{{.*}} [0, 0, 0]
  // CHECK-DAG: vector.insert %[[S00]], %{{.*}} [1, 0, 0]
  // CHECK-DAG: %[[S01:.*]] = vector.extract %[[ACC:.*]][0, 1]
  // CHECK-DAG: vector.insert %[[S01]], %{{.*}} [0, 0, 1]
  // CHECK-DAG: vector.insert %[[S01]], %{{.*}} [1, 0, 1]
  // CHECK-DAG: %[[S02:.*]] = vector.extract %[[ACC:.*]][0, 2]
  // CHECK-DAG: vector.insert %[[S02]], %{{.*}} [0, 0, 2]
  // CHECK-DAG: vector.insert %[[S02]], %{{.*}} [1, 0, 2]
  // CHECK-DAG: %[[S03:.*]] = vector.extract %[[ACC:.*]][0, 3]
  // CHECK-DAG: vector.insert %[[S03]], %{{.*}} [0, 0, 3]
  // CHECK-DAG: vector.insert %[[S03]], %{{.*}} [1, 0, 3]

  // CHECK-DAG: %[[S10:.*]] = vector.extract %[[SOURCE]][1, 0]
  // CHECK-DAG: vector.insert %[[S10]], %{{.*}} [0, 1, 0]
  // CHECK-DAG: vector.insert %[[S10]], %{{.*}} [1, 1, 0]
  // CHECK-DAG: %[[S11:.*]] = vector.extract %[[ACC:.*]][1, 1]
  // CHECK-DAG: vector.insert %[[S11]], %{{.*}} [0, 1, 1]
  // CHECK-DAG: vector.insert %[[S11]], %{{.*}} [1, 1, 1]
  // CHECK-DAG: %[[S12:.*]] = vector.extract %[[ACC:.*]][1, 2]
  // CHECK-DAG: vector.insert %[[S12]], %{{.*}} [0, 1, 2]
  // CHECK-DAG: vector.insert %[[S12]], %{{.*}} [1, 1, 2]
  // CHECK-DAG: %[[S13:.*]] = vector.extract %[[ACC:.*]][1, 3]
  // CHECK-DAG: vector.insert %[[S13]], %{{.*}} [0, 1, 3]
  // CHECK-DAG: vector.insert %[[S13]], %{{.*}} [1, 1, 3]
  func.return %resultl : vector<32x32xf32>
}

func.func @distribute_broadcast_col_row(%source: vector<32xf32>) -> vector<32x32xf32> {
  %result = vector.broadcast %source : vector<32xf32> to vector<32x32xf32>
  %resultl = iree_vector_ext.to_layout %result to #layout_broadcast_2d_t : vector<32x32xf32>
  // CHECK-DAG: %[[S0:.*]] = vector.extract %[[SOURCE:.*]][0]
  // CHECK-DAG: vector.insert %[[S0]], %{{.*}} [0, 0, 0]
  // CHECK-DAG: vector.insert %[[S0]], %{{.*}} [0, 0, 1]
  // CHECK-DAG: vector.insert %[[S0]], %{{.*}} [0, 0, 2]
  // CHECK-DAG: vector.insert %[[S0]], %{{.*}} [0, 0, 3]
  // CHECK-DAG: vector.insert %[[S0]], %{{.*}} [0, 1, 0]
  // CHECK-DAG: vector.insert %[[S0]], %{{.*}} [0, 1, 1]
  // CHECK-DAG: vector.insert %[[S0]], %{{.*}} [0, 1, 2]
  // CHECK-DAG: vector.insert %[[S0]], %{{.*}} [0, 1, 3]

  // CHECK-DAG: %[[S1:.*]] = vector.extract %[[SOURCE:.*]][1]
  // CHECK-DAG: vector.insert %[[S1]], %{{.*}} [1, 0, 0]
  // CHECK-DAG: vector.insert %[[S1]], %{{.*}} [1, 0, 1]
  // CHECK-DAG: vector.insert %[[S1]], %{{.*}} [1, 0, 2]
  // CHECK-DAG: vector.insert %[[S1]], %{{.*}} [1, 0, 3]
  // CHECK-DAG: vector.insert %[[S1]], %{{.*}} [1, 1, 0]
  // CHECK-DAG: vector.insert %[[S1]], %{{.*}} [1, 1, 1]
  // CHECK-DAG: vector.insert %[[S1]], %{{.*}} [1, 1, 2]
  // CHECK-DAG: vector.insert %[[S1]], %{{.*}} [1, 1, 3]
  func.return %resultl : vector<32x32xf32>
}

#layout_broadcast_vectory_1d = #iree_vector_ext.layout<
  <[BATCHY, VECTORX], [1, 4]>
>

#layout_broadcast_vectory_2d = #iree_vector_ext.layout<
  <[BATCHX, VECTORY], [1, 4]>,
  <[BATCHY, VECTORX], [1, 4]>
>

// This test case checks if we distribute correct when we have vectorx frozen
// and we iterate on vectory.
// This previously caused a bug, since calculating SIMT index for broadcast
// needs to know the range of vectorx.
func.func @distribute_broadcast_vectory(%source: vector<4xf32>) -> vector<4x4xf32> {
  %result = vector.broadcast %source : vector<4xf32> to vector<4x4xf32>
  %resultl = iree_vector_ext.to_layout %result to #layout_broadcast_vectory_2d : vector<4x4xf32>
  // CHECK-DAG: %[[S00:.*]] = vector.extract %[[SOURCE:.*]][0, 0] : f32 from vector<1x4xf32>
  // CHECK-DAG: %[[S01:.*]] = vector.extract %[[SOURCE:.*]][0, 1] : f32 from vector<1x4xf32>
  // CHECK-DAG: %[[S02:.*]] = vector.extract %[[SOURCE:.*]][0, 2] : f32 from vector<1x4xf32>
  // CHECK-DAG: %[[S02:.*]] = vector.extract %[[SOURCE:.*]][0, 3] : f32 from vector<1x4xf32>
  // CHECK-DAG: vector.insert %[[S00:.*]] %{{.*}} [0, 0, 0] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S00:.*]] %{{.*}} [0, 0, 4] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S00:.*]] %{{.*}} [0, 0, 8] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S00:.*]] %{{.*}} [0, 0, 12] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S01:.*]] %{{.*}} [0, 0, 1] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S01:.*]] %{{.*}} [0, 0, 5] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S01:.*]] %{{.*}} [0, 0, 9] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S01:.*]] %{{.*}} [0, 0, 13] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S02:.*]] %{{.*}} [0, 0, 2] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S02:.*]] %{{.*}} [0, 0, 6] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S02:.*]] %{{.*}} [0, 0, 10] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S02:.*]] %{{.*}} [0, 0, 14] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S03:.*]] %{{.*}} [0, 0, 3] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S03:.*]] %{{.*}} [0, 0, 7] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S03:.*]] %{{.*}} [0, 0, 11] : f32 into vector<1x1x16xf32>
  // CHECK-DAG: vector.insert %[[S03:.*]] %{{.*}} [0, 0, 15] : f32 into vector<1x1x16xf32>
  func.return %resultl : vector<4x4xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

// This test case checks that chained WMMA contraction is distributable.
// Let C0 = matmul(A0, B0), and OUT = matmul(A1, C0).

// In this case, since C-layout and the RHS-Layout of WMMA has lane conflict,
// and has different numel per lane/thread, we expect compiler to emit code
// that will write back data from C0 to shared memory, before loading it again
// as RHS-layout from shared memory to register.

// We assume in this test that we have distributed it the IR at a subgroup level.

#layoutA = #iree_vector_ext.layout<<[ BATCHX,  LANEX], [1, 16]>, <[ BATCHY,  LANEY,  VECTORX], [1, 1, 16]>>
#layoutB = #iree_vector_ext.layout<<[ BATCHX,  LANEX], [1, 16]>, <[ BATCHY,  LANEY,  VECTORX], [1, 1, 16]>>
#layoutC = #iree_vector_ext.layout<<[ BATCHX,  VECTORY,  LANEY,  VECTORX], [1, 8, 2, 1]>, <[ BATCHY,  LANEX], [1, 16]>>

#layoutA2 = #iree_vector_ext.layout<<[ BATCHX,  LANEX], [1, 16]>, <[ BATCHY,  LANEY,  VECTORX], [1, 1, 16]>>
#layoutB2 = #iree_vector_ext.layout<<[ BATCHX,  LANEY,  VECTORX], [1, 1, 16]>, <[ BATCHY,  LANEX], [1, 16]>>
#layoutC2 = #iree_vector_ext.layout<<[ BATCHX,  VECTORY,  LANEY,  VECTORX], [1, 8, 2, 1]>, <[ BATCHY,  LANEX], [1, 16]>>

// CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + (s0 floordiv 32) * 16)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 mod 16)>
// CHECK-LABEL: func.func @resolve_wmma_layout_conflict_with_shared_memory
func.func @resolve_wmma_layout_conflict_with_shared_memory(%15 : vector<16x16xf16>,
                                                           %14 : vector<16x16xf16>,
                                                           %16 : vector<16x16xf32>,
                                                           %35 : vector<16x16xf16>,
                                                           %33 : vector<16x16xf32>)
                                                           -> vector<16x16xf32>
  attributes {translation_info = #iree_codegen.translation_info<None
                                                                workgroup_size = [32, 2, 1]
                                                                subgroup_size = 32>} {

  %A = iree_vector_ext.to_layout %15 to #layoutA : vector<16x16xf16>
  %B = iree_vector_ext.to_layout %14 to #layoutB : vector<16x16xf16>
  %C = iree_vector_ext.to_layout %16 to #layoutC : vector<16x16xf32>

  %M1 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                          affine_map<(d0, d1, d2) -> (d1, d2)>,
                                          affine_map<(d0, d1, d2) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel", "reduction"],
                         kind = #vector.kind<add>,
                         iree.amdgpu.mma = #iree_gpu.mma_layout<WMMA_F32_16x16x16_F16>}
        %A, %B, %C : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>

  %TM1 = arith.truncf %M1 : vector<16x16xf32> to vector<16x16xf16>

  %A2 = iree_vector_ext.to_layout %35  to #layoutA2 : vector<16x16xf16>
  %B2 = iree_vector_ext.to_layout %TM1 to #layoutB2 : vector<16x16xf16>
  %C2 = iree_vector_ext.to_layout %33  to #layoutC2 : vector<16x16xf32>

  %M2 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                          affine_map<(d0, d1, d2) -> (d2, d1)>,
                                          affine_map<(d0, d1, d2) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel", "reduction"],
                         kind = #vector.kind<add>,
                         iree.amdgpu.mma = #iree_gpu.mma_layout<WMMA_F32_16x16x16_F16>}
       %A2, %B2, %C2  : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>

  func.return %M2 : vector<16x16xf32>
}
// CHECK-NOT: iree_vector_ext.layout_conflict_resolution
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index

// CHECK: %[[VEC_INIT:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x16xf16
// CHECK: %[[TID_X:.+]] = gpu.thread_id  x
// CHECK: %[[TID_Y:.+]] = gpu.thread_id  y
// CHECK: %[[TID_Z:.+]] = gpu.thread_id  z
// CHECK: %[[SUBGROUP_OFFSET:.+]] = affine.apply #[[$MAP0]]()[%[[TID_X]], %[[TID_Y]], %[[TID_Z]]]
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<16x32xf16, #gpu.address_space<workgroup>>
// CHECK: %[[SUBVIEW:.+]] = memref.subview %[[ALLOC]][0, %[[SUBGROUP_OFFSET]]] [16, 16] [1, 1]
// CHECK: %[[HALF_LANE_ID:.+]] = affine.apply #[[$MAP1]]()[%[[TID_X]]]
// CHECK-COUNT-8: vector.store %{{.+}}, %[[SUBVIEW]][%{{.+}}, %[[HALF_LANE_ID]]]
// CHECK-AFTER: gpu.barrier

// CHECK: %[[LANE_OFFSET:.+]] = arith.addi %[[SUBGROUP_OFFSET]], %[[HALF_LANE_ID]]
// CHECK: %[[LOAD0:.+]] = vector.load %[[ALLOC]][%[[C0]], %[[LANE_OFFSET]]]
// CHECK: %[[INSERT0:.+]] = vector.insert_strided_slice %[[LOAD0]], %[[VEC_INIT]] {offsets = [0, 0, 0], strides = [1]} : vector<1xf16> into vector<1x1x16xf16>
// CHECK: %[[LOAD1:.+]] = vector.load %[[ALLOC]][%[[C1]], %[[LANE_OFFSET]]]
// CHECK: %[[INSERT1:.+]] = vector.insert_strided_slice %[[LOAD1]], %[[INSERT0]] {offsets = [0, 0, 1], strides = [1]} : vector<1xf16> into vector<1x1x16xf16>
// CHECK: %[[LOAD2:.+]] = vector.load %[[ALLOC]][%[[C2]], %[[LANE_OFFSET]]]
// CHECK: %[[INSERT2:.+]] = vector.insert_strided_slice %[[LOAD2]], %[[INSERT1]] {offsets = [0, 0, 2], strides = [1]} : vector<1xf16> into vector<1x1x16xf16>
// CHECK: %[[LOAD3:.+]] = vector.load %[[ALLOC]][%[[C3]], %[[LANE_OFFSET]]]
// CHECK: %[[INSERT3:.+]] = vector.insert_strided_slice %[[LOAD3]], %[[INSERT2]] {offsets = [0, 0, 3], strides = [1]} : vector<1xf16> into vector<1x1x16xf16>
// CHECK: %[[LOAD4:.+]] = vector.load %[[ALLOC]][%[[C4]], %[[LANE_OFFSET]]]
// CHECK: %[[INSERT4:.+]] = vector.insert_strided_slice %[[LOAD4]], %[[INSERT3]] {offsets = [0, 0, 4], strides = [1]} : vector<1xf16> into vector<1x1x16xf16>
// CHECK-COUNT-11: %[[LOADN:.+]] = vector.load %[[ALLOC]]
// CHECK-AFTER: vector.insert_strided_slice %[[LOADN]]

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func {experimental = true} : !transform.any_op
    transform.yield
  }
}

// -----

#row_layout = #iree_vector_ext.per_dim_layout<[BATCHX, LANEY, VECTORX], [2, 4, 4]>
#col_layout = #iree_vector_ext.per_dim_layout<[BATCHY, LANEX], [1, 16]>
#layout0 = #iree_vector_ext.layout<#row_layout, #col_layout>
#row_layout2 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEY, VECTORX], [1, 4, 8]>
#layout1 = #iree_vector_ext.layout<#row_layout2, #col_layout>
#row_layout3 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEY, VECTORX], [4, 2, 4]>
#layout2 = #iree_vector_ext.layout<#row_layout3, #col_layout>

func.func @resolved_layout_conflict(%a : memref<32x16xf16>, %b : memref<32x16xf16>) {
  // CHECK: func.func @resolved_layout_conflict(%[[MEM:.*]]: memref<32x16xf16>, %[[MEM1:.*]]: memref<32x16xf16>
  // CHECK-DAG: %[[CST0:.*]] = arith.constant dense<0.000000e+00> : vector<1x1x8xf16>
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  // CHECK-COUNT-8: vector.load %[[MEM]]
  %vec = vector.transfer_read  %a[%c0, %c0], %cst : memref<32x16xf16>, vector<32x16xf16>
  %vecl = iree_vector_ext.to_layout %vec to #layout1 : vector<32x16xf16>
  // CHECK: %[[R0:.+]] = vector.insert_strided_slice {{.*}} {offsets = [0, 0, 7], strides = [1]} : vector<1xf16> into vector<1x1x8xf16>
  // CHECK: %[[ADD:.*]] = arith.addf %[[R0]], %[[R0]] {{.*}} : vector<1x1x8xf16>
  %vec2 = arith.addf %vecl, %vecl : vector<32x16xf16>
  %vec2l = iree_vector_ext.to_layout %vec2 to #layout0 : vector<32x16xf16>
  // CHECK: %[[R1:.*]] = vector.extract_strided_slice %[[ADD]] {offsets = [0, 0, 0], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<1x1x8xf16> to vector<1x1x4xf16>
  // CHECK: %[[R2:.*]] = vector.extract_strided_slice %[[ADD]] {offsets = [0, 0, 4], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<1x1x8xf16> to vector<1x1x4xf16>
  vector.transfer_write %vec2l, %b[%c0, %c0] {in_bounds = [true, true]} : vector<32x16xf16>, memref<32x16xf16>
  // CHECK-COUNT-8: vector.store {{.*}}, vector<1xf16>
  func.return
}

func.func @unresolved_layout_conflict(%a : memref<32x16xf16>, %b : memref<32x16xf16>) {
  // CHECK: func.func @unresolved_layout_conflict(%[[MEM:.*]]: memref<32x16xf16>, %[[MEM1:.*]]: memref<32x16xf16>
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %vcst = arith.constant dense<0.0> : vector<32x16xf16>
  // CHECK-COUNT-8: vector.load %[[MEM]]
  %vec = vector.transfer_read  %a[%c0, %c0], %cst : memref<32x16xf16>, vector<32x16xf16>
  %vecl = iree_vector_ext.to_layout %vec to #layout1 : vector<32x16xf16>
  // CHECK: iree_vector_ext.to_layout {{.*}}
  %vec2 = arith.addf %vecl, %vcst : vector<32x16xf16>
  // CHECK-COUNT-16: vector.store {{.*}}, vector<1xf16>
  %vec2l = iree_vector_ext.to_layout %vec2 to #layout2 : vector<32x16xf16>
  vector.transfer_write %vec2l, %b[%c0, %c0] {in_bounds = [true, true]} : vector<32x16xf16>, memref<32x16xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func {experimental = true} : !transform.any_op
    transform.yield
  }
}
