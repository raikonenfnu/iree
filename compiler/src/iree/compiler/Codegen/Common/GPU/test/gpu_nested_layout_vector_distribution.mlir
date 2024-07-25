// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize --cse %s | FileCheck %s

#layout_col_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [4, 8],
  elements_per_thread     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [8, 1]
>

// CHECK: #[[$MAP:.+]] = affine_map<()[s0] -> ((s0 floordiv 8) * 4 - ((s0 floordiv 8) floordiv 4) * 16)>
// CHECK: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 mod 8)>
// CHECK: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0 mod 8 + 8)>
// CHECK-LABEL: @distribute_transfer_read_col_major
func.func @distribute_transfer_read_col_major(%arg0: memref<32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_result_0" = #layout_col_major}
                  : memref<32x32xf16>, vector<16x16xf16>
  func.return %root : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[X:.+]] = affine.apply #[[$MAP]]()[%[[IDX]]]
// CHECK: %[[Y:.+]] = affine.apply #[[$MAP1]]()[%[[IDX]]]
// CHECK: %[[RD00:.+]] = vector.transfer_read %arg0[%[[X]], %[[Y]]], {{.*}} : memref<32x32xf16>, vector<4x1xf16>
// CHECK: vector.insert_strided_slice %[[RD00]], %{{.*}} {offsets = [0, 0, 0, 0, 0, 0], strides = [1, 1]} : vector<4x1xf16> into vector<1x2x1x1x4x1xf16>
// CHECK: %[[X_PLUS_BATCH:.+]] = affine.apply #[[$MAP2]]()[%[[IDX]]]
// CHECK: vector.transfer_read %arg0[%[[X]], %[[X_PLUS_BATCH]]], %{{.*}} {in_bounds = [true, true]} : memref<32x32xf16>, vector<4x1xf16>
// CHECK: vector.insert_strided_slice {{.*}} {offsets = [0, 1, 0, 0, 0, 0]
// CHECK: iree_vector_ext.to_simd %{{.*}} : vector<1x2x1x1x4x1xf16> -> vector<16x16xf16>

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [2, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [8, 1],
  elements_per_thread     = [1, 8],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 1]
>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1 - (s1 floordiv 8) * 8)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0, s1] -> (s0 + s1 - (s1 floordiv 8) * 8 + 8)>

func.func @distribute_transfer_read_row_major_with_nontrivial_index(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
           "__vector_layout_test_anchor_result_0" = #layout_row_major}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  func.return %root : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
// CHECK-LABEL: @distribute_transfer_read_row_major_with_nontrivial_index
// CHECK-SAME:    %[[I0:.+]]: index, %[[I1:.+]]: index

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[OFF0:.+]] = affine.apply #[[$MAP]]()[%[[I0]], %[[IDX]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[OFF0]], %[[I1]]]
// CHECK: %[[OFF1:.+]] = affine.apply #[[$MAP1]]()[%[[I1]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[OFF0]], %[[OFF1]]]
// CHECK: %[[OFF2:.+]] = affine.apply #[[$MAP2]]()[%[[I0]], %[[IDX]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[OFF2]], %[[I1]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[OFF2]], %[[OFF1]]]

// -----

#layout_col_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [4, 8],
  elements_per_thread     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [8, 1]
>

func.func @distribute_transfer_read_col_major_with_broadcast(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (0, 0)>,
           "__vector_layout_test_anchor_result_0" = #layout_col_major}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  func.return %root : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (0, 0)>

// CHECK-LABEL: @distribute_transfer_read_col_major_with_broadcast
// CHECK-SAME:    %[[I0:.+]]: index, %[[I1:.+]]: index

// CHECK: %[[BROADCAST_READ:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[I0]], %[[I1]]], %{{.*}} permutation_map = #[[$MAP]]
// CHECK: vector.insert_strided_slice %[[BROADCAST_READ]], %{{.*}} {offsets = [0, 0, 0, 0, 0, 0]
// CHECK: vector.insert_strided_slice %[[BROADCAST_READ]], %{{.*}} {offsets = [0, 1, 0, 0, 0, 0]

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [2, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [8, 1],
  elements_per_thread     = [1, 8],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 1]
>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1 - (s1 floordiv 8) * 8)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0, s1] -> (s0 + s1 - (s1 floordiv 8) * 8 + 8)>
// CHECK-DAG: #[[$PERM:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>

func.func @distribute_transfer_read_row_major_transpose(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
           "__vector_layout_test_anchor_result_0" = #layout_row_major}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  func.return %root : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: @distribute_transfer_read_row_major_transpose
// CHECK-SAME:    %[[I0:.+]]: index, %[[I1:.+]]: index

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[LIN_ID0:.+]] = affine.apply #[[$MAP:.+]]()[%[[I1]], %[[IDX]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[I0]], %[[LIN_ID0]]], {{.*}} permutation_map = #[[$PERM]]
// CHECK: %[[I0_PLUS_8:.+]] = affine.apply #[[$MAP1]]()[%[[I0]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[I0_PLUS_8]], %[[LIN_ID0]]], {{.*}} permutation_map = #[[$PERM]]
// CHECK: %[[LIN_ID1:.+]] = affine.apply #[[$MAP2]]()[%[[I1]], %[[IDX]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[I0]], %[[LIN_ID1]]], {{.*}} permutation_map = #[[$PERM]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[I0_PLUS_8]], %[[LIN_ID1]]], %cst_0 {in_bounds = [true, true], permutation_map = #[[$PERM]]} : memref<32x32x32x32xf16>, vector<1x8xf16>

// -----

#layout_col_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [4, 8],
  elements_per_thread     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [8, 1]
>

// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>

// CHECK-LABEL: @distribute_transfer_read_col_major_transpose
func.func @distribute_transfer_read_col_major_transpose(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
           "__vector_layout_test_anchor_result_0" = #layout_col_major}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  func.return %root : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: vector.transfer_read {{.*}} permutation_map = #[[$MAP2]]
// CHECK: vector.transfer_read {{.*}} permutation_map = #[[$MAP2]]

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [7, 3, 1, 1],
  batches_per_subgroup    = [3, 5, 2, 1],
  outers_per_batch        = [1, 1, 2, 4],
  threads_per_outer       = [1, 1, 2, 2],
  elements_per_thread     = [1, 1, 1, 2],

  subgroup_strides        = [3, 1, 1, 1],
  thread_strides          = [1, 1, 1, 2]
>

func.func @distribute_transfer_read_row_major_with_permutations(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<21x15x8x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true, true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d0, d3, 0, d1)>,
           "__vector_layout_test_anchor_result_0" = #layout}
                  : memref<32x32x32x32xf16>, vector<21x15x8x16xf16>
  func.return %root : vector<21x15x8x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
// CHECK-LABEL: @distribute_transfer_read_row_major_with_permutations

// Verify that there are (batch0: 3) * (batch1: 5) * (outer3: 4) = 60 total
// unique transfer read ops. The broadcasted dimension (2) CSEs the duplicate
// reads.
// CHECK-COUNT-60: vector.transfer_read

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1],
  batches_per_subgroup    = [1],
  outers_per_batch        = [1],
  threads_per_outer       = [4],
  elements_per_thread     = [4],

  subgroup_strides        = [1],
  thread_strides          = [16]
>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> ((s0 floordiv 16) * 4 - ((s0 floordiv 16) floordiv 4) * 16)>

// CHECK-LABEL: @distribute_transfer_read_broadcast
func.func @distribute_transfer_read_broadcast(%arg0: memref<32x32xf16>) -> vector<16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst
          {in_bounds = [true],
           "__vector_layout_test_anchor_result_0" = #layout}
                  : memref<32x32xf16>, vector<16xf16>
  func.return %root : vector<16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[LANEY:.+]] = affine.apply #[[$MAP]]()[%[[IDX]]]
// CHECK: %[[RD:.+]] = vector.transfer_read %{{.*}}[%c0, %[[LANEY:.+]]], {{.*}} : memref<32x32xf16>, vector<4xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [2],
  batches_per_subgroup    = [1],
  outers_per_batch        = [1],
  threads_per_outer       = [16],
  elements_per_thread     = [4],

  subgroup_strides        = [1],
  thread_strides          = [1]
>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 64 - ((s0 floordiv 64) floordiv 2) * 128 - (s0 floordiv 16) * 64)>

// CHECK-LABEL: @distribute_transfer_read_broadcast2
func.func @distribute_transfer_read_broadcast2(%arg0: memref<32x128xf16>) -> vector<128xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst
          {in_bounds = [true],
           "__vector_layout_test_anchor_result_0" = #layout}
                  : memref<32x128xf16>, vector<128xf16>
  func.return %root : vector<128xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[LANEY:.+]] = affine.apply #[[$MAP]]()[%[[IDX]]]
// CHECK: %[[RD:.+]] = vector.transfer_read %{{.*}}[%c0, %[[LANEY:.+]]], {{.*}} : memref<32x128xf16>, vector<4xf16>

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [2, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [8, 1],
  elements_per_thread     = [1, 8],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 1]
>

// CHECK: #[[$MAP:.+]]  = affine_map<()[s0] -> (s0 mod 8)>
// CHECK: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 mod 8 + 8)>

// CHECK-LABEL: @distribute_transfer_write_row_major
func.func @distribute_transfer_write_row_major(%root: vector<16x16xf16>, %alloc: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %root, %alloc[%c0, %c0]
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_operand_0" = #layout_row_major}
                  : vector<16x16xf16>, memref<64x64xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[LANEX:.+]] = affine.apply #[[$MAP]]()[%[[IDX]]]
// CHECK: %[[SLICE:.+]] = vector.extract %{{.*}}[0, 0, 0, 0] : vector<1x8xf16> from vector<2x2x1x1x1x8xf16>
// CHECK: vector.transfer_write %[[SLICE]], %{{.*}}[%[[LANEX]], %c0] {in_bounds = [true, true]} : vector<1x8xf16>, memref<64x64xf16>
// CHECK: vector.extract %{{.*}}[0, 1, 0, 0]
// CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[LANEX]], %c8]
// CHECK: %[[LANEX_PLUS_VECDIMX:.+]] = affine.apply #[[$MAP1]]()[%[[IDX]]]
// CHECK: vector.extract %{{.*}}[1, 0, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%[[LANEX_PLUS_VECDIMX]], %c0]
// CHECK: vector.extract %{{.*}}[1, 1, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%[[LANEX_PLUS_VECDIMX]], %c8]

// -----

#layout_col_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [4, 8],
  elements_per_thread     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [8, 1]
>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> ((s0 floordiv 8) * 4 - ((s0 floordiv 8) floordiv 4) * 16)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 mod 8)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0 mod 8 + 8)>

// CHECK-LABEL: @distribute_transfer_write_col_major
func.func @distribute_transfer_write_col_major(%root: vector<16x16xf16>, %alloc: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %root, %alloc[%c0, %c0]
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_operand_0" = #layout_col_major}
                  : vector<16x16xf16>, memref<64x64xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[LANEY:.+]] = affine.apply #[[$MAP]]()[%[[IDX]]]
// CHECK: %[[LANEY2:.+]] = affine.apply #[[$MAP1]]()[%[[IDX]]]
// CHECK: vector.extract %{{.*}}[0, 0, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%[[LANEY]], %[[LANEY2]]]
// CHECK: %[[LANEX:.+]] = affine.apply #[[$MAP2]]()[%[[IDX]]]
// CHECK: vector.extract %{{.*}}[0, 1, 0, 0]
// CHECK: vector.transfer_write {{.*}}[%[[LANEY]], %[[LANEX]]]

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [2, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [8, 1],
  elements_per_thread     = [1, 8],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 1]
>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1 - (s1 floordiv 8) * 8)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<()[s0, s1] -> (s0 + s1 - (s1 floordiv 8) * 8 + 8)>

func.func @distribute_transfer_write_row_major_with_nontrivial_index(%root: vector<16x16xf16>, %a: index, %b: index, %alloc: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %root, %alloc[%c0, %c0, %a, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
           "__vector_layout_test_anchor_operand_0" = #layout_row_major}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: @distribute_transfer_write_row_major_with_nontrivial_index
// CHECK-SAME:    vector<16x16xf16>, %[[I0:.+]]: index, %[[I1:.+]]: index

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[LIN_ID0:.+]] = affine.apply #[[$MAP]]()[%[[I1]], %[[IDX]]]
// CHECK: vector.extract %{{.*}}[0, 0, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%c0, %c0, %[[I0]], %[[LIN_ID0]]] {{.*}} permutation_map = #[[$MAP1]]
// CHECK: %[[LIN_ID1:.+]] = affine.apply #[[$MAP2]]()[%[[I0]]]
// CHECK: vector.extract %{{.*}}[0, 1, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%c0, %c0, %[[LIN_ID1]], %[[LIN_ID0]]] {{.*}} permutation_map = #[[$MAP1]]
// CHECK: %[[LIN_ID2:.+]] = affine.apply #[[$MAP3]]()[%[[I1]], %[[IDX]]]
// CHECK: vector.extract %{{.*}}[1, 0, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%c0, %c0, %[[I0]], %[[LIN_ID2]]] {{.*}} permutation_map = #[[$MAP1]]
// CHECK: vector.extract %{{.*}}[1, 1, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%c0, %c0, %[[LIN_ID1]], %[[LIN_ID2]]] {{.*}} permutation_map = #[[$MAP1]]

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [2, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [8, 1],
  elements_per_thread     = [1, 8],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 1]
>

func.func @distribute_transfer_read_write(%a: index, %b: index,
                                          %arg0: memref<32x32x32x32xf16>,
                                          %arg1: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
           "__vector_layout_test_anchor_result_0" = #layout_row_major}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  vector.transfer_write %root, %arg1[%c0, %c0, %a, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
           "__vector_layout_test_anchor_operand_0" = #layout_row_major}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>
  return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[B00:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[LANEX:[a-zA-Z0-9]+]], %[[OFFSET0:[a-zA-Z0-9]+]]]
// CHECK: %[[B10:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[LANEX]], %[[OFFSET1:[a-zA-Z0-9]+]]]
// CHECK: %[[B01:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[LANEX_PLUS_BATCH:[a-zA-Z0-9]+]], %[[OFFSET0]]]
// CHECK: %[[B11:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[LANEX_PLUS_BATCH]], %[[OFFSET1]]]
// CHECK: vector.transfer_write %[[B00]], %{{.*}}[%c0, %c0, %[[LANEX]], %[[OFFSET0]]]
// CHECK: vector.transfer_write %[[B10]], %{{.*}}[%c0, %c0, %[[LANEX]], %[[OFFSET1]]]
// CHECK: vector.transfer_write %[[B01]], %{{.*}}[%c0, %c0, %[[LANEX_PLUS_BATCH]], %[[OFFSET0]]]
// CHECK: vector.transfer_write %[[B11]], %{{.*}}[%c0, %c0, %[[LANEX_PLUS_BATCH]], %[[OFFSET1]]]

// -----

// A: shape = 128x8, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [4, 1],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [32, 2],
  elements_per_thread     = [1, 4],

  subgroup_strides        = [2, 1],
  thread_strides          = [1, 32]
>

// B: shape = 8x64, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 2],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [2, 32],
  elements_per_thread     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [32, 1]
>

// C: shape = 128x64, layout = layoutC
#layout_c = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [4, 2],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [4, 1],
  threads_per_outer       = [2, 32],
  elements_per_thread     = [4, 1],

  subgroup_strides        = [2, 1],
  thread_strides          = [32, 1]
>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 + (s0 floordiv 128) * 32 - ((s0 floordiv 128) floordiv 4) * 128 - (s0 floordiv 32) * 32)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> ((s0 floordiv 32) * 4 - ((s0 floordiv 32) floordiv 2) * 8)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0 + (s0 floordiv 64) * 32 - ((s0 floordiv 64) floordiv 2) * 64 - (s0 floordiv 32) * 32)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<()[s0] -> ((s0 floordiv 128) * 32 - ((s0 floordiv 128) floordiv 4) * 128 + (s0 floordiv 32) * 4 - ((s0 floordiv 32) floordiv 2) * 8)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<()[s0] -> ((s0 floordiv 128) * 32 - ((s0 floordiv 128) floordiv 4) * 128 + (s0 floordiv 32) * 4 - ((s0 floordiv 32) floordiv 2) * 8 + 8)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<()[s0] -> ((s0 floordiv 128) * 32 - ((s0 floordiv 128) floordiv 4) * 128 + (s0 floordiv 32) * 4 - ((s0 floordiv 32) floordiv 2) * 8 + 16)>
// CHECK-DAG: #[[$MAP6:.+]] = affine_map<()[s0] -> ((s0 floordiv 128) * 32 - ((s0 floordiv 128) floordiv 4) * 128 + (s0 floordiv 32) * 4 - ((s0 floordiv 32) floordiv 2) * 8 + 24)>
// CHECK-LABEL: @mfma_64x128x8_read
func.func @mfma_64x128x8_read(%mem: memref<128x8xf16>,
                              %mem1: memref<8x64xf16>,
                              %mem2: memref<128x64xf16>)
                -> (vector<128x8xf16>, vector<8x64xf16>, vector<128x64xf16>) {

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16

  // CHECK: %[[IDX:.+]] = gpu.thread_id  x

  // CHECK-DAG: %[[LHSM:.+]] = affine.apply #[[$MAP]]()[%[[IDX]]]
  // LHSK = RHSK
  // CHECK-DAG: %[[LHSK:.+]] = affine.apply #[[$MAP1]]()[%[[IDX]]]
  // ACCN = RHSN
  // CHECK-DAG: %[[RHSN:.+]] = affine.apply #[[$MAP2]]()[%[[IDX]]]

  // M is unrolled 4 times.
  // CHECK-DAG: %[[ACCM0:.+]] = affine.apply #[[$MAP3]]()[%[[IDX]]]
  // CHECK-DAG: %[[ACCM1:.+]] = affine.apply #[[$MAP4]]()[%[[IDX]]]
  // CHECK-DAG: %[[ACCM2:.+]] = affine.apply #[[$MAP5]]()[%[[IDX]]]
  // CHECK-DAG: %[[ACCM3:.+]] = affine.apply #[[$MAP6]]()[%[[IDX]]]

  // M, K
  // CHECK-DAG: transfer_read %{{.*}}[%[[LHSM]], %[[LHSK]]]
  // K, N
  // CHECK-DAG: transfer_read %{{.*}}[%[[LHSK]], %[[RHSN]]]
  // M, N
  // CHECK-DAG: transfer_read %{{.*}}[%[[ACCM0]], %[[RHSN]]]
  // CHECK-DAG: transfer_read %{{.*}}[%[[ACCM1]], %[[RHSN]]]
  // CHECK-DAG: transfer_read %{{.*}}[%[[ACCM2]], %[[RHSN]]]
  // CHECK-DAG: transfer_read %{{.*}}[%[[ACCM3]], %[[RHSN]]

  %a = vector.transfer_read %mem[%c0, %c0], %cst
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_result_0" = #layout_a}
  : memref<128x8xf16>, vector<128x8xf16>
  %b = vector.transfer_read %mem1[%c0, %c0], %cst
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_result_0" = #layout_b}
  : memref<8x64xf16>, vector<8x64xf16>
  %c = vector.transfer_read %mem2[%c0, %c0], %cst
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_result_0" = #layout_c}
  : memref<128x64xf16>, vector<128x64xf16>

  return %a, %b, %c : vector<128x8xf16>, vector<8x64xf16>, vector<128x64xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [2, 1],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [32, 2],
  elements_per_thread     = [1, 4],

  subgroup_strides        = [2, 1],
  thread_strides          = [1, 32]
>

func.func @transposed_read_64x8(%mem: memref<8x64xf16>)
                -> (vector<64x8xf16>) {

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16

  %read = vector.transfer_read %mem[%c0, %c0], %cst
          {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
           "__vector_layout_test_anchor_result_0" = #layout}
  : memref<8x64xf16>, vector<64x8xf16>

  return %read : vector<64x8xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 + (s0 floordiv 128) * 32 - ((s0 floordiv 128) floordiv 2) * 64 - (s0 floordiv 32) * 32)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> ((s0 floordiv 32) * 4 - ((s0 floordiv 32) floordiv 2) * 8)>

// CHECK-LABEL: @transposed_read_64x8

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK-DAG: %[[M:.+]] = affine.apply #[[$MAP]]()[%[[IDX]]]
// CHECK-DAG: %[[N:.+]] = affine.apply #[[$MAP1]]()[%[[IDX]]]
// CHECK: transfer_read %{{.*}}[%[[N]], %[[M]]

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [2, 2],
  batches_per_subgroup = [2, 4],
  outers_per_batch = [1, 1],
  threads_per_outer = [4, 16],
  elements_per_thread = [4, 1],
  subgroup_strides = [2, 1],
  thread_strides   = [16, 1]
>

func.func @broadcast(%src: vector<128xf16>) -> (vector<64x128xf16>) {
  %bcast = vector.broadcast %src {"__vector_layout_test_anchor_result_0" = #layout}
    : vector<128xf16> to vector<64x128xf16>
  return %bcast : vector<64x128xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: vector.extract {{.*}}[0, 0] : vector<1xf16> from vector<4x1x1xf16>
// CHECK: vector.broadcast {{.*}} : vector<1xf16> to vector<4x1xf16>
// CHECK: vector.insert {{.*}} [0, 0, 0, 0] : vector<4x1xf16> into vector<2x4x1x1x4x1xf16>
// CHECK: vector.extract {{.*}}[1, 0] : vector<1xf16> from vector<4x1x1xf16>
// CHECK: vector.broadcast {{.*}} : vector<1xf16> to vector<4x1xf16>
// CHECK: vector.insert {{.*}} [0, 1, 0, 0] : vector<4x1xf16> into vector<2x4x1x1x4x1xf16>
// CHECK: vector.extract {{.*}}[2, 0] : vector<1xf16> from vector<4x1x1xf16>
// CHECK: vector.broadcast {{.*}} : vector<1xf16> to vector<4x1xf16>
// CHECK: vector.insert {{.*}} [0, 2, 0, 0] : vector<4x1xf16> into vector<2x4x1x1x4x1xf16>
// CHECK: vector.extract {{.*}}[3, 0] : vector<1xf16> from vector<4x1x1xf16>
// CHECK: vector.broadcast {{.*}} : vector<1xf16> to vector<4x1xf16>

// CHECK: vector.insert {{.*}} [1, 0, 0, 0] : vector<4x1xf16> into vector<2x4x1x1x4x1xf16>
// CHECK: vector.insert {{.*}} [1, 1, 0, 0] : vector<4x1xf16> into vector<2x4x1x1x4x1xf16>
// CHECK: vector.insert {{.*}} [1, 2, 0, 0] : vector<4x1xf16> into vector<2x4x1x1x4x1xf16>
// CHECK: vector.insert {{.*}} [1, 3, 0, 0] : vector<4x1xf16> into vector<2x4x1x1x4x1xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [2, 2, 2],
  batches_per_subgroup = [2, 2, 1],
  outers_per_batch = [2, 1, 1],
  threads_per_outer = [4, 16, 8],
  elements_per_thread = [1, 4, 4],
  subgroup_strides = [4, 2, 1],
  thread_strides   = [128, 8, 1]
>

func.func @broadcast(%src: vector<64xf16>) -> (vector<32x256x64xf16>) {
  %bcast = vector.broadcast %src {"__vector_layout_test_anchor_result_0" = #layout}
    : vector<64xf16> to vector<32x256x64xf16>
  return %bcast : vector<32x256x64xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[EXTRACT:.+]] = vector.extract %{{.*}}[0, 0] : vector<4xf16> from vector<1x1x4xf16>
// CHECK: %[[BCAST:.+]] = vector.broadcast %[[EXTRACT]] : vector<4xf16> to vector<1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [0, 0, 0, 0, 0, 0] : vector<1x4x4xf16> into vector<2x2x1x2x1x1x1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [0, 0, 0, 1, 0, 0] : vector<1x4x4xf16> into vector<2x2x1x2x1x1x1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [0, 1, 0, 0, 0, 0] : vector<1x4x4xf16> into vector<2x2x1x2x1x1x1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [0, 1, 0, 1, 0, 0] : vector<1x4x4xf16> into vector<2x2x1x2x1x1x1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [1, 0, 0, 0, 0, 0] : vector<1x4x4xf16> into vector<2x2x1x2x1x1x1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [1, 0, 0, 1, 0, 0] : vector<1x4x4xf16> into vector<2x2x1x2x1x1x1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [1, 1, 0, 0, 0, 0] : vector<1x4x4xf16> into vector<2x2x1x2x1x1x1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [1, 1, 0, 1, 0, 0] : vector<1x4x4xf16> into vector<2x2x1x2x1x1x1x4x4xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [2, 2, 2],
  batches_per_subgroup = [2, 2, 1],
  outers_per_batch = [2, 1, 1],
  threads_per_outer = [4, 16, 8],
  elements_per_thread = [1, 4, 4],
  subgroup_strides = [4, 2, 1],
  thread_strides = [128, 8, 1]
>

func.func @scalar_broadcast(%src: f16) -> (vector<32x256x64xf16>) {
  %bcast = vector.broadcast %src {"__vector_layout_test_anchor_result_0" = #layout}
    : f16 to vector<32x256x64xf16>
  return %bcast : vector<32x256x64xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @scalar_broadcast
// CHECK-SAME:  (%[[SRC:.*]]: f16)
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[SRC]] : f16 to vector<2x2x1x2x1x1x1x4x4xf16>
// CHECK: iree_vector_ext.to_simd %[[BCAST]] : vector<2x2x1x2x1x1x1x4x4xf16> -> vector<32x256x64xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [2, 2],
  batches_per_subgroup = [2, 4],
  outers_per_batch = [2, 1],
  threads_per_outer = [4, 16],
  elements_per_thread = [2, 2],
  subgroup_strides = [2, 1],
  thread_strides = [16, 1]
>

func.func @transpose(%src: vector<256x64xf16>) -> (vector<64x256xf16>) {
  %transp = vector.transpose %src, [1, 0] {"__vector_layout_test_anchor_result_0" = #layout}
    : vector<256x64xf16> to vector<64x256xf16>
  %sqrt = math.sqrt %transp : vector<64x256xf16>
  return %sqrt : vector<64x256xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @transpose
// CHECK: iree_vector_ext.to_simt %{{.*}} : vector<256x64xf16> -> vector<4x2x1x2x2x2xf16>
// CHECK: vector.transpose %{{.*}}, [1, 0, 3, 2, 5, 4] : vector<4x2x1x2x2x2xf16> to vector<2x4x2x1x2x2xf16>
// CHECK: math.sqrt %{{.*}} : vector<2x4x2x1x2x2xf16>
// CHECK: iree_vector_ext.to_simd %{{.*}} : vector<2x4x2x1x2x2xf16> -> vector<64x256xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [2, 2],
  batches_per_subgroup = [2, 4],
  outers_per_batch = [2, 1],
  threads_per_outer = [4, 16],
  elements_per_thread = [2, 2],
  subgroup_strides = [2, 1],
  thread_strides = [16, 1]
>

func.func @transpose(%src: vector<64x256xf16>) -> (vector<256x64xf16>) {
  %transp = vector.transpose %src, [1, 0] {"__vector_layout_test_anchor_operand_0" = #layout}
    : vector<64x256xf16> to vector<256x64xf16>
  %sqrt = math.sqrt %transp : vector<256x64xf16>
  return %sqrt : vector<256x64xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK:      #[[$LAYOUT:.+]] = #iree_vector_ext.nested_layout
// CHECK-SAME:   subgroups_per_workgroup = [2, 2],
// CHECK-SAME:   batches_per_subgroup = [4, 2]
// CHECK-SAME:   outers_per_batch = [1, 2]
// CHECK-SAME:   threads_per_outer = [16, 4]
// CHECK-SAME:   elements_per_thread = [2, 2]
// CHECK-SAME:   subgroup_strides = [1, 2],
// CHECK-SAME:   thread_strides = [1, 16]

// CHECK-LABEL: func @transpose
// CHECK: iree_vector_ext.to_simt %{{.*}} : vector<64x256xf16> -> vector<2x4x2x1x2x2xf16>
// CHECK: vector.transpose %{{.*}}, [1, 0, 3, 2, 5, 4] : vector<2x4x2x1x2x2xf16> to vector<4x2x1x2x2x2xf16>
// CHECK: math.sqrt %{{.*}} : vector<4x2x1x2x2x2xf16>
// CHECK: iree_vector_ext.to_simd %{{.*}} : vector<4x2x1x2x2x2xf16> -> vector<256x64xf16>
// CHECK: return {{.*}}#[[$LAYOUT]]

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [2, 1, 1],
  batches_per_subgroup = [1, 2, 4],
  outers_per_batch = [1, 1, 1],
  threads_per_outer = [4, 8, 2],
  elements_per_thread = [4, 1, 2],

  subgroup_strides = [1, 1, 1],
  thread_strides = [16, 2, 1]
>

func.func @transpose_3d(%arr: memref<32x32x32xf16>) -> () {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.0 : f16
  %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
  %root = vector.transfer_read %arr[%c0, %c0, %c0], %cst_0 {
    in_bounds = [true, true, true],
    "__vector_layout_test_anchor_result_0" = #layout
  } : memref<32x32x32xf16>, vector<32x16x16xf16>
  %t = vector.transpose %root, [1, 2, 0] : vector<32x16x16xf16> to vector<16x16x32xf16>
  vector.transfer_write %t, %arr[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<16x16x32xf16>, memref<32x32x32xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 2) * 32 + (s0 floordiv 16) * 4 - ((s0 floordiv 16) floordiv 4) * 16)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> ((s0 floordiv 2) mod 8)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 2) * 4)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 2) * 4 + 4)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 2) * 4 + 8)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 2) * 4 + 12)>
// CHECK-DAG: #[[$MAP6:.+]] = affine_map<()[s0] -> ((s0 floordiv 2) mod 8 + 8)>

// CHECK-LABEL: func @transpose_3d
// CHECK-DAG:         %[[IDX:.+]] = gpu.thread_id  x

// CHECK-DAG:         %[[DIM:.+]]  = affine.apply #[[$MAP]]()[%[[IDX]]]
// CHECK-DAG:         %[[DIM1:.+]] = affine.apply #[[$MAP1]]()[%[[IDX]]]
// CHECK-DAG:         %[[DIM2:.+]] = affine.apply #[[$MAP2]]()[%[[IDX]]]
// CHECK-DAG:         %[[DIM3:.+]] = affine.apply #[[$MAP3]]()[%[[IDX]]]
// CHECK-DAG:         %[[DIM4:.+]] = affine.apply #[[$MAP4]]()[%[[IDX]]]
// CHECK-DAG:         %[[DIM5:.+]] = affine.apply #[[$MAP5]]()[%[[IDX]]]
// CHECK-DAG:         %[[DIM6:.+]] = affine.apply #[[$MAP6]]()[%[[IDX]]]
// CHECK-DAG:         %[[RD0:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[DIM1]], %[[DIM2]]], {{.*}} : memref<32x32x32xf16>, vector<4x1x2xf16>
// CHECK-DAG:         %[[RD1:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[DIM1]], %[[DIM3]]]
// CHECK-DAG:         %[[RD2:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[DIM1]], %[[DIM4]]]
// CHECK-DAG:         %[[RD3:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[DIM1]], %[[DIM5]]]
// CHECK-DAG:         %[[RD4:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[DIM6]], %[[DIM2]]]
// CHECK-DAG:         %[[RD5:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[DIM6]], %[[DIM3]]]
// CHECK-DAG:         %[[RD6:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[DIM6]], %[[DIM4]]]
// CHECK-DAG:         %[[RD7:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[DIM6]], %[[DIM5]]]

// CHECK:         vector.transpose %{{.*}}, [1, 2, 0, 4, 5, 3, 7, 8, 6] : vector<1x2x4x1x1x1x4x1x2xf16> to vector<2x4x1x1x1x1x1x2x4xf16>

// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[DIM1]], %[[DIM2]], %[[DIM]]] {{.*}} : vector<1x2x4xf16>, memref<32x32x32xf16>
// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[DIM1]], %[[DIM3]], %[[DIM]]]
// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[DIM1]], %[[DIM4]], %[[DIM]]]
// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[DIM1]], %[[DIM5]], %[[DIM]]]
// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[DIM6]], %[[DIM2]], %[[DIM]]]
// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[DIM6]], %[[DIM3]], %[[DIM]]]
// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[DIM6]], %[[DIM4]], %[[DIM]]]
// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[DIM6]], %[[DIM5]], %[[DIM]]]

// -----

#nested = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  // We are reducing along dim=1, so each thread will reduce
  // 2 batches x 4 elements = 8 elements.
  batches_per_subgroup = [2, 2],
  outers_per_batch = [1, 1],
  // We are reducing on dim=1, which is distributed over 4 threads. Based
  // on the subgroup basis and thread order, the shuffle offset is 16.
  threads_per_outer = [16, 4],
  elements_per_thread = [1, 4],

  subgroup_strides = [1, 1],
  thread_strides = [1, 16]
>

func.func @mfma_16x16x16_out_reduced_dim1(%arg0: vector<32x32xf32>, %arg1: vector<32xf32>) -> vector<32xf32> {
  %0 = vector.multi_reduction <maximumf>, %arg0, %arg1
  {
    __vector_layout_test_anchor_operand_0 = #nested
  } [1] : vector<32x32xf32> to vector<32xf32>
  return %0 : vector<32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @mfma_16x16x16_out_reduced_dim1
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : i32
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : i32
// CHECK-DAG: %[[IDENTITY:.*]] = arith.constant dense<0xFF800000> : vector<2x1x1xf32>
// CHECK-DAG: %[[DARG0:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<32x32xf32> -> vector<2x2x1x1x1x4xf32>
// CHECK-DAG: %[[DARG1:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<32xf32> -> vector<2x1x1xf32>
// Local reduction
// CHECK: vector.multi_reduction <maximumf>, %[[DARG0]], %[[IDENTITY]] [1, 3, 5] : vector<2x2x1x1x1x4xf32> to vector<2x1x1xf32>
// Global reduction
// CHECK: gpu.shuffle  xor %{{.*}}, %[[C16]], %[[C64]] : f32
// CHECK: gpu.shuffle  xor %{{.*}}, %[[C32]], %[[C64]] : f32
// CHECK: gpu.shuffle  xor %{{.*}}, %[[C16]], %[[C64]] : f32
// CHECK: gpu.shuffle  xor %{{.*}}, %[[C32]], %[[C64]] : f32
// Accumulator reduction
// CHECK: %[[ACC_REDUC:.+]] = arith.maximumf %{{.*}}, %[[DARG1]] : vector<2x1x1xf32>
// CHECK: iree_vector_ext.to_simd %[[ACC_REDUC]] : vector<2x1x1xf32> -> vector<32xf32>

// -----

#nested = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  // We are reducing along dim=1, so each thread will reduce
  // 4 batches x 4 elements = 16 elements.
  batches_per_subgroup    = [1, 4],
  outers_per_batch        = [1, 1],
  // We are reducing on dim=1, which is distributed over 2 threads. Based
  // on the subgroup basis and thread order, the shuffle offset is 32.
  threads_per_outer       = [32, 2],
  elements_per_thread     = [1, 4],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 32]
>

func.func @mfma_32x32x8_out_reduced_dim1(%arg0: vector<32x32xf32>, %arg1: vector<32xf32>) -> vector<32xf32> {
  %0 = vector.multi_reduction <maximumf>, %arg0, %arg1
  {
    __vector_layout_test_anchor_operand_0 = #nested
  } [1] : vector<32x32xf32> to vector<32xf32>
  return %0 : vector<32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @mfma_32x32x8_out_reduced_dim1
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : i32
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : i32
// Local reduction
// CHECK: vector.multi_reduction <maximumf>, %{{.*}}, %{{.*}} [1, 3, 5] : vector<1x4x1x1x1x4xf32> to vector<1x1x1xf32>
// Global reduction
// CHECK: gpu.shuffle  xor %{{.*}}, %[[C32]], %[[C64]] : f32
// Accumulator reduction
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : vector<1x1x1xf32>
