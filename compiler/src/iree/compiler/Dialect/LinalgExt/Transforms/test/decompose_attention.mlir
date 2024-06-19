// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-linalg-ext-tile-attention{tileSize=32},iree-linalg-ext-decompose-attention{tileSize=32}),cse)" %s | FileCheck %s

func.func @attention(%query: tensor<1x1024x64xf32>, %key: tensor<1x1024x64xf32>, %value: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> {
  %0 = tensor.empty() : tensor<1x1024x64xf32>
  %scale = arith.constant 0.05 : f32
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale : tensor<1x1024x64xf32>, tensor<1x1024x64xf32>, tensor<1x1024x64xf32>, f32) outs(%0 : tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32>
  return %1 : tensor<1x1024x64xf32>
}

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG:  #[[$MAP2:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @attention
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<1x1024x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> {
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<1x1024x64xf32>
// CHECK:        %[[D1:.+]] = tensor.empty() : tensor<1024x64xf32>
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<1024x64xf32>) ->
// CHECK-SAME:     tensor<1024x64xf32>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant -1.000000e+30 : f32
// CHECK:        %[[D3:.+]] = tensor.empty() : tensor<1024xf32>
// CHECK:        %[[D4:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:        %[[D6:.+]]:3 = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C32]]
// CHECK-SAME:     iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[D2]], %[[ARG5:[a-zA-Z0-9_]+]] = %[[D4]],
// CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>) {
// CHECK:          %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[ARG3]], 0] [1, 32, 64] [1, 1, 1] :
// CHECK-SAME:       tensor<1x1024x64xf32> to tensor<32x64xf32>
// CHECK:          %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG2]][0, %[[ARG3]], 0] [1, 32, 64] [1, 1, 1] :
// CHECK-SAME:       tensor<1x1024x64xf32> to tensor<32x64xf32>
// CHECK:          %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// CHECK-SAME:       tensor<1x1024x64xf32> to tensor<1024x64xf32>
// CHECK:          %[[SCALE_Q:.+]] = linalg.generic {{.+}} ins(%[[EXTRACTED_SLICE_2]] : tensor<1024x64xf32>)
// CHECK:          %[[D8:.+]] = tensor.empty() : tensor<1024x32xf32>
// CHECK:          %[[D9:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D8]] : tensor<1024x32xf32>) ->
// CHECK-SAME:       tensor<1024x32xf32>
// CHECK:          %[[D10:.+]] = linalg.matmul_transpose_b ins(%[[SCALE_Q]], %[[EXTRACTED_SLICE]] :
// CHECK-SAME:       tensor<1024x64xf32>, tensor<32x64xf32>) outs(%[[D9]] : tensor<1024x32xf32>) -> tensor<1024x32xf32>
// CHECK:          %[[D11:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:       "reduction"]} ins(%[[D10]] : tensor<1024x32xf32>) outs(%[[ARG5]] : tensor<1024xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D18:.+]] = arith.maximumf %[[IN]], %[[OUT]] : f32
// CHECK:            linalg.yield %[[D18]] : f32
// CHECK:          } -> tensor<1024xf32>
// CHECK:          %[[D12:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]], iterator_types = ["parallel",
// CHECK-SAME:       "parallel"]} ins(%[[D11]] : tensor<1024xf32>) outs(%[[D10]] : tensor<1024x32xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D18]] = arith.subf %[[OUT]], %[[IN]] : f32
// CHECK:            %[[D19:.+]] = math.exp2 %[[D18]] : f32
// CHECK:            linalg.yield %[[D19]] : f32
// CHECK:          } -> tensor<1024x32xf32>
// CHECK:          %[[D13:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel"]}
// CHECK-SAME:       ins(%[[D11]] : tensor<1024xf32>) outs(%[[ARG5]] : tensor<1024xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D18]] = arith.subf %[[OUT]], %[[IN]] : f32
// CHECK:            %[[D19]] = math.exp2 %[[D18]] : f32
// CHECK:            linalg.yield %[[D19]] : f32
// CHECK:          } -> tensor<1024xf32>
// CHECK:          %[[D14:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel"]}
// CHECK-SAME:       ins(%[[D13]] : tensor<1024xf32>) outs(%[[ARG6]] : tensor<1024xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D18]] = arith.mulf %[[IN]], %[[OUT]] : f32
// CHECK:            linalg.yield %[[D18]] : f32
// CHECK:          } -> tensor<1024xf32>
// CHECK:          %[[D15:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:       "reduction"]} ins(%[[D12]] : tensor<1024x32xf32>) outs(%[[D14]] : tensor<1024xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D18]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK:            linalg.yield %[[D18]] : f32
// CHECK:          } -> tensor<1024xf32>
// CHECK:          %[[D16:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]], iterator_types = ["parallel",
// CHECK-SAME:       "parallel"]} ins(%[[D13]] : tensor<1024xf32>) outs(%[[ARG4]] : tensor<1024x64xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D18]] = arith.mulf %[[IN]], %[[OUT]] : f32
// CHECK:            linalg.yield %[[D18]] : f32
// CHECK:          } -> tensor<1024x64xf32>
// CHECK:          %[[D17:.+]] = linalg.matmul ins(%[[D12]], %[[EXTRACTED_SLICE_1]] : tensor<1024x32xf32>,
// CHECK-SAME:       tensor<32x64xf32>) outs(%[[D16]] : tensor<1024x64xf32>) -> tensor<1024x64xf32>
// CHECK:          scf.yield %[[D17]], %[[D11]], %[[D15]] : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// CHECK:        }
// CHECK:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel"]} ins(%[[D6]]#[[D2:.+]] : tensor<1024xf32>) outs(%[[D6]]#[[D0:.+]] : tensor<1024x64xf32>)
// CHECK-SAME:     {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-DAG:      %[[CST_1:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:          %[[D8:.+]] = arith.divf %[[CST_1]], %[[IN]] : f32
// CHECK:          %[[D9:.+]] = arith.mulf %[[D8]], %[[OUT]] : f32
// CHECK:          linalg.yield %[[D9]] : f32
// CHECK:        } -> tensor<1024x64xf32>
// CHECK:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D7]] into %[[D0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// CHECK-SAME:     tensor<1024x64xf32> into tensor<1x1024x64xf32>
// CHECK:        return %[[INSERTED_SLICE]] : tensor<1x1024x64xf32>
// CHECK:      }

// -----

func.func @attention(%query: tensor<?x?x?xf32>, %key: tensor<?x?x?xf32>, %value: tensor<?x?x?xf32>, %dim0: index, %dim1: index, %dim2: index) -> tensor<?x?x?xf32> {
  %0 = tensor.empty(%dim0, %dim1, %dim2) : tensor<?x?x?xf32>
  %scale = arith.constant 0.05 : f32
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, f32) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG:  #[[$MAP2:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:      func.func @attention
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<?x?x?xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>, %[[ARG3:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: index, %[[ARG5:[a-zA-Z0-9_]+]]: index) -> tensor<?x?x?xf32> {
// CHECK:        %[[D0:.+]] = tensor.empty(%[[ARG3]], %[[ARG4]], %[[ARG5]]) : tensor<?x?x?xf32>
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK:        %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?xf32>
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK:        %[[DIM_0:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:        %[[DIM_1:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:        %[[D1:.+]] = tensor.empty(%[[DIM]], %[[DIM_0]]) : tensor<?x?xf32>
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-DAG:    %[[CST_2:.+]] = arith.constant -1.000000e+30 : f32
// CHECK:        %[[D3:.+]] = tensor.empty(%[[DIM]]) : tensor<?xf32>
// CHECK:        %[[D4:.+]] = linalg.fill ins(%[[CST_2]] : f32) outs(%[[D3]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK:        %[[D6:.+]]:3 = scf.for %[[ARG6:[a-zA-Z0-9_]+]] = %[[C0]] to %[[DIM_1]] step %[[C32]]
// CHECK-SAME:     iter_args(%[[ARG7:[a-zA-Z0-9_]+]] = %[[D2]], %[[ARG8:[a-zA-Z0-9_]+]] = %[[D4]],
// CHECK-SAME:     %[[ARG9:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>) {
// CHECK:          %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[ARG6]], 0] [1, 32, %[[DIM_0]]] [1, 1,
// CHECK-SAME:       1] : tensor<?x?x?xf32> to tensor<32x?xf32>
// CHECK:          %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[ARG2]][0, %[[ARG6]], 0] [1, 32, %[[DIM_0]]] [1,
// CHECK-SAME:       1, 1] : tensor<?x?x?xf32> to tensor<32x?xf32>
// CHECK:          %[[EXTRACTED_SLICE_4:.+]] = tensor.extract_slice %[[ARG0]][0, 0, 0] [1, %[[DIM]], %[[DIM_0]]] [1, 1,
// CHECK-SAME:       1] : tensor<?x?x?xf32> to tensor<?x?xf32>
// CHECK:          %[[DIM_5:.+]] = tensor.dim %[[EXTRACTED_SLICE_4]], %[[C0]] : tensor<?x?xf32>
// CHECK:          %[[SCALE_Q:.+]] = linalg.generic
// CHECK:          %[[D8:.+]] = tensor.empty(%[[DIM_5]]) : tensor<?x32xf32>
// CHECK:          %[[D9:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D8]] : tensor<?x32xf32>) -> tensor<?x32xf32>
// CHECK:          %[[D10:.+]] = linalg.matmul_transpose_b ins(%[[SCALE_Q]], %[[EXTRACTED_SLICE]] :
// CHECK-SAME:       tensor<?x?xf32>, tensor<32x?xf32>) outs(%[[D9]] : tensor<?x32xf32>) -> tensor<?x32xf32>
// CHECK:          %[[D11:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:       "reduction"]} ins(%[[D10]] : tensor<?x32xf32>) outs(%[[ARG8]] : tensor<?xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D18:.+]] = arith.maximumf %[[IN]], %[[OUT]] : f32
// CHECK:            linalg.yield %[[D18]] : f32
// CHECK:          } -> tensor<?xf32>
// CHECK:          %[[D12:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]], iterator_types = ["parallel",
// CHECK-SAME:       "parallel"]} ins(%[[D11]] : tensor<?xf32>) outs(%[[D10]] : tensor<?x32xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D18]] = arith.subf %[[OUT]], %[[IN]] : f32
// CHECK:            %[[D19:.+]] = math.exp2 %[[D18]] : f32
// CHECK:            linalg.yield %[[D19]] : f32
// CHECK:          } -> tensor<?x32xf32>
// CHECK:          %[[D13:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel"]}
// CHECK-SAME:       ins(%[[D11]] : tensor<?xf32>) outs(%[[ARG8]] : tensor<?xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D18]] = arith.subf %[[OUT]], %[[IN]] : f32
// CHECK:            %[[D19]] = math.exp2 %[[D18]] : f32
// CHECK:            linalg.yield %[[D19]] : f32
// CHECK:          } -> tensor<?xf32>
// CHECK:          %[[D14:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel"]}
// CHECK-SAME:       ins(%[[D13]] : tensor<?xf32>) outs(%[[ARG9]] : tensor<?xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D18]] = arith.mulf %[[IN]], %[[OUT]] : f32
// CHECK:            linalg.yield %[[D18]] : f32
// CHECK:          } -> tensor<?xf32>
// CHECK:          %[[D15:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:       "reduction"]} ins(%[[D12]] : tensor<?x32xf32>) outs(%[[D14]] : tensor<?xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D18]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK:            linalg.yield %[[D18]] : f32
// CHECK:          } -> tensor<?xf32>
// CHECK:          %[[D16:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]], iterator_types = ["parallel",
// CHECK-SAME:       "parallel"]} ins(%[[D13]] : tensor<?xf32>) outs(%[[ARG7]] : tensor<?x?xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D18]] = arith.mulf %[[IN]], %[[OUT]] : f32
// CHECK:            linalg.yield %[[D18]] : f32
// CHECK:          } -> tensor<?x?xf32>
// CHECK:          %[[D17:.+]] = linalg.matmul ins(%[[D12]], %[[EXTRACTED_SLICE_3]] : tensor<?x32xf32>,
// CHECK-SAME:       tensor<32x?xf32>) outs(%[[D16]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          scf.yield %[[D17]], %[[D11]], %[[D15]] : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>
// CHECK:        }
// CHECK:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel"]} ins(%[[D6]]#[[D2:.+]] : tensor<?xf32>) outs(%[[D6]]#[[D0:.+]] : tensor<?x?xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-DAG:      %[[CST_3:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:          %[[D8:.+]] = arith.divf %[[CST_3]], %[[IN]] : f32
// CHECK:          %[[D9:.+]] = arith.mulf %[[D8]], %[[OUT]] : f32
// CHECK:          linalg.yield %[[D9]] : f32
// CHECK:        } -> tensor<?x?xf32>
// CHECK:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D7]] into %[[D0]][0, 0, 0] [1, %[[DIM]], %[[DIM_0]]]
// CHECK-SAME:     [1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?xf32>
// CHECK:        return %[[INSERTED_SLICE]] : tensor<?x?x?xf32>
// CHECK:      }

// -----

func.func @attention_f16(%query: tensor<1x1024x64xf16>, %key: tensor<1x1024x64xf16>, %value: tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16> {
  %0 = tensor.empty() : tensor<1x1024x64xf16>
  %scale = arith.constant 0.05 : f16
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, f16) outs(%0 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
  return %1 : tensor<1x1024x64xf16>
}

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG:  #[[$MAP2:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @attention_f16
// CHECK-SAME:   (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1024x64xf16>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<1x1024x64xf16>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16> {
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<1x1024x64xf16>
// CHECK:        %[[D1:.+]] = tensor.empty() : tensor<1024x64xf32>
// CHECK:        %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[D0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// CHECK-SAME:     tensor<1x1024x64xf16> to tensor<1024x64xf16>
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<1024x64xf32>) ->
// CHECK-SAME:     tensor<1024x64xf32>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant -1.000000e+30 : f32
// CHECK:        %[[D3:.+]] = tensor.empty() : tensor<1024xf32>
// CHECK:        %[[D4:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:        %[[D6:.+]]:3 = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C32]]
// CHECK-SAME:     iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[D2]], %[[ARG5:[a-zA-Z0-9_]+]] = %[[D4]],
// CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>) {
// CHECK:          %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG1]][0, %[[ARG3]], 0] [1, 32, 64] [1, 1, 1] :
// CHECK-SAME:       tensor<1x1024x64xf16> to tensor<32x64xf16>
// CHECK:          %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG2]][0, %[[ARG3]], 0] [1, 32, 64] [1, 1, 1] :
// CHECK-SAME:       tensor<1x1024x64xf16> to tensor<32x64xf16>
// CHECK:          %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[ARG0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// CHECK-SAME:       tensor<1x1024x64xf16> to tensor<1024x64xf16>
// CHECK:          %[[SCALE_Q:.+]] = linalg.generic
// CHECK:          %[[D9:.+]] = tensor.empty() : tensor<1024x32xf32>
// CHECK:          %[[D10:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D9]] : tensor<1024x32xf32>) ->
// CHECK-SAME:       tensor<1024x32xf32>
// CHECK:          %[[D11:.+]] = linalg.matmul_transpose_b ins(%[[SCALE_Q]], %[[EXTRACTED_SLICE_1]] :
// CHECK-SAME:       tensor<1024x64xf16>, tensor<32x64xf16>) outs(%[[D10]] : tensor<1024x32xf32>) ->
// CHECK-SAME:       tensor<1024x32xf32>
// CHECK:          %[[D12:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:       "reduction"]} ins(%[[D11]] : tensor<1024x32xf32>) outs(%[[ARG5]] : tensor<1024xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D21:.+]] = arith.maximumf %[[IN]], %[[OUT]] : f32
// CHECK:            linalg.yield %[[D21]] : f32
// CHECK:          } -> tensor<1024xf32>
// CHECK:          %[[D13:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]], iterator_types = ["parallel",
// CHECK-SAME:       "parallel"]} ins(%[[D12]] : tensor<1024xf32>) outs(%[[D11]] : tensor<1024x32xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D21]] = arith.subf %[[OUT]], %[[IN]] : f32
// CHECK:            %[[D22:.+]] = math.exp2 %[[D21]] : f32
// CHECK:            linalg.yield %[[D22]] : f32
// CHECK:          } -> tensor<1024x32xf32>
// CHECK:          %[[D14:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel"]}
// CHECK-SAME:       ins(%[[D12]] : tensor<1024xf32>) outs(%[[ARG5]] : tensor<1024xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D21]] = arith.subf %[[OUT]], %[[IN]] : f32
// CHECK:            %[[D22]] = math.exp2 %[[D21]] : f32
// CHECK:            linalg.yield %[[D22]] : f32
// CHECK:          } -> tensor<1024xf32>
// CHECK:          %[[D15:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel"]}
// CHECK-SAME:       ins(%[[D14]] : tensor<1024xf32>) outs(%[[ARG6]] : tensor<1024xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D21]] = arith.mulf %[[IN]], %[[OUT]] : f32
// CHECK:            linalg.yield %[[D21]] : f32
// CHECK:          } -> tensor<1024xf32>
// CHECK:          %[[D16:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:       "reduction"]} ins(%[[D13]] : tensor<1024x32xf32>) outs(%[[D15]] : tensor<1024xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D21]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK:            linalg.yield %[[D21]] : f32
// CHECK:          } -> tensor<1024xf32>
// CHECK:          %[[D17:.+]] = tensor.empty() : tensor<1024x32xf16>
// CHECK:          %[[D18:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel",
// CHECK-SAME:       "parallel"]} ins(%[[D13]] : tensor<1024x32xf32>) outs(%[[D17]] : tensor<1024x32xf16>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f16):
// CHECK:            %[[D21]] = arith.truncf %[[IN]] : f32 to f16
// CHECK:            linalg.yield %[[D21]] : f16
// CHECK:          } -> tensor<1024x32xf16>
// CHECK:          %[[D19:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]], iterator_types = ["parallel",
// CHECK-SAME:       "parallel"]} ins(%[[D14]] : tensor<1024xf32>) outs(%[[ARG4]] : tensor<1024x64xf32>) {
// CHECK:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:            %[[D21]] = arith.mulf %[[IN]], %[[OUT]] : f32
// CHECK:            linalg.yield %[[D21]] : f32
// CHECK:          } -> tensor<1024x64xf32>
// CHECK:          %[[D20:.+]] = linalg.matmul ins(%[[D18]], %[[EXTRACTED_SLICE_2]] : tensor<1024x32xf16>,
// CHECK-SAME:       tensor<32x64xf16>) outs(%[[D19]] : tensor<1024x64xf32>) -> tensor<1024x64xf32>
// CHECK:          scf.yield %[[D20]], %[[D12]], %[[D16]] : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// CHECK:        }
// CHECK:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel"]} ins(%[[D6]]#[[D2:.+]] : tensor<1024xf32>) outs(%[[D6]]#[[D0:.+]] : tensor<1024x64xf32>)
// CHECK-SAME:     {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-DAG:      %[[CST_1:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:          %[[D9:.+]] = arith.divf %[[CST_1]], %[[IN]] : f32
// CHECK:          %[[D10:.+]] = arith.mulf %[[D9]], %[[OUT]] : f32
// CHECK:          linalg.yield %[[D10]] : f32
// CHECK:        } -> tensor<1024x64xf32>
// CHECK:        %[[D8:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel"]} ins(%[[D7]] : tensor<1024x64xf32>) outs(%[[EXTRACTED_SLICE]] : tensor<1024x64xf16>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f16):
// CHECK:          %[[D9]] = arith.truncf %[[IN]] : f32 to f16
// CHECK:          linalg.yield %[[D9]] : f16
// CHECK:        } -> tensor<1024x64xf16>
// CHECK:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D8]] into %[[D0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// CHECK-SAME:     tensor<1024x64xf16> into tensor<1x1024x64xf16>
// CHECK:        return %[[INSERTED_SLICE]] : tensor<1x1024x64xf16>
// CHECK:      }

// -----

func.func @attention_transpose_v(%query: tensor<1x1024x64xf16>, %key: tensor<1x1024x64xf16>, %value: tensor<1x64x1024xf16>) -> tensor<1x1024x64xf16> {
  %0 = tensor.empty() : tensor<1x1024x64xf16>
  %scale = arith.constant 0.05 : f16
  %1 = iree_linalg_ext.attention {transpose_v = true} ins(%query, %key, %value, %scale : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, tensor<1x64x1024xf16>, f16) outs(%0 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
  return %1 : tensor<1x1024x64xf16>
}

// CHECK-LABEL:  func.func @attention_transpose_v
// There should be two matmul_transpose_b for tranpose_v variant instead of
// only 1.
// CHECK: linalg.matmul_transpose_b
// CHECK-NOT: linalg.matmul
// CHECK: linalg.matmul_transpose_b

// -----

// This test checks that a elementwise-mask operation is inserted
// after QK.T and before the softmax.

func.func @masked_i1_attention(%query: tensor<1x1024x64xf16>, %key: tensor<1x1024x64xf16>, %value: tensor<1x1024x64xf16>, %mask: tensor<1x1024x1024xi1>) -> tensor<1x1024x64xf16> {
  %0 = tensor.empty() : tensor<1x1024x64xf16>
  %scale = arith.constant 0.05 : f16
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale, %mask : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, f16, tensor<1x1024x1024xi1>) outs(%0 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
  return %1 : tensor<1x1024x64xf16>
}

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL:  func.func @masked_i1_attention
// CHECK-SAME: (%[[QUERY:.+]]: tensor<1x1024x64xf16>, %[[KEY:.+]]: tensor<1x1024x64xf16>, %[[VALUE:.+]]: tensor<1x1024x64xf16>, %[[MASK:.+]]: tensor<1x1024x1024xi1>)
//CHECK-DAG:     %[[CST_0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:        scf.for %[[IV:.+]] = %[[C0]] to %[[C1024]] step %[[C32]]
// CHECK:          %[[K_S:.+]] = tensor.extract_slice %[[KEY]][0, %[[IV]], 0] [1, 32, 64] [1, 1, 1]
// CHECK:          %[[V_S:.+]] = tensor.extract_slice %[[VALUE]][0, %[[IV]], 0] [1, 32, 64] [1, 1, 1]
// CHECK:          %[[Q_S:.+]] = tensor.extract_slice %[[QUERY]][0, 0, 0] [1, 1024, 64] [1, 1, 1]
// CHECK:          %[[M_S:.+]] = tensor.extract_slice %[[MASK]][0, 0, %[[IV]]] [1, 1024, 32] [1, 1, 1]
// CHECK:          %[[SCALED_Q:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                          ins(%[[Q_S]] : tensor<1024x64xf16>)
// CHECK:          %[[QK:.+]] = linalg.matmul_transpose_b ins(%[[SCALED_Q]], %[[K_S]]
// CHECK-DAG:      %[[CST_1:.+]] = arith.constant -1.000000e+06 : f32
// CHECK:          %[[MASK_QK:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[QK]], %[[M_S]] : tensor<1024x32xf32>, tensor<1024x32xi1>)
// CHECK:                              ^bb0(%[[IN:.+]]: f32, %[[IN0:.+]]: i1, %[[OUT:.+]]: f32):
// CHECK:                                %[[D0:.+]] = arith.select %[[IN0]], %[[CST_0]], %[[CST_1]] : f32
// CHECK:                                %[[D1:.+]] = arith.addf %[[IN]], %[[D0]] : f32
// CHECK:                                linalg.yield %[[D1]] : f32
// CHECK:                              } -> tensor<1024x32xf32>
// CHECK:           %[[MAX:.+]] linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel", "reduction"]}
// CHECK-SAME:                   ins(%[[MASK_QK]] : tensor<1024x32xf32>)
// CHECK:                       ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                         %[[D2:.+]] = arith.maximumf %[[IN]], %[[OUT]] : f32
// CHECK:                         linalg.yield %[[D2:.+]] : f32
// CHECK:                       } -> tensor<1024xf32>

// -----

// This test checks for non i1 mask since after codegen pipeline we may emit i8 masks.

func.func @masked_i8_attention(%query: tensor<1x1024x64xf16>, %key: tensor<1x1024x64xf16>, %value: tensor<1x1024x64xf16>, %mask: tensor<1x1024x1024xi8>) -> tensor<1x1024x64xf16> {
  %0 = tensor.empty() : tensor<1x1024x64xf16>
  %scale = arith.constant 0.05 : f16
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale, %mask : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, f16, tensor<1x1024x1024xi8>) outs(%0 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
  return %1 : tensor<1x1024x64xf16>
}

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @masked_i8_attention
// CHECK-SAME: (%[[QUERY:.+]]: tensor<1x1024x64xf16>, %[[KEY:.+]]: tensor<1x1024x64xf16>, %[[VALUE:.+]]: tensor<1x1024x64xf16>, %[[MASK:.+]]: tensor<1x1024x1024xi8>)
//CHECK-DAG:     %[[CST_0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:        scf.for %[[IV:.+]] = %[[C0]] to %[[C1024]] step %[[C32]]
// CHECK:          %[[M_S:.+]] = tensor.extract_slice %[[MASK]][0, 0, %[[IV]]] [1, 1024, 32] [1, 1, 1]
// CHECK-DAG:      %[[CST_1:.+]] = arith.constant -1.000000e+06 : f32
// CHECK:          %[[MASK_QK:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%{{.+}}, %[[M_S]] : tensor<1024x32xf32>, tensor<1024x32xi8>)
// CHECK:                              ^bb0(%[[IN:.+]]: f32, %[[IN0:.+]]: i8, %[[OUT:.+]]: f32):
// CHECK:                                %[[D0:.+]] = arith.trunci %[[IN0]] : i8 to i1
// CHECK:                                %[[D1:.+]] = arith.select %[[D0]], %[[CST_0]], %[[CST_1]] : f32
// CHECK:                                %[[D2:.+]] = arith.addf %[[IN]], %[[D1]] : f32
// CHECK:                                linalg.yield %[[D2]] : f32
// CHECK:                              } -> tensor<1024x32xf32>
