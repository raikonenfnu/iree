// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE source.mlir -> execution output test runner.
// This is meant to be called from LIT for FileCheck tests, and tries to match
// the interface of mlir-opt (featuring -split-input-file, etc) so it's easier
// to work with there. If you want a more generalized runner for standalone
// precompiled IREE modules use iree-run-module.
//
// By default all exported functions in the module will be run in order.
// All input values, provided via -function-inputs, will be passed to the
// functions (this means all input signatures must match). Results from the
// executed functions will be printed to stdout for checking.
//
// Example input:
// // RUN: iree-run-mlir %s | IreeFileCheck %s
// // CHECK-LABEL: @foo
// // CHECK: 1xf32: 2
// func @foo() -> tensor<f32> {
//   %0 = constant dense<2.0> : tensor<f32>
//   return %0 : tensor<f32>
// }
//
// Command line arguments are handled by LLVM's parser by default but -- can be
// used to separate the compiler flags from the runtime flags, such as:
//   iree-run-mlir -iree-hal-target-backends=vulkan-spirv -- --logtostderr

#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <sstream>
#include <unordered_map>

#include "iree/base/status_cc.h"
#include "iree/tools/init_dialects.h"
#include "iree/tools/init_passes.h"
#include "iree/tools/init_targets.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MlirOptMain.h"

static llvm::cl::opt<std::string> input_file_flag{
    llvm::cl::Positional,
    llvm::cl::desc("<input .mlir file>"),
    llvm::cl::init("-"),
};

static llvm::cl::opt<std::string> output_file_flag{
    "output-file",
    llvm::cl::desc("<output .txt file for matmul shape count>"),
    llvm::cl::init("/tmp/output_count.txt"),
};


static llvm::cl::list<std::string> run_args_flag{
    "run-arg",
    llvm::cl::desc("Argument passed to the execution flag parser"),
    llvm::cl::ZeroOrMore,
};

namespace iree {
namespace {

// TODO(raikonenfnu): Cleanup the many reuse of code between two getBatchMatmulSize.
// Returns {m, n, k}.
std::string getMatmulSize(mlir::linalg::MatmulOp op) {
    auto lhs_shape = op.inputs()[0].getType().dyn_cast<mlir::RankedTensorType>().getShape();
    auto rhs_shape = op.inputs()[1].getType().dyn_cast<mlir::RankedTensorType>().getShape();
    auto out_shape = op.result_tensors()[0].getType().dyn_cast<mlir::RankedTensorType>().getShape();
    int m = out_shape[0];
    int n = out_shape[1];
    int k = lhs_shape[1];
    const std::string kMult = "x";
    std::string mm_shape = std::to_string(m)+kMult+std::to_string(n)+kMult+std::to_string(k);
    return mm_shape;
}

// Returns {batch, m, n, k}.
std::string getBatchMatmulSize(mlir::linalg::BatchMatmulOp op) {
    auto lhs_shape = op.inputs()[0].getType().dyn_cast<mlir::RankedTensorType>().getShape();
    auto rhs_shape = op.inputs()[1].getType().dyn_cast<mlir::RankedTensorType>().getShape();
    auto out_shape = op.result_tensors()[0].getType().dyn_cast<mlir::RankedTensorType>().getShape();
    int b = out_shape[0];
    int m = out_shape[1];
    int n = out_shape[2];
    int k = lhs_shape[2];
    const std::string kMult = "x";
    std::string mm_shape = std::to_string(b)+kMult+std::to_string(m)+kMult+std::to_string(n)+kMult+std::to_string(k);
    return mm_shape;
}


Status CountFile(const std::string& mlir_filename, const std::string& output_file, mlir::DialectRegistry &registry) {
  // MLIR Parsing and Module Setup
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  context.allowUnregisteredDialects();
  mlir::OwningModuleRef moduleRef = parseSourceFile(mlir_filename, &context);
  if (!moduleRef) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "could not open file");
  mlir::ModuleOp module = *moduleRef;
  llvm::StringMap<int> umap;
  module->walk([&](mlir::Operation *nestedOp) {
    // TODO(raikonenfnu): Cleanup the region in here.
    auto mm_op = llvm::dyn_cast<mlir::linalg::MatmulOp>(nestedOp);
    auto bmm_op = llvm::dyn_cast<mlir::linalg::BatchMatmulOp>(nestedOp);
    std::string mm_shapes;
    if(mm_op) {
        mm_shapes = getMatmulSize(mm_op);
    } else if(bmm_op) {
        mm_shapes = getBatchMatmulSize(bmm_op);
    }
    if(!mm_shapes.empty()) {
        if(umap.find(mm_shapes) != umap.end()) {
            umap[mm_shapes] += 1;
        } else{
            umap[mm_shapes] = 1;
        }
    }
  });
// TODO(raikonenfnu): Add sorting by num of occurence
   for (auto &it : umap) {
        llvm::outs()<<it.first()<<":"<<it.second<<"\n";
    }

  return iree_ok_status();
}

} // namespace

extern "C" int main(int argc, char** argv) {
  int argc_llvm = argc;
  char** argv_llvm = argv;
  int argc_iree = 1;
  std::vector<char*> argv_iree = {argv[0]};
  for (int i = 0; i < argc; ++i) {
    if (std::strcmp(argv[i], "--") == 0) {
      argc_llvm = i;
      argc_iree = argc - i;
      for (int j = i + 1; j < argc; ++j) {
        argv_iree.push_back(argv[i + 1]);
      }
      break;
    }
  }

  mlir::DialectRegistry registry;
  mlir::iree_compiler::registerAllDialects(registry);
  // Make sure command line options are registered.
  (void)mlir::iree_compiler::IREE::HAL::getTargetOptionsFromFlags();

  // Register MLIRContext command-line options like
  // -mlir-print-op-on-diagnostic.
  mlir::registerMLIRContextCLOptions();
  // Register pass manager command-line options like -print-ir-*.
  mlir::registerPassManagerCLOptions();

  llvm::InitLLVM init_llvm(argc_llvm, argv_llvm);
  llvm::cl::ParseCommandLineOptions(argc_llvm, argv_llvm);

  for (auto& run_arg : run_args_flag) {
    argv_iree.push_back(const_cast<char*>(run_arg.c_str()));
  }
  argc_iree += run_args_flag.size();
  auto status = CountFile(input_file_flag, output_file_flag, registry);
  if (!status.ok()) {
    std::cerr << "ERROR running file (" << input_file_flag << "): " << status
              << "\n";
    return 1;
  }
  return 0;
}

}  // namespace iree
