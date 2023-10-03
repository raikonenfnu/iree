// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/Utils/MMAUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <queue>

#define DEBUG_TYPE "iree-codegen-gpu-scheduler"

namespace mlir {
namespace iree_compiler {

bool isGraphBreak(Operation *op) {
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  return !interface;
}

bool hasReadWriteEffects(Operation *op) {
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface) {
    return false;
  }
  bool hasWriteEffect = interface.hasEffect<MemoryEffects::Write>();
  bool hasReadEffect = interface.hasEffect<MemoryEffects::Read>();
  return hasWriteEffect || hasReadEffect;
}

// A node can consist of several ops.
// For example, we group:
//     %124 = vector.load %subview_2[%122, %123]
//     %125 = vector.insert_strided_slice %124, %cst_5
//     %168 = vector.extract %125[0, 0] : vector<4x4x4xf32>
//     as a single node since they essentially represent a "load" dependency.
struct Node {
  using Ptr = std::shared_ptr<Node>;
  SmallVector<Operation *> ops;
};

struct Graph {
  using Ptr = std::shared_ptr<Graph>;
  SetVector<Operation *> nodes;
  bool freeze{true};
};

static bool isOverwritten(Operation *op, ArrayAttr &offsets) {
  if (auto newInsertOp = dyn_cast<vector::InsertStridedSliceOp>(op)) {
    auto newOffsets = newInsertOp.getOffsets();
    return newOffsets == offsets;
  }
  return false;
}

static bool isExtracted(Operation *op, ArrayAttr offsets) {
  if (auto extractOp = dyn_cast<vector::ExtractOp>(op)) {
    auto newOffsets = extractOp.getPositionAttr();
    if (newOffsets.size() < offsets.size()) {
      bool match{true};
      for (int i = 0; i < newOffsets.size(); i++) {
        auto offset = dyn_cast<IntegerAttr>(offsets[i]);
        if (!offset) return false;
        match = match && (newOffsets[i] == offset.getInt());
      }
      return match;
    }
    return newOffsets == offsets;
  }
  return false;
}

static SmallVector<Operation *> createCompositeNode(Operation *op, DenseSet<Operation *> &ignore) {
  SmallVector<Operation *> groupedOps{op};
  if (isa<vector::LoadOp>(op)) {
    for (Operation *user : op->getUsers()) {
      if (auto insertOp = dyn_cast<vector::InsertStridedSliceOp>(user)) {
        llvm::dbgs() << "Got an insert op!\n";
        insertOp.dump();
        groupedOps.push_back(insertOp);
        auto offsets = insertOp.getOffsets();

        // Find nodes at the end of the insert chain
        vector::InsertStridedSliceOp lastInsert{insertOp};
        do {
          SmallVector<Operation *> newUsers(insertOp->getUsers().begin(), insertOp->getUsers().end());
          if (newUsers.size() != 1) break;
          if (isOverwritten(newUsers[0], offsets)) break;
          lastInsert = insertOp;
          insertOp = dyn_cast_or_null<vector::InsertStridedSliceOp>(newUsers[0]);
        } while (insertOp);

        if (!insertOp)
          insertOp = lastInsert;

        // Find matching extract ops
        for (Operation *endUser : insertOp->getUsers()) {
          endUser->dump();
          if (isExtracted(endUser, offsets)) {
            // Replace op with new op that uses the intermediate insert result
            auto extractOp = dyn_cast<vector::ExtractOp>(endUser);
            extractOp.dump();
            auto insertOp = dyn_cast<vector::InsertStridedSliceOp>(groupedOps.back());
            ignore.insert(extractOp);
            extractOp.setOperand(insertOp.getResult());
            groupedOps.push_back(extractOp);
          }
        }
      }
    }
  }
  return groupedOps;
}

static bool isComposite(Node::Ptr node) {
  return node->ops.size() > 1;
}

static bool isNodeReadyToSchedule(Node::Ptr node, SmallVector<Operation *> &unscheduledNodes,
                                  DenseMap<Operation *, Node::Ptr> &operatorToNode) {
  // Node is ready to schedule if all of its operands are ready.
  const auto isReady = [&](Value value) {
    Operation *parent = value.getDefiningOp();
    // If it is a block argument
    if (!parent) return true;
    // Or if it is not defined by an unscheduled op and not nested
    // within an unscheduled op
    do {
      if (std::find(unscheduledNodes.begin(), unscheduledNodes.end(), parent) != unscheduledNodes.end()) {
        llvm::dbgs() << "Not scheduled because this op has not been scheduled yet ...\n";
        parent->dump();
        return false;
      }
      if (operatorToNode[parent] == node) {
        return true;
      }
    } while ((parent = parent->getParentOp()));
    // No unscheduled op found
    return true;
  };

  // An operation is recursively ready to be scheduled of it and its nested
  // operations are ready.
  Operation *op = node->ops[0];
  WalkResult readyToSchedule = op->walk([&](Operation *nestedOp) {
    return llvm::all_of(nestedOp->getOperands(),
                        [&](Value operand) { return isReady(operand); })
               ? WalkResult::advance()
               : WalkResult::interrupt();
  });
  return !readyToSchedule.wasInterrupted();
}

struct NodeComparator {
  bool operator()(Node::Ptr lhs, Node::Ptr rhs) {
    return cost.at(lhs) < cost.at(rhs);
  }
  std::map<Node::Ptr, int> cost;
};

static bool prioritizedTopologicalSort(Block *block) {
  if (block->empty()) return true;
  llvm::iterator_range<Block::iterator> ops = block->without_terminator();
  if (ops.empty()) return true;

  block->dump();
  // Create simple and composite nodes for ops
  DenseSet<Operation *> unscheduledOps, ignoreOps;
  DenseMap<Operation *, Node::Ptr> operationToNode;
  std::vector<Node::Ptr> nodes;

  struct Graph {
    SmallVector<Operation *> nodes;
    bool freeze{true};
  };
  SmallVector<Graph> unscheduledGraphs;
  unscheduledGraphs.push_back(Graph());

  for (Operation &op : ops) {
    llvm::dbgs() << "Processing ... \n";
    if (unscheduledOps.contains(&op)) {
      llvm::dbgs() << "Already in unscheduled!\n";
      op.dump();
      continue;
    }
    if (ignoreOps.contains(&op)) {
      llvm::dbgs() << "in ignore list!\n";
      op.dump();
      continue;
    }
    if (hasReadWriteEffects(&op)) {
      llvm::dbgs() << "has read/write effects\n";
      op.dump();
      auto compositeNode = std::make_shared<Node>();
      compositeNode->ops = createCompositeNode(&op, ignoreOps);
      llvm::dbgs() << "----------\n";
      for (Operation *child : compositeNode->ops) {
        operationToNode[child] = compositeNode;
        unscheduledOps.insert(child);
        unscheduledGraphs.back().nodes.push_back(child);
        child->dump();
      }
      nodes.push_back(compositeNode);
      if (compositeNode->ops.size() > 1)
        unscheduledGraphs.back().freeze = false;
      llvm::dbgs() << "----------\n";
      continue;
    }
    if (isGraphBreak(&op)) {
      unscheduledGraphs.push_back(Graph());
    }
    op.dump();
    auto newNode = std::make_shared<Node>();
    newNode->ops.push_back(&op);
    unscheduledOps.insert(&op);
    operationToNode[&op] = newNode;
    nodes.push_back(newNode);
    unscheduledGraphs.back().nodes.push_back(&op);
  }

  llvm::dbgs() << "Partitioned graph into " << unscheduledGraphs.size() << " subgraphs \n";
  for (Graph graph : unscheduledGraphs) {
    if (graph.freeze) {
      llvm::dbgs() << "Graph is frozen\n";
    } else {
      llvm::dbgs() << "Graph is not frozen\n";
    }
  }

  // Assign costs to nodes, starting with mfmas
  // The lower the cost, the higher the priority
  int count{0};
  int offset{10};
  int barrierCost{-1};
  std::map<Node::Ptr, int> nodeCost;
  std::map<Operation *, int> opCost;
  for (Node::Ptr node : nodes) {
    for (Operation *op : node->ops) {
      if (auto mfmaOp = dyn_cast<amdgpu::MFMAOp>(op)) {
        op->dump();
        llvm::dbgs() << "MFMA op = ";
        nodeCost[node] = count + offset;
        llvm::dbgs() << "Node cost = " << nodeCost[node] << "\n";
        Operation *parentA = mfmaOp.getSourceA().getDefiningOp();
        Operation *parentB = mfmaOp.getSourceB().getDefiningOp();
        Operation *parentC = mfmaOp.getDestC().getDefiningOp();
        opCost[parentA] = opCost[parentB] = opCost[parentC] = nodeCost[node] - 1;
        count += offset;
        break;
      }
      if (auto barrierOp = dyn_cast<gpu::BarrierOp>(op)) {
        nodeCost[node] = barrierCost;
        break;
      }
    }
  }

  int baselineValue{1000};
  for (Node::Ptr node : nodes) {
    if (nodeCost.count(node)) continue;
    for (Operation *op : node->ops) {
      if (opCost.count(op)) {
        op->dump();
        llvm::dbgs() << "MFMA Dependency op = ";
        nodeCost[node] = opCost[op];
        llvm::dbgs() << "Node cost = " << nodeCost[node] << "\n";
        break;
      } else {
        op->dump();
        llvm::dbgs() << "Ordinary op = ";
        nodeCost[node] = baselineValue;
        llvm::dbgs() << "Node cost = " << nodeCost[node] << "\n";
      }
    }
  }

  llvm::dbgs() << "Begin scheduling\n";
  Block::iterator nextScheduledOp = ops.begin();
  //Block::iterator endOp = ops.end();
  llvm::dbgs() << "Before ...\n";
  block->dump();
  llvm::dbgs() << "----------\n";
  for (Graph graph : unscheduledGraphs) {
    if (graph.freeze) {
      llvm::dbgs() << "Encountered frozen graph. Forwarding iterator ...\n";
      llvm::dbgs() << "Here are the ops in the current graph ... \n";
      for (Operation *op : graph.nodes) {
        op->dump();
      }
      llvm::dbgs() << "====================\n";
      for (Operation *op : graph.nodes) {
        llvm::dbgs() << " current op = \n";
        op->dump();
        llvm::dbgs() << " nextScheduledOp = \n";
        nextScheduledOp->dump();
        if (op == &*nextScheduledOp) nextScheduledOp++;
      }
      llvm::dbgs() << "====================\n";
      continue;
    }
    while (!graph.nodes.empty()) {
        // Find the min cost node that can be scheduled
        Node::Ptr minCostNode{nullptr};
        for (Operation *op : graph.nodes) {
          Node::Ptr node = operationToNode[op];
          llvm::dbgs() << "Attempting to schedule ... \n";
          op->dump();
          if (!isNodeReadyToSchedule(node, graph.nodes, operationToNode)) {
            llvm::dbgs() << "Op not ready to be scheduled \n";
            for (Operation *nodeOp : node->ops) {
              nodeOp->dump();
            }
            llvm::dbgs() << "----------\n";
            continue;
          }
          if (!minCostNode) {
            minCostNode = node;
          }
          if (nodeCost[node] < nodeCost[minCostNode]) {
            minCostNode = node;
          }
        }

        llvm::dbgs() << " nextScheduledOp = \n";
        nextScheduledOp->dump();

        // Schedule the operation by moving it to the start
        for (Operation *nodeOp : minCostNode->ops) {
          llvm::dbgs() << "Scheduling ...\n";
          nodeOp->dump();
          llvm::dbgs() << "----------\n";
          nodeOp->moveBefore(block, nextScheduledOp);
          auto it = std::find(graph.nodes.begin(), graph.nodes.end(), nodeOp);
          if (it == graph.nodes.end())
            llvm::dbgs() << "Element not found!\n";
          graph.nodes.erase(it);
          if (nodeOp == &*nextScheduledOp)
            ++nextScheduledOp;
        }
        llvm::dbgs() << "After ...\n";
        block->dump();
        llvm::dbgs() << "----------\n";
    }
  }

  // Create a new dependency graph
  //std::vector<Graph::Ptr> graphs;
  //graphs.push_back(std::make_shared<Graph>());
  //for (Operation &op : ops) {
  //  llvm::dbgs() << "--------\n";
  //  if (isGraphBreak(&op)) {
  //    graphs.push_back(std::make_shared<Graph>());
  //    continue;
  //  }
  //  if (isa<vector::LoadOp>(op) || isa<vector::StoreOp>(op))
  //    graphs.back()->freeze = false;
  //}

  llvm::dbgs() << "--------------------\n";

  // First, create new nodes

  return false;
}

void scheduleOperations(func::FuncOp funcOp) {
  SmallVector<scf::ForOp> forOps;
  funcOp.walk([&](Operation *op) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      forOps.push_back(forOp);
    }
    return WalkResult::advance();
  });

  // Only schedule body of inner-most for loop for now
  for (scf::ForOp forOp : forOps) {
    prioritizedTopologicalSort(&forOp.getLoopBody().front());
  }
}

} // namespace iree_compiler
} // namespace mlir