// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtEnums.cpp.inc" // IWYU pragma: keep

#define GET_ATTRDEF_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.cpp.inc" // IWYU pragma: keep
                                                                     //
struct IREEVectorExtDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<LayoutAttr>(attr)) {
      os << "layout";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

void IREEVectorExtDialect::initialize() {
  addInterfaces<IREEVectorExtDialectOpAsmInterface>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.cpp.inc"
      >();

#define GET_OP_LIST
  addOperations<
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.cpp.inc"
      >();
}

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.cpp.inc"

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrInterfaces.cpp.inc"

bool PerDimLayoutAttr::contains(const LayoutDimension &dim) {
  for (LayoutDimensionAttr label : getLabels()) {
    if (label.getValue() == dim)
      return true;
  }
  return false;
}

std::optional<int64_t> PerDimLayoutAttr::getShape(const LayoutDimension &dim) {
  for (auto value : llvm::zip(getLabels(), getShapes())) {
    if (dim == std::get<0>(value).getValue())
      return std::get<1>(value);
  }
  return std::nullopt;
}

// Get the SIMT Vector shape in the order specified by dims. If no dims are
// specified, then return an empty vector.
SmallVector<int64_t>
LayoutAttr::getSIMTVectorShape(ArrayRef<LayoutDimension> dims) const {
  SmallVector<int64_t> simtVectorShape;
  for (LayoutDimension dim : dims) {
    ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
    for (PerDimLayoutAttr layout : layouts) {
      if (!layout.contains(dim))
        continue;
      simtVectorShape.push_back(layout.getShape(dim).value());
    }
  }
  return simtVectorShape;
}

PerDimLayoutAttr LayoutAttr::getDimLayout(int64_t dim) const {
  assert(dim >= 0 && dim < getLayouts().size());
  return getLayouts()[dim];
}

bool LayoutAttr::isValidLayout(ArrayRef<int64_t> shape) const {
  for (auto perDimLayout : llvm::enumerate(getLayouts())) {
    ArrayRef<int64_t> layoutShape = perDimLayout.value().getShapes();
    int64_t computedShape = std::reduce(layoutShape.begin(), layoutShape.end(),
                                        1, std::multiplies<int64_t>());
    int64_t expectedShape = shape[perDimLayout.index()];
    if (computedShape != expectedShape) {
      return false;
    }
  }
  return true;
}

static bool isLane(LayoutDimension dim) {
  return (dim == LayoutDimension::LANEX) ||
         (dim == LayoutDimension::LANEY) ||
         (dim == LayoutDimension::LANEZ);
}

static bool isVector(LayoutDimension dim) {
  return (dim == LayoutDimension::VECTORX) ||
         (dim == LayoutDimension::VECTORY) ||
         (dim == LayoutDimension::VECTORZ);
}

// Returns true if iterator is at the end and false otherwise.
bool PerDimLayoutAttr::DimensionIterator::next() {
  current += step;
  bool done = current >= end;
  if (done) current = 0;
  return done;
}

static int64_t getInnermostVectorShape(ArrayRef<LayoutDimensionAttr> labels, ArrayRef<int64_t> shapes) {
  return isVector(labels.back().getValue()) ? shapes.back() : 1;
}

PerDimLayoutAttr::Iterator PerDimLayoutAttr::getIterator() {
  PerDimLayoutAttr::Iterator iterator;
  int64_t step = getInnermostVectorShape(getLabels(), getShapes());
  for (auto [nameAttr, shape] : llvm::zip(llvm::reverse(getLabels()), llvm::reverse(getShapes()))) {
    LayoutDimension name = nameAttr.getValue();
    if (isLane(name)) continue;
    iterator.state[name] = PerDimLayoutAttr::DimensionIterator(0, shape, step);
  }
  return iterator;
}

bool PerDimLayoutAttr::Iterator::next() {
  bool done{true};
  for (auto &[name, iterator] : state) {
    if (!iterator.next()) {
      done = false;
      break;
    }
  }
  return done;
}

AffineExpr PerDimLayoutAttr::computeSIMDIndex(PerDimLayoutAttr::Iterator &iterator) {
  DenseSet<LayoutDimension> layoutDims;
  for (auto label : getLabels()) {
    if (isLane(label.getValue())) layoutDims.insert(label.getValue());
  }
  MLIRContext *ctx = getContext();
  SmallVector<AffineExpr> dims(layoutDims.size());
  bindDimsList(ctx, MutableArrayRef{dims});
  AffineExpr offset = getAffineConstantExpr(0, ctx);
  AffineExpr stride = getAffineConstantExpr(1, ctx);
  int i = 0;
  for (const auto &[nameAttr, shape] : llvm::zip(llvm::reverse(getLabels()), llvm::reverse(getShapes()))) {
    LayoutDimension name = nameAttr.getValue();
    if (layoutDims.contains(name)) {
      offset = offset + stride * dims[i++];
      stride = stride * getAffineConstantExpr(shape, ctx);
      continue;
    }
    if (!iterator.state.contains(name))
      continue;
    offset =
        offset + stride * getAffineConstantExpr(iterator.state[name].current, ctx);
    stride = stride * getAffineConstantExpr(shape, ctx);
  }
  return offset;
}

SmallVector<int64_t> LayoutAttr::computeSIMTIndex(LayoutAttr::Iterator &iterator,
                                                  ArrayRef<LayoutDimension> labels) {
  SmallVector<int64_t> offset(labels.size(), 0);
  for (int i = 0; i < labels.size(); i++) {
    int64_t stride{1};
    for (auto pair : llvm::zip(getLayouts(), iterator.states)) {
      PerDimLayoutAttr layout = std::get<0>(pair);
      PerDimLayoutAttr::Iterator iterator = std::get<1>(pair);
      for (auto [nameAttr, size] : llvm::zip(llvm::reverse(layout.getLabels()), llvm::reverse(layout.getShapes()))) {
        LayoutDimension name = nameAttr.getValue();
        if ((name == labels[i]) && (iterator.state.contains(name))) {
          offset[i] = iterator.state[name].current * stride + offset[i];
          stride = size;
        }
      }
    }
  }
  return offset;
}

LayoutAttr::Iterator LayoutAttr::getIterator() {
  LayoutAttr::Iterator iterator;
  for (auto layout : getLayouts()) {
    iterator.states.push_back(layout.getIterator());
  }
  return iterator;
}

bool LayoutAttr::Iterator::next() {
  bool done{true};
  for (auto iterator : states) {
    done = done || iterator.next();
  }
  return done;
}

void PerDimLayoutAttr::Iterator::print() {
  for (auto [label, value] : state) {
    if (isLane(label)) continue;
    llvm::outs() << stringifyLayoutDimension(label) << " : " << value.current << "/" << value.end << " ,";
  }
}

void LayoutAttr::Iterator::print() {
  for (PerDimLayoutAttr::Iterator iterator : states) {
    iterator.print();
    llvm::outs() << "\n";
  }
  llvm::outs() << "====================\n";
}

void LayoutAttr::map(std::function<void(LayoutAttr::Iterator &)> callback, Iterator &iterator) {
  do {
    iterator.print();
    callback(iterator);
  } while (!iterator.next());
}

// Project out the layout for the specified dimensions
// resulting in the layout for a lower dimensional vector.
VectorLayoutInterface LayoutAttr::project(ArrayRef<bool> projectedDims) const {
  assert(projectedDims.size() == getLayouts().size() &&
         "projectedDims size must match layout size");

  ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
  assert(projectedDims.size() == layouts.size());
  SmallVector<PerDimLayoutAttr> newLayouts;
  for (auto pair : llvm::zip(projectedDims, layouts)) {
    if (!std::get<0>(pair))
      newLayouts.push_back(std::get<1>(pair));
  }
  return LayoutAttr::get(getContext(), newLayouts);
}

// Permute the layout according to the provided permutation
// vector. The dimensionality of the layout remains the same.
VectorLayoutInterface LayoutAttr::permute(ArrayRef<int64_t> permutation) const {
  assert(permutation.size() == getLayouts().size() &&
         "permutation size must match layout size");

  ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
  assert(permutation.size() == layouts.size());
  SmallVector<PerDimLayoutAttr> newLayouts;
  for (unsigned index : permutation) {
    assert(index >= 0 && index < layouts.size());
    newLayouts.push_back(layouts[index]);
  }
  return LayoutAttr::get(getContext(), newLayouts);
}
