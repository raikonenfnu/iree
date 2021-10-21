#ifndef IREE_COMPILER_DIALECT_NOD_TRANSFORMS_PASS_DETAIL_H_
#define IREE_COMPILER_DIALECT_NOD_TRANSFORMS_PASS_DETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace Nod {

#define GEN_PASS_CLASSES
#include "iree/compiler/Dialect/Nod/Transforms/Passes.h.inc"  // IWYU pragma: keep

}  // namespace Nod
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_NOD_TRANSFORMS_PASS_DETAIL_H_
