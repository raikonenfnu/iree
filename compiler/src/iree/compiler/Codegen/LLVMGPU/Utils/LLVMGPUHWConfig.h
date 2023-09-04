
#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUHWCONFIG_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUHWCONFIG_H_

#include "LLVMGPULayout.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::iree_compiler {

enum class MatrixType {
  A, B, C, D
};

struct LLVMGPUHWConfig {
  LLVMGPUHWConfig() {}
  LLVMGPUHWConfig(LLVMGPULayout::ContractType contractType) : contractType(contractType) {}
  virtual LLVMGPULayout getLayout(MatrixType matrixType, Value matrix) {
    return LLVMGPULayout();
  }
  virtual bool verifyOperandTypes(Value a, Value b, Value c, Value d) { return false; }
  virtual bool verifyContract(vector::ContractionOp contractOp) { return false; }
  SmallVector<int64_t> getIndices(MatrixType matrixType, int i, int j);

  LLVMGPULayout::ContractType contractType;
};

// For more information, see link below:
// https://www.amd.com/system/files/TechDocs/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf
struct AMDWMMAConfig : public LLVMGPUHWConfig {
  enum class WMMAType {
    F16_16X16X16_F16,
    F32_16X16X16_F16,
  };
  AMDWMMAConfig(WMMAType wmmaType, LLVMGPULayout::ContractType contractType, uint32_t warpSize) :
    LLVMGPUHWConfig(contractType), wmmaType(wmmaType), warpSize(warpSize) {}

  LLVMGPULayout getLayout(MatrixType matrixType, Value matrix) override;
  bool verifyOperandTypes(Value a, Value b, Value c, Value d) override;
  bool verifyContract(vector::ContractionOp contractOp) override;

  LLVMGPULayout createWMMAF16Layout(MatrixType matrixType,
                                    ArrayRef<int64_t> matrixShape);

  WMMAType wmmaType;
  uint32_t warpSize;
};
}

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUHWCONFIG_H_