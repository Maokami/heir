#include "lib/Dialect/ModArith/IR/ModArithDialect.h"

#include <cassert>
#include <optional>

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"       // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

// NOLINTBEGIN(misc-include-cleaner): Required to define
// ModArithDialect, ModArithTypes, ModArithOps,
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/CommonFolders.h"   // from @llvm-project
// NOLINTEND(misc-include-cleaner)

// Generated definitions
#include "lib/Dialect/ModArith/IR/ModArithDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithOps.cpp.inc"

#define DEBUG_TYPE "mod-arith"

namespace mlir {
namespace heir {
namespace mod_arith {

void ModArithDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/ModArith/IR/ModArithTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/ModArith/IR/ModArithOps.cpp.inc"
      >();
}

/// Ensures that the underlying integer type is wide enough for the coefficient
template <typename OpType>
LogicalResult verifyModArithType(OpType op, ModArithType type) {
  APInt modulus = type.getModulus().getValue();
  unsigned bitWidth = modulus.getBitWidth();
  unsigned modWidth = modulus.getActiveBits();
  if (modWidth > bitWidth - 1)
    return op.emitOpError()
           << "underlying type's bitwidth must be 1 bit larger than "
           << "the modulus bitwidth, but got " << bitWidth
           << " while modulus requires width " << modWidth << ".";
  return success();
}

template <typename OpType>
LogicalResult verifySameWidth(OpType op, ModArithType modArithType,
                              IntegerType integerType) {
  unsigned bitWidth = modArithType.getModulus().getValue().getBitWidth();
  unsigned intWidth = integerType.getWidth();
  if (intWidth != bitWidth)
    return op.emitOpError()
           << "the result integer type should be of the same width as the "
           << "mod arith type width, but got " << intWidth
           << " while mod arith type width " << bitWidth << ".";
  return success();
}

LogicalResult ExtractOp::verify() {
  auto modArithType = getOperandModArithType(*this);
  auto integerType = getResultIntegerType(*this);
  auto result = verifySameWidth(*this, modArithType, integerType);
  if (result.failed()) return result;
  return verifyModArithType(*this, modArithType);
}

LogicalResult ReduceOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult AddOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult SubOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult MulOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult MacOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult BarrettReduceOp::verify() {
  auto inputType = getInput().getType();
  unsigned bitWidth;
  if (auto tensorType = dyn_cast<RankedTensorType>(inputType)) {
    bitWidth = tensorType.getElementTypeBitWidth();
  } else {
    auto integerType = dyn_cast<IntegerType>(inputType);
    assert(integerType &&
           "expected input to be a ranked tensor type or integer type");
    bitWidth = integerType.getWidth();
  }
  auto expectedBitWidth = (getModulus() - 1).getActiveBits();
  if (bitWidth < expectedBitWidth || 2 * expectedBitWidth < bitWidth) {
    return emitOpError()
           << "input bitwidth is required to be in the range [w, 2w], where w "
              "is the smallest bit-width that contains the range [0, modulus). "
              "Got "
           << bitWidth << " but w is " << expectedBitWidth << ".";
  }
  if (getModulus().slt(0))
    return emitOpError() << "provided modulus " << getModulus().getSExtValue()
                         << " is not a positive integer.";
  return success();
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  APInt parsedInt;
  Type parsedType;

  if (parser.parseInteger(parsedInt) || parser.parseColonType(parsedType))
    return failure();

  result.addAttribute(
      "value", IntegerAttr::get(IntegerType::get(parser.getContext(),
                                                 parsedInt.getBitWidth()),
                                parsedInt));
  result.addTypes(parsedType);
  return success();
}

LogicalResult ConstantOp::verify() {
  auto valueBW = getValue().getBitWidth();
  auto modBW = getType().getModulus().getValue().getBitWidth();
  if (valueBW > modBW)
    return emitOpError(
        "Constant value's bitwidth must be smaller than underlying type.");

  return success();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  getValue().print(p.getStream(), true);
  p << " : ";
  p.printType(getOutput().getType());
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  // constant(c0) -> c0 mod q
  APInt cst = adaptor.getValue();

  auto modType = dyn_cast_if_present<ModArithType>(getType());
  if (!modType) return {};

  // Retrieve the modulus value and its bit width
  APInt modulus = modType.getModulus().getValue();
  unsigned modBitWidth = modulus.getBitWidth();

  // Adjust cst's bit width to match modulus if necessary
  if (cst.getBitWidth() != modBitWidth) {
    cst = cst.zext(modBitWidth);
  }

  // Fold the constant value
  APInt foldedVal = cst.urem(modulus);

  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    llvm::dbgs() << "========================================\n";
    llvm::dbgs() << "  Folding Operation: Constant\n";
    llvm::dbgs() << "----------------------------------------\n";
    llvm::dbgs() << "  Value   : " << cst << "\n";
    llvm::dbgs() << "  Modulus : " << modulus << "\n";
    llvm::dbgs() << "  Folded  : " << foldedVal << "\n";
    llvm::dbgs() << "========================================\n";
  });

  // Create the result
  auto elementType = modType.getModulus().getType();
  return IntegerAttr::get(elementType, foldedVal);
}

/// Helper function to handle common folding logic for binary arithmetic
/// operations.
/// - `opName` is used for debug output.
/// - `foldBinFn` defines how the actual binary operation (+, -, *) should be
/// performed.
template <typename FoldAdaptor, typename FoldBinFn>
static OpFoldResult foldBinModOp(Operation *op, FoldAdaptor adaptor,
                                 FoldBinFn &&foldBinFn,
                                 llvm::StringRef opName) {
  // Check if lhs and rhs are ModArithAttr
  auto lhs = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs) return {};

  auto modType = dyn_cast<ModArithType>(op->getResultTypes().front());
  if (!modType) return {};

  // Retrieve the modulus value and its bit width
  APInt modulus = modType.getModulus().getValue();
  unsigned modBitWidth = modulus.getBitWidth();

  // Extract the actual integer values
  APInt lhsVal = lhs.getValue();
  APInt rhsVal = rhs.getValue();

  // Adjust lhsVal and rhsVal bit widths to match modulus if necessary
  if (lhsVal.getBitWidth() != modBitWidth) {
    lhsVal = lhsVal.zext(modBitWidth);
  }
  if (rhsVal.getBitWidth() != modBitWidth) {
    rhsVal = rhsVal.zext(modBitWidth);
  }

  // Perform the operation using the provided foldBinFn
  APInt foldedVal = foldBinFn(lhsVal, rhsVal, modulus);

  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    llvm::dbgs() << "========================================\n";
    llvm::dbgs() << "  Folding Operation: " << opName << "\n";
    llvm::dbgs() << "----------------------------------------\n";
    llvm::dbgs() << "  LHS     : " << lhsVal << "\n";
    llvm::dbgs() << "  RHS     : " << rhsVal << "\n";
    llvm::dbgs() << "  Modulus : " << modulus << "\n";
    llvm::dbgs() << "  Folded  : " << foldedVal << "\n";
    llvm::dbgs() << "========================================\n";
  });

  // Create the result
  auto elementType = modType.getModulus().getType();
  return IntegerAttr::get(elementType, foldedVal);
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  // add(c0, c1) -> (c0 + c1) mod q
  return foldBinModOp(
      getOperation(), adaptor,
      [](APInt lhs, APInt rhs, APInt modulus) {
        APInt sum = lhs + rhs;
        return sum.urem(modulus);
      },
      "Add");
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  // sub(c0, c1) -> (c0 - c1) mod q
  return foldBinModOp(
      getOperation(), adaptor,
      [](APInt lhs, APInt rhs, APInt modulus) {
        APInt diff = lhs - rhs;
        if (diff.isNegative()) {
          diff += modulus;
        }
        return diff.urem(modulus);
      },
      "Sub");
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  // mul(c0, c1) -> (c0 * c1) mod q
  return foldBinModOp(
      getOperation(), adaptor,
      [](APInt lhs, APInt rhs, APInt modulus) {
        APInt product = lhs * rhs;
        return product.urem(modulus);
      },
      "Mul");
}

Operation *ModArithDialect::materializeConstant(OpBuilder &builder,
                                                Attribute value, Type type,
                                                Location loc) {
  auto intAttr = dyn_cast_if_present<IntegerAttr>(value);
  if (!intAttr) return nullptr;
  auto modType = dyn_cast_if_present<ModArithType>(type);
  if (!modType) return nullptr;
  auto op = builder.create<mod_arith::ConstantOp>(loc, modType, intAttr);
  return op.getOperation();
}

//===----------------------------------------------------------------------===//
// TableGen'd canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
#include "lib/Dialect/ModArith/IR/ModArithCanonicalization.cpp.inc"
}  // namespace

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<AddZero, AddAddConstant, AddSubConstantRHS, AddSubConstantLHS,
              AddMulNegativeOneRhs, AddMulNegativeOneLhs>(context);
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SubZero, SubMulNegativeOneRhs, SubMulNegativeOneLhs,
              SubRHSAddConstant, SubLHSAddConstant, SubRHSSubConstantRHS,
              SubRHSSubConstantLHS, SubLHSSubConstantRHS, SubLHSSubConstantLHS,
              SubSubLHSRHSLHS>(context);
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<MulZero, MulOne, MulMulConstant>(context);
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
