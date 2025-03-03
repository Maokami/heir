#include "lib/Dialect/ModArith/IR/ModArithDialect.h"

#include <cassert>
#include <optional>

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
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

// NOLINTBEGIN(misc-include-cleaner): Required to define ModArithDialect,
// ModArithTypes, ModArithOps, ModArithAttributes
#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/CommonFolders.h"   // from @llvm-project
// NOLINTEND(misc-include-cleaner)

// Generated definitions
#include "lib/Dialect/ModArith/IR/ModArithDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithOps.cpp.inc"

#define DEBUG_TYPE "mod-arith"

namespace mlir {
namespace heir {
namespace mod_arith {

class ModArithOpAsmDialectInterface : public OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    auto res = llvm::TypeSwitch<Type, AliasResult>(type)
                   .Case<ModArithType>([&](auto &modArithType) {
                     os << "Z";
                     os << modArithType.getModulus().getValue();
                     os << "_";
                     os << modArithType.getModulus().getType();
                     return AliasResult::FinalAlias;
                   })
                   .Default([&](Type) { return AliasResult::NoAlias; });
    return res;
  }
};

void ModArithDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/ModArith/IR/ModArithTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/ModArith/IR/ModArithAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/ModArith/IR/ModArithOps.cpp.inc"
      >();

  addInterface<ModArithOpAsmDialectInterface>();
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
  APInt parsedValue(64, 0);
  Type parsedType;

  if (failed(parser.parseInteger(parsedValue))) {
    parser.emitError(parser.getCurrentLocation(),
                     "found invalid integer value");
    return failure();
  }

  if (parser.parseColon() || parser.parseType(parsedType)) return failure();

  auto modArithType = dyn_cast<ModArithType>(parsedType);
  if (!modArithType) return failure();

  auto outputBitWidth =
      modArithType.getModulus().getType().getIntOrFloatBitWidth();
  if (parsedValue.getActiveBits() > outputBitWidth)
    return parser.emitError(parser.getCurrentLocation(),
                            "constant value is too large for the modulus");

  auto intValue = IntegerAttr::get(modArithType.getModulus().getType(),
                                   parsedValue.trunc(outputBitWidth));
  result.addAttribute(
      "value", ModArithAttr::get(parser.getContext(), modArithType, intValue));
  result.addTypes(modArithType);
  return success();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  // getValue chain:
  // op's ModArithAttribute value
  //   -> ModArithAttribute's IntegerAttr value
  //   -> IntegerAttr's APInt value
  getValue().getValue().getValue().print(p.getStream(), true);
  p << " : ";
  p.printType(getOutput().getType());
}

LogicalResult ConstantOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> loc,
    ConstantOpAdaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  returnTypes.push_back(adaptor.getValue().getType());
  return success();
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  // constant(c0) -> c0 mod q
  auto constant = dyn_cast_if_present<ModArithAttr>(adaptor.getValue());
  if (!constant)
    return {};
  auto modType = dyn_cast_if_present<ModArithType>(getType());
  if (!modType)
    return {};

  // Retrieve the modulus value
  int64_t modulus = modType.getModulus().getValue().getSExtValue();

  // Extract the actual integer value
  auto value = constant.getValue().getInt();

  // Fold the constant value
  int64_t foldedVal = ((value % modulus) + modulus) % modulus;

  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    llvm::dbgs() << "========================================\n";
    llvm::dbgs() << "  Folding Operation: Constant\n";
    llvm::dbgs() << "----------------------------------------\n";
    llvm::dbgs() << "  Value   : " << value << "\n";
    llvm::dbgs() << "  Modulus : " << modulus << "\n";
    llvm::dbgs() << "  Folded  : " << foldedVal << "\n";
    llvm::dbgs() << "========================================\n";
  });
  
  // Create the resulting ModArithAttr
  auto elementType = modType.getModulus().getType();
  auto foldedIntAttr = IntegerAttr::get(elementType, foldedVal);
  auto ctx = getContext();
  return ModArithAttr::get(ctx, modType, foldedIntAttr);
}

/// Helper function to handle common folding logic for binary arithmetic operations.
/// - `opName` is used for debug output.
/// - `foldBinFn` defines how the actual binary operation (+, -, *) should be performed.
template <typename FoldAdaptor, typename FoldBinFn>
static OpFoldResult foldBinModOp(Operation *op,
                              FoldAdaptor adaptor,
                              FoldBinFn &&foldBinFn,
                              llvm::StringRef opName) {
  // Check if lhs and rhs are ModArithAttr
  auto lhs = dyn_cast_if_present<ModArithAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_if_present<ModArithAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
    return {};

  // Ensure the result type is ModArithType
  auto modType = dyn_cast<ModArithType>(op->getResultTypes().front());
  if (!modType)
    return {};

  // Retrieve the modulus value
  int64_t modulus = modType.getModulus().getValue().getSExtValue();

  // Extract the actual integer values
  int64_t lhsVal = lhs.getValue().getInt();
  int64_t rhsVal = rhs.getValue().getInt();

  // Perform the operation using the provided foldBinFn
  int64_t foldedVal = foldBinFn(lhsVal, rhsVal, modulus);

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

  // Create the resulting ModArithAttr
  auto elementType = modType.getModulus().getType();
  auto foldedIntAttr = IntegerAttr::get(elementType, foldedVal);
  auto ctx = op->getContext();
  return ModArithAttr::get(ctx, modType, foldedIntAttr);
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  // add(c0, c1) -> (c0 + c1) mod q
  return foldBinModOp(getOperation(), adaptor,
                   [](int64_t lhs, int64_t rhs, int64_t modulus) {
                     int64_t sum = lhs + rhs;
                     // Subtract modulus if the result exceeds it
                     return (sum >= modulus) ? (sum - modulus) : sum;
                   },
                   "Add");
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  // sub(c0, c1) -> (c0 - c1) mod q
  return foldBinModOp(getOperation(), adaptor,
                   [](int64_t lhs, int64_t rhs, int64_t modulus) {
                     int64_t diff = lhs - rhs;
                     // Add modulus if the result is negative
                     return (diff < 0) ? (diff + modulus) : diff;
                   },
                   "Sub");
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  // mul(c0, c1) -> (c0 * c1) mod q
  return foldBinModOp(getOperation(), adaptor,
                   [](int64_t lhs, int64_t rhs, int64_t modulus) {
                     // Perform modular multiplication
                     return (lhs * rhs) % modulus;
                   },
                   "Mul");
}


Operation *ModArithDialect::materializeConstant(
  OpBuilder &builder, 
  Attribute value, 
  Type type, 
  Location loc) {
auto modArithTy = dyn_cast<ModArithType>(type);
if (!modArithTy)
  return nullptr;

if (auto modArithAttr = dyn_cast<ModArithAttr>(value)) {
  if (modArithAttr.getType() != modArithTy)
    return nullptr;
  auto op = builder.create<mod_arith::ConstantOp>(loc, type, modArithAttr);
  return op.getOperation();
}

return nullptr;
}

//===----------------------------------------------------------------------===//
// TableGen'd canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
#include "lib/Dialect/ModArith/IR/ModArithCanonicalization.cpp.inc"
}  // namespace

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<AddZero, AddAddConstant, AddSubConstantRHS, AddSubConstantLHS, AddMulNegativeOneRhs, AddMulNegativeOneLhs>(context);
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SubZero, SubRHSAddConstant, SubLHSAddConstant, SubRHSSubConstantRHS, SubRHSSubConstantLHS, SubLHSSubConstantRHS, SubLHSSubConstantLHS, SubSubLHSRHSLHS>(context);
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<MulZero, MulOne, MulMulConstant>(context);
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
