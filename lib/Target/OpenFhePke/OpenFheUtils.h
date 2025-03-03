#ifndef LIB_TARGET_OPENFHEPKE_OPENFHEUTILS_H_
#define LIB_TARGET_OPENFHEPKE_OPENFHEUTILS_H_

#include <string>

#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

enum class OpenfheScheme { BGV, CKKS };

// OpenFHE's installation process moves headers around in the install directory,
// as well as changing the import paths from the development repository. This
// option controls which type of import should be used on the generated code.
enum class OpenfheImportType {
  // Import paths are relative to the openfhe development repository, i.e.,
  // paths like #include "src/pke/include/openfhe.h". This is primarily useful
  // for development within HEIR, where the openfhe source repository is cloned
  // by bazel and otherwise not installed on the system.
  SOURCE_RELATIVE,

  // Import paths are relative to the openfhe installation process, i.e., paths
  // like #include "openfhe/pke/openfhe.h". This is useful for user-facing code
  // generation, where the openfhe backend is installed by the user or shipped
  // as a shared library dependency of a heir frontend.
  INSTALL_RELATIVE
};

std::string getModulePrelude(OpenfheScheme scheme,
                             OpenfheImportType importType);

/// Convert a type to a string.
::mlir::FailureOr<std::string> convertType(::mlir::Type type,
                                           ::mlir::Location loc);

/// Find the CryptoContext SSA value in the input operation's parent func
/// arguments.
::mlir::FailureOr<::mlir::Value> getContextualCryptoContext(
    ::mlir::Operation *op);

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_OPENFHEUTILS_H_
