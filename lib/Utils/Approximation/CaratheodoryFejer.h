#ifndef LIB_UTILS_APPROXIMATION_CARATHEODORYFEJER_H_
#define LIB_UTILS_APPROXIMATION_CARATHEODORYFEJER_H_

#include <cstdint>
#include <functional>
#include <optional>

#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APFloat.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace approximation {

/// Construct the Caratheodory-Fejer approximation of a given function. The
/// result polynomial is represented in the monomial basis and has degree
/// `degree`.
///
/// A port of chebfun's cf.m, cf.
/// https://github.com/chebfun/chebfun/blob/69c12cf75f93cb2f36fd4cfd5e287662cd2f1091/%40chebfun/cf.m
/// Specifically, the path through `cf` that invokes `polynomialCF`
/// (rational functions are not supported here).
///
/// Cf. https://doi.org/10.1007/s10543-011-0331-7 for a mathematical
/// explanation.
/// https://people.maths.ox.ac.uk/trefethen/publication/PDF/2011_138.pdf
///
/// The argument chebDegree provides control over the degree of the Chebyshev
/// interpolation used to seed the CF approximation. If unset, it
/// is heuristically chosen as twice the desired CF approximant degree.
::mlir::heir::polynomial::FloatPolynomial caratheodoryFejerApproximation(
    const std::function<::llvm::APFloat(::llvm::APFloat)> &func, int32_t degree,
    std::optional<int32_t> chebDegree = std::nullopt);

}  // namespace approximation
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_APPROXIMATION_CARATHEODORYFEJER_H_
