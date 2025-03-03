#include "lib/Utils/Approximation/CaratheodoryFejer.h"

#include <cstdint>
#include <functional>
#include <optional>

#include "Eigen/Dense"  // from @eigen
#include "lib/Utils/Approximation/Chebyshev.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APFloat.h"      // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace approximation {

using ::Eigen::MatrixXd;
using ::Eigen::SelfAdjointEigenSolver;
using ::Eigen::VectorXd;
using ::llvm::APFloat;
using ::llvm::SmallVector;
using ::mlir::heir::polynomial::FloatPolynomial;

FloatPolynomial caratheodoryFejerApproximation(
    const std::function<APFloat(APFloat)> &func, int32_t degree,
    std::optional<int32_t> chebDegree) {
  // Construct the Chebyshev interpolant.
  SmallVector<APFloat> chebPts, chebEvalPts, chebCoeffs;
  int32_t actualChebDegree = chebDegree.value_or(2 * degree);
  int32_t numChebPts = 1 + actualChebDegree;
  assert(numChebPts >= 2 * degree + 1 &&
         "Chebyshev degree must be at least twice the CF degree plus 1");
  chebPts.reserve(numChebPts);
  getChebyshevPoints(numChebPts, chebPts);
  for (auto &pt : chebPts) {
    chebEvalPts.push_back(func(pt));
  }
  interpolateChebyshev(chebEvalPts, chebCoeffs);

  // Use the tail coefficients to construct a Hankel matrix
  // where A[i, j] = c[i+j]
  // Cf. https://en.wikipedia.org/wiki/Hankel_matrix
  SmallVector<APFloat> tailChebCoeffs(chebCoeffs.begin() + (degree + 1),
                                      chebCoeffs.end());
  int32_t hankelSize = tailChebCoeffs.size();
  MatrixXd hankel(hankelSize, hankelSize);
  for (int i = 0; i < hankelSize; ++i) {
    for (int j = 0; j < hankelSize; ++j) {
      // upper left triangular region, including diagonal
      if (i + j < hankelSize)
        hankel(i, j) = tailChebCoeffs[i + j].convertToDouble();
      else
        hankel(i, j) = 0;
    }
  }

  // Compute the eigenvalues and eigenvectors of the Hankel matrix
  SelfAdjointEigenSolver<MatrixXd> solver(hankel);

  const VectorXd &eigenvalues = solver.eigenvalues();
  // Eigenvectors are columns of the matrix.
  const MatrixXd &eigenvectors = solver.eigenvectors();

  // Extract the eigenvector for the (absolute value) largest eigenvalue.
  int32_t maxIndex = 0;
  double maxEigenvalue = eigenvalues(0);
  for (int32_t i = 1; i < eigenvalues.size(); ++i) {
    if (std::abs(eigenvalues(i)) > maxEigenvalue) {
      maxEigenvalue = std::abs(eigenvalues(i));
      maxIndex = i;
    }
  }
  VectorXd maxEigenvector = eigenvectors.col(maxIndex);

  // A debug for comparing the eigenvalue solver with the reference
  // implementation.
  // std::cout << "Max eigenvector:" << std::endl;
  // for (int32_t i = 0; i < maxEigenvector.size(); ++i) {
  //   std::cout << std::setprecision(18) << maxEigenvector(i) << ", ";
  // }
  // std::cout << std::endl;

  double v1 = maxEigenvector(0);
  VectorXd vv = maxEigenvector.tail(maxEigenvector.size() - 1);

  SmallVector<APFloat> b =
      SmallVector<APFloat>(tailChebCoeffs.begin(), tailChebCoeffs.end());

  int32_t t = actualChebDegree - degree - 1;
  for (int32_t i = degree; i > -degree - 1; --i) {
    SmallVector<APFloat> sliceB(b.begin(), b.begin() + t);

    APFloat sum = APFloat(0.0);
    for (int32_t j = 0; j < sliceB.size(); ++j) {
      double vvVal = vv(j);
      sum = sum + sliceB[j] * APFloat(vvVal);
    }

    APFloat z = -sum / APFloat(v1);

    // I suspect this insert is slow. Once it's working we can optimize this
    // loop to avoid the insert.
    b.insert(b.begin(), z);
  }

  SmallVector<APFloat> bb(b.begin() + degree, b.begin() + (2 * degree + 1));
  for (int32_t i = 1; i < bb.size(); ++i) {
    bb[i] = bb[i] + b[degree - 1 - (i - 1)];
  }

  SmallVector<APFloat> pk;
  pk.reserve(bb.size());
  for (int32_t i = 0; i < bb.size(); ++i) {
    pk.push_back(chebCoeffs[i] - bb[i]);
  }

  return chebyshevToMonomial(pk);
}

}  // namespace approximation
}  // namespace heir
}  // namespace mlir
