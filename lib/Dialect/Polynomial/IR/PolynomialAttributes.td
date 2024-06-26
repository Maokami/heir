#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_TD_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_TD_

include "lib/Dialect/Polynomial/IR/PolynomialDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

class Polynomial_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Polynomial_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Polynomial_Attr : Polynomial_Attr<"Polynomial", "polynomial"> {
  let summary = "An attribute containing a single-variable polynomial.";
  let description = [{
     #poly = #_polynomial.polynomial<x**1024 + 1>
  }];

  let parameters = (ins "Polynomial":$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Polynomial":$value), [{
      return $_get(value.getContext(), value);
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = Polynomial;
    Polynomial getPolynomial() const { return getValue(); }
  }];

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
}

def Ring_Attr : Polynomial_Attr<"Ring", "ring"> {
  let summary = "An attribute specifying a ring.";
  let description = [{
    An attribute specifying a polynomial quotient ring with integer
    coefficients, $\mathbb{Z}/n\mathbb{Z}[x] / (p(x))$.

    `cmod` is the coefficient modulus $n$, `ideal` is the ring ideal
    $(p(x))$, and `root` is a primitive 2d-th root of unity of `cmod` where d is
    the degree of `p(x)`. Because all ideals in a single-variable polynomial
    ring are principal, the ideal is defined by a single polynomial. `root` is
    optionally specified to a desired 2n-th primitive root of unity for the
    coefficient ring of the polynomial ring. If it is not provided then a
    pre-computed value is used instead for ops that need it. The semantics of
    this root (e.g., its degree) are determined by the lowerings.

      #ring = #_polynomial.ring<cmod=1234, ideal=#_polynomial.polynomial<x**1024 + 1>>
      #ring = #_polynomial.ring<cmod=256, ideal=#_polynomial.polynomial<x**4 + 1>, root=31>
  }];

  let parameters = (ins
    "IntegerAttr": $cmod,
    "Polynomial": $ideal,
    OptionalParameter<"IntegerAttr">: $root
  );

  let builders = [
    AttrBuilderWithInferredContext<
        (ins "const APInt &": $cmod, "Polynomial":$ideal), [{
      return $_get(
        ideal.getContext(),
        IntegerAttr::get(IntegerType::get(ideal.getContext(), cmod.getBitWidth()), cmod),
        ideal,
        nullptr
      );
    }]>,
    AttrBuilderWithInferredContext<
        (ins "const APInt &": $cmod, "Polynomial":$ideal, "const APInt&": $root), [{
      return $_get(
        ideal.getContext(),
        IntegerAttr::get(IntegerType::get(ideal.getContext(), cmod.getBitWidth()), cmod),
        ideal,
        IntegerAttr::get(IntegerType::get(ideal.getContext(), root.getBitWidth()), root)
      );
    }]>
  ];
  let extraClassDeclaration = [{
    Polynomial ideal() const { return getIdeal(); }
    APInt coefficientModulus() const { return getCmod().getValue(); }
    std::optional<APInt> primitive2NthRoot() const {
      auto root = getRoot();
      return root ? std::optional<APInt>(root.getValue()) : std::nullopt;
    }
  }];

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
}


#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_TD_
