#ifndef __NESO_SPECIAL_FUNCTIONS_H_
#define __NESO_SPECIAL_FUNCTIONS_H_

#ifndef MAPPING_CROSS_PRODUCT_3D
#define MAPPING_CROSS_PRODUCT_3D(a1, a2, a3, b1, b2, b3, c1, c2, c3)           \
  c1 = ((a2) * (b3)) - ((a3) * (b2));                                          \
  c2 = ((a3) * (b1)) - ((a1) * (b3));                                          \
  c3 = ((a1) * (b2)) - ((a2) * (b1));
#endif

#ifndef MAPPING_DOT_PRODUCT_3D
#define MAPPING_DOT_PRODUCT_3D(a1, a2, a3, b1, b2, b3)                         \
  ((a1) * (b1) + (a2) * (b2) + (a3) * (b3))
#endif

#ifndef KERNEL_MIN
#define KERNEL_MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

#ifndef KERNEL_MAX
#define KERNEL_MAX(x, y) (((x) < (y)) ? (y) : (x))
#endif

#include <cmath>

namespace NESO {

/**
 *  Compute the Pochhammer symbol (m)_n.
 *
 *  @param m Value of m.
 *  @param n Value of n.
 *  @returns (m)_n.
 */
inline int pochhammer(const int m, const int n) {
  int output = 1;
  for (int offset = 0; offset <= (n - 1); offset++) {
    output *= (m + offset);
  }
  return output;
};

/**
 *  Computes jacobi polynomial P{alpha, beta}_p(z).
 *
 *  @param p Order.
 *  @param z Evaluation point.
 *  @param alpha Alpha value.
 *  @param beta Beta value.
 */
inline double jacobi(const int p, const double z, const int alpha,
                     const int beta) {
  if (p == 0) {
    return 1.0;
  }
  double pnm1 = 0.5 * (2 * (alpha + 1) + (alpha + beta + 2) * (z - 1.0));
  if (p == 1) {
    return pnm1;
  }
  double pn =
      0.125 * (4 * (alpha + 1) * (alpha + 2) +
               4 * (alpha + beta + 3) * (alpha + 2) * (z - 1.0) +
               (alpha + beta + 3) * (alpha + beta + 4) * (z - 1.0) * (z - 1.0));
  if (p == 2) {
    return pn;
  }

  double pnp1;
  for (int px = 3; px <= p; px++) {
    const int n = (px - 1);
    const double coeff_pnp1 =
        2 * (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta);
    const double coeff_pn =
        (2 * n + alpha + beta + 1) * (alpha * alpha - beta * beta) +
        pochhammer(2 * n + alpha + beta, 3) * z;
    const double coeff_pnm1 =
        -2.0 * (n + alpha) * (n + beta) * (2 * n + alpha + beta + 2);

    pnp1 = (1.0 / coeff_pnp1) * (coeff_pn * pn + coeff_pnm1 * pnm1);

    pnm1 = pn;
    pn = pnp1;
  }

  return pnp1;
};

/**
 *  Compute relative error between a correct value and a test value.
 *
 *  @param correct Correct value to test against.
 *  @param to_test Value to compare with the correct value.
 *  @returns relative error.
 */
inline double relative_error(const double correct, const double to_test) {
  const double abs_correct = std::abs(correct);
  const double abs_error = std::abs(correct - to_test);
  return abs_correct == 0 ? abs_error : abs_error / abs_correct;
}

/**
 *  Compute absolute error between a correct value and a test value.
 *
 *  @param correct Correct value to test against.
 *  @param to_test Value to compare with the correct value.
 *  @returns absolute error.
 */
inline double absolute_error(const double correct, const double to_test) {
  const double abs_error = std::abs(correct - to_test);
  return abs_error;
}

/**
 *  Compute absolute and relative error between a correct value and a test value
 * and return the minimum.
 *
 *  @param correct Correct value to test against.
 *  @param to_test Value to compare with the correct value.
 *  @returns Minimum of relative and absolute error.
 */
inline double minimum_absrel_error(const double correct, const double to_test) {
  return std::min(absolute_error(correct, to_test),
                  relative_error(correct, to_test));
}

} // namespace NESO

#endif
