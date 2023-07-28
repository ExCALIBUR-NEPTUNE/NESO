#ifndef __JACOBI_COEFF_MOD_BASIS_H_
#define __JACOBI_COEFF_MOD_BASIS_H_

#include <cstdlib>
#include <map>
#include <memory>
#include <neso_particles.hpp>

#include <LocalRegions/QuadExp.h>
#include <LocalRegions/TriExp.h>
#include <StdRegions/StdExpansion2D.h>

using namespace NESO::Particles;

namespace NESO {

/**
 *  Pre-compute the coefficients required to compute series of Jacobi
 *  polynomials via recursion.
 */
class JacobiCoeffModBasis {

protected:
public:
  /// Disable (implicit) copies.
  JacobiCoeffModBasis(const JacobiCoeffModBasis &st) = delete;
  /// Disable (implicit) copies.
  JacobiCoeffModBasis &operator=(JacobiCoeffModBasis const &a) = delete;

  /**
   *  Coefficients such that
   *  P_^{alpha, 1}_{n} =
   *      (coeffs_pnm10) * P_^{alpha, 1}_{n-1} * z
   *    + (coeffs_pnm11) * P_^{alpha, 1}_{n-1}
   *    + (coeffs_pnm2) * P_^{alpha, 1}_{n-2}
   *
   *  Coefficients are stored in a matrix (row major) where each row gives the
   *  coefficients for a fixed alpha. i.e. the columns are the orders.
   */
  std::vector<REAL> coeffs_pnm10;
  std::vector<REAL> coeffs_pnm11;
  std::vector<REAL> coeffs_pnm2;

  const int max_n;
  const int max_alpha;
  const int stride_n;

  /**
   *  Compute coefficients for computing Jacobi polynomial values via recursion
   *  relation. Coefficients are computed such that:
   * P_^{alpha, 1}_{n} =
   *      (coeffs_pnm10) * P_^{alpha, 1}_{n-1} * z
   *    + (coeffs_pnm11) * P_^{alpha, 1}_{n-1}
   *    + (coeffs_pnm2) * P_^{alpha, 1}_{n-2}
   *
   * @param max_n Maximum polynomial order required.
   * @param max_alpha Maximum alpha value required.
   */
  JacobiCoeffModBasis(const int max_n, const int max_alpha);

  /**
   *  Compute P^{alpha,1}_n(z) using recursion.
   *
   *  @param n Order of Jacobi polynomial
   *  @param alpha Alpha value.
   *  @param z Point to evaluate at.
   *  @returns P^{alpha,1}_n(z).
   */
  double host_evaluate(const int n, const int alpha, const double z);
};

} // namespace NESO

#endif
