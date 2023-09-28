#include <nektar_interface/expansion_looping/jacobi_coeff_mod_basis.hpp>

namespace NESO {

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
JacobiCoeffModBasis::JacobiCoeffModBasis(const int max_n, const int max_alpha)
    : max_n(max_n), max_alpha(max_alpha), stride_n(max_n + 1) {

  const int beta = 1;
  this->coeffs_pnm10.reserve((max_n + 1) * (max_alpha + 1));
  this->coeffs_pnm11.reserve((max_n + 1) * (max_alpha + 1));
  this->coeffs_pnm2.reserve((max_n + 1) * (max_alpha + 1));

  for (int alphax = 0; alphax <= max_alpha; alphax++) {
    for (int nx = 0; nx <= max_n; nx++) {
      const double a = nx + alphax;
      const double b = nx + beta;
      const double c = a + b;
      const double n = nx;

      const double c_pn = 2.0 * n * (c - n) * (c - 2.0);
      const double c_pnm10 = (c - 1.0) * c * (c - 2);
      const double c_pnm11 = (c - 1.0) * (a - b) * (c - 2 * n);
      const double c_pnm2 = -2.0 * (a - 1.0) * (b - 1.0) * c;
      const double ic_pn = 1.0 / c_pn;

      this->coeffs_pnm10.push_back(ic_pn * c_pnm10);
      this->coeffs_pnm11.push_back(ic_pn * c_pnm11);
      this->coeffs_pnm2.push_back(ic_pn * c_pnm2);
    }
  }
}

/**
 *  Compute P^{alpha,1}_n(z) using recursion.
 *
 *  @param n Order of Jacobi polynomial
 *  @param alpha Alpha value.
 *  @param z Point to evaluate at.
 *  @returns P^{alpha,1}_n(z).
 */
double JacobiCoeffModBasis::host_evaluate(const int n, const int alpha,
                                          const double z) {

  NESOASSERT((0 <= n) && (n <= this->max_n), "Bad order - not in [0, max_n].");
  NESOASSERT((0 <= alpha) && (alpha <= this->max_alpha),
             "Bad alpha - not in [0, max_alpha].");

  double pnm2 = 1.0;
  if (n == 0) {
    return pnm2;
  }
  const int beta = 1;
  double pnm1 = 0.5 * (2 * (alpha + 1) + (alpha + beta + 2) * (z - 1.0));
  if (n == 1) {
    return pnm1;
  }

  double pn;
  for (int nx = 2; nx <= n; nx++) {
    const double c_pnm10 = this->coeffs_pnm10[this->stride_n * alpha + nx];
    const double c_pnm11 = this->coeffs_pnm11[this->stride_n * alpha + nx];
    const double c_pnm2 = this->coeffs_pnm2[this->stride_n * alpha + nx];
    pn = c_pnm10 * pnm1 * z + c_pnm11 * pnm1 + c_pnm2 * pnm2;
    pnm2 = pnm1;
    pnm1 = pn;
  }

  return pn;
}

} // namespace NESO
