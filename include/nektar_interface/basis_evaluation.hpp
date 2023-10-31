#ifndef __BASIS_EVALUATION_H_
#define __BASIS_EVALUATION_H_
#include "particle_interface.hpp"
#include <cstdlib>
#include <map>
#include <memory>
#include <neso_particles.hpp>

#include <LocalRegions/QuadExp.h>
#include <LocalRegions/TriExp.h>
#include <StdRegions/StdExpansion2D.h>

#include "basis_reference.hpp"
#include "expansion_looping/geom_to_expansion_builder.hpp"
#include "expansion_looping/jacobi_coeff_mod_basis.hpp"
#include "geometry_transport/shape_mapping.hpp"
#include "special_functions.hpp"
#include "utility_sycl.hpp"

using namespace NESO::Particles;
using namespace Nektar::LocalRegions;
using namespace Nektar::StdRegions;

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tgmath.h>

namespace NESO {

namespace BasisJacobi {

/**
 *  Evaluate the eModified_B basis functions up to a given order placing the
 *  evaluations in an output array. For reference see the function eval_modB_ij.
 *  Jacobi polynomials are evaluated using recusion relations:
 *
 *  For brevity the (alpha, beta) superscripts are dropped. i.e. P_n(z) =
 * P_n^{alpha, beta}(z). P_n(z) = C_{n-1}^0 P_{n-1}(z) * z + C_{n-1}^1
 * P_{n-1}(z) + C_{n-2} * P_{n-2}(z) P_0(z) = 1 P_1(z) = 2 + 2 * (z - 1)
 *
 * @param[in] nummodes Number of modes to compute, i.e. p modes evaluates at
 * most an order p-1 polynomial.
 * @param[in] z Evaluation point to evaluate basis at.
 * @param[in] k_stride_n Stride between sets of coefficients for different
 * alpha values in the coefficient arrays.
 * @param[in] k_coeffs_pnm10 Coefficients for C_{n-1}^0 for different alpha
 * values stored row wise for each alpha.
 * @param[in] k_coeffs_pnm11 Coefficients for C_{n-1}^1 for different alpha
 * values stored row wise for each alpha.
 * @param[in] k_coeffs_pnm2 Coefficients for C_{n-2} for different alpha values
 * stored row wise for each alpha.
 * @param[in, out] output entry i contains the i-th eModified_B basis function
 * evaluated at z. This particular basis function runs over two indices p and q
 * and we linearise this two dimensional indexing to match the Nektar++
 * ordering.
 */
inline void mod_B(const int nummodes, const REAL z, const int k_stride_n,
                  const REAL *const k_coeffs_pnm10,
                  const REAL *const k_coeffs_pnm11,
                  const REAL *const k_coeffs_pnm2, REAL *output) {
  int modey = 0;
  const REAL b0 = 0.5 * (1.0 - z);
  const REAL b1 = 0.5 * (1.0 + z);
  REAL b1_pow = 1.0 / b0;
  for (int px = 0; px < nummodes; px++) {
    REAL pn, pnm1, pnm2;
    b1_pow *= b0;
    const int alpha = 2 * px - 1;
    for (int qx = 0; qx < (nummodes - px); qx++) {
      REAL etmp1;
      // evaluate eModified_B at eta1
      if (px == 0) {
        // evaluate eModified_A(q, eta1)
        if (qx == 0) {
          etmp1 = b0;
        } else if (qx == 1) {
          etmp1 = b1;
        } else if (qx == 2) {
          etmp1 = b0 * b1;
          pnm2 = 1.0;
        } else if (qx == 3) {
          pnm1 = (2.0 + 2.0 * (z - 1.0));
          etmp1 = b0 * b1 * pnm1;
        } else {
          const int nx = qx - 2;
          const REAL c_pnm10 = k_coeffs_pnm10[k_stride_n * 1 + nx];
          const REAL c_pnm11 = k_coeffs_pnm11[k_stride_n * 1 + nx];
          const REAL c_pnm2 = k_coeffs_pnm2[k_stride_n * 1 + nx];
          pn = c_pnm10 * pnm1 * z + c_pnm11 * pnm1 + c_pnm2 * pnm2;
          pnm2 = pnm1;
          pnm1 = pn;
          etmp1 = pn * b0 * b1;
        }
      } else if (qx == 0) {
        etmp1 = b1_pow;
      } else {
        const int nx = qx - 1;
        if (qx == 1) {
          pnm2 = 1.0;
          etmp1 = b1_pow * b1;
        } else if (qx == 2) {
          pnm1 = 0.5 * (2.0 * (alpha + 1) + (alpha + 3) * (z - 1.0));
          etmp1 = b1_pow * b1 * pnm1;
        } else {
          const REAL c_pnm10 = k_coeffs_pnm10[k_stride_n * alpha + nx];
          const REAL c_pnm11 = k_coeffs_pnm11[k_stride_n * alpha + nx];
          const REAL c_pnm2 = k_coeffs_pnm2[k_stride_n * alpha + nx];

          pn = c_pnm10 * pnm1 * z + c_pnm11 * pnm1 + c_pnm2 * pnm2;
          pnm2 = pnm1;
          pnm1 = pn;
          etmp1 = b1_pow * b1 * pn;
        }
      }
      const int mode = modey++;
      output[mode] = etmp1;
    }
  }
}

/**
 *  Evaluate the eModified_A basis functions up to a given order placing the
 *  evaluations in an output array. For reference see the function eval_modA_i.
 *  Jacobi polynomials are evaluated using recusion relations:
 *
 *  For brevity the (alpha, beta) superscripts are dropped. i.e. P_n(z) =
 * P_n^{alpha, beta}(z). P_n(z) = C_{n-1}^0 P_{n-1}(z) * z + C_{n-1}^1
 * P_{n-1}(z) + C_{n-2} * P_{n-2}(z) P_0(z) = 1 P_1(z) = 2 + 2 * (z - 1)
 *
 * @param[in] nummodes Number of modes to compute, i.e. p modes evaluates at
 * most an order p-1 polynomial.
 * @param[in] z Evaluation point to evaluate basis at.
 * @param[in] k_stride_n Stride between sets of coefficients for different
 * alpha values in the coefficient arrays.
 * @param[in] k_coeffs_pnm10 Coefficients for C_{n-1}^0 for different alpha
 * values stored row wise for each alpha.
 * @param[in] k_coeffs_pnm11 Coefficients for C_{n-1}^1 for different alpha
 * values stored row wise for each alpha.
 * @param[in] k_coeffs_pnm2 Coefficients for C_{n-2} for different alpha values
 * stored row wise for each alpha.
 * @param[in, out] output entry i contains the i-th eModified_A basis function
 * evaluated at z.
 */
inline void mod_A(const int nummodes, const REAL z, const int k_stride_n,
                  const REAL *const k_coeffs_pnm10,
                  const REAL *const k_coeffs_pnm11,
                  const REAL *const k_coeffs_pnm2, REAL *output) {
  const REAL b0 = 0.5 * (1.0 - z);
  const REAL b1 = 0.5 * (1.0 + z);
  output[0] = b0;
  output[1] = b1;
  REAL pn;
  REAL pnm2 = 1.0;
  REAL pnm1 = 2.0 + 2.0 * (z - 1.0);
  if (nummodes > 2) {
    output[2] = b0 * b1;
  }
  if (nummodes > 3) {
    output[3] = b0 * b1 * pnm1;
  }
  for (int modex = 4; modex < nummodes; modex++) {
    const int nx = modex - 2;
    const REAL c_pnm10 = k_coeffs_pnm10[k_stride_n * 1 + nx];
    const REAL c_pnm11 = k_coeffs_pnm11[k_stride_n * 1 + nx];
    const REAL c_pnm2 = k_coeffs_pnm2[k_stride_n * 1 + nx];
    pn = c_pnm10 * pnm1 * z + c_pnm11 * pnm1 + c_pnm2 * pnm2;
    pnm2 = pnm1;
    pnm1 = pn;
    output[modex] = b0 * b1 * pn;
  }
}

/**
 *  Evaluate the eModified_C basis functions up to a given order placing the
 *  evaluations in an output array. For reference see the function
 * eval_modC_ijk. Jacobi polynomials are evaluated using recusion relations:
 *
 *  For brevity the (alpha, beta) superscripts are dropped. i.e. P_n(z) =
 * P_n^{alpha, beta}(z). P_n(z) = C_{n-1}^0 P_{n-1}(z) * z + C_{n-1}^1
 * P_{n-1}(z) + C_{n-2} * P_{n-2}(z) P_0(z) = 1 P_1(z) = 2 + 2 * (z - 1)
 *
 * @param[in] nummodes Number of modes to compute, i.e. p modes evaluates at
 * most an order p-1 polynomial.
 * @param[in] z Evaluation point to evaluate basis at.
 * @param[in] k_stride_n Stride between sets of coefficients for different
 * alpha values in the coefficient arrays.
 * @param[in] k_coeffs_pnm10 Coefficients for C_{n-1}^0 for different alpha
 * values stored row wise for each alpha.
 * @param[in] k_coeffs_pnm11 Coefficients for C_{n-1}^1 for different alpha
 * values stored row wise for each alpha.
 * @param[in] k_coeffs_pnm2 Coefficients for C_{n-2} for different alpha values
 * stored row wise for each alpha.
 * @param[in, out] output entry i contains the i-th eModified_C basis function
 * evaluated at z.
 */
inline void mod_C(const int nummodes, const REAL z, const int k_stride_n,
                  const REAL *const k_coeffs_pnm10,
                  const REAL *const k_coeffs_pnm11,
                  const REAL *const k_coeffs_pnm2, REAL *output) {

  int mode = 0;
  const REAL b0 = 0.5 * (1.0 - z);
  const REAL b1 = 0.5 * (1.0 + z);
  REAL outer_b1_pow = 1.0 / b0;

  for (int p = 0; p < nummodes; p++) {
    outer_b1_pow *= b0;
    REAL inner_b1_pow = outer_b1_pow;

    for (int q = 0; q < (nummodes - p); q++) {
      const int px = p + q;
      const int alpha = 2 * px - 1;
      REAL pn, pnm1, pnm2;

      for (int r = 0; r < (nummodes - p - q); r++) {
        const int qx = r;
        REAL etmp1;
        // evaluate eModified_B at eta
        if (px == 0) {
          // evaluate eModified_A(q, eta1)
          if (qx == 0) {
            etmp1 = b0;
          } else if (qx == 1) {
            etmp1 = b1;
          } else if (qx == 2) {
            etmp1 = b0 * b1;
            pnm2 = 1.0;
          } else if (qx == 3) {
            pnm1 = (2.0 + 2.0 * (z - 1.0));
            etmp1 = b0 * b1 * pnm1;
          } else {
            const int nx = qx - 2;
            const REAL c_pnm10 = k_coeffs_pnm10[k_stride_n * 1 + nx];
            const REAL c_pnm11 = k_coeffs_pnm11[k_stride_n * 1 + nx];
            const REAL c_pnm2 = k_coeffs_pnm2[k_stride_n * 1 + nx];
            pn = c_pnm10 * pnm1 * z + c_pnm11 * pnm1 + c_pnm2 * pnm2;
            pnm2 = pnm1;
            pnm1 = pn;
            etmp1 = pn * b0 * b1;
          }
        } else if (qx == 0) {
          etmp1 = inner_b1_pow;
        } else {
          const int nx = qx - 1;
          if (qx == 1) {
            pnm2 = 1.0;
            etmp1 = inner_b1_pow * b1;
          } else if (qx == 2) {
            pnm1 = 0.5 * (2.0 * (alpha + 1) + (alpha + 3) * (z - 1.0));
            etmp1 = inner_b1_pow * b1 * pnm1;
          } else {
            const REAL c_pnm10 = k_coeffs_pnm10[k_stride_n * alpha + nx];
            const REAL c_pnm11 = k_coeffs_pnm11[k_stride_n * alpha + nx];
            const REAL c_pnm2 = k_coeffs_pnm2[k_stride_n * alpha + nx];
            pn = c_pnm10 * pnm1 * z + c_pnm11 * pnm1 + c_pnm2 * pnm2;
            pnm2 = pnm1;
            pnm1 = pn;
            etmp1 = inner_b1_pow * b1 * pn;
          }
        }

        output[mode] = etmp1;
        mode++;
      }
      inner_b1_pow *= b0;
    }
  }
}

/**
 * Evaluate the eModified_PyrC basis functions up to a given order placing the
 * evaluations in an output array. For reference see the function
 * eval_modPyrC_ijk. Jacobi polynomials are evaluated using recusion relations:
 *
 * For brevity the (alpha, beta) superscripts are dropped. i.e. P_n(z) =
 * P_n^{alpha, beta}(z). P_n(z) = C_{n-1}^0 P_{n-1}(z) * z + C_{n-1}^1
 * P_{n-1}(z) + C_{n-2} * P_{n-2}(z) P_0(z) = 1 P_1(z) = 2 + 2 * (z - 1)
 *
 * @param[in] nummodes Number of modes to compute, i.e. p modes evaluates at
 * most an order p-1 polynomial.
 * @param[in] z Evaluation point to evaluate basis at.
 * @param[in] k_stride_n Stride between sets of coefficients for different
 * alpha values in the coefficient arrays.
 * @param[in] k_coeffs_pnm10 Coefficients for C_{n-1}^0 for different alpha
 * values stored row wise for each alpha.
 * @param[in] k_coeffs_pnm11 Coefficients for C_{n-1}^1 for different alpha
 * values stored row wise for each alpha.
 * @param[in] k_coeffs_pnm2 Coefficients for C_{n-2} for different alpha values
 * stored row wise for each alpha.
 * @param[in, out] output entry i contains the i-th eModified_PyrC basis
 * function evaluated at z.
 */
inline void mod_PyrC(const int nummodes, const REAL z, const int k_stride_n,
                     const REAL *const k_coeffs_pnm10,
                     const REAL *const k_coeffs_pnm11,
                     const REAL *const k_coeffs_pnm2, REAL *output) {

  REAL *output_base = output + nummodes;

  // The p==0 case if an eModified_B basis over indices q,r
  mod_B(nummodes, z, k_stride_n, k_coeffs_pnm10, k_coeffs_pnm11, k_coeffs_pnm2,
        output);
  int mode = nummodes * (nummodes + 1) / 2;
  output += mode;

  // The p==1, q==0 case is eModified_B over 1,r
  for (int cx = 0; cx < (nummodes - 1); cx++) {
    output[cx] = output_base[cx];
  }
  mode += nummodes - 1;
  output += nummodes - 1;

  // The p==1, q!=0 case is eModified_B over q,r
  const int l_tmp = ((nummodes - 1) * (nummodes) / 2);
  for (int cx = 0; cx < l_tmp; cx++) {
    output[cx] = output_base[cx];
  }
  output_base += nummodes - 1;
  output += l_tmp;
  mode += l_tmp;

  REAL one_m_z = 0.5 * (1.0 - z);
  REAL one_p_z = 0.5 * (1.0 + z);
  REAL r0_pow = 1.0;

  for (int p = 2; p < nummodes; ++p) {
    r0_pow *= one_m_z;
    REAL r0_pow_inner = r0_pow;

    // q < 2 case is eModified_B over p,r
    const int l_tmp = (nummodes - p);
    for (int cx = 0; cx < l_tmp; cx++) {
      output[cx] = output_base[cx];
      output[l_tmp + cx] = output_base[cx];
    }
    output += 2 * l_tmp;
    output_base += l_tmp;

    for (int q = 2; q < nummodes; ++q) {
      r0_pow_inner *= one_m_z;

      // p > 1, q > 1, r == 0 term (0.5 * (1 - z) ** (p + q - 2))
      output[0] = r0_pow_inner;
      output++;

      REAL pn, pnm1, pnm2;
      const int alpha = 2 * p + 2 * q - 3;
      int maxpq = max(p, q);

      /*
       * The remaining terms are of the form
       *    std::pow(0.5 * (1.0 - z), p + q - 2) * (0.5 * (1.0 + z)) *
       *      jacobi(r - 1, z, 2 * p + 2 * q - 3, 1)
       *
       *  where we compute the Jacobi polynomials using the recursion
       *  relations.
       */
      for (int r = 1; r < nummodes - maxpq; ++r) {
        // this is the std::pow(0.5 * (1.0 - z), p + q - 2) * (0.5 * (1.0 + z))
        const REAL b0b1_coefficient = r0_pow_inner * one_p_z;
        // compute the P_{r-1}^{2p+2q-3, 1} terms using recursion
        REAL etmp;
        if (r == 1) {
          etmp = b0b1_coefficient;
          pnm2 = 1.0;
        } else if (r == 2) {
          pnm1 = 0.5 * (2.0 * (alpha + 1) + (alpha + 3) * (z - 1.0));
          etmp = pnm1 * b0b1_coefficient;
        } else {
          const int nx = r - 1;
          const REAL c_pnm10 = k_coeffs_pnm10[k_stride_n * alpha + nx];
          const REAL c_pnm11 = k_coeffs_pnm11[k_stride_n * alpha + nx];
          const REAL c_pnm2 = k_coeffs_pnm2[k_stride_n * alpha + nx];
          pn = c_pnm10 * pnm1 * z + c_pnm11 * pnm1 + c_pnm2 * pnm2;
          pnm2 = pnm1;
          pnm1 = pn;
          etmp = pn * b0b1_coefficient;
        }
        output[0] = etmp;
        output++;
      }
    }
  }
}

/**
 *  Abstract base class for 1D basis evaluation functions which are based on
 *  Jacobi polynomials.
 */
template <typename SPECIALISATION> struct Basis1D {

  /**
   * Method called in sycl kernel to evaluate a set of basis functions at a
   * point. Jacobi polynomials are evaluated using recusion relations:
   *
   * For brevity the (alpha, beta) superscripts are dropped. i.e. P_n(z) =
   * P_n^{alpha, beta}(z). P_n(z) = C_{n-1}^0 P_{n-1}(z) * z + C_{n-1}^1
   * P_{n-1}(z) + C_{n-2} * P_{n-2}(z) P_0(z) = 1 P_1(z) = 2 + 2 * (z - 1)
   *
   * @param[in] nummodes Number of modes to compute, i.e. p modes evaluates at
   * most an order p-1 polynomial.
   * @param[in] z Evaluation point to evaluate basis at.
   * @param[in] k_stride_n Stride between sets of coefficients for different
   * alpha values in the coefficient arrays.
   * @param[in] k_coeffs_pnm10 Coefficients for C_{n-1}^0 for different alpha
   * values stored row wise for each alpha.
   * @param[in] k_coeffs_pnm11 Coefficients for C_{n-1}^1 for different alpha
   * values stored row wise for each alpha.
   * @param[in] k_coeffs_pnm2 Coefficients for C_{n-2} for different alpha
   * values stored row wise for each alpha.
   * @param[in, out] Output array for evaluations.
   */
  static inline void evaluate(const int nummodes, const REAL z,
                              const int k_stride_n, const REAL *k_coeffs_pnm10,
                              const REAL *k_coeffs_pnm11,
                              const REAL *k_coeffs_pnm2, REAL *output) {
    SPECIALISATION::evaluate(nummodes, z, k_stride_n, k_coeffs_pnm10,
                             k_coeffs_pnm11, k_coeffs_pnm2, output);
  }
};

/**
 *  Specialisation of Basis1D that calls the mod_A function that implements
 *  eModified_A.
 */
struct ModifiedA : Basis1D<ModifiedA> {
  static inline void evaluate(const int nummodes, const REAL z,
                              const int k_stride_n, const REAL *k_coeffs_pnm10,
                              const REAL *k_coeffs_pnm11,
                              const REAL *k_coeffs_pnm2, REAL *output) {
    mod_A(nummodes, z, k_stride_n, k_coeffs_pnm10, k_coeffs_pnm11,
          k_coeffs_pnm2, output);
  }
};

/**
 *  Specialisation of Basis1D that calls the mod_B function that implements
 *  eModified_B.
 */
struct ModifiedB : Basis1D<ModifiedB> {
  static inline void evaluate(const int nummodes, const REAL z,
                              const int k_stride_n, const REAL *k_coeffs_pnm10,
                              const REAL *k_coeffs_pnm11,
                              const REAL *k_coeffs_pnm2, REAL *output) {
    mod_B(nummodes, z, k_stride_n, k_coeffs_pnm10, k_coeffs_pnm11,
          k_coeffs_pnm2, output);
  }
};

/**
 *  Specialisation of Basis1D that calls the mod_B function that implements
 *  eModified_C.
 */
struct ModifiedC : Basis1D<ModifiedC> {
  static inline void evaluate(const int nummodes, const REAL z,
                              const int k_stride_n, const REAL *k_coeffs_pnm10,
                              const REAL *k_coeffs_pnm11,
                              const REAL *k_coeffs_pnm2, REAL *output) {
    mod_C(nummodes, z, k_stride_n, k_coeffs_pnm10, k_coeffs_pnm11,
          k_coeffs_pnm2, output);
  }
};

/**
 *  Specialisation of Basis1D that calls the mod_B function that implements
 *  eModifiedPyr_C.
 */
struct ModifiedPyrC : Basis1D<ModifiedPyrC> {
  static inline void evaluate(const int nummodes, const REAL z,
                              const int k_stride_n, const REAL *k_coeffs_pnm10,
                              const REAL *k_coeffs_pnm11,
                              const REAL *k_coeffs_pnm2, REAL *output) {
    mod_PyrC(nummodes, z, k_stride_n, k_coeffs_pnm10, k_coeffs_pnm11,
             k_coeffs_pnm2, output);
  }
};

} // namespace BasisJacobi

} // namespace NESO

#endif
