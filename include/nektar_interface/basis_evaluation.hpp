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

namespace Templated {

template <size_t n> inline size_t pochhammer(const size_t m) {
  return (m + (n - 1)) * pochhammer<n - 1>(m);
}
template <> inline size_t pochhammer<0>([[maybe_unused]] const size_t m) {
  return 1;
}

template <size_t p, size_t alpha, size_t beta> struct jacobis;
template <size_t pc, size_t alphac, size_t betac, typename T>
inline const REAL &getj(T *t) {
  return static_cast<jacobis<pc, alphac, betac> *>(t)->value;
}
template <size_t pc, size_t alphac, size_t betac, typename T>
inline const REAL getj(T t) {
  return static_cast<jacobis<pc, alphac, betac>>(t).value;
}

template <size_t alpha, size_t beta> struct jacobis<0, alpha, beta> {
  const REAL value = 1.0;
  jacobis(const REAL z){};
};
template <size_t alpha, size_t beta>
struct jacobis<1, alpha, beta> : public jacobis<0, alpha, beta> {
  const REAL value;
  jacobis(const REAL z)
      : jacobis<0, alpha, beta>(z),
        value(0.5 * (2.0 * (alpha + 1.0) + (alpha + beta + 2.0) * (z - 1.0))){};
};
template <size_t alpha, size_t beta>
struct jacobis<2, alpha, beta> : public jacobis<1, alpha, beta> {
  const REAL value;
  jacobis(const REAL z)
      : jacobis<1, alpha, beta>(z),
        value(0.125 *
              ((4.0 * (alpha + 1.0) * (alpha + 2.0)) +
               (4.0 * (alpha + beta + 3.0) * (alpha + 2.0)) * (z - (1.0)) +
               ((alpha + beta + 3.0)) * ((alpha + beta + 4.0)) * (z - (1.0)) *
                   (z - (1.0)))){};
};

template <size_t p, size_t alpha, size_t beta>
struct jacobis : public jacobis<p - 1, alpha, beta> {
  const size_t n = p - 1;
  const REAL coeff_pnp1 =
      (2.0 * (n + 1.0) * (n + alpha + beta + 1.0) * (2.0 * n + alpha + beta));
  const REAL coeff_pn =
      (2.0 * n + alpha + beta + 1.0) * (alpha * alpha - beta * beta);
  const REAL coeff_pnm1 =
      (-2.0 * (n + alpha) * (n + beta) * (2.0 * n + alpha + beta + 2.0));

  const REAL value = 1.0;
  jacobis(const REAL z)
      : jacobis<p - 1, alpha, beta>(z),
        value((1.0 / coeff_pnp1) * (coeff_pn * getj<p - 1, alpha, beta>(this)) +
              (coeff_pnm1 + pochhammer<2 * (p - 1) + alpha + beta>(3) * z) *
                  getj<p - 2, alpha, beta>(this)){};
};

template <size_t p, size_t alpha, size_t beta> constexpr auto jacobi() {
  if constexpr (p == 0) {
    return [](const auto z) { return 1.0; };
  } else if constexpr (p == 1) {
    return [](const auto z) {
      return 0.5 * (2.0 * (alpha + 1.0) + (alpha + beta + 2.0) * (z - 1.0));
    };
  } else if constexpr (p == 2) {
    return [](const auto z) {
      return 0.125 *
             ((4.0 * (alpha + 1.0) * (alpha + 2.0)) +
              (4.0 * (alpha + beta + 3.0) * (alpha + 2.0)) * (z - (1.0)) +
              ((alpha + beta + 3.0)) * ((alpha + beta + 4.0)) * (z - (1.0)) *
                  (z - (1.0)));
    };
  } else {
    return [](const auto z) {
      const auto n = p - 1;
      auto pn = jacobi<n, alpha, beta>();
      auto pnm1 = jacobi<n - 1, alpha, beta>();
      const auto coeff_pnp1 = (2.0 * (n + 1.0) * (n + alpha + beta + 1.0) *
                               (2.0 * n + alpha + beta));
      const auto coeff_pn =
          (2.0 * n + alpha + beta + 1.0) * (alpha * alpha - beta * beta) +
          pochhammer<2 * n + alpha + beta>(3) * z;
      const auto coeff_pnm1 =
          (-2.0 * (n + alpha) * (n + beta) * (2.0 * n + alpha + beta + 2.0));
      const REAL v =
          (1.0 / coeff_pnp1) * (coeff_pn * pn(z) + coeff_pnm1 * pnm1(z));
      return v;
    };
  }
}

template <size_t nummodes, size_t px, size_t JPX>
constexpr auto eModifiedA(const jacobis<JPX, 1, 1> j0) {
  if constexpr (px == 0) {
    return [](const auto z) { return 0.5 * (1.0 - z); };
  } else if constexpr (px == 1) {
    return [](const auto z) { return 0.5 * (1.0 + z); };
  } else {
    return [=](const auto z) {
      return 0.5 * (1.0 - z) * 0.5 * (1.0 + z) * getj<px - 2, 1, 1>(j0);
    };
  }
}

namespace Quadrilateral {

template <size_t PX, size_t QX, size_t px, size_t qx, size_t mode, size_t JPX,
          size_t JQX>
inline REAL evaluate_inner(const jacobis<JPX, 1, 1> j0,
                           const jacobis<JQX, 1, 1> j1, const REAL *dofs,
                           const REAL eta0, const REAL eta1) {
  return dofs[mode] * eModifiedA<PX, px>(j0)(eta0) *
         eModifiedA<QX, qx>(j1)(eta1);
}

template <size_t PX, size_t QX, size_t px, size_t qx, size_t mode, size_t JPX,
          size_t JQX>
inline REAL inner(const jacobis<JPX, 1, 1> j0, const jacobis<JQX, 1, 1> j1,
                  const REAL *dofs, const REAL eta0, const REAL eta1) {
  REAL v = evaluate_inner<PX, QX, px, qx, mode>(j0, j1, dofs, eta0, eta1);
  if constexpr (px < (PX - 1)) {
    v += inner<PX, QX, px + 1, qx, mode + 1>(j0, j1, dofs, eta0, eta1);
  } else if constexpr (px == (PX - 1) && qx < (QX - 1)) {
    v += inner<PX, QX, 0, qx + 1, mode + 1>(j0, j1, dofs, eta0, eta1);
  }
  return v;
}

template <size_t PX, size_t QX>
inline REAL evaluate(const REAL *dofs, const REAL eta0, const REAL eta1) {
  jacobis<PX, 1, 1> j0(eta0);
  jacobis<PX, 1, 1> j1(eta1);
  return inner<PX, QX, 0, 0, 0>(j0, j1, dofs, eta0, eta1);
}

} // namespace Quadrilateral

/*
template <size_t p, size_t alpha, size_t beta> constexpr auto jacobi() {
  if constexpr (p == 0) {
    return [](const auto z) { return 1.0; };
  } else if constexpr (p == 1) {
    return [](const auto z) {
      return 0.5 * (2.0 * (alpha + 1.0) + (alpha + beta + 2.0) * (z - 1.0));
    };
  } else if constexpr (p == 2) {
    return [](const auto z) {
      return 0.125 *
             ((4.0 * (alpha + 1.0) * (alpha + 2.0)) +
              (4.0 * (alpha + beta + 3.0) * (alpha + 2.0)) * (z - (1.0)) +
              ((alpha + beta + 3.0)) * ((alpha + beta + 4.0)) * (z - (1.0)) *
                  (z - (1.0)));
    };
  } else {
    return [](const auto z) {
      const auto n = p - 1;
      auto pn = jacobi<n, alpha, beta>();
      auto pnm1 = jacobi<n - 1, alpha, beta>();
      const auto coeff_pnp1 = (2.0 * (n + 1.0) * (n + alpha + beta + 1.0) *
                               (2.0 * n + alpha + beta));
      const auto coeff_pn =
          (2.0 * n + alpha + beta + 1.0) * (alpha * alpha - beta * beta) +
          pochhammer<2 * n + alpha + beta>(3) * z;
      const auto coeff_pnm1 =
          (-2.0 * (n + alpha) * (n + beta) * (2.0 * n + alpha + beta + 2.0));
      const REAL v =
          (1.0 / coeff_pnp1) * (coeff_pn * pn(z) + coeff_pnm1 * pnm1(z));
      return v;
    };
  }
}

template <size_t nummodes, size_t px> constexpr auto eModifiedA() {
  if constexpr (px == 0) {
    return [](const auto z) { return 0.5 * (1.0 - z); };
  } else if constexpr (px == 1) {
    return [](const auto z) { return 0.5 * (1.0 + z); };
  } else {
    return [](const auto z) {
      return 0.5 * (1.0 - z) * 0.5 * (1.0 + z) * jacobi<px - 2, 1, 1>()(z);
    };
  }
}

namespace Quadrilateral {

template <size_t PX, size_t QX, size_t px, size_t qx, size_t mode>
inline REAL evaluate_inner(const REAL *dofs, const REAL eta0, const REAL eta1) {
  return dofs[mode] * eModifiedA<PX, px>()(eta0) * eModifiedA<QX, qx>()(eta1);
}

template <size_t PX, size_t QX, size_t px, size_t qx, size_t mode>
inline REAL inner(const REAL *dofs, const REAL eta0, const REAL eta1) {
  REAL v = evaluate_inner<PX, QX, px, qx, mode>(dofs, eta0, eta1);
  if constexpr (px < (PX - 1)) {
    v += inner<PX, QX, px + 1, qx, mode + 1>(dofs, eta0, eta1);
  } else if constexpr (px == (PX - 1) && qx < (QX - 1)) {
    v += inner<PX, QX, 0, qx + 1, mode + 1>(dofs, eta0, eta1);
  }
  return v;
}

template <size_t PX, size_t QX>
inline REAL evaluate(const REAL *dofs, const REAL eta0, const REAL eta1) {
  return inner<PX, QX, 0, 0, 0>(dofs, eta0, eta1);
}

} // namespace Quadrilateral
  //
*/

} // namespace Templated

} // namespace BasisJacobi

} // namespace NESO

#endif
