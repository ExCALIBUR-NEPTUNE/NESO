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
#include "function_coupling_base.hpp"
#include "geometry_transport_3d.hpp"
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

#include <CL/sycl.hpp>

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
                  const REAL *k_coeffs_pnm10, const REAL *k_coeffs_pnm11,
                  const REAL *k_coeffs_pnm2, REAL *output) {
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
                  const REAL *k_coeffs_pnm10, const REAL *k_coeffs_pnm11,
                  const REAL *k_coeffs_pnm2, REAL *output) {
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
 * @param[in, out] output entry i contains the i-th eModified_A basis function
 * evaluated at z.
 */
inline void mod_C(const int nummodes, const REAL z, const int k_stride_n,
                  const REAL *k_coeffs_pnm10, const REAL *k_coeffs_pnm11,
                  const REAL *k_coeffs_pnm2, REAL *output) {
  /*
  int mode = 0;
  for (int p = 0; p < P; p++) {
    for (int q = 0; q < (P - p); q++) {
      for (int r = 0; r < (P - p - q); r++) {
        const double contrib_2 = eval_modC_ijk(p, q, r, eta2);

        mode++;
      }
    }
  }
  */
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

} // namespace BasisJacobi

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
  JacobiCoeffModBasis(const int max_n, const int max_alpha)
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
  inline double host_evaluate(const int n, const int alpha, const double z) {

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
};

/**
 * Base class for derived classes that evaluate eModified_A and eModified_B
 * Nektar++ basis.
 */
template <typename T> class BasisEvaluateBase : GeomToExpansionBuilder {
protected:
  std::shared_ptr<T> field;
  ParticleMeshInterfaceSharedPtr mesh;
  CellIDTranslationSharedPtr cell_id_translation;
  SYCLTargetSharedPtr sycl_target;

  BufferDeviceHost<int> dh_nummodes;

  std::map<ShapeType, int> map_shape_to_count;
  std::map<ShapeType, std::vector<int>> map_shape_to_cells;
  std::map<ShapeType, std::unique_ptr<BufferDeviceHost<int>>>
      map_shape_to_dh_cells;

  BufferDeviceHost<int> dh_coeffs_offsets;
  BufferDeviceHost<REAL> dh_global_coeffs;
  BufferDeviceHost<REAL> dh_coeffs_pnm10;
  BufferDeviceHost<REAL> dh_coeffs_pnm11;
  BufferDeviceHost<REAL> dh_coeffs_pnm2;
  int stride_n;
  std::map<ShapeType, std::array<int, 3>> map_total_nummodes;

public:
  /// Disable (implicit) copies.
  BasisEvaluateBase(const BasisEvaluateBase &st) = delete;
  /// Disable (implicit) copies.
  BasisEvaluateBase &operator=(BasisEvaluateBase const &a) = delete;

  /**
   * Create new instance. Expected to be called by a derived class - not a user.
   *
   * @param field Example field this class will be used to evaluate basis
   * functions for.
   * @param mesh Interface between NESO-Particles and Nektar++ meshes.
   * @param cell_id_translation Map between NESO-Particles cells and Nektar++
   * cells.
   */
  BasisEvaluateBase(std::shared_ptr<T> field,
                    ParticleMeshInterfaceSharedPtr mesh,
                    CellIDTranslationSharedPtr cell_id_translation)
      : field(field), mesh(mesh), cell_id_translation(cell_id_translation),
        sycl_target(cell_id_translation->sycl_target),
        dh_nummodes(sycl_target, 1), dh_global_coeffs(sycl_target, 1),
        dh_coeffs_offsets(sycl_target, 1), dh_coeffs_pnm10(sycl_target, 1),
        dh_coeffs_pnm11(sycl_target, 1), dh_coeffs_pnm2(sycl_target, 1) {

    // build the map from geometry ids to expansion ids
    std::map<int, int> geom_to_exp;
    build_geom_to_expansion_map(this->field, geom_to_exp);

    auto geom_type_lookup =
        this->cell_id_translation->dh_map_to_geom_type.h_buffer.ptr;

    const int index_tri_geom =
        shape_type_to_int(LibUtilities::ShapeType::eTriangle);
    const int index_quad_geom =
        shape_type_to_int(LibUtilities::ShapeType::eQuadrilateral);

    const int neso_cell_count = mesh->get_cell_count();

    this->dh_nummodes.realloc_no_copy(neso_cell_count);
    this->dh_coeffs_offsets.realloc_no_copy(neso_cell_count);

    int max_n = 1;
    int max_alpha = 1;

    std::array<ShapeType, 6> shapes = {eTriangle, eQuadrilateral, eHexahedron,
                                       ePrism,    ePyramid,       eTetrahedron};
    for (auto shape : shapes) {
      this->map_shape_to_count[shape] = 0;
      this->map_shape_to_count[shape] = 0;
      for (int dimx = 0; dimx < 3; dimx++) {
        this->map_total_nummodes[shape][dimx] = 0;
      }
    }

    for (int neso_cellx = 0; neso_cellx < neso_cell_count; neso_cellx++) {

      const int nektar_geom_id =
          this->cell_id_translation->map_to_nektar[neso_cellx];
      const int expansion_id = geom_to_exp[nektar_geom_id];
      // get the nektar expansion
      auto expansion = this->field->GetExp(expansion_id);
      auto basis = expansion->GetBase();
      const int expansion_ndim = basis.size();

      // build the map from shape types to neso cells
      auto shape_type = expansion->DetShapeType();
      this->map_shape_to_cells[shape_type].push_back(neso_cellx);

      for (int dimx = 0; dimx < expansion_ndim; dimx++) {
        const int basis_nummodes = basis[dimx]->GetNumModes();
        const int basis_total_nummodes = basis[dimx]->GetTotNumModes();
        max_n = std::max(max_n, basis_nummodes - 1);
        if (dimx == 0) {
          this->dh_nummodes.h_buffer.ptr[neso_cellx] = basis_nummodes;
        } else {
          NESOASSERT(this->dh_nummodes.h_buffer.ptr[neso_cellx] ==
                         basis_nummodes,
                     "Differing numbers of modes in coordinate directions.");
        }
        this->map_total_nummodes.at(shape_type).at(dimx) =
            std::max(this->map_total_nummodes.at(shape_type).at(dimx),
                     basis_total_nummodes);
      }

      // determine the maximum Jacobi order and alpha value required to
      // evaluate the basis functions for this expansion
      int alpha_tmp = 0;
      int n_tmp = 0;
      BasisReference::get_total_num_modes(
          shape_type, this->dh_nummodes.h_buffer.ptr[neso_cellx], &alpha_tmp,
          &n_tmp);
      max_alpha = std::max(max_alpha, alpha_tmp);
      max_n = std::max(max_n, n_tmp);

      // record offsets and number of coefficients
      this->dh_coeffs_offsets.h_buffer.ptr[neso_cellx] =
          this->field->GetCoeff_Offset(expansion_id);
    }

    int expansion_count = 0;
    // create the maps from shape types to NESO::Particles cells which
    // correpond to the shape type.

    for (auto &item : this->map_shape_to_cells) {
      expansion_count += item.second.size();
      auto shape_type = item.first;
      auto &cells = item.second;
      const int num_cells = cells.size();
      // allocate and build the map.
      this->map_shape_to_dh_cells[shape_type] =
          std::make_unique<BufferDeviceHost<int>>(this->sycl_target, num_cells);
      for (int cellx = 0; cellx < num_cells; cellx++) {
        const int cell = cells[cellx];
        this->map_shape_to_dh_cells[shape_type]->h_buffer.ptr[cellx] = cell;
      }
      this->map_shape_to_dh_cells[shape_type]->host_to_device();
      this->map_shape_to_count[shape_type] = num_cells;
    }

    NESOASSERT(expansion_count == neso_cell_count,
               "Missmatch in number of cells found and total number of cells.");

    JacobiCoeffModBasis jacobi_coeff(max_n, max_alpha);

    const int num_coeffs = jacobi_coeff.coeffs_pnm10.size();
    this->dh_coeffs_pnm10.realloc_no_copy(num_coeffs);
    this->dh_coeffs_pnm11.realloc_no_copy(num_coeffs);
    this->dh_coeffs_pnm2.realloc_no_copy(num_coeffs);
    for (int cx = 0; cx < num_coeffs; cx++) {
      this->dh_coeffs_pnm10.h_buffer.ptr[cx] = jacobi_coeff.coeffs_pnm10[cx];
      this->dh_coeffs_pnm11.h_buffer.ptr[cx] = jacobi_coeff.coeffs_pnm11[cx];
      this->dh_coeffs_pnm2.h_buffer.ptr[cx] = jacobi_coeff.coeffs_pnm2[cx];
    }
    this->stride_n = jacobi_coeff.stride_n;

    this->dh_coeffs_offsets.host_to_device();
    this->dh_nummodes.host_to_device();
    this->dh_coeffs_pnm10.host_to_device();
    this->dh_coeffs_pnm11.host_to_device();
    this->dh_coeffs_pnm2.host_to_device();
  }
};

} // namespace NESO

#endif
