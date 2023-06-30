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
inline void mod_B(const int nummodes, const double z, const int k_stride_n,
                  const double *k_coeffs_pnm10, const double *k_coeffs_pnm11,
                  const double *k_coeffs_pnm2, double *output) {
  int modey = 0;
  const double b0 = 0.5 * (1.0 - z);
  const double b1 = 0.5 * (1.0 + z);
  double b1_pow = 1.0 / b0;
  for (int px = 0; px < nummodes; px++) {
    double pn, pnm1, pnm2;
    b1_pow *= b0;
    const int alpha = 2 * px - 1;
    for (int qx = 0; qx < (nummodes - px); qx++) {
      double etmp1;
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
          const double c_pnm10 = k_coeffs_pnm10[k_stride_n * 1 + nx];
          const double c_pnm11 = k_coeffs_pnm11[k_stride_n * 1 + nx];
          const double c_pnm2 = k_coeffs_pnm2[k_stride_n * 1 + nx];
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
          const double c_pnm10 = k_coeffs_pnm10[k_stride_n * alpha + nx];
          const double c_pnm11 = k_coeffs_pnm11[k_stride_n * alpha + nx];
          const double c_pnm2 = k_coeffs_pnm2[k_stride_n * alpha + nx];
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
inline void mod_A(const int nummodes, const double z, const int k_stride_n,
                  const double *k_coeffs_pnm10, const double *k_coeffs_pnm11,
                  const double *k_coeffs_pnm2, double *output) {
  const double b0 = 0.5 * (1.0 - z);
  const double b1 = 0.5 * (1.0 + z);
  output[0] = b0;
  output[1] = b1;
  double pn;
  double pnm2 = 1.0;
  double pnm1 = 2.0 + 2.0 * (z - 1.0);
  if (nummodes > 2) {
    output[2] = b0 * b1;
  }
  if (nummodes > 3) {
    output[3] = b0 * b1 * pnm1;
  }
  for (int modex = 4; modex < nummodes; modex++) {
    const int nx = modex - 2;
    const double c_pnm10 = k_coeffs_pnm10[k_stride_n * 1 + nx];
    const double c_pnm11 = k_coeffs_pnm11[k_stride_n * 1 + nx];
    const double c_pnm2 = k_coeffs_pnm2[k_stride_n * 1 + nx];
    pn = c_pnm10 * pnm1 * z + c_pnm11 * pnm1 + c_pnm2 * pnm2;
    pnm2 = pnm1;
    pnm1 = pn;
    output[modex] = b0 * b1 * pn;
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
  static inline void evaluate(const int nummodes, const double z,
                              const int k_stride_n,
                              const double *k_coeffs_pnm10,
                              const double *k_coeffs_pnm11,
                              const double *k_coeffs_pnm2, double *output) {
    SPECIALISATION::evaluate(nummodes, z, k_stride_n, k_coeffs_pnm10,
                             k_coeffs_pnm11, k_coeffs_pnm2, output);
  }
};

/**
 *  Specialisation of Basis1D that calls the mod_A function that implements
 *  eModified_A.
 */
struct ModifiedA : Basis1D<ModifiedA> {
  static inline void evaluate(const int nummodes, const double z,
                              const int k_stride_n,
                              const double *k_coeffs_pnm10,
                              const double *k_coeffs_pnm11,
                              const double *k_coeffs_pnm2, double *output) {
    mod_A(nummodes, z, k_stride_n, k_coeffs_pnm10, k_coeffs_pnm11,
          k_coeffs_pnm2, output);
  }
};

/**
 *  Specialisation of Basis1D that calls the mod_B function that implements
 *  eModified_B.
 */
struct ModifiedB : Basis1D<ModifiedB> {
  static inline void evaluate(const int nummodes, const double z,
                              const int k_stride_n,
                              const double *k_coeffs_pnm10,
                              const double *k_coeffs_pnm11,
                              const double *k_coeffs_pnm2, double *output) {
    mod_B(nummodes, z, k_stride_n, k_coeffs_pnm10, k_coeffs_pnm11,
          k_coeffs_pnm2, output);
  }
};

/**
 *  Abstract base class for kernels which require basis function evaluations
 *  indexed by p and q, e.g. eModified_A/B.
 */
template <typename SPECIALISATION> struct LoopingKernelBase {

  /**
   *  Call a kernel function with indices p and q along with the linearised
   *  mode corresponding to the indices p and q.
   *
   *  @param p Basis function p index.
   *  @param p Basis function q index.
   *  @param mode Linearised mode corresponding to p and q.
   */
  inline void kernel(const int px, const int qx, const int mode) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.kernel(px, qx, mode);
  }
};

/**
 *  Abstract base class for looping over all modes in dimension0 with all modes
 *  in dimension1.
 */
template <typename SPECIALISATION> struct Indexing2D {
  template <typename LOOP_FUNC>

  /**
   *  Double loop over modes:
   *  For px in [0, nummodes0-1]:
   *    For qx in [0, nummodes1-1]:
   *      <apply kernel>
   *
   *  @param nummodes0 Number of modes in first dimension.
   *  @param nummodes1 Number of modes in second dimension.
   *  @param kernel Kernel to be called for each iteration of the double loop.
   */
  static inline void loop(const int nummodes0, const int nummodes1,
                          LoopingKernelBase<LOOP_FUNC> &kernel) {
    SPECIALISATION::loop(nummodes0, nummodes1, kernel);
  }
};

/**
 *  Loop over modes in dimension 0 and 1 for Quadrilateral elements where there
 *  is an eModified_A basis in dimensions 0 and 1.
 */
struct IndexingQuad : Indexing2D<IndexingQuad> {
  template <typename LOOP_FUNC>
  static inline void loop(const int nummodes0, const int nummodes1,
                          LoopingKernelBase<LOOP_FUNC> &kernel) {
    for (int qx = 0; qx < nummodes1; qx++) {
      for (int px = 0; px < nummodes0; px++) {
        const int mode = qx * nummodes0 + px;
        kernel.kernel(px, qx, mode);
      }
    }
  }
};

/**
 *  Loop over modes in dimension 0 and 1 for Triangle elements where there
 *  is an eModified_A basis and an eModified_B basis.
 */
struct IndexingTriangle : Indexing2D<IndexingTriangle> {
  template <typename LOOP_FUNC>
  static inline void loop(const int nummodes0, const int nummodes1,
                          LoopingKernelBase<LOOP_FUNC> &kernel) {
    int modey = 0;
    for (int px = 0; px < nummodes0; px++) {
      for (int qx = 0; qx < nummodes1 - px; qx++) {
        const int mode = modey++;
        kernel.kernel(px, qx, mode);
      }
    }
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
  std::vector<double> coeffs_pnm10;
  std::vector<double> coeffs_pnm11;
  std::vector<double> coeffs_pnm2;

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
 *  Reference implementation to map xi to eta for TriGeoms.
 *
 *  @param xi XI value.
 *  @param eta Output pointer for eta.
 */
inline void to_collapsed_triangle(Array<OneD, NekDouble> &xi, double *eta) {
  const REAL xi0 = xi[0];
  const REAL xi1 = xi[1];

  const NekDouble d1_original = 1.0 - xi1;
  const bool mask_small_cond = (fabs(d1_original) < NekConstants::kNekZeroTol);
  NekDouble d1 = d1_original;

  d1 =
      (mask_small_cond && (d1 >= 0.0))
          ? NekConstants::kNekZeroTol
          : ((mask_small_cond && (d1 < 0.0)) ? -NekConstants::kNekZeroTol : d1);
  eta[0] = 2. * (1. + xi0) / d1 - 1.0;
  eta[1] = xi1;
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

  std::vector<int> cells_quads;
  std::vector<int> cells_tris;

  BufferDeviceHost<int> dh_nummodes0;
  BufferDeviceHost<int> dh_nummodes1;
  BufferDeviceHost<int> dh_cells_quads;
  BufferDeviceHost<int> dh_cells_tris;

  BufferDeviceHost<int> dh_coeffs_offsets;
  BufferDeviceHost<double> dh_global_coeffs;

  BufferDeviceHost<double> dh_coeffs_pnm10;
  BufferDeviceHost<double> dh_coeffs_pnm11;
  BufferDeviceHost<double> dh_coeffs_pnm2;
  int stride_n;
  int max_nummodes_0;
  int max_nummodes_1;

  int max_total_nummodes0;
  int max_total_nummodes1;

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
        dh_nummodes0(sycl_target, 1), dh_nummodes1(sycl_target, 1),
        dh_cells_quads(sycl_target, 1), dh_cells_tris(sycl_target, 1),
        dh_global_coeffs(sycl_target, 1), dh_coeffs_offsets(sycl_target, 1),
        dh_coeffs_pnm10(sycl_target, 1), dh_coeffs_pnm11(sycl_target, 1),
        dh_coeffs_pnm2(sycl_target, 1) {

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

    this->dh_nummodes0.realloc_no_copy(neso_cell_count);
    this->dh_nummodes1.realloc_no_copy(neso_cell_count);
    this->dh_coeffs_offsets.realloc_no_copy(neso_cell_count);

    int max_n = 1;
    int max_alpha = 1;
    this->max_nummodes_0 = 0;
    this->max_nummodes_1 = 0;
    this->max_total_nummodes0 = 0;
    this->max_total_nummodes1 = 0;

    for (int neso_cellx = 0; neso_cellx < neso_cell_count; neso_cellx++) {

      const int nektar_geom_id =
          this->cell_id_translation->map_to_nektar[neso_cellx];
      const int expansion_id = geom_to_exp[nektar_geom_id];
      // get the nektar expansion
      auto expansion = this->field->GetExp(expansion_id);

      auto basis0 = expansion->GetBasis(0);
      auto basis1 = expansion->GetBasis(1);
      const int nummodes0 = basis0->GetNumModes();
      const int nummodes1 = basis1->GetNumModes();

      max_total_nummodes0 =
          std::max(max_total_nummodes0, basis0->GetTotNumModes());
      max_total_nummodes1 =
          std::max(max_total_nummodes1, basis1->GetTotNumModes());

      this->dh_nummodes0.h_buffer.ptr[neso_cellx] = nummodes0;
      this->dh_nummodes1.h_buffer.ptr[neso_cellx] = nummodes1;

      max_n = std::max(max_n, nummodes0 - 1);
      max_n = std::max(max_n, nummodes1 - 1);
      max_alpha = std::max(max_alpha, (nummodes0 - 1) * 2 - 1);
      this->max_nummodes_0 = std::max(this->max_nummodes_0, nummodes0);
      this->max_nummodes_1 = std::max(this->max_nummodes_1, nummodes1);

      // is this a tri expansion?
      if (geom_type_lookup[neso_cellx] == index_tri_geom) {
        this->cells_tris.push_back(neso_cellx);
      }
      // is this a quad expansion?
      if (geom_type_lookup[neso_cellx] == index_quad_geom) {
        this->cells_quads.push_back(neso_cellx);
      }

      // record offsets and number of coefficients
      this->dh_coeffs_offsets.h_buffer.ptr[neso_cellx] =
          this->field->GetCoeff_Offset(expansion_id);
    }

    NESOASSERT((this->cells_tris.size() + this->cells_quads.size()) ==
                   neso_cell_count,
               "Missmatch in number of quad cells triangle cells and total "
               "number of cells.");

    JacobiCoeffModBasis jacobi_coeff(max_n, max_alpha);

    this->dh_cells_tris.realloc_no_copy(this->cells_tris.size());
    this->dh_cells_quads.realloc_no_copy(this->cells_quads.size());
    for (int px = 0; px < this->cells_tris.size(); px++) {
      this->dh_cells_tris.h_buffer.ptr[px] = this->cells_tris[px];
    }
    for (int px = 0; px < this->cells_quads.size(); px++) {
      this->dh_cells_quads.h_buffer.ptr[px] = this->cells_quads[px];
    }

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
    this->dh_nummodes0.host_to_device();
    this->dh_nummodes1.host_to_device();
    this->dh_cells_tris.host_to_device();
    this->dh_cells_quads.host_to_device();
    this->dh_coeffs_pnm10.host_to_device();
    this->dh_coeffs_pnm11.host_to_device();
    this->dh_coeffs_pnm2.host_to_device();
  }
};

} // namespace NESO

#endif
