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

#include "function_coupling_base.hpp"
#include "special_functions.hpp"

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
#include <LibUtilities/Foundations/Basis.h>
#include <LibUtilities/Polylib/Polylib.h>
#include <MultiRegions/ContField.h>
#include <MultiRegions/DisContField.h>

using namespace Nektar;
using namespace Nektar::SpatialDomains;
using namespace Nektar::MultiRegions;
using namespace Nektar::LibUtilities;

namespace NESO {

/**
 *  Reference implementation to compute eModified_A at an order p and point z.
 *
 *  @param p Polynomial order.
 *  @param z Point in [-1, 1] to evaluate at.
 *  @returns Basis function evaluated at point.
 */
inline double eval_modA_i(const int p, const double z) {
  const double b0 = 0.5 * (1.0 - z);
  const double b1 = 0.5 * (1.0 + z);
  if (p == 0) {
    return b0;
  }
  if (p == 1) {
    return b1;
  }
  return b0 * b1 * jacobi(p - 2, z, 1, 1);
}

/**
 *  Reference implementation to compute eModified_B at an order p,q and point z.
 *
 *  @param p First index for basis.
 *  @param q Second index for basis.
 *  @param z Point in [-1, 1] to evaluate at.
 *  @returns Basis function evaluated at point.
 */
inline double eval_modB_ij(const int p, const int q, const double z) {

  double output;

  if (p == 0) {
    output = eval_modA_i(q, z);
  } else if (q == 0) {
    output = std::pow(0.5 * (1.0 - z), (double)p);
  } else {
    output = std::pow(0.5 * (1.0 - z), (double)p) * 0.5 * (1.0 + z) *
             jacobi(q - 1, z, 2 * p - 1, 1);
  }
  return output;
}

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

inline void to_collapsed(Array<OneD, NekDouble> &xi, double *eta) {
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
 *  TODO
 *
 */
template <typename T>
inline double evaluate_poly_scalar_2d(std::shared_ptr<T> field, const double x,
                                      const double y) {
  Array<OneD, NekDouble> xi(2);
  Array<OneD, NekDouble> eta_array(2);
  Array<OneD, NekDouble> coords(2);

  coords[0] = x;
  coords[1] = y;
  int elmtIdx = field->GetExpIndex(coords, xi);
  auto elmtPhys = field->GetPhys() + field->GetPhys_Offset(elmtIdx);

  // const double eval = field->GetExp(elmtIdx)->StdPhysEvaluate(xi, elmtPhys);
  auto expansion = field->GetExp(elmtIdx);

  auto global_coeffs = field->GetCoeffs();
  const int coeff_offset = field->GetCoeff_Offset(elmtIdx);

  auto coeffs = &global_coeffs[coeff_offset];

  const int num_modes = expansion->GetNcoeffs();

  auto basis0 = expansion->GetBasis(0);
  auto basis1 = expansion->GetBasis(1);
  const int nummodes0 = basis0->GetNumModes();
  const int nummodes1 = basis1->GetNumModes();

  nprint(basis0->GetBasisType() == eModified_A,
         basis1->GetBasisType() == eModified_A,
         basis1->GetBasisType() == eModified_B);

  const bool quad = (basis0->GetBasisType() == eModified_A) &&
                    (basis1->GetBasisType() == eModified_A);

  // std::cout << num_modes << " ---------------------------" << std::endl;

  if (quad) {

    std::vector<double> b0(nummodes0);
    std::vector<double> b1(nummodes1);

    for (int px = 0; px < nummodes0; px++) {
      b0[px] = eval_modA_i(px, xi[0]);
      nprint(0, px, eval_modA_i(px, xi[0]));
    }
    for (int px = 0; px < nummodes1; px++) {
      b1[px] = eval_modA_i(px, xi[1]);
      nprint(1, px, eval_modA_i(px, xi[1]));
    }

    double eval = 0.0;
    for (int px = 0; px < nummodes0; px++) {
      for (int py = 0; py < nummodes1; py++) {

        const double inner_coeff = coeffs[py * nummodes0 + px];
        const double basis_eval = b0[px] * b1[py];
        eval += inner_coeff * basis_eval;
      }
    }

    const double eval_correct =
        field->GetExp(elmtIdx)->StdPhysEvaluate(xi, elmtPhys);
    const double err = abs(eval_correct - eval);
    nprint("OUTER EVAL:", err, eval_correct, eval);

    auto bdata0 = basis0->GetBdata();
    auto bdata1 = basis1->GetBdata();
    auto Z0 = basis0->GetZ();
    auto Z1 = basis1->GetZ();

    nprint("Z0 num points", basis0->GetNumPoints());
    nprint("Z0 size:", Z0.size());
    nprint("Z1 size:", Z1.size());
    nprint("nummodes0", nummodes0, "nummodes1", nummodes1);
    nprint("bdata0 size:", bdata0.size());
    nprint("bdata1 size:", bdata1.size());

    const int numpoints0 = basis0->GetNumPoints();
    int tindex = 0;
    for (int px = 0; px < nummodes0; px++) {
      for (int qx = 0; qx < numpoints0; qx++) {
        const double ztmp = Z0[qx];
        const double btmp = bdata0[tindex++];
        const double etmp = eval_modA_i(px, Z0[qx]);
        const double err = abs(btmp - etmp);
        if (err > 1.0e-12) {
          nprint("BAD EVAL quad dir0 err:\t", err, "\t", btmp, "\t", etmp, "\t",
                 ztmp);
        }
      }
    }

  } else {
    double eta[2];
    to_collapsed(xi, eta);
    eta_array[0] = eta[0];
    eta_array[1] = eta[1];

    // eta[0] = xi[0];
    // eta[1] = xi[1];

    // nprint("~~~~~~~~~~~~~~~~");
    // nprint("(4)_5:", pochhammer(4,5));
    // nprint("P^(3,4)_5(0.3)", jacobi(5, 0.3, 3,4));
    // nprint("P^(7,11)_13(0.3)", jacobi(13, 0.3, 7,11));
    // nprint("P^(7,11)_13(-0.5)", jacobi(13, -0.5, 7,11));

    auto bdata0 = basis0->GetBdata();
    auto bdata1 = basis1->GetBdata();
    auto Z0 = basis0->GetZ();
    auto Z1 = basis1->GetZ();
    int tindex = 0;

    /*
    nprint("Z0 num points", basis0->GetNumPoints());
    nprint("Z1 num points", basis1->GetNumPoints());
    nprint("Z0 size:", Z0.size());
    nprint("Z1 size:", Z1.size());
    nprint("nummodes0", nummodes0, "nummodes1", nummodes1);
    nprint("bdata0 size:", bdata0.size());
    nprint("bdata1 size:", bdata1.size());

    for (int nx = 0; nx < nummodes0; nx++) {
      nprint("n0:", nx, Z0[nx]);
    }
    for (int nx = 0; nx < nummodes1; nx++) {
      nprint("n1:", nx, Z1[nx]);
    }

    const int numpoints1 = basis1->GetNumPoints();
    for (int px = 0; px < nummodes1; px++) {
      for (int qx = 0; qx < (nummodes1 - px); qx++) {
        for (int pointx = 0; pointx < numpoints1; pointx++) {
          const double ztmp = Z1[pointx];
          const double btmp = bdata1[tindex++];
          const double etmp = eval_modB_ij(px, qx, ztmp);
          const double err = abs(btmp - etmp);
          // if (err > 1.0e-12){
          nprint(px, qx, "BAD EVAL tqp dir1 err:\t", err, "\t", btmp, "\t",
                 etmp, "\t", ztmp);
          //}
        }
      }
    }
    */

    const int nummodes_total = expansion->GetNcoeffs();
    const double eval_correct =
        field->GetExp(elmtIdx)->StdPhysEvaluate(xi, elmtPhys);

    double eval_modes = 0.0;
    nprint("N coeffs", nummodes_total);
    for (int modex = 0; modex < nummodes_total; modex++) {
      const double basis_eval = expansion->PhysEvaluateBasis(xi, modex);
      eval_modes += basis_eval * coeffs[modex];
    }

    const double err_modes = abs(eval_correct - eval_modes);
    nprint("TRI EVAL MODES:", err_modes, eval_correct, eval_modes);

    nprint("eta:", eta[0], eta[1]);

    double eval_basis = 0.0;
    tindex = 0;
    for (int px = 0; px < nummodes1; px++) {
      for (int qx = 0; qx < (nummodes1 - px); qx++) {
        const int modex = tindex++;
        const double ztmp0 = eta[0];
        const double ztmp1 = eta[1];
        const double btmp = expansion->PhysEvaluateBasis(xi, modex);
        double etmp0 = eval_modA_i(px, ztmp0);
        double etmp1 = eval_modB_ij(px, qx, ztmp1);
        // if(modex==1){
        //   etmp1 *= 1.0 / eval_modA_i(px, ztmp0);
        // }
        //  or
        if (modex == 1) {
          etmp0 = 1.0;
        }
        const double etmp = etmp0 * etmp1;
        const double err = abs(btmp - etmp);
        // if (err > 1.0e-12){
        // nprint(px, qx, "BAD TB EVAL err:\t", err, "\t", btmp, "\t", etmp);
        nprint(px, qx, etmp0, etmp1, coeffs[modex] * etmp);
        eval_basis += coeffs[modex] * etmp;
        //}
      }
    }

    nprint("TMP EVAL:", abs(eval_basis - eval_correct), eval_basis);

    nprint("nummodes total:", nummodes_total, "tindex", tindex);
  }

  return 0.0;
}

/**
 * TODO
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
  BufferDeviceHost<NekDouble> dh_global_coeffs;

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
   * TODO
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

    const int index_tri_geom = this->cell_id_translation->index_tri_geom;
    const int index_quad_geom = this->cell_id_translation->index_quad_geom;

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

  /**
   *  Get a number of local work items that should not exceed the maximum
   *  available local memory on the device.
   *
   *  @param num_bytes Number of bytes requested per work item.
   *  @param default_num Default number of work items.
   *  @returns Number of work items.
   */
  inline size_t get_num_local_work_items(const size_t num_bytes,
                                         const size_t default_num) {
    sycl::device device = this->sycl_target->device;
    auto local_mem_exists =
        device.is_host() ||
        (device.get_info<sycl::info::device::local_mem_type>() !=
         sycl::info::local_mem_type::none);
    auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();

    const size_t max_num_workitems = local_mem_size / num_bytes;
    // find the max power of two that does not exceed the number of work items.
    const size_t two_power = log2(max_num_workitems);
    const size_t max_base_two_num_workitems = std::pow(2, two_power);

    const size_t deduced_num_work_items =
        std::min(default_num, max_base_two_num_workitems);
    NESOASSERT((deduced_num_work_items > 0),
               "Deduced number of work items is not strictly positive.");

    const size_t local_mem_bytes = deduced_num_work_items * num_bytes;
    if ((!local_mem_exists) || (local_mem_size < local_mem_bytes)) {
      NESOASSERT(false, "Not enough local memory");
    }
    return deduced_num_work_items;
  }
};

} // namespace NESO

#endif
