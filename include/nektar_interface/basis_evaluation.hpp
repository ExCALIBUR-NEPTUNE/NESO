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

  //std::cout << num_modes << " ---------------------------" << std::endl;

  if (quad) {
    /*
      std::vector<double> b0(nummodes0);
      std::vector<double> b1(nummodes1);

      eval_modA(nummodes0, xi[0], b0);
      eval_modA(nummodes1, xi[1], b1);

      double eval = 0.0;
      for(int px=0 ; px<nummodes0 ; px++){
        for(int py=0 ; py<nummodes1 ; py++){

          const double inner_coeff = coeffs[py * nummodes0 + px];
          const double basis_eval = b0[px] * b1[py];
          eval += inner_coeff * basis_eval;
        }
      }

      const double eval_correct = field->GetExp(elmtIdx)->StdPhysEvaluate(xi,
      elmtPhys); const double err = abs(eval_correct - eval);

      if (err > 1.0e-12){
        nprint("BAD EVAL:", err, eval_correct, eval);
      }

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
      int tindex=0;
      for(int px=0 ; px<nummodes0 ; px++){
        for(int qx=0 ; qx<numpoints0 ; qx++){
          const double ztmp = Z0[qx];
          const double btmp = bdata0[tindex++];
          const double etmp = eval_modA_i(px, Z0[qx]);
          const double err = abs(btmp - etmp);
          if (err > 1.0e-12){
            nprint("BAD EVAL quad dir0 err:\t", err, "\t", btmp, "\t", etmp,
      "\t", ztmp);
          }
        }
      }
  */

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
        //nprint(px, qx, "BAD TB EVAL err:\t", err, "\t", btmp, "\t", etmp);
        nprint(px, qx, etmp0, etmp1);
        //}
      }
    }

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

    NESOASSERT((this->cells_tris.size() + this->cells_quads.size()) == neso_cell_count, "Missmatch in number of quad cells triangle cells and total number of cells.");

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
   * TODO
   */
  template <typename U, typename V>
  inline void evaluate(ParticleGroupSharedPtr particle_group, Sym<U> sym,
                       const int component, V &global_coeffs) {
    const int num_global_coeffs = global_coeffs.size();
    this->dh_global_coeffs.realloc_no_copy(num_global_coeffs);
    for (int px = 0; px < num_global_coeffs; px++) {
      this->dh_global_coeffs.h_buffer.ptr[px] = global_coeffs[px];
    }
    this->dh_global_coeffs.host_to_device();

    auto mpi_rank_dat = particle_group->mpi_rank_dat;
    const int local_size = 128;
    const int max_cell_occupancy = mpi_rank_dat->cell_dat.get_nrow_max();
    const auto div_mod = std::div(max_cell_occupancy, local_size);
    const int outer_size = div_mod.quot + (div_mod.rem == 0 ? 0 : 1);

    const auto k_ref_positions =
        (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
            ->cell_dat.device_ptr();

    auto k_output = (*particle_group)[sym]->cell_dat.device_ptr();
    const int k_component = component;

    const auto d_npart_cell = mpi_rank_dat->d_npart_cell;
    const auto k_cells_quads = this->dh_cells_quads.d_buffer.ptr;
    const auto k_cells_tris = this->dh_cells_tris.d_buffer.ptr;

    const auto k_global_coeffs = this->dh_global_coeffs.d_buffer.ptr;
    const auto k_coeffs_offsets = this->dh_coeffs_offsets.d_buffer.ptr;
    const auto k_nummodes0 = this->dh_nummodes0.d_buffer.ptr;
    const auto k_nummodes1 = this->dh_nummodes1.d_buffer.ptr;

    // jacobi coefficients
    const auto k_coeffs_pnm10 = this->dh_coeffs_pnm10.d_buffer.ptr;
    const auto k_coeffs_pnm11 = this->dh_coeffs_pnm11.d_buffer.ptr;
    const auto k_coeffs_pnm2 = this->dh_coeffs_pnm2.d_buffer.ptr;
    const int k_stride_n = this->stride_n;
    const int k_max_nummodes_0 = this->max_nummodes_0;

    sycl::range<2> cell_iterset_quad{
      static_cast<size_t>(outer_size) * static_cast<size_t>(local_size),
      static_cast<size_t>(this->cells_quads.size())
    };
    sycl::range<2> cell_iterset_tri{
      static_cast<size_t>(outer_size) * static_cast<size_t>(local_size),
      static_cast<size_t>(this->cells_tris.size())
    };
    sycl::range<2> local_iterset{local_size, 1};

    auto device = sycl_target->device;
    auto local_mem_exists =
        device.is_host() ||
        (device.get_info<sycl::info::device::local_mem_type>() !=
         sycl::info::local_mem_type::none);
    auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    
    const int local_mem_num_items = this->max_nummodes_0 * local_size;
    const int local_mem_bytes = local_mem_num_items * sizeof(double);
    if (!local_mem_exists || local_mem_size < (local_mem_bytes)) {
      NESOASSERT(false, "Not enough local memory");
    }

    auto event_quad = this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          sycl::accessor<double, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>
              local_mem(sycl::range<1>(local_mem_num_items), cgh);

          cgh.parallel_for<>(
              sycl::nd_range<2>(cell_iterset_quad, local_iterset),
              [=](sycl::nd_item<2> idx) {
                const int iter_cell = idx.get_global_id(1);
                const int idx_local = idx.get_local_id(0);

                const INT cellx = k_cells_quads[iter_cell];
                const INT layerx = idx.get_global_id(0);
                if (layerx < d_npart_cell[cellx]) {
                  const auto dofs = &k_global_coeffs[k_coeffs_offsets[cellx]];
                  const int nummodes0 = k_nummodes0[cellx];
                  const int nummodes1 = k_nummodes1[cellx];

                  const double xi0 = k_ref_positions[cellx][0][layerx];
                  const double xi1 = k_ref_positions[cellx][1][layerx];

                  auto local_space = &local_mem[idx_local * k_max_nummodes_0];

                  // evaluate basis in x direction
                  const double b0_0 = 0.5 * (1.0 - xi0);
                  const double b0_1 = 0.5 * (1.0 + xi0);
                  local_space[0] = b0_0;
                  local_space[1] = b0_1;

                  double p0n;
                  double p0nm2 = 1.0;
                  double p0nm1 = 2.0 + 2.0 * (xi0 - 1.0);
                  if (nummodes0 > 2) {
                    local_space[2] = b0_0 * b0_1 * p0nm2;
                  }
                  if (nummodes0 > 3) {
                    local_space[3] = b0_0 * b0_1 * p0nm1;
                  }
                  for(int modex=4 ; modex<nummodes0 ; modex++){
                    const int nx = modex - 2;
                    const double c_pnm10 = k_coeffs_pnm10[k_stride_n * 1 + nx];
                    const double c_pnm11 = k_coeffs_pnm11[k_stride_n * 1 + nx];
                    const double c_pnm2 =  k_coeffs_pnm2[k_stride_n * 1 + nx];
                    p0n = c_pnm10 * p0nm1 * xi0 + c_pnm11 * p0nm1 + c_pnm2 * p0nm2;
                    p0nm2 = p0nm1;
                    p0nm1 = p0n;
                    local_space[modex] = p0n;
                  }

                  double evaluation = 0.0;
                  // evaluate in the y direction
                  int modey;
                  const double b1_0 = 0.5 * (1.0 - xi1);
                  modey = 0;
                  for(int modex=0 ; modex<nummodes0 ; modex++){
                    const double coeff = dofs[modey * nummodes0 + modex];
                    evaluation += coeff * local_space[modex] * b1_0;
                  }
                  const double b1_1 = 0.5 * (1.0 + xi1);
                  modey = 1;
                  for(int modex=0 ; modex<nummodes0 ; modex++){
                    const double coeff = dofs[modey * nummodes0 + modex];
                    evaluation += coeff * local_space[modex] * b1_1;
                  }
                  double p1n;
                  double p1nm1;
                  double p1nm2;
                  if (nummodes1 > 2) {
                    p1nm2 = 1.0;
                    const double b1_2 = p1nm2 * b1_0 * b1_1;
                    modey = 2;
                    for(int modex=0 ; modex<nummodes0 ; modex++){
                      const double coeff = dofs[modey * nummodes0 + modex];
                      evaluation += coeff * local_space[modex] * b1_2;
                    }
                  }
                  if (nummodes1 > 3) {
                    p1nm1 = 2.0 + 2.0 * (xi1 - 1.0);
                    const double b1_3 = p1nm1 * b1_0 * b1_1;
                    modey = 3;
                    for(int modex=0 ; modex<nummodes0 ; modex++){
                      const double coeff = dofs[modey * nummodes0 + modex];
                      evaluation += coeff * local_space[modex] * b1_3;
                    }
                  }
                  for(modey=4 ; modey<nummodes1 ; modey++){
                    const int nx = modey - 2;
                    const double c_pnm10 = k_coeffs_pnm10[k_stride_n * 1 + nx];
                    const double c_pnm11 = k_coeffs_pnm11[k_stride_n * 1 + nx];
                    const double c_pnm2 =  k_coeffs_pnm2[k_stride_n * 1 + nx];
                    p1n = c_pnm10 * p1nm1 * xi1 + c_pnm11 * p1nm1 + c_pnm2 * p1nm2;
                    p1nm2 = p1nm1;
                    p1nm1 = p1n;
                    const double b1_modey = p1n * b1_0 * b1_1;
                    for(int modex=0 ; modex<nummodes0 ; modex++){
                      const double coeff = dofs[modey * nummodes0 + modex];
                      evaluation += coeff * local_space[modex] * b1_modey;
                    }
                  }

                  k_output[cellx][k_component][layerx] = evaluation;
                }
              });
        });

    auto event_tri = this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          sycl::accessor<double, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>
              local_mem(sycl::range<1>(local_mem_num_items), cgh);

          cgh.parallel_for<>(
              sycl::nd_range<2>(cell_iterset_tri, local_iterset),
              [=](sycl::nd_item<2> idx) {
                const int iter_cell = idx.get_global_id(1);
                const int idx_local = idx.get_local_id(0);

                const INT cellx = k_cells_tris[iter_cell];
                const INT layerx = idx.get_global_id(0);

                //printf("----- %ld %ld ------\n", cellx, layerx);
                if (layerx < d_npart_cell[cellx]) {
                  const auto dofs = &k_global_coeffs[k_coeffs_offsets[cellx]];
                  const int nummodes0 = k_nummodes0[cellx];
                  const int nummodes1 = k_nummodes1[cellx];

                  const double xi0 = k_ref_positions[cellx][0][layerx];
                  const double xi1 = k_ref_positions[cellx][1][layerx];
                  const NekDouble d1_original = 1.0 - xi1;
                  const bool mask_small_cond = (fabs(d1_original) < NekConstants::kNekZeroTol);
                  NekDouble d1 = d1_original;
                  d1 =
                      (mask_small_cond && (d1 >= 0.0))
                          ? NekConstants::kNekZeroTol
                          : ((mask_small_cond && (d1 < 0.0)) ? -NekConstants::kNekZeroTol : d1);
                  const double eta0 = 2. * (1. + xi0) / d1 - 1.0;
                  const double eta1 = xi1;


                  auto local_space = &local_mem[idx_local * k_max_nummodes_0];

                  // evaluate basis in x direction
                  const double b0_0 = 0.5 * (1.0 - eta0);
                  const double b0_1 = 0.5 * (1.0 + eta0);
                  local_space[0] = b0_0;
                  local_space[1] = b0_1;

                  double p0n;
                  double p0nm2 = 1.0;
                  double p0nm1 = 2.0 + 2.0 * (eta0 - 1.0);
                  if (nummodes0 > 2) {
                    local_space[2] = b0_0 * b0_1 * p0nm2;
                  }
                  if (nummodes0 > 3) {
                    local_space[3] = b0_0 * b0_1 * p0nm1;
                  }
                  for(int modex=4 ; modex<nummodes0 ; modex++){
                    const int nx = modex - 2;
                    const double c_pnm10 = k_coeffs_pnm10[k_stride_n * 1 + nx];
                    const double c_pnm11 = k_coeffs_pnm11[k_stride_n * 1 + nx];
                    const double c_pnm2 =  k_coeffs_pnm2[k_stride_n * 1 + nx];
                    p0n = c_pnm10 * p0nm1 * eta0 + c_pnm11 * p0nm1 + c_pnm2 * p0nm2;
                    p0nm2 = p0nm1;
                    p0nm1 = p0n;
                    local_space[modex] = p0n;
                  }

                  double evaluation = 0.0;
                  //nprint("keta", eta0, eta1);
                  //out << "keta " << eta0 << " " << eta1 << sycl::endl;
                  //printf("keta %f %f\n", eta0, eta1);
                  // evaluate in the y direction
                  int modey = 0;
                  const double b1_0 = 0.5 * (1.0 - eta1);
                  const double b1_1 = 0.5 * (1.0 + eta1);
                  double b1_pow = 1.0 / b1_0;
                  for(int px=0 ; px<nummodes1 ; px++){
                    double p1n, p1nm1, p1nm2;
                    b1_pow *= b1_0;
                    const int alpha = 2*px - 1;
                    for(int qx=0 ; qx<(nummodes1 - px) ; qx++){

                      double etmp1;
                      // evaluate eModified_B at eta1
                      if (px == 0){
                        // evaluate eModified_A(q, eta1)
                        if (qx == 0){
                          etmp1 = b1_0;
                        } else if (qx == 1) {
                          etmp1 = b1_1;
                        } else if (qx == 2) {
                          etmp1 = b1_0 * b1_1;
                          p1nm2 = 1.0;
                        } else if (qx == 3) {
                          etmp1 = b1_0 * b1_1 * (2.0 + 2.0 * (eta1 - 1.0));
                          p1nm1 = etmp1;
                        } else {
                          const int nx = qx - 2;
                          const double c_pnm10 = k_coeffs_pnm10[k_stride_n * 1 + nx];
                          const double c_pnm11 = k_coeffs_pnm11[k_stride_n * 1 + nx];
                          const double c_pnm2 =  k_coeffs_pnm2[k_stride_n * 1 + nx];
                          p1n = c_pnm10 * p1nm1 * eta1 + c_pnm11 * p1nm1 + c_pnm2 * p1nm2;
                          p1nm2 = p1nm1;
                          p1nm1 = p1n;
                          etmp1 = p1n;
                        }
                      } else if (qx == 0) {
                        etmp1 = b1_pow;
                      } else {
                        const int nx = qx - 1;
                        if (qx == 1){
                          p1nm2 = 1.0;
                          etmp1 = b1_pow * b1_1;
                        } else if (qx == 2) {
                          p1nm1 = 0.5 * (2.0 * (alpha + 1) + (alpha + 3) * (eta1 - 1.0));
                          etmp1 = b1_pow * b1_1 * p1nm1;
                        } else {
                          const double c_pnm10 = k_coeffs_pnm10[k_stride_n * alpha + nx];
                          const double c_pnm11 = k_coeffs_pnm11[k_stride_n * alpha + nx];
                          const double c_pnm2 =  k_coeffs_pnm2[k_stride_n * alpha + nx];
                          p1n = c_pnm10 * p1nm1 * eta1 + c_pnm11 * p1nm1 + c_pnm2 * p1nm2;
                          p1nm2 = p1nm1;
                          p1nm1 = p1n;
                          etmp1 = b1_pow * b1_1 * p1n;
                        }
                      }
                      // here have etmp1
                      const int mode = modey++;
                      const double coeff = dofs[mode];
                      const double etmp0 = (mode == 1) ? 1.0 : local_space[px];
                      evaluation += coeff * etmp0 * etmp1;
                      //out <<px << " " << qx << " " << etmp0 << " " << etmp1 << sycl::endl;
                      //printf("%f %f %d %d %f %f\n", eta0, eta1, px, qx, etmp0, etmp1);
                    }
                  }

                  k_output[cellx][k_component][layerx] = evaluation;
                }
              });
        });

    event_quad.wait_and_throw();
    event_tri.wait_and_throw();
    
  }
};

} // namespace NESO

#endif
