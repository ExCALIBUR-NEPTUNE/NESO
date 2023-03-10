#ifndef __FUNCTION_BASIS_EVALUATION_H_
#define __FUNCTION_BASIS_EVALUATION_H_
#include "particle_interface.hpp"
#include <cstdlib>
#include <map>
#include <memory>
#include <neso_particles.hpp>

#include <LocalRegions/QuadExp.h>
#include <LocalRegions/TriExp.h>
#include <StdRegions/StdExpansion2D.h>

#include "basis_evaluation.hpp"
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

namespace NESO {

/**
 * TODO
 */
template <typename T>
class FunctionEvaluateBasis : public BasisEvaluateBase<T> {
protected:
public:
  /// Disable (implicit) copies.
  FunctionEvaluateBasis(const FunctionEvaluateBasis &st) = delete;
  /// Disable (implicit) copies.
  FunctionEvaluateBasis &operator=(FunctionEvaluateBasis const &a) = delete;

  /**
   * TODO
   */
  FunctionEvaluateBasis(std::shared_ptr<T> field,
                        ParticleMeshInterfaceSharedPtr mesh,
                        CellIDTranslationSharedPtr cell_id_translation)
      : BasisEvaluateBase<T>(field, mesh, cell_id_translation) {}

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

    const int k_max_total_nummodes0 = this->max_total_nummodes0;
    const int k_max_total_nummodes1 = this->max_total_nummodes1;

    sycl::range<2> cell_iterset_quad{
        static_cast<size_t>(outer_size) * static_cast<size_t>(local_size),
        static_cast<size_t>(this->cells_quads.size())};
    sycl::range<2> cell_iterset_tri{
        static_cast<size_t>(outer_size) * static_cast<size_t>(local_size),
        static_cast<size_t>(this->cells_tris.size())};
    sycl::range<2> local_iterset{local_size, 1};

    sycl::device device = this->sycl_target->device;
    auto local_mem_exists =
        device.is_host() ||
        (device.get_info<sycl::info::device::local_mem_type>() !=
         sycl::info::local_mem_type::none);
    auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();

    const int local_mem_num_items =
        (k_max_total_nummodes0 + k_max_total_nummodes1) * local_size;
    const int local_mem_bytes = local_mem_num_items * sizeof(double);
    if (!local_mem_exists || local_mem_size < (local_mem_bytes)) {
      NESOASSERT(false, "Not enough local memory");
    }

    auto event_quad = this->sycl_target->queue.submit([&](sycl::handler &cgh) {
      sycl::accessor<double, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          local_mem(sycl::range<1>(local_mem_num_items), cgh);

      cgh.parallel_for<>(
          sycl::nd_range<2>(cell_iterset_quad, local_iterset),
          [=](sycl::nd_item<2> idx) {
            auto lambda_mod_A = [&](const int nummodes, const double z,
                                    double *output) {
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
            };

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

              auto local_space_0 =
                  &local_mem[idx_local *
                             (k_max_total_nummodes0 + k_max_total_nummodes1)];
              auto local_space_1 = local_space_0 + k_max_total_nummodes0;

              lambda_mod_A(nummodes0, xi0, local_space_0);
              lambda_mod_A(nummodes1, xi1, local_space_1);

              double evaluation = 0.0;
              for (int qx = 0; qx < nummodes1; qx++) {
                const double basis1 = local_space_1[qx];
                for (int px = 0; px < nummodes0; px++) {
                  const double coeff = dofs[qx * nummodes0 + px];
                  const double basis0 = local_space_0[px];
                  evaluation += coeff * basis0 * basis1;
                }
              }

              k_output[cellx][k_component][layerx] = evaluation;
            }
          });
    });

    auto event_tri = this->sycl_target->queue.submit([&](sycl::handler &cgh) {
      sycl::accessor<double, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          local_mem(sycl::range<1>(local_mem_num_items), cgh);

      cgh.parallel_for<>(
          sycl::nd_range<2>(cell_iterset_tri, local_iterset),
          [=](sycl::nd_item<2> idx) {
            // evaluate basis in x direction
            auto lambda_mod_A = [&](const int nummodes, const double z,
                                    double *output) {
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
            };

            auto lambda_mod_B = [&](const int nummodes, const double z,
                                    double *output) {
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
                      const double c_pnm10 =
                          k_coeffs_pnm10[k_stride_n * 1 + nx];
                      const double c_pnm11 =
                          k_coeffs_pnm11[k_stride_n * 1 + nx];
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
                      pnm1 =
                          0.5 * (2.0 * (alpha + 1) + (alpha + 3) * (z - 1.0));
                      etmp1 = b1_pow * b1 * pnm1;
                    } else {
                      const double c_pnm10 =
                          k_coeffs_pnm10[k_stride_n * alpha + nx];
                      const double c_pnm11 =
                          k_coeffs_pnm11[k_stride_n * alpha + nx];
                      const double c_pnm2 =
                          k_coeffs_pnm2[k_stride_n * alpha + nx];
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
            };

            const int iter_cell = idx.get_global_id(1);
            const int idx_local = idx.get_local_id(0);

            const INT cellx = k_cells_tris[iter_cell];
            const INT layerx = idx.get_global_id(0);

            if (layerx < d_npart_cell[cellx]) {
              const auto dofs = &k_global_coeffs[k_coeffs_offsets[cellx]];
              const int nummodes0 = k_nummodes0[cellx];
              const int nummodes1 = k_nummodes1[cellx];

              const double xi0 = k_ref_positions[cellx][0][layerx];
              const double xi1 = k_ref_positions[cellx][1][layerx];
              const NekDouble d1_original = 1.0 - xi1;
              const bool mask_small_cond =
                  (fabs(d1_original) < NekConstants::kNekZeroTol);
              NekDouble d1 = d1_original;
              d1 = (mask_small_cond && (d1 >= 0.0))
                       ? NekConstants::kNekZeroTol
                       : ((mask_small_cond && (d1 < 0.0))
                              ? -NekConstants::kNekZeroTol
                              : d1);
              const double eta0 = 2. * (1. + xi0) / d1 - 1.0;
              const double eta1 = xi1;

              auto local_space_0 =
                  &local_mem[idx_local *
                             (k_max_total_nummodes0 + k_max_total_nummodes1)];
              auto local_space_1 = local_space_0 + k_max_total_nummodes0;

              lambda_mod_A(nummodes0, eta0, local_space_0);
              lambda_mod_B(nummodes1, eta1, local_space_1);

              double evaluation = 0.0;
              //  evaluate in the y direction
              int modey = 0;
              for (int px = 0; px < nummodes1; px++) {
                for (int qx = 0; qx < nummodes1 - px; qx++) {
                  const int mode = modey++;
                  const double coeff = dofs[mode];
                  const double etmp0 = (mode == 1) ? 1.0 : local_space_0[px];
                  const double etmp1 = local_space_1[mode];
                  evaluation += coeff * etmp0 * etmp1;
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
