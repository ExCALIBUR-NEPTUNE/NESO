#ifndef __FUNCTION_BASIS_PROJECTION_H_
#define __FUNCTION_BASIS_PROJECTION_H_
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
template <typename T> class FunctionProjectBasis : public BasisEvaluateBase<T> {
protected:
public:
  /// Disable (implicit) copies.
  FunctionProjectBasis(const FunctionProjectBasis &st) = delete;
  /// Disable (implicit) copies.
  FunctionProjectBasis &operator=(FunctionProjectBasis const &a) = delete;

  /**
   * TODO
   */
  FunctionProjectBasis(std::shared_ptr<T> field,
                       ParticleMeshInterfaceSharedPtr mesh,
                       CellIDTranslationSharedPtr cell_id_translation)
      : BasisEvaluateBase<T>(field, mesh, cell_id_translation) {}

  /**
   * TODO
   */
  template <typename U, typename V>
  inline void project(ParticleGroupSharedPtr particle_group, Sym<U> sym,
                      const int component, V &global_coeffs) {

    const int num_global_coeffs = global_coeffs.size();
    this->dh_global_coeffs.realloc_no_copy(num_global_coeffs);
    for (int px = 0; px < num_global_coeffs; px++) {
      this->dh_global_coeffs.h_buffer.ptr[px] = 0.0;
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

    const auto k_input = (*particle_group)[sym]->cell_dat.device_ptr();
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

    const int local_mem_num_items = this->max_nummodes_0 * local_size;
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
              const double value = k_input[cellx][k_component][layerx];

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
              for (int modex = 4; modex < nummodes0; modex++) {
                const int nx = modex - 2;
                const double c_pnm10 = k_coeffs_pnm10[k_stride_n * 1 + nx];
                const double c_pnm11 = k_coeffs_pnm11[k_stride_n * 1 + nx];
                const double c_pnm2 = k_coeffs_pnm2[k_stride_n * 1 + nx];
                p0n = c_pnm10 * p0nm1 * xi0 + c_pnm11 * p0nm1 + c_pnm2 * p0nm2;
                p0nm2 = p0nm1;
                p0nm1 = p0n;
                local_space[modex] = p0n;
              }

              // evaluate in the y direction
              int modey;
              const double b1_0 = 0.5 * (1.0 - xi1);
              modey = 0;
              for (int modex = 0; modex < nummodes0; modex++) {
                const double evaluation = value * local_space[modex] * b1_0;
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    coeff_atomic_ref(dofs[modey * nummodes0 + modex]);
                coeff_atomic_ref.fetch_add(evaluation);
              }
              const double b1_1 = 0.5 * (1.0 + xi1);
              modey = 1;
              for (int modex = 0; modex < nummodes0; modex++) {
                const double evaluation = value * local_space[modex] * b1_1;
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    coeff_atomic_ref(dofs[modey * nummodes0 + modex]);
                coeff_atomic_ref.fetch_add(evaluation);
              }
              double p1n;
              double p1nm1;
              double p1nm2;
              if (nummodes1 > 2) {
                p1nm2 = 1.0;
                const double b1_2 = p1nm2 * b1_0 * b1_1;
                modey = 2;
                for (int modex = 0; modex < nummodes0; modex++) {
                  const double evaluation = value * local_space[modex] * b1_2;
                  sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      coeff_atomic_ref(dofs[modey * nummodes0 + modex]);
                  coeff_atomic_ref.fetch_add(evaluation);
                }
              }
              if (nummodes1 > 3) {
                p1nm1 = 2.0 + 2.0 * (xi1 - 1.0);
                const double b1_3 = p1nm1 * b1_0 * b1_1;
                modey = 3;
                for (int modex = 0; modex < nummodes0; modex++) {
                  const double evaluation = value * local_space[modex] * b1_3;
                  sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      coeff_atomic_ref(dofs[modey * nummodes0 + modex]);
                  coeff_atomic_ref.fetch_add(evaluation);
                }
              }
              for (modey = 4; modey < nummodes1; modey++) {
                const int nx = modey - 2;
                const double c_pnm10 = k_coeffs_pnm10[k_stride_n * 1 + nx];
                const double c_pnm11 = k_coeffs_pnm11[k_stride_n * 1 + nx];
                const double c_pnm2 = k_coeffs_pnm2[k_stride_n * 1 + nx];
                p1n = c_pnm10 * p1nm1 * xi1 + c_pnm11 * p1nm1 + c_pnm2 * p1nm2;
                p1nm2 = p1nm1;
                p1nm1 = p1n;
                const double b1_modey = p1n * b1_0 * b1_1;
                for (int modex = 0; modex < nummodes0; modex++) {
                  const double evaluation =
                      value * local_space[modex] * b1_modey;
                  sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      coeff_atomic_ref(dofs[modey * nummodes0 + modex]);
                  coeff_atomic_ref.fetch_add(evaluation);
                }
              }
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
            const int iter_cell = idx.get_global_id(1);
            const int idx_local = idx.get_local_id(0);

            const INT cellx = k_cells_tris[iter_cell];
            const INT layerx = idx.get_global_id(0);

            // printf("----- %ld %ld ------\n", cellx, layerx);
            if (layerx < d_npart_cell[cellx]) {
              const auto dofs = &k_global_coeffs[k_coeffs_offsets[cellx]];
              const int nummodes0 = k_nummodes0[cellx];
              const int nummodes1 = k_nummodes1[cellx];
              const double value = k_input[cellx][k_component][layerx];
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
              for (int modex = 4; modex < nummodes0; modex++) {
                const int nx = modex - 2;
                const double c_pnm10 = k_coeffs_pnm10[k_stride_n * 1 + nx];
                const double c_pnm11 = k_coeffs_pnm11[k_stride_n * 1 + nx];
                const double c_pnm2 = k_coeffs_pnm2[k_stride_n * 1 + nx];
                p0n = c_pnm10 * p0nm1 * eta0 + c_pnm11 * p0nm1 + c_pnm2 * p0nm2;
                p0nm2 = p0nm1;
                p0nm1 = p0n;
                local_space[modex] = p0n;
              }

              // nprint("keta", eta0, eta1);
              // out << "keta " << eta0 << " " << eta1 << sycl::endl;
              // printf("keta %f %f\n", eta0, eta1);
              //  evaluate in the y direction
              int modey = 0;
              const double b1_0 = 0.5 * (1.0 - eta1);
              const double b1_1 = 0.5 * (1.0 + eta1);
              double b1_pow = 1.0 / b1_0;
              for (int px = 0; px < nummodes1; px++) {
                double p1n, p1nm1, p1nm2;
                b1_pow *= b1_0;
                const int alpha = 2 * px - 1;
                for (int qx = 0; qx < (nummodes1 - px); qx++) {

                  double etmp1;
                  // evaluate eModified_B at eta1
                  if (px == 0) {
                    // evaluate eModified_A(q, eta1)
                    if (qx == 0) {
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
                      const double c_pnm10 =
                          k_coeffs_pnm10[k_stride_n * 1 + nx];
                      const double c_pnm11 =
                          k_coeffs_pnm11[k_stride_n * 1 + nx];
                      const double c_pnm2 = k_coeffs_pnm2[k_stride_n * 1 + nx];
                      p1n = c_pnm10 * p1nm1 * eta1 + c_pnm11 * p1nm1 +
                            c_pnm2 * p1nm2;
                      p1nm2 = p1nm1;
                      p1nm1 = p1n;
                      etmp1 = p1n;
                    }
                  } else if (qx == 0) {
                    etmp1 = b1_pow;
                  } else {
                    const int nx = qx - 1;
                    if (qx == 1) {
                      p1nm2 = 1.0;
                      etmp1 = b1_pow * b1_1;
                    } else if (qx == 2) {
                      p1nm1 = 0.5 *
                              (2.0 * (alpha + 1) + (alpha + 3) * (eta1 - 1.0));
                      etmp1 = b1_pow * b1_1 * p1nm1;
                    } else {
                      const double c_pnm10 =
                          k_coeffs_pnm10[k_stride_n * alpha + nx];
                      const double c_pnm11 =
                          k_coeffs_pnm11[k_stride_n * alpha + nx];
                      const double c_pnm2 =
                          k_coeffs_pnm2[k_stride_n * alpha + nx];
                      p1n = c_pnm10 * p1nm1 * eta1 + c_pnm11 * p1nm1 +
                            c_pnm2 * p1nm2;
                      p1nm2 = p1nm1;
                      p1nm1 = p1n;
                      etmp1 = b1_pow * b1_1 * p1n;
                    }
                  }
                  // here have etmp1
                  const int mode = modey++;
                  const double etmp0 = (mode == 1) ? 1.0 : local_space[px];
                  const double evaluation = value * etmp0 * etmp1;

                  sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      coeff_atomic_ref(dofs[mode]);
                  coeff_atomic_ref.fetch_add(evaluation);

                  // out <<px << " " << qx << " " << etmp0 << " " << etmp1 <<
                  // sycl::endl; printf("%f %f %d %d %f %f\n", eta0, eta1, px,
                  // qx, etmp0, etmp1);
                }
              }
            }
          });
    });

    event_quad.wait_and_throw();
    event_tri.wait_and_throw();
    this->dh_global_coeffs.device_to_host();
    for (int px = 0; px < num_global_coeffs; px++) {
      global_coeffs[px] = this->dh_global_coeffs.h_buffer.ptr[px];
    }
  }
};

} // namespace NESO

#endif
