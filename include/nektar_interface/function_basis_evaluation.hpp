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

#include <CL/sycl.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

namespace NESO {

/**
 * Class to evaluate Nektar++ fields by evaluating basis functions.
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
   * Constructor to create instance to evaluate Nektar++ fields.
   *
   * @param field Example Nektar++ field of the same mesh and function space as
   * the destination fields that this instance will be called with.
   * @param mesh ParticleMeshInterface constructed over same mesh as the
   * function.
   * @param cell_id_translation Map between NESO-Particles cells and Nektar++
   * cells.
   */
  FunctionEvaluateBasis(std::shared_ptr<T> field,
                        ParticleMeshInterfaceSharedPtr mesh,
                        CellIDTranslationSharedPtr cell_id_translation)
      : BasisEvaluateBase<T>(field, mesh, cell_id_translation) {}

  /**
   * Evaluate nektar++ function at particle locations.
   *
   * @param particle_group Source container of particles.
   * @param sym Symbol of ParticleDat within the ParticleGroup.
   * @param component Determine which component of the ParticleDat is
   * the output for function evaluations.
   * @param global_coeffs source DOFs which are evaluated.
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

    const size_t local_size = get_num_local_work_items(
        this->sycl_target,
        static_cast<size_t>(k_max_total_nummodes0 + k_max_total_nummodes1) *
            sizeof(double),
        128);

    const int local_mem_num_items =
        (k_max_total_nummodes0 + k_max_total_nummodes1) * local_size;
    const size_t outer_size =
        get_particle_loop_global_size(mpi_rank_dat, local_size);

    sycl::range<2> cell_iterset_quad{
        static_cast<size_t>(outer_size) * static_cast<size_t>(local_size),
        static_cast<size_t>(this->cells_quads.size())};
    sycl::range<2> cell_iterset_tri{
        static_cast<size_t>(outer_size) * static_cast<size_t>(local_size),
        static_cast<size_t>(this->cells_tris.size())};
    sycl::range<2> local_iterset{local_size, 1};

    // Evaluate the function at all particles in quads
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

              // Get the number of modes in x and y
              const int nummodes0 = k_nummodes0[cellx];
              const int nummodes1 = k_nummodes1[cellx];

              const double xi0 = k_ref_positions[cellx][0][layerx];
              const double xi1 = k_ref_positions[cellx][1][layerx];

              // Get the local space for the 1D evaluations in dim0 and dim1
              auto local_space_0 =
                  &local_mem[idx_local *
                             (k_max_total_nummodes0 + k_max_total_nummodes1)];
              auto local_space_1 = local_space_0 + k_max_total_nummodes0;

              // Compute the basis functions in dim0 and dim1
              BasisEvaluateBase<T>::mod_A(nummodes0, xi0, k_stride_n,
                                          k_coeffs_pnm10, k_coeffs_pnm11,
                                          k_coeffs_pnm2, local_space_0);
              BasisEvaluateBase<T>::mod_A(nummodes1, xi1, k_stride_n,
                                          k_coeffs_pnm10, k_coeffs_pnm11,
                                          k_coeffs_pnm2, local_space_1);

              // Multiply out the basis functions along with the DOF value
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

              double eta0, eta1;
              Coordinate::Mapping::xi_to_eta(0, 0, xi0, xi1, &eta0, &eta1);

              auto local_space_0 =
                  &local_mem[idx_local *
                             (k_max_total_nummodes0 + k_max_total_nummodes1)];
              auto local_space_1 = local_space_0 + k_max_total_nummodes0;

              // Compute the basis functions in dim0 and dim1
              BasisEvaluateBase<T>::mod_A(nummodes0, eta0, k_stride_n,
                                          k_coeffs_pnm10, k_coeffs_pnm11,
                                          k_coeffs_pnm2, local_space_0);
              BasisEvaluateBase<T>::mod_B(nummodes1, eta1, k_stride_n,
                                          k_coeffs_pnm10, k_coeffs_pnm11,
                                          k_coeffs_pnm2, local_space_1);

              // Multiply the basis functions together along with the DOF
              double evaluation = 0.0;
              int modey = 0;
              for (int px = 0; px < nummodes1; px++) {
                for (int qx = 0; qx < nummodes1 - px; qx++) {
                  const int mode = modey++;
                  const double coeff = dofs[mode];
                  // There exists a correction for mode == 1 in the Nektar++
                  // definition of this 2D basis which we apply here.
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
