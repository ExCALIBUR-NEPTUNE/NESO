#ifndef __FUNCTION_BASIS_EVALUATION_H_
#define __FUNCTION_BASIS_EVALUATION_H_
#include "coordinate_mapping.hpp"
#include "particle_interface.hpp"
#include <cstdlib>
#include <map>
#include <memory>
#include <neso_particles.hpp>

#include <LocalRegions/QuadExp.h>
#include <LocalRegions/TriExp.h>
#include <StdRegions/StdExpansion2D.h>

#include "basis_evaluation.hpp"
#include "expansion_looping/basis_evaluate_base.hpp"
#include "expansion_looping/expansion_looping.hpp"
#include "expansion_looping/geom_to_expansion_builder.hpp"
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
  /**
   *  Templated evaluation function for CRTP.
   */
  template <typename EVALUATE_TYPE, typename COMPONENT_TYPE>
  inline sycl::event evaluate_inner(
      ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
          evaluation_type,
      ParticleGroupSharedPtr particle_group, Sym<COMPONENT_TYPE> sym,
      const int component) {

    const ShapeType shape_type = evaluation_type.get_shape_type();
    const int cells_iterset_size = this->map_shape_to_count.at(shape_type);
    if (cells_iterset_size == 0) {
      return sycl::event{};
    }

    const auto k_cells_iterset =
        this->map_shape_to_dh_cells.at(shape_type)->d_buffer.ptr;

    auto mpi_rank_dat = particle_group->mpi_rank_dat;

    const auto k_ref_positions =
        (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
            ->cell_dat.device_ptr();

    auto k_output = (*particle_group)[sym]->cell_dat.device_ptr();
    const int k_component = component;

    const auto d_npart_cell = mpi_rank_dat->d_npart_cell;
    const auto k_global_coeffs = this->dh_global_coeffs.d_buffer.ptr;
    const auto k_coeffs_offsets = this->dh_coeffs_offsets.d_buffer.ptr;
    const auto k_nummodes = this->dh_nummodes.d_buffer.ptr;

    // jacobi coefficients
    const auto k_coeffs_pnm10 = this->dh_coeffs_pnm10.d_buffer.ptr;
    const auto k_coeffs_pnm11 = this->dh_coeffs_pnm11.d_buffer.ptr;
    const auto k_coeffs_pnm2 = this->dh_coeffs_pnm2.d_buffer.ptr;
    const int k_stride_n = this->stride_n;

    const int k_max_total_nummodes0 =
        this->map_total_nummodes.at(shape_type).at(0);
    const int k_max_total_nummodes1 =
        this->map_total_nummodes.at(shape_type).at(1);
    const int k_max_total_nummodes2 =
        this->map_total_nummodes.at(shape_type).at(2);

    const size_t local_size = get_num_local_work_items(
        this->sycl_target,
        static_cast<size_t>(k_max_total_nummodes0 + k_max_total_nummodes1 +
                            k_max_total_nummodes2) *
            sizeof(REAL),
        128);

    const int local_mem_num_items =
        (k_max_total_nummodes0 + k_max_total_nummodes1 +
         k_max_total_nummodes2) *
        local_size;
    const size_t outer_size =
        get_particle_loop_global_size(mpi_rank_dat, local_size);

    const int k_ndim = evaluation_type.get_ndim();

    sycl::range<2> cell_iterset_range{static_cast<size_t>(cells_iterset_size),
                                      static_cast<size_t>(outer_size) *
                                          static_cast<size_t>(local_size)};
    sycl::range<2> local_iterset{1, local_size};

    auto event_loop = this->sycl_target->queue.submit([&](sycl::handler &cgh) {
      sycl::accessor<REAL, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          local_mem(sycl::range<1>(local_mem_num_items), cgh);

      cgh.parallel_for<>(
          sycl::nd_range<2>(cell_iterset_range, local_iterset),
          [=](sycl::nd_item<2> idx) {
            const int iter_cell = idx.get_global_id(0);
            const int idx_local = idx.get_local_id(1);

            const INT cellx = k_cells_iterset[iter_cell];
            const INT layerx = idx.get_global_id(1);
            ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
                loop_type{};

            if (layerx < d_npart_cell[cellx]) {
              const REAL *dofs = &k_global_coeffs[k_coeffs_offsets[cellx]];

              // Get the number of modes in x,y and z.
              const int nummodes = k_nummodes[cellx];

              REAL xi0, xi1, xi2, eta0, eta1, eta2;
              xi0 = k_ref_positions[cellx][0][layerx];
              if (k_ndim > 1) {
                xi1 = k_ref_positions[cellx][1][layerx];
              }
              if (k_ndim > 2) {
                xi2 = k_ref_positions[cellx][2][layerx];
              }

              loop_type.loc_coord_to_loc_collapsed(xi0, xi1, xi2, &eta0, &eta1,
                                                   &eta2);

              // Get the local space for the 1D evaluations in each dimension.
              REAL *local_space_0 =
                  &local_mem[idx_local *
                             (k_max_total_nummodes0 + k_max_total_nummodes1 +
                              k_max_total_nummodes2)];
              REAL *local_space_1 = local_space_0 + k_max_total_nummodes0;
              REAL *local_space_2 = local_space_1 + k_max_total_nummodes1;

              // Compute the basis functions in each dimension.
              loop_type.evaluate_basis_0(nummodes, eta0, k_stride_n,
                                         k_coeffs_pnm10, k_coeffs_pnm11,
                                         k_coeffs_pnm2, local_space_0);
              loop_type.evaluate_basis_1(nummodes, eta1, k_stride_n,
                                         k_coeffs_pnm10, k_coeffs_pnm11,
                                         k_coeffs_pnm2, local_space_1);
              loop_type.evaluate_basis_2(nummodes, eta2, k_stride_n,
                                         k_coeffs_pnm10, k_coeffs_pnm11,
                                         k_coeffs_pnm2, local_space_2);

              REAL evaluation = 0.0;
              loop_type.loop_evaluate(nummodes, dofs, local_space_0,
                                      local_space_1, local_space_2,
                                      &evaluation);

              k_output[cellx][k_component][layerx] = evaluation;
            }
          });
    });

    return event_loop;
  }

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

    EventStack event_stack{};

    if (this->mesh->get_ndim() == 2) {
      event_stack.push(evaluate_inner(ExpansionLooping::Quadrilateral{},
                                      particle_group, sym, component));

      event_stack.push(evaluate_inner(ExpansionLooping::Triangle{},
                                      particle_group, sym, component));
    } else {
      event_stack.push(evaluate_inner(ExpansionLooping::Hexahedron{},
                                      particle_group, sym, component));
      event_stack.push(evaluate_inner(ExpansionLooping::Pyramid{},
                                      particle_group, sym, component));
      event_stack.push(evaluate_inner(ExpansionLooping::Prism{}, particle_group,
                                      sym, component));
      event_stack.push(evaluate_inner(ExpansionLooping::Tetrahedron{},
                                      particle_group, sym, component));
    }

    event_stack.wait();
  }
};

} // namespace NESO

#endif
