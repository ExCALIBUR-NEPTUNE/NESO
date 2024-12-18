#ifndef __FUNCTION_BASIS_PROJECTION_H_
#define __FUNCTION_BASIS_PROJECTION_H_
#include "coordinate_mapping.hpp"
#include "particle_interface.hpp"
#include <cstdlib>
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
#include "utility_sycl.hpp"

using namespace NESO::Particles;
using namespace Nektar::LocalRegions;
using namespace Nektar::StdRegions;

#include <map>
#include <memory>
#include <string>

namespace NESO {

/**
 * Class to project onto Nektar++ fields by evaluating basis functions.
 */
template <typename T> class FunctionProjectBasis : public BasisEvaluateBase<T> {
protected:
  /**
   *  Templated projection function for CRTP.
   */
  template <typename PROJECT_TYPE, typename COMPONENT_TYPE>
  inline sycl::event
  project_inner(ExpansionLooping::JacobiExpansionLoopingInterface<PROJECT_TYPE>
                    project_type,
                ParticleGroupSharedPtr particle_group, Sym<COMPONENT_TYPE> sym,
                const int component) {

    const ShapeType shape_type = project_type.get_shape_type();
    const int cells_iterset_size = this->map_shape_to_count.at(shape_type);
    if (cells_iterset_size == 0) {
      return sycl::event{};
    }
    const auto loop_data = this->get_loop_data(project_type);
    const auto k_cells_iterset =
        this->map_shape_to_dh_cells.at(shape_type)->d_buffer.ptr;
    auto mpi_rank_dat = particle_group->mpi_rank_dat;
    const auto k_ref_positions =
        particle_group->get_dat(Sym<REAL>("NESO_REFERENCE_POSITIONS"))
            ->cell_dat.device_ptr();

    auto k_input = particle_group->get_dat(sym)->cell_dat.device_ptr();
    const int k_component = component;
    const auto d_npart_cell = mpi_rank_dat->d_npart_cell;
    const auto max_total_nummodes_sum =
        PrivateBasisEvaluateBaseKernel::sum_max_modes(loop_data);

    const std::size_t default_local_size =
        this->sycl_target->parameters
            ->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;
    const size_t local_size = get_num_local_work_items(
        this->sycl_target,
        static_cast<size_t>(max_total_nummodes_sum) * sizeof(REAL),
        default_local_size);

    const int local_mem_num_items = max_total_nummodes_sum * local_size;
    const size_t outer_size =
        get_particle_loop_global_size(mpi_rank_dat, local_size);

    sycl::range<2> cell_iterset_range{static_cast<size_t>(cells_iterset_size),
                                      static_cast<size_t>(outer_size)};
    sycl::range<2> local_iterset{1, local_size};

    auto event_loop = this->sycl_target->queue.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<REAL, 1> local_mem(
          sycl::range<1>(local_mem_num_items), cgh);

      cgh.parallel_for<>(
          this->sycl_target->device_limits.validate_nd_range(
              sycl::nd_range<2>(cell_iterset_range, local_iterset)),
          [=](sycl::nd_item<2> idx) {
            const int iter_cell = idx.get_global_id(0);
            const int idx_local = idx.get_local_id(1);

            const INT cellx = k_cells_iterset[iter_cell];
            const INT layerx = idx.get_global_id(1);
            ExpansionLooping::JacobiExpansionLoopingInterface<PROJECT_TYPE>
                loop_type{};

            REAL *local_mem_ptr = static_cast<REAL *>(&local_mem[0]) +
                                  idx_local * max_total_nummodes_sum;

            if (layerx < d_npart_cell[cellx]) {
              // Get the number of modes in x and y
              const int nummodes = loop_data.nummodes[cellx];
              REAL *dofs =
                  &loop_data.global_coeffs[loop_data.coeffs_offsets[cellx]];
              REAL *local_space_0, *local_space_1, *local_space_2;

              REAL xi[3];
              PrivateBasisEvaluateBaseKernel::extract_ref_positions_ptr(
                  loop_data.ndim, k_ref_positions, cellx, layerx, xi);
              PrivateBasisEvaluateBaseKernel::prepare_per_dim_basis(
                  nummodes, loop_data, loop_type, xi, local_mem_ptr,
                  &local_space_0, &local_space_1, &local_space_2);

              const double value = k_input[cellx][k_component][layerx];
              loop_type.loop_project(nummodes, value, local_space_0,
                                     local_space_1, local_space_2, dofs);
            }
          });
    });

    return event_loop;
  }

  /**
   *  Templated projection function for CRTP for ParticleSubGroup.
   */
  template <typename PROJECT_TYPE, typename COMPONENT_TYPE>
  inline sycl::event
  project_inner(ExpansionLooping::JacobiExpansionLoopingInterface<PROJECT_TYPE>
                    project_type,
                ParticleSubGroupSharedPtr particle_sub_group,
                Sym<COMPONENT_TYPE> sym, const int component) {

    auto particle_group = particle_sub_group->get_particle_group();
    if (particle_sub_group->is_entire_particle_group()) {
      return this->project_inner(project_type, particle_group, sym, component);
    }

    const ShapeType shape_type = project_type.get_shape_type();
    const int cells_iterset_size = this->map_shape_to_count.at(shape_type);
    if (cells_iterset_size == 0) {
      return sycl::event{};
    }
    const auto loop_data = this->get_loop_data(project_type);
    const auto h_cells_iterset =
        this->map_shape_to_dh_cells.at(shape_type)->h_buffer.ptr;

    const auto max_total_nummodes_sum =
        PrivateBasisEvaluateBaseKernel::sum_max_modes(loop_data);

    const std::size_t local_size_bytes =
        static_cast<size_t>(max_total_nummodes_sum) * sizeof(REAL);
    auto local_space = std::make_shared<LocalMemoryBlock>(local_size_bytes);

    const int k_component = component;

    for (std::size_t cx = 0; cx < cells_iterset_size; cx++) {
      const int cellx = h_cells_iterset[cx];
      particle_loop(
          "FunctionProjectionBasis::ParticleSubGroup", particle_sub_group,
          [=](auto LOCAL_SPACE, auto REF_POSITIONS, auto VALUE) {
            ExpansionLooping::JacobiExpansionLoopingInterface<PROJECT_TYPE>
                loop_type{};

            // Get the number of modes in x and y
            const int nummodes = loop_data.nummodes[cellx];
            REAL *dofs =
                &loop_data.global_coeffs[loop_data.coeffs_offsets[cellx]];
            REAL *local_space_0, *local_space_1, *local_space_2;

            REAL xi[3];
            PrivateBasisEvaluateBaseKernel::extract_ref_positions_dat(
                loop_data.ndim, REF_POSITIONS, xi);
            PrivateBasisEvaluateBaseKernel::prepare_per_dim_basis(
                nummodes, loop_data, loop_type, xi,
                static_cast<REAL *>(LOCAL_SPACE.data()), &local_space_0,
                &local_space_1, &local_space_2);

            const auto value = VALUE.at(k_component);
            loop_type.loop_project(nummodes, value, local_space_0,
                                   local_space_1, local_space_2, dofs);
          },
          Access::write(local_space),
          Access::read(Sym<REAL>("NESO_REFERENCE_POSITIONS")),
          Access::read(sym))
          ->execute(cellx);
    }

    return sycl::event{};
  }

public:
  /// Disable (implicit) copies.
  FunctionProjectBasis(const FunctionProjectBasis &st) = delete;
  /// Disable (implicit) copies.
  FunctionProjectBasis &operator=(FunctionProjectBasis const &a) = delete;

  /**
   * Constructor to create instance to project onto Nektar++ fields.
   *
   * @param field Example Nektar++ field of the same mesh and function space as
   * the destination fields that this instance will be called with.
   * @param mesh ParticleMeshInterface constructed over same mesh as the
   * function.
   * @param cell_id_translation Map between NESO-Particles cells and Nektar++
   * cells.
   */
  FunctionProjectBasis(std::shared_ptr<T> field,
                       ParticleMeshInterfaceSharedPtr mesh,
                       CellIDTranslationSharedPtr cell_id_translation)
      : BasisEvaluateBase<T>(field, mesh, cell_id_translation) {}

  /**
   * Project particle data onto a function.
   *
   * @param particle_group Source container of particles.
   * @param sym Symbol of ParticleDat within the ParticleGroup.
   * @param component Determine which component of the ParticleDat is
   * projected.
   * @param global_coeffs[in,out] RHS in the Ax=b L2 projection system.
   */
  template <typename GROUP_TYPE, typename U, typename V>
  inline void project(std::shared_ptr<GROUP_TYPE> particle_group, Sym<U> sym,
                      const int component, V &global_coeffs) {

    static_assert((std::is_same_v<GROUP_TYPE, ParticleGroup> ||
                   std::is_same_v<GROUP_TYPE, ParticleSubGroup>),
                  "Expected ParticleGroup or ParticleSubGroup");

    const int num_global_coeffs = global_coeffs.size();
    this->dh_global_coeffs.realloc_no_copy(num_global_coeffs);
    this->sycl_target->queue
        .fill(this->dh_global_coeffs.d_buffer.ptr, static_cast<REAL>(0.0),
              num_global_coeffs)
        .wait_and_throw();

    EventStack event_stack{};
    if (this->mesh->get_ndim() == 2) {
      event_stack.push(project_inner(ExpansionLooping::Quadrilateral{},
                                     particle_group, sym, component));

      event_stack.push(project_inner(ExpansionLooping::Triangle{},
                                     particle_group, sym, component));
    } else {
      event_stack.push(project_inner(ExpansionLooping::Hexahedron{},
                                     particle_group, sym, component));
      event_stack.push(project_inner(ExpansionLooping::Pyramid{},
                                     particle_group, sym, component));
      event_stack.push(project_inner(ExpansionLooping::Prism{}, particle_group,
                                     sym, component));
      event_stack.push(project_inner(ExpansionLooping::Tetrahedron{},
                                     particle_group, sym, component));
    }

    event_stack.wait();
    this->dh_global_coeffs.device_to_host();
    for (int px = 0; px < num_global_coeffs; px++) {
      global_coeffs[px] = this->dh_global_coeffs.h_buffer.ptr[px];
    }
  }
};

} // namespace NESO

#endif
