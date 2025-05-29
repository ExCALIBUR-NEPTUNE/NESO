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
  inline void evaluate_inner(
      [[maybe_unused]] EventStack &event_stack,
      ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
          evaluation_type,
      ParticleGroupSharedPtr particle_group,
      [[maybe_unused]] ParticleDatImplGetConstT<REAL> k_ref_positions,
      [[maybe_unused]] ParticleDatImplGetT<COMPONENT_TYPE> k_output,
      [[maybe_unused]] Sym<COMPONENT_TYPE> sym, const int component) {

    const ShapeType shape_type = evaluation_type.get_shape_type();
    const int cells_iterset_size = this->map_shape_to_count.at(shape_type);
    if (cells_iterset_size == 0) {
      return;
    }

    const auto loop_data = this->get_loop_data(evaluation_type);
    const auto k_cells_iterset =
        this->map_shape_to_dh_cells.at(shape_type)->d_buffer.ptr;
    auto mpi_rank_dat = particle_group->mpi_rank_dat;

    const int k_component = component;
    const auto d_npart_cell = mpi_rank_dat->d_npart_cell;
    const auto max_total_nummodes_sum =
        PrivateBasisEvaluateBaseKernel::sum_max_modes(loop_data);

    const std::size_t default_local_size =
        this->sycl_target->parameters
            ->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;

    const size_t local_size = this->sycl_target->get_num_local_work_items(
        static_cast<size_t>(max_total_nummodes_sum) * sizeof(REAL),
        default_local_size);

    const int local_mem_num_items = max_total_nummodes_sum * local_size;
    const size_t outer_size =
        get_particle_loop_global_size(mpi_rank_dat, local_size);

    sycl::range<2> cell_iterset_range{static_cast<size_t>(cells_iterset_size),
                                      static_cast<size_t>(outer_size)};
    sycl::range<2> local_iterset{1, local_size};

    event_stack.push(this->sycl_target->queue.submit([&](sycl::handler &cgh) {
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
            ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
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

              REAL evaluation = 0.0;
              loop_type.loop_evaluate(nummodes, dofs, local_space_0,
                                      local_space_1, local_space_2,
                                      &evaluation);

              k_output[cellx][k_component][layerx] = evaluation;
            }
          });
    }));

    return;
  }

  /**
   *  Templated evaluation function for CRTP for ParticleSubGroup.
   */
  template <typename EVALUATE_TYPE, typename COMPONENT_TYPE>
  inline void evaluate_inner(
      [[maybe_unused]] EventStack &event_stack,
      ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
          evaluation_type,
      ParticleSubGroupSharedPtr particle_sub_group,
      [[maybe_unused]] ParticleDatImplGetConstT<REAL> k_ref_positions,
      [[maybe_unused]] ParticleDatImplGetT<COMPONENT_TYPE> k_output,
      [[maybe_unused]] Sym<COMPONENT_TYPE> sym, const int component) {

    auto particle_group = particle_sub_group->get_particle_group();
    if (particle_sub_group->is_entire_particle_group()) {
      return this->evaluate_inner(event_stack, evaluation_type, particle_group,
                                  k_ref_positions, k_output, sym, component);
    }

    const ShapeType shape_type = evaluation_type.get_shape_type();
    const int cells_iterset_size = this->map_shape_to_count.at(shape_type);
    if (cells_iterset_size == 0) {
      return;
    }
    const auto loop_data = this->get_loop_data(evaluation_type);
    const auto h_cells_iterset =
        this->map_shape_to_dh_cells.at(shape_type)->h_buffer.ptr;

    const auto max_total_nummodes_sum =
        PrivateBasisEvaluateBaseKernel::sum_max_modes(loop_data);
    auto local_space =
        std::make_shared<LocalMemoryBlock<REAL>>(max_total_nummodes_sum);

    const int k_component = component;

    for (std::size_t cx = 0; cx < cells_iterset_size; cx++) {
      const int cellx = h_cells_iterset[cx];
      particle_loop(
          "FunctionEvaluateBasis::ParticleSubGroup", particle_sub_group,
          [=](auto LOCAL_SPACE, auto REF_POSITIONS, auto OUTPUT) {
            ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
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
                nummodes, loop_data, loop_type, xi, LOCAL_SPACE.data(),
                &local_space_0, &local_space_1, &local_space_2);

            REAL evaluation = 0.0;
            loop_type.loop_evaluate(nummodes, dofs, local_space_0,
                                    local_space_1, local_space_2, &evaluation);

            OUTPUT.at(k_component) = evaluation;
          },
          Access::write(local_space),
          Access::read(Sym<REAL>("NESO_REFERENCE_POSITIONS")),
          Access::write(sym))
          ->execute(cellx);
    }

    return;
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
  template <typename GROUP_TYPE, typename U, typename V>
  inline void evaluate(std::shared_ptr<GROUP_TYPE> particle_group, Sym<U> sym,
                       const int component, V &global_coeffs) {

    static_assert((std::is_same_v<GROUP_TYPE, ParticleGroup> ||
                   std::is_same_v<GROUP_TYPE, ParticleSubGroup>),
                  "Expected ParticleGroup or ParticleSubGroup");
    const int num_global_coeffs = global_coeffs.size();
    this->dh_global_coeffs.realloc_no_copy(num_global_coeffs);
    for (int px = 0; px < num_global_coeffs; px++) {
      this->dh_global_coeffs.h_buffer.ptr[px] = global_coeffs[px];
    }
    this->dh_global_coeffs.host_to_device();

    auto k_ref_positions = Access::direct_get(
        Access::read(get_particle_group(particle_group)
                         ->get_dat(Sym<REAL>("NESO_REFERENCE_POSITIONS"))));
    auto k_output = Access::direct_get(
        Access::write(get_particle_group(particle_group)->get_dat(sym)));

    EventStack event_stack{};
    if (this->mesh->get_ndim() == 2) {
      evaluate_inner(event_stack, ExpansionLooping::Quadrilateral{},
                     particle_group, k_ref_positions, k_output, sym, component);

      evaluate_inner(event_stack, ExpansionLooping::Triangle{}, particle_group,
                     k_ref_positions, k_output, sym, component);
    } else {
      evaluate_inner(event_stack, ExpansionLooping::Hexahedron{},
                     particle_group, k_ref_positions, k_output, sym, component);
      evaluate_inner(event_stack, ExpansionLooping::Pyramid{}, particle_group,
                     k_ref_positions, k_output, sym, component);
      evaluate_inner(event_stack, ExpansionLooping::Prism{}, particle_group,
                     k_ref_positions, k_output, sym, component);
      evaluate_inner(event_stack, ExpansionLooping::Tetrahedron{},
                     particle_group, k_ref_positions, k_output, sym, component);
    }
    event_stack.wait();

    Access::direct_restore(
        Access::write(get_particle_group(particle_group)->get_dat(sym)),
        k_output);
    Access::direct_restore(
        Access::read(get_particle_group(particle_group)
                         ->get_dat(Sym<REAL>("NESO_REFERENCE_POSITIONS"))),
        k_ref_positions);
  }
};

} // namespace NESO

#endif
