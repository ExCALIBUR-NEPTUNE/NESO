#ifndef __FUNCTION_BASIS_EVALUATION_H_
#define __FUNCTION_BASIS_EVALUATION_H_
#include "coordinate_mapping.hpp"
#include "particle_interface.hpp"
#include <cstdlib>
#include <map>
#include <memory>
#include <neso_particles.hpp>
#include <type_traits>

#include <LocalRegions/QuadExp.h>
#include <LocalRegions/TriExp.h>
#include <MultiRegions/DisContField.h>
#include <StdRegions/StdExpansion2D.h>

#include "basis_evaluation.hpp"
#include "basis_evaluation_templated.hpp"
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

#include <nektar_interface/expansion_looping/generated/generated_evaluate.hpp>

#define NESO_MAX_TEMPLATED_MODES 4

namespace NESO::TemplateTest {

template <size_t NUM_MODES, typename EVALUATE_TYPE_GENERIC,
          typename EVALUATE_TYPE_TEMPLATE>
inline bool templated_evaluate(
    const int num_modes,
    ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE_GENERIC>
        evaluation_type_generic,
    BasisJacobi::Templated::ExpansionLoopingInterface<EVALUATE_TYPE_TEMPLATE>
        evaluation_type_template,
    SYCLTargetSharedPtr sycl_target, ParticleGroupSharedPtr particle_group,
    Sym<REAL> sym, const int component, const int shape_count,
    const REAL *k_global_coeffs, const int *h_coeffs_offsets,
    const int *h_cells_iterset, EventStack &event_stack) {

  if (num_modes != NUM_MODES) {
    return false;
  }

  const int cells_iterset_size = shape_count;
  if (cells_iterset_size == 0) {
    return true;
  }

  auto mpi_rank_dat = particle_group->mpi_rank_dat;

  const auto k_ref_positions =
      (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
          ->cell_dat.device_ptr();

  auto k_output = (*particle_group)[sym]->cell_dat.device_ptr();
  const int k_component = component;

  for (int cell_idx = 0; cell_idx < cells_iterset_size; cell_idx++) {
    const int cellx = h_cells_iterset[cell_idx];
    const int dof_offset = h_coeffs_offsets[cellx];
    const REAL *dofs = k_global_coeffs + dof_offset;

    const int num_particles = mpi_rank_dat->h_npart_cell[cellx];

    const auto div_mod = std::div(static_cast<long long>(num_particles),
                                  static_cast<long long>(1));
    const std::size_t num_blocks =
        static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1));

    const size_t ls = 128;
    const size_t gs = get_global_size((std::size_t)num_blocks, ls);
    const int k_ndim = evaluation_type_generic.get_ndim();

    auto event_loop = sycl_target->queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<>(
          // sycl::range<1>(static_cast<size_t>(num_blocks)),
          //[=](sycl::id<1> idx) {
          sycl::nd_range<1>(sycl::range<1>(gs), sycl::range<1>(ls)),
          [=](sycl::nd_item<1> nd_idx) {
            const size_t idx = nd_idx.get_global_linear_id();
            const int ix = idx;
            if (idx < num_particles) {
              ExpansionLooping::JacobiExpansionLoopingInterface<
                  EVALUATE_TYPE_GENERIC>
                  loop_type_generic{};
              BasisJacobi::Templated::ExpansionLoopingInterface<
                  EVALUATE_TYPE_TEMPLATE>
                  loop_type_template{};

              REAL xi0, xi1, xi2, eta0, eta1, eta2;
              xi0 = k_ref_positions[cellx][0][ix];
              if (k_ndim > 1) {
                xi1 = k_ref_positions[cellx][1][ix];
              }
              if (k_ndim > 2) {
                xi2 = k_ref_positions[cellx][2][ix];
              }

              loop_type_generic.loc_coord_to_loc_collapsed(xi0, xi1, xi2, &eta0,
                                                           &eta1, &eta2);

              const REAL eval = loop_type_template.template evaluate<NUM_MODES>(
                  dofs, eta0, eta1, eta2);
              k_output[cellx][k_component][ix] = eval;
            }
          });
    });

    event_stack.push(event_loop);
  }
  return true;
}

template <size_t NUM_MODES, typename EVALUATE_TYPE_GENERIC,
          typename EVALUATE_TYPE_TEMPLATE>
inline bool templated_evaluate_wrapper_inner(
    const int num_modes,
    ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE_GENERIC>
        evaluation_type_generic,
    BasisJacobi::Templated::ExpansionLoopingInterface<EVALUATE_TYPE_TEMPLATE>
        evaluation_type_template,
    SYCLTargetSharedPtr sycl_target, ParticleGroupSharedPtr particle_group,
    Sym<REAL> sym, const int component, const int shape_count,
    const REAL *k_global_coeffs, const int *h_coeffs_offsets,
    const int *h_cells_iterset, EventStack &event_stack) {

  bool ran1 = templated_evaluate<NUM_MODES>(
      num_modes, evaluation_type_generic, evaluation_type_template, sycl_target,
      particle_group, sym, component, shape_count, k_global_coeffs,
      h_coeffs_offsets, h_cells_iterset, event_stack);

  if constexpr (NUM_MODES < NESO_MAX_TEMPLATED_MODES) {
    ran1 = ran1 ||
           templated_evaluate<NUM_MODES + 1>(
               num_modes, evaluation_type_generic, evaluation_type_template,
               sycl_target, particle_group, sym, component, shape_count,
               k_global_coeffs, h_coeffs_offsets, h_cells_iterset, event_stack);
  }
  return ran1;
}

template <typename EVALUATE_TYPE_GENERIC, typename EVALUATE_TYPE_TEMPLATE>
inline bool templated_evaluate_wrapper(
    const int num_modes,
    ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE_GENERIC>
        evaluation_type_generic,
    BasisJacobi::Templated::ExpansionLoopingInterface<EVALUATE_TYPE_TEMPLATE>
        evaluation_type_template,
    SYCLTargetSharedPtr sycl_target, ParticleGroupSharedPtr particle_group,
    Sym<REAL> sym, const int component, const int shape_count,
    const REAL *k_global_coeffs, const int *h_coeffs_offsets,
    const int *h_cells_iterset, EventStack &event_stack) {

  if (num_modes < 2 || num_modes > NESO_MAX_TEMPLATED_MODES) {
    return false;
  } else {
    return templated_evaluate_wrapper_inner<2>(
        num_modes, evaluation_type_generic, evaluation_type_template,
        sycl_target, particle_group, sym, component, shape_count,
        k_global_coeffs, h_coeffs_offsets, h_cells_iterset, event_stack);
  }
}

} // namespace NESO::TemplateTest

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
      ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
          evaluation_type,
      ParticleGroupSharedPtr particle_group, Sym<COMPONENT_TYPE> sym,
      const int component, EventStack &event_stack) {

    const ShapeType shape_type = evaluation_type.get_shape_type();
    const int cells_iterset_size = this->map_shape_to_count.at(shape_type);
    if (cells_iterset_size == 0) {
      return;
    }

    auto mpi_rank_dat = particle_group->mpi_rank_dat;

    const auto k_ref_positions =
        (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
            ->cell_dat.device_ptr();

    auto k_output = (*particle_group)[sym]->cell_dat.device_ptr();
    const int k_component = component;

    const auto d_npart_cell = mpi_rank_dat->d_npart_cell;
    const auto k_global_coeffs = this->dh_global_coeffs.d_buffer.ptr;
    const auto k_coeffs_offsets = this->dh_coeffs_offsets.d_buffer.ptr;
    const auto h_nummodes = this->dh_nummodes.h_buffer.ptr;

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

    const int k_ndim = evaluation_type.get_ndim();

    for (int cell_host = 0; cell_host < cells_iterset_size; cell_host++) {
      const int cellx =
          this->map_shape_to_dh_cells.at(shape_type)->h_buffer.ptr[cell_host];
      const INT num_particles = mpi_rank_dat->h_npart_cell[cellx];
      if (num_particles == 0) {
        continue;
      }

      const size_t outer_size =
          get_global_size(static_cast<size_t>(num_particles), local_size);
      // Get the number of modes in x,y and z.
      const int nummodes = h_nummodes[cellx];

      sycl::range<1> cell_iterset_range{outer_size};
      sycl::range<1> local_iterset{local_size};

      auto event_loop = this->sycl_target->queue.submit([&](sycl::handler
                                                                &cgh) {
        sycl::accessor<REAL, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            local_mem(sycl::range<1>(local_mem_num_items), cgh);

        cgh.parallel_for<>(
            sycl::nd_range<1>(cell_iterset_range, local_iterset),
            [=](sycl::nd_item<1> idx) {
              const int idx_local = idx.get_local_id(0);
              const INT layerx = idx.get_global_id(0);

              if (layerx < num_particles) {

                ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
                    loop_type{};

                const REAL *dofs = &k_global_coeffs[k_coeffs_offsets[cellx]];

                REAL xi0, xi1, xi2, eta0, eta1, eta2;
                xi0 = k_ref_positions[cellx][0][layerx];
                if (k_ndim > 1) {
                  xi1 = k_ref_positions[cellx][1][layerx];
                }
                if (k_ndim > 2) {
                  xi2 = k_ref_positions[cellx][2][layerx];
                }

                loop_type.loc_coord_to_loc_collapsed(xi0, xi1, xi2, &eta0,
                                                     &eta1, &eta2);

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
      event_stack.push(event_loop);
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
   * @param bypass_generated Bypass generated evaluation code, default false.
   */
  template <typename U, typename V>
  inline void evaluate(ParticleGroupSharedPtr particle_group, Sym<U> sym,
                       const int component, V &global_coeffs,
                       bool bypass_generated = false) {
    const int num_global_coeffs = global_coeffs.size();
    this->dh_global_coeffs.realloc_no_copy(num_global_coeffs);
    for (int px = 0; px < num_global_coeffs; px++) {
      this->dh_global_coeffs.h_buffer.ptr[px] = global_coeffs[px];
    }
    this->dh_global_coeffs.host_to_device();
    EventStack event_stack{};

    const auto num_modes = this->dh_nummodes.h_buffer.ptr[0];
    if (!this->common_nummodes) {
      bypass_generated = true;
    }
    if (std::is_same_v<U, REAL> != true) {
      bypass_generated = true;
    }

    nprint("HERE");

    typedef std::function<bool(const int, SYCLTargetSharedPtr,
                               ParticleGroupSharedPtr, Sym<REAL>, const int,
                               const int, const REAL *, const int *,
                               const int *, EventStack &)>
        generated_call_exists_t;

    // generated_call_exists
    std::map<ShapeType, generated_call_exists_t> map_shape_to_func;
    map_shape_to_func[eQuadrilateral] =
        GeneratedEvaluation::Quadrilateral::generated_call_exists;
    map_shape_to_func[eTriangle] =
        GeneratedEvaluation::Triangle::generated_call_exists;
    map_shape_to_func[eHexahedron] =
        GeneratedEvaluation::Hexahedron::generated_call_exists;
    map_shape_to_func[ePrism] =
        GeneratedEvaluation::Prism::generated_call_exists;
    map_shape_to_func[eTetrahedron] =
        GeneratedEvaluation::Tetrahedron::generated_call_exists;
    map_shape_to_func[ePyramid] =
        GeneratedEvaluation::Pyramid::generated_call_exists;

    auto lambda_call = [&](const auto shape_type, auto crtp_shape_type) {
      auto func = map_shape_to_func.at(shape_type);
      bool gen_exists = false;
      if (!bypass_generated) {
        const int num_elements = this->map_shape_to_count.at(shape_type);
        gen_exists =
            func(num_modes, particle_group->sycl_target, particle_group, sym,
                 component, num_elements, this->dh_global_coeffs.d_buffer.ptr,
                 this->dh_coeffs_offsets.h_buffer.ptr,
                 this->map_shape_to_dh_cells.at(shape_type)->h_buffer.ptr,
                 event_stack);
      }
      if ((!gen_exists) || bypass_generated) {
        evaluate_inner(crtp_shape_type, particle_group, sym, component,
                       event_stack);
      }
    };

    auto lamba_call_templated = [&](const auto shape_type,
                                    auto evaluation_type_generic,
                                    auto evaluation_type_template) -> bool {
      const int num_elements = this->map_shape_to_count.at(shape_type);
      return TemplateTest::templated_evaluate_wrapper(
          num_modes, evaluation_type_generic, evaluation_type_template,
          particle_group->sycl_target, particle_group, sym, component,
          num_elements, this->dh_global_coeffs.d_buffer.ptr,
          this->dh_coeffs_offsets.h_buffer.ptr,
          this->map_shape_to_dh_cells.at(shape_type)->h_buffer.ptr,
          event_stack);
    };

    if (this->mesh->get_ndim() == 2) {
      nprint("2D");
      const auto shape_type = eQuadrilateral;
      const int num_elements = this->map_shape_to_count.at(shape_type);

      bool templated_ran = lamba_call_templated(
          eQuadrilateral, ExpansionLooping::Quadrilateral{},
          BasisJacobi::Templated::TemplatedQuadrilateral{});

      if (templated_ran) {
        nprint("TEST CASE RAN");
      } else {
        nprint("NOT TEST CASE", num_modes);
        lambda_call(eQuadrilateral, ExpansionLooping::Quadrilateral{});
      }

      // lambda_call(eQuadrilateral, ExpansionLooping::Quadrilateral{});
      lambda_call(eTriangle, ExpansionLooping::Triangle{});
    } else {
      lambda_call(eHexahedron, ExpansionLooping::Hexahedron{});
      lambda_call(ePrism, ExpansionLooping::Prism{});
      lambda_call(eTetrahedron, ExpansionLooping::Tetrahedron{});
      lambda_call(ePyramid, ExpansionLooping::Pyramid{});
    }

    event_stack.wait();
  }
};

} // namespace NESO

#endif
