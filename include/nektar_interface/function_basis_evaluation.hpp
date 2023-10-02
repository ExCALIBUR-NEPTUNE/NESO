#ifndef __FUNCTION_BASIS_EVALUATION_H_
#define __FUNCTION_BASIS_EVALUATION_H_
#include "coordinate_mapping.hpp"
#include "particle_interface.hpp"
#include <cstdlib>
#include <map>
#include <memory>
#include <neso_particles.hpp>

#include <MultiRegions/DisContField.h>
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

namespace NESO::GeneratedEvaluation::Quadrilateral {

/// The minimum number of modes which implementations are generated for.
inline constexpr int mode_min = 2;
/// The maximum number of modes which implementations are generated for.
inline constexpr int mode_max = 4;

/**
 * TODO
 */
template <size_t NUM_MODES>
inline sycl::vec<REAL, NESO_VECTOR_LENGTH> evaluate(
  const sycl::vec<REAL, NESO_VECTOR_LENGTH> eta0,
  const sycl::vec<REAL, NESO_VECTOR_LENGTH> eta1,
  [[maybe_unused]] const sycl::vec<REAL, NESO_VECTOR_LENGTH> eta2,
  const REAL * dofs
){
  return sycl::vec<REAL, NESO_VECTOR_LENGTH>(0);
}
/**
 * TODO
 */
template <size_t NUM_MODES>
inline int flop_count(){
  return (0);
}

/**
 * TODO
 */
template <>
inline int flop_count<2>(){
  return 19;
}
/**
 * TODO
 * Sympy flop count: 19
 */
template <>
inline sycl::vec<REAL, NESO_VECTOR_LENGTH> evaluate<2>(
  const sycl::vec<REAL, NESO_VECTOR_LENGTH> eta0,
  const sycl::vec<REAL, NESO_VECTOR_LENGTH> eta1,
  [[maybe_unused]] const sycl::vec<REAL, NESO_VECTOR_LENGTH> eta2,
  const REAL * dofs
){
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_0_eta0(0.5*(1 - eta0));
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_1_eta0(0.5*(eta0 + 1));
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_0_eta1(0.5*(1 - eta1));
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_1_eta1(0.5*(eta1 + 1));
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_0(dofs[0]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_1(dofs[1]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_2(dofs[2]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_3(dofs[3]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_0(dof_0*modA_0_eta0*modA_0_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_1(dof_1*modA_0_eta1*modA_1_eta0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_2(dof_2*modA_0_eta0*modA_1_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_3(dof_3*modA_1_eta0*modA_1_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1(eval_eta0_eta1_0 + eval_eta0_eta1_1 + eval_eta0_eta1_2 + eval_eta0_eta1_3);
  return eval_eta0_eta1;
}

/**
 * TODO
 */
template <>
inline int flop_count<3>(){
  return 44;
}
/**
 * TODO
 * Sympy flop count: 44
 */
template <>
inline sycl::vec<REAL, NESO_VECTOR_LENGTH> evaluate<3>(
  const sycl::vec<REAL, NESO_VECTOR_LENGTH> eta0,
  const sycl::vec<REAL, NESO_VECTOR_LENGTH> eta1,
  [[maybe_unused]] const sycl::vec<REAL, NESO_VECTOR_LENGTH> eta2,
  const REAL * dofs
){
  sycl::vec<REAL, NESO_VECTOR_LENGTH> P_0_1_1_eta0(1.0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> x0(eta0 - 1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> x1(eta0 + 1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_0_eta0(-0.5*x0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_1_eta0(0.5*x1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_2_eta0(-0.25*P_0_1_1_eta0*x0*x1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> P_0_1_1_eta1(1.0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> x2(eta1 - 1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> x3(eta1 + 1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_0_eta1(-0.5*x2);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_1_eta1(0.5*x3);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_2_eta1(-0.25*P_0_1_1_eta1*x2*x3);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_0(dofs[0]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_1(dofs[1]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_2(dofs[2]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_3(dofs[3]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_4(dofs[4]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_5(dofs[5]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_6(dofs[6]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_7(dofs[7]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_8(dofs[8]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_0(dof_0*modA_0_eta0*modA_0_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_1(dof_1*modA_0_eta1*modA_1_eta0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_2(dof_2*modA_0_eta1*modA_2_eta0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_3(dof_3*modA_0_eta0*modA_1_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_4(dof_4*modA_1_eta0*modA_1_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_5(dof_5*modA_1_eta1*modA_2_eta0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_6(dof_6*modA_0_eta0*modA_2_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_7(dof_7*modA_1_eta0*modA_2_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_8(dof_8*modA_2_eta0*modA_2_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1(eval_eta0_eta1_0 + eval_eta0_eta1_1 + eval_eta0_eta1_2 + eval_eta0_eta1_3 + eval_eta0_eta1_4 + eval_eta0_eta1_5 + eval_eta0_eta1_6 + eval_eta0_eta1_7 + eval_eta0_eta1_8);
  return eval_eta0_eta1;
}

/**
 * TODO
 */
template <>
inline int flop_count<4>(){
  return 71;
}
/**
 * TODO
 * Sympy flop count: 71
 */
template <>
inline sycl::vec<REAL, NESO_VECTOR_LENGTH> evaluate<4>(
  const sycl::vec<REAL, NESO_VECTOR_LENGTH> eta0,
  const sycl::vec<REAL, NESO_VECTOR_LENGTH> eta1,
  [[maybe_unused]] const sycl::vec<REAL, NESO_VECTOR_LENGTH> eta2,
  const REAL * dofs
){
  sycl::vec<REAL, NESO_VECTOR_LENGTH> P_0_1_1_eta0(1.0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> P_1_1_1_eta0(2.0*eta0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> x0(eta0 - 1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> x1(eta0 + 1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> x2(0.25*x0*x1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_0_eta0(-0.5*x0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_1_eta0(0.5*x1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_2_eta0(-P_0_1_1_eta0*x2);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_3_eta0(-P_1_1_1_eta0*x2);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> P_0_1_1_eta1(1.0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> P_1_1_1_eta1(2.0*eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> x3(eta1 - 1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> x4(eta1 + 1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> x5(0.25*x3*x4);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_0_eta1(-0.5*x3);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_1_eta1(0.5*x4);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_2_eta1(-P_0_1_1_eta1*x5);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> modA_3_eta1(-P_1_1_1_eta1*x5);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_0(dofs[0]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_1(dofs[1]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_2(dofs[2]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_3(dofs[3]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_4(dofs[4]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_5(dofs[5]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_6(dofs[6]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_7(dofs[7]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_8(dofs[8]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_9(dofs[9]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_10(dofs[10]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_11(dofs[11]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_12(dofs[12]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_13(dofs[13]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_14(dofs[14]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> dof_15(dofs[15]);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_0(dof_0*modA_0_eta0*modA_0_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_1(dof_1*modA_0_eta1*modA_1_eta0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_2(dof_2*modA_0_eta1*modA_2_eta0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_3(dof_3*modA_0_eta1*modA_3_eta0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_4(dof_4*modA_0_eta0*modA_1_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_5(dof_5*modA_1_eta0*modA_1_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_6(dof_6*modA_1_eta1*modA_2_eta0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_7(dof_7*modA_1_eta1*modA_3_eta0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_8(dof_8*modA_0_eta0*modA_2_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_9(dof_9*modA_1_eta0*modA_2_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_10(dof_10*modA_2_eta0*modA_2_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_11(dof_11*modA_2_eta1*modA_3_eta0);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_12(dof_12*modA_0_eta0*modA_3_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_13(dof_13*modA_1_eta0*modA_3_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_14(dof_14*modA_2_eta0*modA_3_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1_15(dof_15*modA_3_eta0*modA_3_eta1);
  sycl::vec<REAL, NESO_VECTOR_LENGTH> eval_eta0_eta1(eval_eta0_eta1_0 + eval_eta0_eta1_1 + eval_eta0_eta1_10 + eval_eta0_eta1_11 + eval_eta0_eta1_12 + eval_eta0_eta1_13 + eval_eta0_eta1_14 + eval_eta0_eta1_15 + eval_eta0_eta1_2 + eval_eta0_eta1_3 + eval_eta0_eta1_4 + eval_eta0_eta1_5 + eval_eta0_eta1_6 + eval_eta0_eta1_7 + eval_eta0_eta1_8 + eval_eta0_eta1_9);
  return eval_eta0_eta1;
}
} // namespace NESO::GeneratedEvaluation::Quadrilateral


namespace NESO {


template <size_t NUM_MODES, typename EVALUATE_TYPE, typename COMPONENT_TYPE>
inline void evaluate_inner_per_cell_vector_test(
    SYCLTargetSharedPtr sycl_target,
    ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE> evaluation_type,
    ParticleGroupSharedPtr particle_group, 
    Sym<COMPONENT_TYPE> sym,
    const int component,
    std::map<ShapeType, int> &map_shape_to_count,
    const REAL * k_global_coeffs,
    const int * h_coeffs_offsets,
    const int * h_cells_iterset,
    EventStack &event_stack
  ) {

  const ShapeType shape_type = evaluation_type.get_shape_type();
  const int cells_iterset_size = map_shape_to_count.at(shape_type);
  if (cells_iterset_size == 0) {
    return;
  }

  auto mpi_rank_dat = particle_group->mpi_rank_dat;

  const auto k_ref_positions =
      (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
          ->cell_dat.device_ptr();

  auto k_output = (*particle_group)[sym]->cell_dat.device_ptr();
  const int k_component = component;

  for(int cell_idx=0 ; cell_idx<cells_iterset_size ; cell_idx++){
    const int cellx = h_cells_iterset[cell_idx];
    const int dof_offset = h_coeffs_offsets[cellx];
    const REAL *dofs = k_global_coeffs + dof_offset;

    auto event_loop = sycl_target->queue.submit([&](sycl::handler &cgh) {
      const int num_particles = mpi_rank_dat->h_npart_cell[cellx]; 

      const auto div_mod =
          std::div(static_cast<long long>(num_particles), static_cast<long long>(NESO_VECTOR_LENGTH));
      const std::size_t num_blocks =
          static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1));

      cgh.parallel_for<>(
          sycl::range<1>(static_cast<size_t>(num_blocks)),
          [=](sycl::id<1> idx) {

            const INT layer_start = idx * NESO_VECTOR_LENGTH;
            const INT layer_end = std::min(INT(layer_start + NESO_VECTOR_LENGTH), INT(num_particles));
            ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
                loop_type{};
            const int k_ndim = loop_type.get_ndim();

            REAL eta0_local[NESO_VECTOR_LENGTH];
            REAL eta1_local[NESO_VECTOR_LENGTH];
            REAL eta2_local[NESO_VECTOR_LENGTH];
            REAL eval_local[NESO_VECTOR_LENGTH];
            for(int ix=0 ; ix<NESO_VECTOR_LENGTH ; ix++){
              eta0_local[ix] = 0.0;
              eta1_local[ix] = 0.0;
              eta2_local[ix] = 0.0;
              eval_local[ix] = 0.0;
            }
            int cx = 0;
            for(int ix=layer_start ; ix<layer_end ; ix++){

              REAL xi0, xi1, xi2, eta0, eta1, eta2;
              xi0 = k_ref_positions[cellx][0][ix];
              if (k_ndim > 1) {
                xi1 = k_ref_positions[cellx][1][ix];
              }
              if (k_ndim > 2) {
                xi2 = k_ref_positions[cellx][2][ix];
              }
              loop_type.loc_coord_to_loc_collapsed(xi0, xi1, xi2, &eta0, &eta1,
                                                   &eta2);
              eta0_local[cx] = eta0;
              eta1_local[cx] = eta1;
              eta2_local[cx] = eta2;
              cx++;
            }

            sycl::local_ptr<const REAL> eta0_ptr(eta0_local);
            sycl::local_ptr<const REAL> eta1_ptr(eta1_local);
            sycl::local_ptr<const REAL> eta2_ptr(eta2_local);

            sycl::vec<REAL, NESO_VECTOR_LENGTH> eta0, eta1, eta2;
            eta0.load(0, eta0_ptr);
            eta1.load(0, eta1_ptr);
            eta2.load(0, eta2_ptr);

            const sycl::vec<REAL, NESO_VECTOR_LENGTH> eval = 
              NESO::GeneratedEvaluation::Quadrilateral::evaluate<NUM_MODES>(eta0, eta1, eta2, dofs);

            sycl::local_ptr<REAL> eval_ptr(eval_local);
            eval.store(0, eval_ptr);

            cx = 0;
            for(int ix=layer_start ; ix<layer_end ; ix++){
              k_output[cellx][k_component][ix] = eval_local[cx];
              cx++;
            }
          });
    });

    event_stack.push(event_loop);

  }
  return;
}


template <typename EVALUATE_TYPE, typename COMPONENT_TYPE>
inline bool vector_call_exists(
    const int num_modes,
    SYCLTargetSharedPtr sycl_target,
    ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE> evaluation_type,
    ParticleGroupSharedPtr particle_group, 
    Sym<COMPONENT_TYPE> sym,
    const int component,
    std::map<ShapeType, int> &map_shape_to_count,
    const REAL * k_global_coeffs,
    const int * h_coeffs_offsets,
    const int * h_cells_iterset,
    EventStack &event_stack
  ) {

  if (
      (num_modes >= NESO::GeneratedEvaluation::Quadrilateral::mode_min) && (num_modes <= NESO::GeneratedEvaluation::Quadrilateral::mode_max)
  ) {
    switch(num_modes) {
      case 2:
        evaluate_inner_per_cell_vector_test<2>(
          sycl_target,
          evaluation_type,
          particle_group, 
          sym,
          component,
          map_shape_to_count,
          k_global_coeffs,
          h_coeffs_offsets,
          h_cells_iterset,
          event_stack
        );
        return true;
      case 3:
        evaluate_inner_per_cell_vector_test<3>(
          sycl_target,
          evaluation_type,
          particle_group, 
          sym,
          component,
          map_shape_to_count,
          k_global_coeffs,
          h_coeffs_offsets,
          h_cells_iterset,
          event_stack
        );
        return true;
      case 4:
        evaluate_inner_per_cell_vector_test<4>(
          sycl_target,
          evaluation_type,
          particle_group, 
          sym,
          component,
          map_shape_to_count,
          k_global_coeffs,
          h_coeffs_offsets,
          h_cells_iterset,
          event_stack
        );
        return true;

      default:
        return false;
    }
    return true;
  } else {
    return false;
  }

}

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
    
    for(int cell_host=0 ; cell_host<cells_iterset_size ; cell_host++){
      const int cellx =
        this->map_shape_to_dh_cells.at(shape_type)->h_buffer.ptr[cell_host];
      const INT num_particles = mpi_rank_dat->h_npart_cell[cellx];

      const auto div_mod =
          std::div(static_cast<long long>(num_particles), static_cast<long long>(local_size));
      const std::size_t outer_size =
          static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1));

      sycl::range<1> cell_iterset_range{static_cast<size_t>(outer_size) *
                                          static_cast<size_t>(local_size)};
      sycl::range<1> local_iterset{local_size};

      auto event_loop = this->sycl_target->queue.submit([&](sycl::handler &cgh) {
        sycl::accessor<REAL, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            local_mem(sycl::range<1>(local_mem_num_items), cgh);

        cgh.parallel_for<>(
            sycl::nd_range<1>(cell_iterset_range, local_iterset),
            [=](sycl::nd_item<1> idx) {
              const int idx_local = idx.get_local_id(0);
              const INT layerx = idx.get_global_id(0);

              if (layerx < d_npart_cell[cellx]) {

                ExpansionLooping::JacobiExpansionLoopingInterface<EVALUATE_TYPE>
                    loop_type{};

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
      evaluate_inner(ExpansionLooping::Quadrilateral{},
                     particle_group, sym, component, event_stack);

      evaluate_inner(ExpansionLooping::Triangle{},
                     particle_group, sym, component, event_stack);
    } else {
      evaluate_inner(ExpansionLooping::Hexahedron{},
                     particle_group, sym, component, event_stack);
      evaluate_inner(ExpansionLooping::Pyramid{},
                     particle_group, sym, component, event_stack);
      evaluate_inner(ExpansionLooping::Prism{}, particle_group,
                     sym, component, event_stack);
      evaluate_inner(ExpansionLooping::Tetrahedron{},
                     particle_group, sym, component, event_stack);
    }

    event_stack.wait();
  }


  template <typename U, typename V>
  inline void evaluate_test_init(ParticleGroupSharedPtr particle_group, Sym<U> sym,
                       const int component, V &global_coeffs) {
    const int num_global_coeffs = global_coeffs.size();
    this->dh_global_coeffs.realloc_no_copy(num_global_coeffs);
    for (int px = 0; px < num_global_coeffs; px++) {
      this->dh_global_coeffs.h_buffer.ptr[px] = global_coeffs[px];
    }
    this->dh_global_coeffs.host_to_device();
  }

  template <typename U, typename V>
  void evaluate_test(ParticleGroupSharedPtr particle_group, Sym<U> sym,
                       const int component, V &global_coeffs) {

    const auto num_modes = this->dh_nummodes.h_buffer.ptr[0];

    EventStack event_stack{};

    const bool vector_exists = vector_call_exists(
      num_modes,
      particle_group->sycl_target,
      ExpansionLooping::Quadrilateral{},
      particle_group, 
      sym,
      component,
      this->map_shape_to_count,
      this->dh_global_coeffs.d_buffer.ptr,
      this->dh_coeffs_offsets.h_buffer.ptr,
      this->map_shape_to_dh_cells.at(eQuadrilateral)->h_buffer.ptr,
      event_stack
    );
    
    if (!vector_exists){
      evaluate_inner(ExpansionLooping::Quadrilateral{},
                     particle_group, sym, component, event_stack);
    }

    event_stack.wait();

  }
};

} // namespace NESO

#endif
