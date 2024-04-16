#pragma once
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
#include "utility_sycl.hpp"

#include <CL/sycl.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include "projection/device_data.hpp"
#include "projection/project_tpp_2d.hpp"
#include "projection/shapes.hpp"
#include "projection/algorithm_types.hpp"
#include "projection/auto_switch.hpp"
#include "projection/constants.hpp"

using REAL = double;

namespace sycl = cl::sycl;

namespace NESO {

template <typename T>
void fill_device_buffer_and_wait(T *ptr, T val, int size, sycl::queue &queue) {
  sycl::range range(size);
  queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(range, [=](sycl::id<1> id) { ptr[id] = val; });
      })
      .wait();
}


template <typename T> class FunctionProjectBasis : public BasisEvaluateBase<T> {

  template <typename U>
  Project::DeviceData<U>
  get_device_data(ParticleGroupSharedPtr &particle_group, Sym<U> sym, ShapeType const shape_type) {
    auto mpi_rank_dat = particle_group->mpi_rank_dat;
    auto cell_ids =
        this->map_shape_to_dh_cells.at(shape_type)->d_buffer.ptr;
    auto ncell = this->map_shape_to_count.at(shape_type);
    return Project::DeviceData<U>(
        this->dh_global_coeffs.d_buffer.ptr,
        this->dh_coeffs_offsets.d_buffer.ptr, ncell,
        mpi_rank_dat->cell_dat.get_nrow_max(), cell_ids,
        // this->map_shape_to_dh_cells.at(SHAPE::shape_type)->d_buffer.ptr,
        mpi_rank_dat->d_npart_cell,
        (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
            ->cell_dat.device_ptr(),
        (*particle_group)[sym]->cell_dat.device_ptr());
  }

  template <typename Shape, typename COMPONENT_TYPE>
  inline sycl::event project_inner(ParticleGroupSharedPtr particle_group,
                                   Sym<COMPONENT_TYPE> sym,
                                   int const component) {

    ShapeType const shape_type = Shape::shape_type;

    const int cells_iterset_size = this->map_shape_to_count.at(shape_type);
    if (cells_iterset_size == 0) {
      return sycl::event{};
    }

    auto device_data = this->get_device_data<COMPONENT_TYPE>(
        particle_group, sym, shape_type);

    const auto k_nummodes =
        this->dh_nummodes.h_buffer
            .ptr[this->map_shape_to_dh_cells.at(shape_type)->h_buffer.ptr[0]];
    sycl::event event;
    if constexpr (std::is_same<typename Shape::algorithm, Project::ThreadPerCell>::value) {
      AUTO_SWITCH(
          // template param for generated switch/case
          k_nummodes,
          // return value
          event,
          // function to call
          Project::project_tpp,
          // function arguments
          FUNCTION_ARGS(device_data, component, this->sycl_target->queue),
          // Start of constant template arguments
          COMPONENT_TYPE, Project::Constants::alpha, Project::Constants::beta,
          Shape);
    } else {
      AUTO_SWITCH(
          // template param for generated switch/case
          k_nummodes,
          // return value
          event,
          // function to call
          // TODO: Change this the the sync version once written
          Project::project_tpdof,
          // function arguments
          FUNCTION_ARGS(device_data, component, this->sycl_target->queue),
          // Start of constant template arguments
          COMPONENT_TYPE, Project::Constants::alpha, Project::Constants::beta,
          Shape);
    }
    return event;
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

  template <typename U, typename V>
  void project(
      ParticleGroupSharedPtr particle_group, Sym<U> sym,
      int const component, // TODO <component> should be a vector or something
      V &global_coeffs)    // process multiple componants at once
                           // wasteful to do one at a time probably
  {

    this->dh_global_coeffs.realloc_no_copy(global_coeffs.size());
    fill_device_buffer_and_wait(this->dh_global_coeffs.d_buffer.ptr, U(0.0),
                                global_coeffs.size(), this->sycl_target->queue);

    if (this->sycl_target->queue.get_device().is_gpu()) {
      project_inner<Project::eQuad<Project::ThreadPerDof>, U>(particle_group, sym, component)
          .wait();
      project_inner<Project::eTriangle<Project::ThreadPerDof>, U>(particle_group, sym, component)
          .wait();
    } else {
      project_inner<Project::eQuad<Project::ThreadPerCell>, U>(particle_group, sym, component)
          .wait();

      project_inner<Project::eTriangle<Project::ThreadPerCell>, U>(particle_group, sym, component)
          .wait();
    }

    // copyback
    this->sycl_target->queue
        .memcpy(global_coeffs.begin(), this->dh_global_coeffs.d_buffer.ptr,
                global_coeffs.size() * sizeof(U))
        .wait();
  }
};
} // namespace NESO
