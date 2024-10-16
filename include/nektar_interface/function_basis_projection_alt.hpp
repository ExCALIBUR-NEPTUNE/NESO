#pragma once
#include <cstdlib>
#include <memory>
#include <neso_particles.hpp>
#include <optional>

#include <LocalRegions/QuadExp.h>
#include <LocalRegions/TriExp.h>
#include <StdRegions/StdExpansion2D.h>

#include "expansion_looping/basis_evaluate_base.hpp"

#include <sycl/sycl.hpp>
#include <string>

#include "projection/algorithm_types.hpp"
#include "projection/auto_switch.hpp"
#include "projection/constants.hpp"
#include "projection/device_data.hpp"
#include "projection/shapes.hpp"

using REAL = double;

namespace NESO {

template <typename T>
void fill_device_buffer_and_wait(T *ptr, T val, int size,
                                 sycl::queue &queue) {
  sycl::range range(size);
  queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(range, [=](sycl::id<1> id) { ptr[id] = val; });
      })
      .wait();
}

template <typename T> class FunctionProjectBasis : public BasisEvaluateBase<T> {

  template <typename U>
  Project::DeviceData<U> get_device_data(ParticleGroupSharedPtr &particle_group,
                                         Sym<U> sym,
                                         ShapeType const shape_type) {
    auto mpi_rank_dat = particle_group->mpi_rank_dat;
    auto cell_ids = this->map_shape_to_dh_cells.at(shape_type)->d_buffer.ptr;
    auto ncell = this->map_shape_to_count.at(shape_type);
    return Project::DeviceData<U>(
        this->dh_global_coeffs.d_buffer.ptr,
        this->dh_coeffs_offsets.d_buffer.ptr, ncell,
        mpi_rank_dat->cell_dat.get_nrow_max(), cell_ids,
        mpi_rank_dat->d_npart_cell,
        (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
            ->cell_dat.device_ptr(),
        (*particle_group)[sym]->cell_dat.device_ptr());
  }

  template <typename Shape, typename U>
  inline sycl::event project_inner(ParticleGroupSharedPtr particle_group,
                                       Sym<U> sym, int const component) {

    ShapeType const shape_type = Shape::shape_type;

    const int cells_iterset_size = this->map_shape_to_count.at(shape_type);
    if (cells_iterset_size == 0) {
      return sycl::event{};
    }

    auto device_data =
        this->get_device_data<U>(particle_group, sym, shape_type);

    const auto k_nummodes =
        this->dh_nummodes.h_buffer
            .ptr[this->map_shape_to_dh_cells.at(shape_type)->h_buffer.ptr[0]];
    std::optional<sycl::event> event;
    AUTO_SWITCH(
        // template param for generated switch/case
        k_nummodes,
        // return value
        event,
        // function to call
        Shape::algorithm::template project,
        // function arguments
        FUNCTION_ARGS(device_data, component, this->sycl_target->queue),
        // template arguments to append to param to switch
        U, Project::Constants::alpha, Project::Constants::beta, Shape);
    // TODO: Do something if project fails i.e. option is empty
    // For example could try run the other algorithm
    return event.value_or(sycl::event());
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
  void
  project(ParticleGroupSharedPtr particle_group, Sym<U> sym,
          int const component, // TODO: <component> should be a vector or
                               // something process multiple componants at once
                               // wasteful to do one at a time probably
          V &global_coeffs, bool force_thread_per_dof = false) {

    this->dh_global_coeffs.realloc_no_copy(global_coeffs.size());
    fill_device_buffer_and_wait(this->dh_global_coeffs.d_buffer.ptr, U(0.0),
                                global_coeffs.size(), this->sycl_target->queue);

    if (this->mesh->get_ndim() == 2) {
      if (this->sycl_target->queue.get_device().is_gpu() ||
          force_thread_per_dof) {
        project_inner<Project::eQuad<Project::ThreadPerDof>, U>(particle_group,
                                                                sym, component)
            .wait();
        project_inner<Project::eTriangle<Project::ThreadPerDof>, U>(
            particle_group, sym, component)
            .wait();
      } else {
        project_inner<Project::eQuad<Project::ThreadPerCell>, U>(particle_group,
                                                                 sym, component)
            .wait();

        project_inner<Project::eTriangle<Project::ThreadPerCell>, U>(
            particle_group, sym, component)
            .wait();
      }
    } else {
      if (this->sycl_target->queue.get_device().is_gpu() ||
          force_thread_per_dof) {
        project_inner<Project::eHex<Project::ThreadPerDof>, U>(particle_group,
                                                               sym, component)
            .wait();
        project_inner<Project::ePrism<Project::ThreadPerDof>, U>(particle_group,
                                                                 sym, component)
            .wait();
        project_inner<Project::eTet<Project::ThreadPerDof>, U>(particle_group,
                                                               sym, component)
            .wait();
        project_inner<Project::ePyramid<Project::ThreadPerDof>, U>(
            particle_group, sym, component)
            .wait();
      } else {
        project_inner<Project::eHex<Project::ThreadPerCell>, U>(particle_group,
                                                                sym, component)
            .wait();
        project_inner<Project::ePrism<Project::ThreadPerCell>, U>(
            particle_group, sym, component)
            .wait();
        project_inner<Project::eTet<Project::ThreadPerCell>, U>(particle_group,
                                                                sym, component)
            .wait();
        project_inner<Project::ePyramid<Project::ThreadPerCell>, U>(
            particle_group, sym, component)
            .wait();
      }
    }

    // copyback
    this->sycl_target->queue
        .memcpy(global_coeffs.begin(), this->dh_global_coeffs.d_buffer.ptr,
                global_coeffs.size() * sizeof(U))
        .wait();
  }
};
} // namespace NESO
