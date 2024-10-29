#ifndef _NESO_NEKTAR_INTERFACE_FUNCTION_BASIS_PROJECTION_ALT_HPP
#define _NESO_NEKTAR_INTERFACE_FUNCTION_BASIS_PROJECTION_ALT_HPP
#include <cstdlib>
#include <memory>
#include <neso_particles.hpp>
#include <optional>

#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LocalRegions/QuadExp.h>
#include <LocalRegions/TriExp.h>
#include <StdRegions/StdExpansion2D.h>

#include "expansion_looping/basis_evaluate_base.hpp"

#include "projection/algorithm_types.hpp"
#include "projection/device_data.hpp"
#include "projection/shapes.hpp"
#include <neso_constants.hpp>
#include <string>
#include <sycl_typedefs.hpp>
#include <utilities/static_case.hpp>
namespace NESO {

// Convert to

template <typename T> class FunctionProjectBasis : public BasisEvaluateBase<T> {
  template <typename Shape> auto constexpr get_nektar_shape_type() {
    using alg = typename Shape::algorithm;
    if constexpr (std::is_same<Shape, Project::eQuad<alg>>::value) {
      return Nektar::LibUtilities::eQuadrilateral;
    } else if constexpr (std::is_same<Shape, Project::eTriangle<alg>>::value) {
      return Nektar::LibUtilities::eTriangle;
    } else if constexpr (std::is_same<Shape, Project::eTet<alg>>::value) {
      return Nektar::LibUtilities::eTetrahedron;
    } else if constexpr (std::is_same<Shape, Project::ePyramid<alg>>::value) {
      return Nektar::LibUtilities::ePyramid;
    } else if constexpr (std::is_same<Shape, Project::ePrism<alg>>::value) {
      return Nektar::LibUtilities::ePrism;
    } else if constexpr (std::is_same<Shape, Project::eHex<alg>>::value) {
      return Nektar::LibUtilities::eHexahedron;
    } else {
      static_assert(false, "Unsupported shape type");
      return Nektar::LibUtilities::eNoShapeType;
    }
  }
  template <typename U>
  Project::DeviceData<U>
  get_device_data(ParticleGroupSharedPtr &particle_group, Sym<U> sym,
                  Nektar::LibUtilities::ShapeType const shape_type) {
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

    auto const shape_type = get_nektar_shape_type<Shape>();

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
    Utilities::static_case<Constants::min_nummodes, Constants::max_nummodes>(
        k_nummodes, [&](auto I) {
          event = Shape::algorithm::template project<I, U, Constants::alpha,
                                                     Constants::beta, Shape>(
              device_data, component, this->sycl_target->queue);
        });
    NESOASSERT(event.has_value(), "Projection Failed");
    return event.value();
  }

  template <typename U, typename V, typename Alg>
  void dispatch_to_alg(
      ParticleGroupSharedPtr &particle_group, Sym<U> sym,
      int const component, // TODO: <component> should be a vector or
                           // something process multiple componants at once
                           // wasteful to do one at a time probably
      V &global_coeffs) {
    if (this->mesh->get_ndim() == 2) {
      project_inner<Project::eQuad<Alg>, U>(particle_group, sym, component)
          .wait();
      project_inner<Project::eTriangle<Alg>, U>(particle_group, sym, component)
          .wait();
    } else {
      project_inner<Project::eHex<Alg>, U>(particle_group, sym, component)
          .wait();
      project_inner<Project::ePrism<Alg>, U>(particle_group, sym, component)
          .wait();
      project_inner<Project::eTet<Alg>, U>(particle_group, sym, component)
          .wait();
      project_inner<Project::ePyramid<Alg>, U>(particle_group, sym, component)
          .wait();
    }
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
  project(ParticleGroupSharedPtr &particle_group, Sym<U> sym,
          int const component, // TODO: <component> should be a vector or
                               // something process multiple componants at once
                               // wasteful to do one at a time probably
          V &global_coeffs, bool force_thread_per_dof = false) {

    this->dh_global_coeffs.realloc_no_copy(global_coeffs.size());

    this->sycl_target->queue
        .fill(this->dh_global_coeffs.d_buffer.ptr, U(0.0), global_coeffs.size())
        .wait();

    if (this->sycl_target->queue.get_device().is_gpu() ||
        force_thread_per_dof) {
      dispatch_to_alg<U, V, Project::ThreadPerDof>(particle_group, sym,
                                                   component, global_coeffs);
    } else {
      dispatch_to_alg<U, V, Project::ThreadPerCell>(particle_group, sym,
                                                    component, global_coeffs);
    }

    // copyback
    this->sycl_target->queue
        .memcpy(global_coeffs.begin(), this->dh_global_coeffs.d_buffer.ptr,
                global_coeffs.size() * sizeof(U))
        .wait();
  }
};
} // namespace NESO
#endif
