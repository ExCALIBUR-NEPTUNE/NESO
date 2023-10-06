///////////////////////////////////////////////////////////////////////////////
//
// Description: Entrypoint for the evaluation benchmark.
//
///////////////////////////////////////////////////////////////////////////////
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <cstdio>
#include <iostream>
#include <map>
#include <mpi.h>
#include <string>

#include "nektar_interface/function_evaluation.hpp"
#include <nektar_interface/equation_system_wrapper.hpp>
#include <nektar_interface/geometry_transport/halo_extension.hpp>
#include <nektar_interface/particle_interface.hpp>
using namespace NESO;

#include <neso_particles.hpp>
using namespace NESO::Particles;

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

int main_evaluation(int argc, char *argv[],
                    LibUtilities::SessionReaderSharedPtr session);

/**
 *  Benchmarking class to evaluate element types separately.
 */
template <typename T>
class BenchmarkEvaluate : public FunctionEvaluateBasis<T> {

protected:
  inline int get_count(ParticleGroupSharedPtr particle_group,
                       const ShapeType shape_type) {

    auto cell_iterset =
        this->map_shape_to_dh_cells.at(shape_type)->h_buffer.ptr;
    const int num_elements = this->map_shape_to_count.at(shape_type);
    int count = 0;
    auto cell_counts = particle_group->mpi_rank_dat->h_npart_cell;

    for (int cx = 0; cx < num_elements; cx++) {
      const auto cell = cell_iterset[cx];
      count += cell_counts[cell];
    }
    return count;
  }

public:
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
  BenchmarkEvaluate(std::shared_ptr<T> field,
                    ParticleMeshInterfaceSharedPtr mesh,
                    CellIDTranslationSharedPtr cell_id_translation)
      : FunctionEvaluateBasis<T>(field, mesh, cell_id_translation) {}

  /**
   * Get the number of particles in each element type. Order:
   * Quadrilateral
   * Triangle
   * Hexahedron
   * Prism
   * Tetrahedron
   * Pyramid
   *
   * @param[in] particle_group ParticleGroup to get particle counts from.
   * @param[out] counts Output counts in the order in the description.
   */
  inline void get_particle_counts(ParticleGroupSharedPtr particle_group,
                                  std::vector<int> &counts) {

    std::vector<ShapeType> shapes = {eQuadrilateral, eTriangle,    eHexahedron,
                                     ePrism,         eTetrahedron, ePyramid};
    int ix = 0;
    for (auto shape : shapes) {
      counts.at(ix) = this->get_count(particle_group, shape);
      ix++;
    }
  }

  /**
   * Get the number of flops required per evaluation per particle for each
   * element type. Order:
   * Quadrilateral
   * Triangle
   * Hexahedron
   * Prism
   * Tetrahedron
   * Pyramid
   *
   * @param[out] counts Flop counts per evaluation.
   */
  inline void get_flop_counts(std::vector<int> &counts) {
    const auto num_modes = this->dh_nummodes.h_buffer.ptr[0];
    NESOASSERT(this->common_nummodes,
               "Mesh has varying mode count across elements.");
    counts.at(0) =
        GeneratedEvaluation::Quadrilateral::get_flop_count(num_modes);
    counts.at(1) = GeneratedEvaluation::Triangle::get_flop_count(num_modes);
    counts.at(2) = GeneratedEvaluation::Hexahedron::get_flop_count(num_modes);
    counts.at(3) = GeneratedEvaluation::Prism::get_flop_count(num_modes);
    counts.at(4) = GeneratedEvaluation::Tetrahedron::get_flop_count(num_modes);
    counts.at(5) = GeneratedEvaluation::Pyramid::get_flop_count(num_modes);
    for (auto cx : counts) {
      NESOASSERT(cx >= 0, "FLOP count requested for a number of modes for "
                          "which there is no generated kernel.");
    }
  }

  /**
   * Initialise the evaluate method.
   *
   * @param global_coeffs source DOFs which are evaluated.
   */
  template <typename V> inline void evaluate_init(V &global_coeffs) {
    const int num_global_coeffs = global_coeffs.size();
    this->dh_global_coeffs.realloc_no_copy(num_global_coeffs);
    for (int px = 0; px < num_global_coeffs; px++) {
      this->dh_global_coeffs.h_buffer.ptr[px] = global_coeffs[px];
    }
    this->dh_global_coeffs.host_to_device();
  }

  /**
   *  @returns Number of modes used for each element
   */
  inline int get_num_modes() { return this->dh_nummodes.h_buffer.ptr[0]; }

  /**
   * Evaluate nektar++ function at particle locations.
   *
   * @param particle_group Source container of particles.
   * @param sym Symbol of ParticleDat within the ParticleGroup.
   * @param component Determine which component of the ParticleDat is
   * the output for function evaluations.
   * @param bypass_generated Bypass generated evaluation code, default false.
   */
  template <typename U>
  inline REAL evaluate_quadrilaterals(ParticleGroupSharedPtr particle_group,
                                      Sym<U> sym, const int component,
                                      bool bypass_generated = false) {

    const auto num_modes = this->dh_nummodes.h_buffer.ptr[0];
    if (!this->common_nummodes) {
      bypass_generated = true;
    }
    if (this->mesh->get_ndim() != 2) {
      return 0.0;
    }
    EventStack event_stack{};

    auto t0 = profile_timestamp();
    bool vector_exists;
    if (!bypass_generated) {
      const int num_elements = this->map_shape_to_count.at(eQuadrilateral);
      vector_exists = GeneratedEvaluation::Quadrilateral::vector_call_exists(
          num_modes, particle_group->sycl_target, particle_group, sym,
          component, num_elements, this->dh_global_coeffs.d_buffer.ptr,
          this->dh_coeffs_offsets.h_buffer.ptr,
          this->map_shape_to_dh_cells.at(eQuadrilateral)->h_buffer.ptr,
          event_stack);
    }
    if ((!vector_exists) || bypass_generated) {
      FunctionEvaluateBasis<T>::evaluate_inner(
          ExpansionLooping::Quadrilateral{}, particle_group, sym, component,
          event_stack);
    }

    event_stack.wait();
    auto t1 = profile_timestamp();
    auto tt = profile_elapsed(t0, t1);
    return tt;
  }

  /**
   * Evaluate nektar++ function at particle locations.
   *
   * @param particle_group Source container of particles.
   * @param sym Symbol of ParticleDat within the ParticleGroup.
   * @param component Determine which component of the ParticleDat is
   * the output for function evaluations.
   * @param bypass_generated Bypass generated evaluation code, default false.
   */
  template <typename U>
  inline REAL evaluate_triangles(ParticleGroupSharedPtr particle_group,
                                 Sym<U> sym, const int component,
                                 bool bypass_generated = false) {

    const auto num_modes = this->dh_nummodes.h_buffer.ptr[0];
    if (!this->common_nummodes) {
      bypass_generated = true;
    }
    if (this->mesh->get_ndim() != 2) {
      return 0.0;
    }
    EventStack event_stack{};

    auto t0 = profile_timestamp();
    bool vector_exists;
    if (!bypass_generated) {
      const int num_elements = this->map_shape_to_count.at(eTriangle);
      vector_exists = GeneratedEvaluation::Triangle::vector_call_exists(
          num_modes, particle_group->sycl_target, particle_group, sym,
          component, num_elements, this->dh_global_coeffs.d_buffer.ptr,
          this->dh_coeffs_offsets.h_buffer.ptr,
          this->map_shape_to_dh_cells.at(eTriangle)->h_buffer.ptr, event_stack);
    }
    if ((!vector_exists) || bypass_generated) {
      FunctionEvaluateBasis<T>::evaluate_inner(ExpansionLooping::Triangle{},
                                               particle_group, sym, component,
                                               event_stack);
    }

    event_stack.wait();
    auto t1 = profile_timestamp();
    auto tt = profile_elapsed(t0, t1);
    return tt;
  }

  /**
   * Evaluate nektar++ function at particle locations.
   *
   * @param particle_group Source container of particles.
   * @param sym Symbol of ParticleDat within the ParticleGroup.
   * @param component Determine which component of the ParticleDat is
   * the output for function evaluations.
   * @param bypass_generated Bypass generated evaluation code, default false.
   */
  template <typename U>
  inline REAL evaluate_hexahedrons(ParticleGroupSharedPtr particle_group,
                                   Sym<U> sym, const int component,
                                   bool bypass_generated = false) {

    const auto num_modes = this->dh_nummodes.h_buffer.ptr[0];
    if (!this->common_nummodes) {
      bypass_generated = true;
    }
    if (this->mesh->get_ndim() != 3) {
      return 0.0;
    }
    EventStack event_stack{};

    auto t0 = profile_timestamp();
    bool vector_exists;
    if (!bypass_generated) {
      const int num_elements = this->map_shape_to_count.at(eHexahedron);
      vector_exists = GeneratedEvaluation::Hexahedron::vector_call_exists(
          num_modes, particle_group->sycl_target, particle_group, sym,
          component, num_elements, this->dh_global_coeffs.d_buffer.ptr,
          this->dh_coeffs_offsets.h_buffer.ptr,
          this->map_shape_to_dh_cells.at(eHexahedron)->h_buffer.ptr,
          event_stack);
    }
    if ((!vector_exists) || bypass_generated) {
      FunctionEvaluateBasis<T>::evaluate_inner(ExpansionLooping::Hexahedron{},
                                               particle_group, sym, component,
                                               event_stack);
    }

    event_stack.wait();
    auto t1 = profile_timestamp();
    auto tt = profile_elapsed(t0, t1);
    return tt;
  }

  /**
   * Evaluate nektar++ function at particle locations.
   *
   * @param particle_group Source container of particles.
   * @param sym Symbol of ParticleDat within the ParticleGroup.
   * @param component Determine which component of the ParticleDat is
   * the output for function evaluations.
   * @param bypass_generated Bypass generated evaluation code, default false.
   */
  template <typename U>
  inline REAL evaluate_prisms(ParticleGroupSharedPtr particle_group, Sym<U> sym,
                              const int component,
                              bool bypass_generated = false) {

    const auto num_modes = this->dh_nummodes.h_buffer.ptr[0];
    if (!this->common_nummodes) {
      bypass_generated = true;
    }
    if (this->mesh->get_ndim() != 3) {
      return 0.0;
    }
    EventStack event_stack{};

    auto t0 = profile_timestamp();
    bool vector_exists;
    if (!bypass_generated) {
      const int num_elements = this->map_shape_to_count.at(ePrism);
      vector_exists = GeneratedEvaluation::Prism::vector_call_exists(
          num_modes, particle_group->sycl_target, particle_group, sym,
          component, num_elements, this->dh_global_coeffs.d_buffer.ptr,
          this->dh_coeffs_offsets.h_buffer.ptr,
          this->map_shape_to_dh_cells.at(ePrism)->h_buffer.ptr, event_stack);
    }
    if ((!vector_exists) || bypass_generated) {
      FunctionEvaluateBasis<T>::evaluate_inner(ExpansionLooping::Prism{},
                                               particle_group, sym, component,
                                               event_stack);
    }

    event_stack.wait();
    auto t1 = profile_timestamp();
    auto tt = profile_elapsed(t0, t1);
    return tt;
  }

  /**
   * Evaluate nektar++ function at particle locations.
   *
   * @param particle_group Source container of particles.
   * @param sym Symbol of ParticleDat within the ParticleGroup.
   * @param component Determine which component of the ParticleDat is
   * the output for function evaluations.
   * @param bypass_generated Bypass generated evaluation code, default false.
   */
  template <typename U>
  inline REAL evaluate_tetrahedrons(ParticleGroupSharedPtr particle_group,
                                    Sym<U> sym, const int component,
                                    bool bypass_generated = false) {

    const auto num_modes = this->dh_nummodes.h_buffer.ptr[0];
    if (!this->common_nummodes) {
      bypass_generated = true;
    }
    if (this->mesh->get_ndim() != 3) {
      return 0.0;
    }
    EventStack event_stack{};

    auto t0 = profile_timestamp();
    bool vector_exists;
    if (!bypass_generated) {
      const int num_elements = this->map_shape_to_count.at(eTetrahedron);
      vector_exists = GeneratedEvaluation::Tetrahedron::vector_call_exists(
          num_modes, particle_group->sycl_target, particle_group, sym,
          component, num_elements, this->dh_global_coeffs.d_buffer.ptr,
          this->dh_coeffs_offsets.h_buffer.ptr,
          this->map_shape_to_dh_cells.at(eTetrahedron)->h_buffer.ptr,
          event_stack);
    }
    if ((!vector_exists) || bypass_generated) {
      FunctionEvaluateBasis<T>::evaluate_inner(ExpansionLooping::Tetrahedron{},
                                               particle_group, sym, component,
                                               event_stack);
    }

    event_stack.wait();
    auto t1 = profile_timestamp();
    auto tt = profile_elapsed(t0, t1);
    return tt;
  }

  /**
   * Evaluate nektar++ function at particle locations.
   *
   * @param particle_group Source container of particles.
   * @param sym Symbol of ParticleDat within the ParticleGroup.
   * @param component Determine which component of the ParticleDat is
   * the output for function evaluations.
   * @param bypass_generated Bypass generated evaluation code, default false.
   */
  template <typename U>
  inline REAL evaluate_pyramids(ParticleGroupSharedPtr particle_group,
                                Sym<U> sym, const int component,
                                bool bypass_generated = false) {

    const auto num_modes = this->dh_nummodes.h_buffer.ptr[0];
    if (!this->common_nummodes) {
      bypass_generated = true;
    }
    if (this->mesh->get_ndim() != 3) {
      return 0.0;
    }
    EventStack event_stack{};

    auto t0 = profile_timestamp();
    bool vector_exists;
    if (!bypass_generated) {
      const int num_elements = this->map_shape_to_count.at(ePyramid);
      vector_exists = GeneratedEvaluation::Pyramid::vector_call_exists(
          num_modes, particle_group->sycl_target, particle_group, sym,
          component, num_elements, this->dh_global_coeffs.d_buffer.ptr,
          this->dh_coeffs_offsets.h_buffer.ptr,
          this->map_shape_to_dh_cells.at(ePyramid)->h_buffer.ptr, event_stack);
    }
    if ((!vector_exists) || bypass_generated) {
      FunctionEvaluateBasis<T>::evaluate_inner(ExpansionLooping::Pyramid{},
                                               particle_group, sym, component,
                                               event_stack);
    }

    event_stack.wait();
    auto t1 = profile_timestamp();
    auto tt = profile_elapsed(t0, t1);
    return tt;
  }
};
