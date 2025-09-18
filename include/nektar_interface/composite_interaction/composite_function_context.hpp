#ifndef _NESO_COMPOSITE_INTERACTION_COMPOSITE_FUNCTION_CONTEXT_HPP_
#define _NESO_COMPOSITE_INTERACTION_COMPOSITE_FUNCTION_CONTEXT_HPP_

#include "composite_function.hpp"

#include <vector>

#include <MultiRegions/DisContField.h>
#include <MultiRegions/ExpList.h>
using namespace Nektar;

#include <neso_particles.hpp>
using namespace NESO::Particles;

namespace NESO::CompositeInteraction {

/**
 * Construct the map from composite labels to boundary expansion indices.
 *
 * @param graph The mesh the field is defined over.
 * @param boundary_groups The boundary groups used for particle-composite
 * interactions.
 * @param dis_cont_field Prototype field to create boundary functions from.
 * @returns Map from composite label to index in boundary expansions in the
 * prototype field.
 */
std::map<int, int> get_map_composite_label_to_bnd_exp_index(
    SpatialDomains::MeshGraphSharedPtr graph,
    std::map<int, std::vector<int>> &boundary_groups,
    MultiRegions::DisContFieldSharedPtr dis_cont_field);

/**
 * Type which holds the state and methods for evaluation/projection onto surface
 * functions.
 */
class CompositeFunctionContext {
protected:
  std::shared_ptr<BufferDevice<REAL>> d_coeffs_pnm10;
  std::shared_ptr<BufferDevice<REAL>> d_coeffs_pnm11;
  std::shared_ptr<BufferDevice<REAL>> d_coeffs_pnm2;
  int stride_n;

public:
  /// Disable (implicit) copies.
  CompositeFunctionContext(const CompositeFunctionContext &st) = delete;
  /// Disable (implicit) copies.
  CompositeFunctionContext &
  operator=(CompositeFunctionContext const &a) = delete;
  ~CompositeFunctionContext() = default;

  /// Compute device the function is stored on.
  SYCLTargetSharedPtr sycl_target;
  /// The mesh functions are created on the boundary of.
  SpatialDomains::MeshGraphSharedPtr graph;
  /// The expansions that define the function.
  MultiRegions::DisContFieldSharedPtr prototype_field;
  /// The boundary groups functions may be created on
  std::map<int, std::vector<int>> boundary_groups;
  /// Map from composite labels to the boundary group.
  std::map<int, int> map_composite_label_to_bnd_index;
  /// Map from shape type to the number of modes in the expansion.
  std::map<int, int> map_shape_type_to_num_modes;
  /// Map from shape type to the total number of modes in each dimension.
  std::map<int, std::array<int, 2>> map_shape_type_to_total_num_modes;
  /// Map from shape type to the sum of the total number of modes in each
  /// dimension.
  std::map<int, int> map_shape_type_to_sum_total_num_modes;
  /// Maximum of sum of total num modes. Used as stride between expansions.
  int max_num_dofs{0};

  /**
   * Create surface function context over the specified boundary groups.
   *
   * @param sycl_target Compute device for created functions.
   * @param graph Underlying mesh functions are created on the boundary of.
   * @param prototype_field Prototype field with boundary expansions that define
   * the function space of the boundary functions.
   * @param boundary_groups The boundary groups functions can be created on.
   */
  CompositeFunctionContext(SYCLTargetSharedPtr sycl_target,
                           SpatialDomains::MeshGraphSharedPtr graph,
                           MultiRegions::DisContFieldSharedPtr prototype_field,
                           std::map<int, std::vector<int>> boundary_groups);

  /**
   * Create a surface function over a boundary group. Must be called
   * collectively on the communicator.
   *
   * @param boundary_group Boundary group to create function over.
   */
  CompositeFunctionSharedPtr create_function(const int boundary_group);

  /**
   * Performs the initialisation of the RHS of the mass matrix solve. Collective
   * on the communicator.
   *
   * @param func Function to project onto.
   * @param boundary_mesh_interface BoundaryMeshInterface for boundary group.
   */
  void function_project_initialise(
      CompositeFunctionSharedPtr func,
      std::shared_ptr<BoundaryMeshInterface> boundary_mesh_interface);

  /**
   * Add particle data onto the RHS of the mass matrix solve in the projection.
   *
   * @param particle_sub_group ParticleSubGroup to project onto function.
   * @param sym Sym<REAL> Particle property to use as source weights.
   * @param component Component of particle property to use as source weights.
   * @param is_ephemeral Indicate if the particle weights are in an EphemeralDat
   * or ParticleDat.
   * @param func Function to project onto.
   * @param boundary_mesh_interface BoundaryMeshInterface for boundary group.
   */
  void function_project_contribute(
      ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
      const int component, const bool is_ephemeral,
      CompositeFunctionSharedPtr func,
      std::shared_ptr<BoundaryMeshInterface> boundary_mesh_interface);

  /**
   * Performs the reduction of the RHS of the mass matrix solve. Collective on
   * the communicator.
   *
   * @param func Function to project onto.
   * @param boundary_mesh_interface BoundaryMeshInterface for boundary group.
   */
  void function_project_finalise_reduce(
      CompositeFunctionSharedPtr func,
      std::shared_ptr<BoundaryMeshInterface> boundary_mesh_interface);

  /**
   * Performs the mass matrix solve of the projection. Collective on the
   * communicator.
   *
   * @param func Function to project onto.
   * @param boundary_mesh_interface BoundaryMeshInterface for boundary group.
   */
  void function_project_finalise_mass_solve(
      CompositeFunctionSharedPtr func,
      std::shared_ptr<BoundaryMeshInterface> boundary_mesh_interface);

  /**
   * Finalises the projection by reducing the RHS of the mass matrix solve then
   * performing the mass-matrix solve. Equivalent to calling
   *
   * function_project_finalise_reduce(func, boundary_mesh_interface);
   * function_project_finalise_mass_solve(func, boundary_mesh_interface);
   *
   * Collective on the communicator.
   *
   * @param func Function to project onto.
   * @param boundary_mesh_interface BoundaryMeshInterface for boundary group.
   */
  void function_project_finalise(
      CompositeFunctionSharedPtr func,
      std::shared_ptr<BoundaryMeshInterface> boundary_mesh_interface);

  /**
   * Project particle data onto a function defined on the surface. Uses the
   * standardarised boundary interface on the sub group.
   *
   * function_project_initialise(func);
   * function_project_contribute(particle_sub_group, sym, component,
   *                             is_ephemeral, func, boundary_mesh_interface);
   * function_project_finalise(func);
   *
   * Must be called collectively on the communicator.
   *
   * @param particle_sub_group ParticleSubGroup to project onto function.
   * @param sym Sym<REAL> Particle property to use as source weights.
   * @param component Component of particle property to use as source weights.
   * @param is_ephemeral Indicate if the particle weights are in an EphemeralDat
   * or ParticleDat.
   * @param func Function to project onto.
   * @param boundary_mesh_interface BoundaryMeshInterface for boundary group.
   */
  void function_project(
      ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
      const int component, const bool is_ephemeral,
      CompositeFunctionSharedPtr func,
      std::shared_ptr<BoundaryMeshInterface> boundary_mesh_interface);

  /**
   * Evaluate particle data from a function defined on the surface. Uses the
   * standardarised boundary interface on the sub group.
   *
   * @param particle_sub_group ParticleSubGroup to containing destination
   * particles for evaluation.
   * @param sym Sym<REAL> Particle property to overwrite with function
   * evaluations.
   * @param component Component of particle property to write evalauations to.
   * @param is_ephemeral Indicate if the particle evaluations are in an
   * EphemeralDat or ParticleDat.
   * @param func Function to evaluate at particle locations.
   * @param boundary_mesh_interface BoundaryMeshInterface for boundary group.
   */
  void function_evaluate(
      ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
      const int component, const bool is_ephemeral,
      CompositeFunctionSharedPtr func,
      std::shared_ptr<BoundaryMeshInterface> boundary_mesh_interface);

  /**
   * Get the owned geometry objects for a boundary group.
   *
   * @param boundary_group Boundary group to get owned geometry objects for.
   * @returns Vector of owned geometry ids.
   */
  std::vector<INT> get_owned_geoms(const int boundary_group);
};

} // namespace NESO::CompositeInteraction

#endif
