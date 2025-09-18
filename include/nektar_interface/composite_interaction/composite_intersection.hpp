#ifndef __COMPOSITE_INTERSECTION_H_
#define __COMPOSITE_INTERSECTION_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <nektar_interface/geometry_transport/packed_geom_2d.hpp>
#include <nektar_interface/particle_cell_mapping/x_map_newton_kernel.hpp>
#include <nektar_interface/particle_mesh_interface.hpp>
#include <nektar_interface/special_functions.hpp>
#include <nektar_interface/typedefs.hpp>

#include "composite_collections.hpp"
#include "composite_function.hpp"
#include "composite_function_context.hpp"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace NESO::CompositeInteraction {

/**
 *  High-level class to detect and compute the intersection of a particle
 *  trajectory and a Nektar++ composite.
 */
class CompositeIntersection {
protected:
  const int ndim;
  const int num_cells;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  std::unique_ptr<BufferDevice<int>> d_cell_min_maxes;
  std::unique_ptr<MeshHierarchyMapper> mesh_hierarchy_mapper;
  std::unique_ptr<BufferDeviceHost<int>> dh_max_bounding_box_size;
  std::unique_ptr<BufferDeviceHost<INT>> dh_mh_cells;
  std::unique_ptr<BufferDeviceHost<int>> dh_mh_cells_index;
  /// Tolerance for line-line intersections
  REAL line_intersection_tol;
  /// Exit tolerance for Newton iteration.
  REAL newton_tol;
  /// Maximum number of Newton iterations.
  INT newton_max_iteration;
  /// Tolerance to determine if a reference point is in [-1, 1]
  REAL contained_tol;
  /// Modifier for grid size in reference space.
  int num_modes_factor;

  template <typename T> inline void check_iteration_set(std::shared_ptr<T>) {
    static_assert(std::is_same_v<T, ParticleGroup> ||
                  std::is_same_v<T, ParticleSubGroup>);
  }

  template <typename T>
  void find_cells(std::shared_ptr<T> iteration_set, std::set<INT> &cells);

  template <typename T>
  void find_intersections_2d(std::shared_ptr<T> iteration_set, REAL *d_real,
                             INT *d_int);
  template <typename T>
  void find_intersections_3d(std::shared_ptr<T> iteration_set, REAL *d_real,
                             INT *d_int);

  // Composite function members
  std::map<int, std::shared_ptr<UnseenValueExtractor>>
      map_groups_unseen_value_extractor;

  /// The map from boundary groups to boundary mesh interfaces.
  std::map<int, std::shared_ptr<BoundaryMeshInterface>>
      map_groups_boundary_interface;

public:
  /// The CompositeCollections used to detect intersections.
  std::shared_ptr<CompositeCollections> composite_collections;
  /// Disable (implicit) copies.
  CompositeIntersection(const CompositeIntersection &st) = delete;
  /// Disable (implicit) copies.
  CompositeIntersection &operator=(CompositeIntersection const &a) = delete;

  /// SYCLTarget to use for computation.
  SYCLTargetSharedPtr sycl_target;

  /// The NESO::Particles Sym<REAL> used to store the previous particle
  /// position.
  const static inline Sym<REAL> previous_position_sym =
      Sym<REAL>("NESO_COMP_INT_PREV_POS");

  /// Map from boundary group id to composites in the group.
  std::map<int, std::vector<int>> boundary_groups;

  /// Prototype field for boundary functions.
  MultiRegions::DisContFieldSharedPtr prototype_field;

  /// The context used to create, project and evaluate boundary functions.
  std::shared_ptr<CompositeFunctionContext> composite_function_context;

  /**
   * Free the intersection object. Must be called collectively on the
   * communicator.
   */
  void free();

  /**
   *  Create a new intersection object for a compute device, mesh and vector of
   *  composite indices.
   *
   *  @param sycl_target Compute device to find intersections on.
   *  @param particle_mesh_interface Mesh interface all particle groups will be
   *  based on.
   *  @param boundary_groups Map from boundary group id to composite ids which
   *  form the group.
   *  @param prototype_field Prototype function/field to use for creating
   * functions on boundaries.
   *  @param config Optional configuration for intersection algorithms, e.g.
   *  Newton iterations.
   */
  CompositeIntersection(
      SYCLTargetSharedPtr sycl_target,
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      std::map<int, std::vector<int>> boundary_groups,
      MultiRegions::DisContFieldSharedPtr prototype_field,
      ParameterStoreSharedPtr config = std::make_shared<ParameterStore>());

  /**
   *  Create a new intersection object for a compute device, mesh and vector of
   *  composite indices.
   *
   *  @param sycl_target Compute device to find intersections on.
   *  @param particle_mesh_interface Mesh interface all particle groups will be
   *  based on.
   *  @param boundary_groups Map from boundary group id to composite ids which
   *  form the group.
   *  @param config Optional configuration for intersection algorithms, e.g.
   *  Newton iterations.
   */
  CompositeIntersection(
      SYCLTargetSharedPtr sycl_target,
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      std::map<int, std::vector<int>> boundary_groups,
      ParameterStoreSharedPtr config = std::make_shared<ParameterStore>());

  /**
   *  Method to store the current particle positions before an integration step.
   *
   *  @param iteration_set Particles to store current positions of.
   *  @param output_sym_composite_name Optionally specifiy the property to
   *  store composite intersection information in. Otherwise use the default.
   */
  template <typename T> void pre_integration(std::shared_ptr<T> iteration_set);

  /**
   *  Find intersections between particle trajectories and composites.
   *
   * @param iteration_set ParticleGroup or ParticleSubGroup which defines the
   * set of particles.
   * @param output_sym_composite Optionally place the information of which
   * composite is hit into a different particle dat.
   * @param output_sym_position Optionally place the information of where the
   * intersection occurred in a different particle dat.
   * @returns Map from composite indices to a ParticleSubGroup containing
   * particles which have a trajectory that intersected the composite.
   */
  template <typename T>
  std::map<int, ParticleSubGroupSharedPtr>
  get_intersections(std::shared_ptr<T> iteration_set);

  /**
   * Create a function on a boundary group.
   *
   * @param group ID of boundary group to create function on.
   * @returns Function object on boundary.
   */
  CompositeFunctionSharedPtr create_function(const int group);

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
   */
  void function_evaluate(ParticleSubGroupSharedPtr particle_sub_group,
                         Sym<REAL> sym, const int component,
                         const bool is_ephemeral,
                         CompositeFunctionSharedPtr func);

  /**
   * Performs the initialisation of the RHS of the mass matrix solve. Collective
   * on the communicator.
   *
   * @param func Function to project onto.
   */
  void function_project_initialise(CompositeFunctionSharedPtr func);

  /**
   * Add particle data onto the RHS of the mass matrix solve in the projection.
   *
   * @param particle_sub_group ParticleSubGroup to project onto function.
   * @param sym Sym<REAL> Particle property to use as source weights.
   * @param component Component of particle property to use as source weights.
   * @param is_ephemeral Indicate if the particle weights are in an EphemeralDat
   * or ParticleDat.
   * @param func Function to project onto.
   */
  void function_project_contribute(ParticleSubGroupSharedPtr particle_sub_group,
                                   Sym<REAL> sym, const int component,
                                   const bool is_ephemeral,
                                   CompositeFunctionSharedPtr func);

  /**
   * Performs the reduction of the RHS of the mass matrix solve. Collective on
   * the communicator.
   *
   * @param func Function to project onto.
   */
  void function_project_finalise_reduce(CompositeFunctionSharedPtr func);

  /**
   * Performs the mass matrix solve of the projection. Collective on the
   * communicator.
   *
   * @param func Function to project onto.
   */
  void function_project_finalise_mass_solve(CompositeFunctionSharedPtr func);

  /**
   * Finalises the projection by reducing the RHS of the mass matrix solve then
   * performing the mass-matrix solve. Equivalent to calling
   *
   * function_project_finalise_reduce(func);
   * function_project_finalise_mass_solve(func);
   *
   * Collective on the communicator.
   *
   * @param func Function to project onto.
   */
  void function_project_finalise(CompositeFunctionSharedPtr func);

  /**
   * Project particle data onto a function defined on the surface. Uses the
   * standardarised boundary interface on the sub group.
   *
   * function_project_initialise(func);
   * function_project_contribute(particle_sub_group, sym, component,
   *                             is_ephemeral, func);
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
   */
  void function_project(ParticleSubGroupSharedPtr particle_sub_group,
                        Sym<REAL> sym, const int component,
                        const bool is_ephemeral,
                        CompositeFunctionSharedPtr func);

  /**
   * Get the BoundaryMeshInterface instance for a boundary group.
   *
   * @param group Boundary group to get mesh interface for.
   * @returns BoundaryMeshInterface for passed group.
   */
  std::shared_ptr<BoundaryMeshInterface>
  get_boundary_mesh_interface(const int group);
};

extern template void
CompositeIntersection::find_cells(std::shared_ptr<ParticleGroup> iteration_set,
                                  std::set<INT> &cells);

extern template void CompositeIntersection::find_intersections_2d(
    std::shared_ptr<ParticleGroup> iteration_set, REAL *d_real, INT *d_int);

extern template void CompositeIntersection::find_intersections_3d(
    std::shared_ptr<ParticleGroup> iteration_set, REAL *d_real, INT *d_int);

extern template void CompositeIntersection::pre_integration(
    std::shared_ptr<ParticleGroup> iteration_set);

extern template std::map<int, ParticleSubGroupSharedPtr>
CompositeIntersection::get_intersections(
    std::shared_ptr<ParticleGroup> iteration_set);

extern template void CompositeIntersection::find_cells(
    std::shared_ptr<ParticleSubGroup> iteration_set, std::set<INT> &cells);

extern template void CompositeIntersection::find_intersections_2d(
    std::shared_ptr<ParticleSubGroup> iteration_set, REAL *d_real, INT *d_int);

extern template void CompositeIntersection::find_intersections_3d(
    std::shared_ptr<ParticleSubGroup> iteration_set, REAL *d_real, INT *d_int);

extern template void CompositeIntersection::pre_integration(
    std::shared_ptr<ParticleSubGroup> iteration_set);

extern template std::map<int, ParticleSubGroupSharedPtr>
CompositeIntersection::get_intersections(
    std::shared_ptr<ParticleSubGroup> iteration_set);

} // namespace NESO::CompositeInteraction

#endif
