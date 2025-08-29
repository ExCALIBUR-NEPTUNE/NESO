#ifndef _NESO_COMPOSITE_INTERACTION_COMPOSITE_FUNCTION_CONTEXT_HPP_
#define _NESO_COMPOSITE_INTERACTION_COMPOSITE_FUNCTION_CONTEXT_HPP_

#include "composite_function.hpp"

#include <string>
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
 * TODO
 */
class CompositeFunctionContext {
protected:
  std::map<int, int> map_composite_label_to_bnd_index;

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
   * Create a surface function over a boundary group.
   *
   * @param boundary_group Boundary group to create function over.
   */
  CompositeFunctionSharedPtr create_function(const int boundary_group);

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
