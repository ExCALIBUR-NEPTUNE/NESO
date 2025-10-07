#ifndef _NESO_COMPOSITE_INTERACTION_COMPOSITE_INTERACTION_BOUNDARY_CONDITIONS_HPP_
#define _NESO_COMPOSITE_INTERACTION_COMPOSITE_INTERACTION_BOUNDARY_CONDITIONS_HPP_

#include <SpatialDomains/Conditions.h>
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar;

namespace NESO::CompositeInteraction {

/**
 * Interface class for Nektar++ BoundaryConditions class.
 */
struct CompositeIntersectionBoundaryConditions
    : SpatialDomains::BoundaryConditions {
protected:
  std::map<int, int> map_composite_to_exp_index;
  std::map<int, std::vector<int>> boundary_groups;

public:
  /**
   * Interface class for Nektar++ BoundaryConditions class.
   *
   * @param session Nektar Session instance.
   * @param graph Nektar MeshGraph instance.
   */
  CompositeIntersectionBoundaryConditions(
      const LibUtilities::SessionReaderSharedPtr &session,
      const SpatialDomains::MeshGraphSharedPtr &graph);

  /**
   * @returns The boundary groups map which is identical to the boundary regions
   * specified in the Nektar++ session file.
   */
  std::map<int, std::vector<int>> get_boundary_groups();

  /**
   * Get the expansion index for a composite index.
   *
   * @param composite_index Composite index to retrieve expansion index.
   * @returns Expansion index. Returns -1 if the expansion index is not found.
   */
  int get_expansion_index(const int composite_index);

  /**
   * @returns The MeshGraph passed at construction.
   */
  SpatialDomains::MeshGraphSharedPtr get_mesh_graph();
};

} // namespace NESO::CompositeInteraction

#endif
