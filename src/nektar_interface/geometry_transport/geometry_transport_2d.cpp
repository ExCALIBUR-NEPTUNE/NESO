#include <nektar_interface/geometry_transport/geometry_transport_2d.hpp>

namespace NESO {

/**
 * Get all 2D geometry objects from a Nektar++ MeshGraph
 *
 * @param[in] graph MeshGraph instance.
 * @param[in,out] std::map of Nektar++ Geometry2D pointers.
 */
void get_all_elements_2d(
    Nektar::SpatialDomains::MeshGraphSharedPtr &graph,
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>> &geoms) {
  geoms.clear();

  for (auto &e : graph->GetAllTriGeoms()) {
    geoms[e.first] =
        std::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry2D>(e.second);
  }
  for (auto &e : graph->GetAllQuadGeoms()) {
    geoms[e.first] =
        std::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry2D>(e.second);
  }
}

/**
 * Get a local 2D geometry object from a Nektar++ MeshGraph
 *
 * @param graph Nektar++ MeshGraph to return geometry object from.
 * @returns Local 2D geometry object.
 */
Geometry2DSharedPtr
get_element_2d(Nektar::SpatialDomains::MeshGraphSharedPtr &graph) {
  {
    auto geoms = graph->GetAllQuadGeoms();
    if (geoms.size() > 0) {
      return std::dynamic_pointer_cast<Geometry2D>(geoms.begin()->second);
    }
  }
  auto geoms = graph->GetAllTriGeoms();
  NESOASSERT(geoms.size() > 0, "No local 2D geometry objects found.");
  return std::dynamic_pointer_cast<Geometry2D>(geoms.begin()->second);
}

} // namespace NESO
