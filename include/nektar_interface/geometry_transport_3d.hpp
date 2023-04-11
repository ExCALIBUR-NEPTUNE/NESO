#ifndef __GEOMETRY_TRANSPORT_3D_H__
#define __GEOMETRY_TRANSPORT_3D_H__

// Nektar++ Includes
#include <SpatialDomains/MeshGraph.h>

// System includes
#include <iostream>
#include <mpi.h>

using namespace std;
using namespace Nektar;
using namespace Nektar::SpatialDomains;

#include "geometry_transport_2d.hpp"

namespace NESO {

class RemoteGeom3D {
protected:
public:
  /// The remote rank that owns the geometry object (i.e. holds it in its
  /// MeshGraph).
  int rank = -1;
  /// The geometry id on the remote rank.
  int id = -1;
  /// A local copy of the geometry object.
  std::shared_ptr<Nektar::SpatialDomains::Geometry3D> geom;
  /// The underlying Nektar++ shape type
  LibUtilities::ShapeType shape_type;

  /**
   * Wrapper around remote 3D geometry object.
   *
   * @param rank Owning rank.
   * @param id Geometry id on remote rank.
   * @param geom Shared pointer to geometry object.
   */
  template <typename T>
  RemoteGeom3D(const int rank, const int id, std::shared_ptr<T> geom)
      : rank(rank), id(id),
        geom(std::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry3D>(
            geom)),
        shape_type(geom->GetShapeType()){};
};

/**
 * Get all 3D geometry objects from a Nektar++ MeshGraph
 *
 * @param[in] graph MeshGraph instance.
 * @param[in,out] std::map of Nektar++ Geometry3D pointers.
 */
inline void get_all_elements_3d(
    Nektar::SpatialDomains::MeshGraphSharedPtr &graph,
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> &geoms) {
  geoms.clear();

  for (auto &e : graph->GetAllTetGeoms()) {
    geoms[e.first] =
        std::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry3D>(e.second);
  }
  for (auto &e : graph->GetAllPyrGeoms()) {
    geoms[e.first] =
        std::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry3D>(e.second);
  }
  for (auto &e : graph->GetAllPrismGeoms()) {
    geoms[e.first] =
        std::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry3D>(e.second);
  }
  for (auto &e : graph->GetAllHexGeoms()) {
    geoms[e.first] =
        std::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry3D>(e.second);
  }
}

} // namespace NESO

#endif
