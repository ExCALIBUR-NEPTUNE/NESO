#ifndef __GEOMETRY_LOCAL_REMOTE_3D_H__
#define __GEOMETRY_LOCAL_REMOTE_3D_H__

#include "remote_geom_3d.hpp"
#include <map>
#include <memory>

// Nektar++ Includes
#include <SpatialDomains/MeshGraph.h>
using namespace Nektar::SpatialDomains;

namespace NESO {

/**
 * Struct to hold local and remote 3D geoms.
 */
struct GeometryLocalRemote3D {
  /// Local geometry objects which are owned by this MPI rank.
  std::map<int, std::shared_ptr<Geometry3D>> local;
  /// Remote geometry objects where a copy is stored on this MPI rank.
  std::vector<std::shared_ptr<RemoteGeom3D>> remote;

  /**
   * Push a geometry object onto the container depending on if the geometry
   * object is local or remote.
   *
   * @param geom Geometry object.
   */
  inline void push_back(std::pair<int, std::shared_ptr<Geometry3D>> geom) {
    this->local[geom.first] = geom.second;
  }

  /**
   * Push a geometry object onto the container depending on if the geometry
   * object is local or remote.
   *
   * @param geom Geometry object.
   */
  inline void push_back(std::shared_ptr<RemoteGeom3D> geom) {
    this->remote.push_back(geom);
  }

  /**
   *  Returns number of remote and local geometry objects.
   */
  inline std::size_t size() { return this->local.size() + this->remote.size(); }
};

} // namespace NESO

#endif
