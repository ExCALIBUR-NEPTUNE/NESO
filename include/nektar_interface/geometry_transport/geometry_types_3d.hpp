#ifndef __GEOMETRY_TYPES_3D_H__
#define __GEOMETRY_TYPES_3D_H__

#include "geometry_local_remote_3d.hpp"
#include "remote_geom_3d.hpp"
#include <map>
#include <memory>

// Nektar++ Includes
#include <SpatialDomains/MeshGraph.h>
using namespace Nektar::SpatialDomains;
using namespace Nektar::LibUtilities;

#include <neso_particles.hpp>
using namespace NESO::Particles;

namespace NESO {

/**
 * Struct to hold shared pointers to the different types of 3D geometry
 * objects in terms of classification of shape.
 */
class GeometryTypes3D {
protected:
  inline GeometryLocalRemote3D &classify(std::shared_ptr<Geometry3D> &geom) {

    auto shape_type = geom->GetShapeType();
    if (shape_type == eTetrahedron) {
      return this->tet;
    } else if (shape_type == ePyramid) {
      return this->pyr;
    } else if (shape_type == ePrism) {
      return this->prism;
    } else if (shape_type == eHexahedron) {
      return this->hex;
    } else {
      NESOASSERT(false, "could not classify geometry type");
      return this->tet; // supresses warnings, unreachable.
    }
  }

public:
  /// Store of local and remote tetrahedrons.
  GeometryLocalRemote3D tet;
  /// Store of local and remote pyramids.
  GeometryLocalRemote3D pyr;
  /// Store of local and remote prism.
  GeometryLocalRemote3D prism;
  /// Store of local and remote hexahedrons.
  GeometryLocalRemote3D hex;

  /**
   * Push a geometry object onto the correct container.
   *
   * @param geom Geometry object to push onto correct container.
   */
  inline void push_back(std::pair<int, std::shared_ptr<Geometry3D>> &geom) {
    auto &container = this->classify(geom.second);
    container.push_back(geom);
  }
  /**
   * Push a geometry object onto the correct container.
   *
   * @param geom Geometry object to push onto correct container.
   */
  inline void push_back(std::shared_ptr<RemoteGeom3D> &geom) {
    auto &container = this->classify(geom->geom);
    container.push_back(geom);
  }

  /**
   *  @returns Number of elements across all types.
   */
  inline std::size_t size() {
    return this->tet.size() + this->pyr.size() + this->prism.size() +
           this->hex.size();
  }
};

} // namespace NESO

#endif
