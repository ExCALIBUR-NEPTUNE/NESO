#ifndef __GEOMETRY_CONTAINER_3D_H__
#define __GEOMETRY_CONTAINER_3D_H__

// Nektar++ Includes
#include <SpatialDomains/MeshGraph.h>

// System includes
#include <iostream>
#include <mpi.h>

using namespace Nektar;
using namespace Nektar::SpatialDomains;
using namespace Nektar::LibUtilities;

#include "geometry_transport_2d.hpp"
#include "geometry_types_3d.hpp"
#include "remote_geom_3d.hpp"
#include "shape_mapping.hpp"
#include "utility_geometry.hpp"

namespace NESO {

/**
 * Struct to hold shared pointers to the different types of 3D geometry
 * objects in terms of classification of shape and type, e.g. Regular,
 * Deformed, linear, non-linear.
 */
class GeometryContainer3D {
protected:
  inline GeometryTypes3D &classify(std::shared_ptr<Geometry3D> &geom) {
    auto g_type = geom->GetMetricInfo()->GetGtype();
    if (g_type == eRegular) {
      return this->regular;
    } else {
      if (geometry_is_linear(geom)) {
        return this->deformed_linear;
      } else {
        return this->deformed_non_linear;
      }
    }
  }

public:
  /// Elements with linear sides that are considered eRegular by Nektar++.
  GeometryTypes3D regular;
  /// Elements with linear sides that are considered eDeformed by Nektar++.
  GeometryTypes3D deformed_linear;
  /// Elements with non-linear sides that are considered eDeformed by Nektar++.
  GeometryTypes3D deformed_non_linear;

  /**
   * Push a geometry object onto the correct container.
   *
   * @param geom Geometry object to push onto correct container.
   */
  inline void push_back(std::pair<int, std::shared_ptr<Geometry3D>> geom) {
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
   * Print information about the number of contained geometry objects.
   */
  inline void print() {
    nprint("Num regular            :", regular.size());
    nprint("\tTetrahedrons:", regular.tet.size());
    nprint("\tPyramid     :", regular.pyr.size());
    nprint("\tPrism       :", regular.prism.size());
    nprint("\tHexahedrons :", regular.hex.size());
    nprint("Num deformed linear    :", deformed_linear.size());
    nprint("\tTetrahedrons:", deformed_linear.tet.size());
    nprint("\tPyramid     :", deformed_linear.pyr.size());
    nprint("\tPrism       :", deformed_linear.prism.size());
    nprint("\tHexahedrons :", deformed_linear.hex.size());
    nprint("Num deformed non-linear:", deformed_non_linear.size());
    nprint("\tTetrahedrons:", deformed_non_linear.tet.size());
    nprint("\tPyramid     :", deformed_non_linear.pyr.size());
    nprint("\tPrism       :", deformed_non_linear.prism.size());
    nprint("\tHexahedrons :", deformed_non_linear.hex.size());
  }
};

} // namespace NESO

#endif
