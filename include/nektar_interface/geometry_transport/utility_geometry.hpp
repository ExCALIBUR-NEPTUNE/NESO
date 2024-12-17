#ifndef _NESO_GEOMETRY_TRANSPORT_UTILITY_GEOMETRY_HPP_
#define _NESO_GEOMETRY_TRANSPORT_UTILITY_GEOMETRY_HPP_

#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <memory>
#include <string>

namespace NESO {

/**
 * @returns True if the geometry object is linear.
 */
template <typename GEOM_TYPE>
inline bool geometry_is_linear(std::shared_ptr<GEOM_TYPE> geom) {
  const auto xmap = geom->GetXmap();
  const int ndim = xmap->GetBase().size();
  for (int dimx = 0; dimx < ndim; dimx++) {
    if (xmap->GetBasisNumModes(dimx) != 2) {
      return false;
    }
  }
  return true;
}

/**
 * @param geom Geometry object.
 * @returns String describing geometry.
 */
template <typename GEOM_TYPE>
inline std::string get_geometry_name(GEOM_TYPE geom) {
  using namespace Nektar::LibUtilities;

  const auto shape_type = geom->GetShapeType();

  if (shape_type == ShapeType::eTriangle) {
    return "Triangle";
  }
  if (shape_type == ShapeType::eQuadrilateral) {
    return "Quadrilateral";
  }
  if (shape_type == ShapeType::eHexahedron) {
    return "Hexahedron";
  }
  if (shape_type == ShapeType::ePrism) {
    return "Prism";
  }
  if (shape_type == ShapeType::eTetrahedron) {
    return "Tetrahedron";
  }
  if (shape_type == ShapeType::ePyramid) {
    return "Pyramid";
  }

  return "Unknown ShapeType";
}

} // namespace NESO

#endif
