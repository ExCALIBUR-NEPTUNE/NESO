#ifndef __SHAPE_MAPPING_H__
#define __SHAPE_MAPPING_H__

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar::SpatialDomains;

/**
 * Get the unqiue integer (cast) that corresponds to a Nektar++ shape type.
 *
 * @param shape_type ShapeType Enum value.
 * @returns Static cast of enum value.
 */
inline constexpr int
shape_type_to_int(Nektar::LibUtilities::ShapeType shape_type) {
  return static_cast<int>(shape_type);
}

/**
 * Get the unqiue Enum (cast) that corresponds to an integer returned form
 * shape_type_to_int.
 *
 * @param type_int Int returned from shape_type_to_int.
 * @returns Static cast to Enum value.
 */
inline constexpr Nektar::LibUtilities::ShapeType
int_to_shape_type(const int type_int) {
  return static_cast<Nektar::LibUtilities::ShapeType>(type_int);
}

#endif
