#ifndef __LINE_PLANE_INTERSECTION_H_
#define __LINE_PLANE_INTERSECTION_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include "composite_utility.hpp"
#include "nektar_interface/special_functions.hpp"
#include "nektar_interface/typedefs.hpp"
#include <cmath>

namespace NESO::CompositeInteraction {

/**
 * Class to hold the definition of a plane as defined by a 2D linear sided
 * Nektar++ geometry object.
 */
class LinePlaneIntersection {
protected:
  inline REAL distance_from_centre_squared(const REAL r0, const REAL r1,
                                           const REAL r2) const {
    const REAL d0 = point0 - r0;
    const REAL d1 = point1 - r1;
    const REAL d2 = point2 - r2;
    const REAL dd = d0 * d0 + d1 * d1 + d2 * d2;
    return dd;
  }

  REAL radius_squared;

public:
  /// A point in the plane, x component.
  REAL point0;
  /// A point in the plane, y component.
  REAL point1;
  /// A point in the plane, z component.
  REAL point2;
  /// Normal vector for the plane, x component.
  REAL normal0;
  /// Normal vector for the plane, y component.
  REAL normal1;
  /// Normal vector for the plane, z component.
  REAL normal2;

  LinePlaneIntersection() = default;

  /**
   * Create intance from Nektar++ geometry object.
   *
   * @param geom 2D linear sided Nektar++ geometry object.
   */
  template <typename T> LinePlaneIntersection(std::shared_ptr<T> geom) {
    std::vector<REAL> normal;
    get_normal_vector(
        std::static_pointer_cast<SpatialDomains::Geometry2D>(geom), normal);
    this->normal0 = normal.at(0);
    this->normal1 = normal.at(1);
    this->normal2 = normal.at(2);

    // The average of the vertices is a point in the plane.
    get_vertex_average(std::static_pointer_cast<SpatialDomains::Geometry>(geom),
                       normal);

    this->point0 = normal.at(0);
    this->point1 = normal.at(1);
    this->point2 = normal.at(2);

    const int num_verts = geom->GetNumVerts();
    REAL radius_squared_tmp = 0.0;
    for (int vx = 0; vx < num_verts; vx++) {
      auto vertex = geom->GetVertex(vx);
      NekDouble t0, t1, t2;
      vertex->GetCoords(t0, t1, t2);
      radius_squared_tmp = std::max(
          radius_squared_tmp, this->distance_from_centre_squared(t0, t1, t2));
    }
    this->radius_squared = radius_squared_tmp;
  };

  /**
   *  Determine if a test point is close to the geometry object which defines
   *  the plane. This function returns true for all points within the geometry
   *  object and may return true for points outside the geometry object.
   *
   *  @param r0 Input point to test, x component.
   *  @param r1 Input point to test, y component.
   *  @param r2 Input point to test, z component.
   *  @returns True if point is close to original geometry object.
   */
  inline bool point_near_to_geom(const REAL r0, const REAL r1,
                                 const REAL r2) const {
    const REAL dd = this->distance_from_centre_squared(r0, r1, r2);
    return dd <= this->radius_squared;
  }

  /**
   * For a line segment defined by two points p0 and p1 determine if the line
   * segment intersects the plane and compute the intersection point if it
   * exists. Output values are only meaningful if the method returns true.
   *
   * @param[in] p00 First point, x component.
   * @param[in] p01 First point, y component.
   * @param[in] p02 First point, z component.
   * @param[in] p10 Second point, x component.
   * @param[in] p11 Second point, y component.
   * @param[in] p12 Second point, z component.
   * @param[out] i0 Intersection point, x component.
   * @param[out] i1 Intersection point, y component.
   * @param[out] i2 Intersection point, z component.
   * @returns True if the line segment intersects the plane otherwise false.
   */
  inline bool line_segment_intersection(const REAL p00, const REAL p01,
                                        const REAL p02, const REAL p10,
                                        const REAL p11, const REAL p12,
                                        REAL *i0, REAL *i1, REAL *i2) const {
    // direction of line
    const REAL l0 = p10 - p00;
    const REAL l1 = p11 - p01;
    const REAL l2 = p12 - p02;

    // if l_dot_n == 0 then the line is parallel to the plane
    const REAL l_dot_n =
        MAPPING_DOT_PRODUCT_3D(l0, l1, l2, normal0, normal1, normal2);

    const REAL point_plane_m_point_line0 = point0 - p00;
    const REAL point_plane_m_point_line1 = point1 - p01;
    const REAL point_plane_m_point_line2 = point2 - p02;

    const REAL ppmpl_dot_n = MAPPING_DOT_PRODUCT_3D(
        point_plane_m_point_line0, point_plane_m_point_line1,
        point_plane_m_point_line2, normal0, normal1, normal2);

    const REAL d = (l_dot_n == 0) ? 0.0 : ppmpl_dot_n / l_dot_n;

    const REAL intersection0 = p00 + d * l0;
    const REAL intersection1 = p01 + d * l1;
    const REAL intersection2 = p02 + d * l2;

    *i0 = intersection0;
    *i1 = intersection1;
    *i2 = intersection2;

    return (((l_dot_n == 0) && (ppmpl_dot_n != 0)) || (d < 0.0) || (d > 1.0))
               ? false
               : true;
  }
};

} // namespace NESO::CompositeInteraction

#endif
