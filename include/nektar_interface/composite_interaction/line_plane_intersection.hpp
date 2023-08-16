#ifndef __LINE_PLANE_INTERSECTION_H_
#define __LINE_PLANE_INTERSECTION_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;
using namespace Nektar::SpatialDomains;
using namespace Nektar::LibUtilities;

#include <neso_particles.hpp>
using namespace NESO::Particles;

#include "nektar_interface/special_functions.hpp"

namespace NESO::CompositeInteraction {

/**
 * Class to hold the definition of a plane as defined by a 2D linear sided
 * Nektar++ geometry object.
 */
struct LinePlaneIntersection {

  REAL point0;
  REAL point1;
  REAL point2;
  REAL normal0;
  REAL normal1;
  REAL normal2;

  /**
   * Create intance from Nektar++ geometry object.
   *
   * @param geom 2D linear sided Nektar++ geometry object.
   */
  template <typename T> LinePlaneIntersection(std::shared_ptr<T> geom) {
    auto shape_type = geom->GetShapeType();
    NESOASSERT(shape_type == eQuadrilateral || shape_type == eTriangle,
               "Plane deduction not implemented for this shape type.");

    PointGeomSharedPtr v0, v1, vlast;
    v0 = geom->GetVertex(0);
    v1 = geom->GetVertex(1);
    if (shape_type == eQuadrilateral) {
      vlast = geom->GetVertex(3);
    } else {
      vlast = geom->GetVertex(2);
    }

    // compute a normal vector for the plane defined by the 2D geom
    PointGeom p0(3, 0, 0.0, 0.0, 0.0);
    PointGeom p1(3, 1, 0.0, 0.0, 0.0);
    PointGeom nx(3, 2, 0.0, 0.0, 0.0);
    p0.Sub(*v1, *v0);
    p1.Sub(*vlast, *v0);
    nx.Mult(p0, p1);
    NekDouble tn0, tn1, tn2, tp0, tp1, tp2;
    nx.GetCoords(tn0, tn1, tn2);
    this->normal0 = tn0;
    this->normal1 = tn1;
    this->normal2 = tn2;
    // v0 is a point in the plane which we use to define the plane
    v0->GetCoords(tp0, tp1, tp2);
    this->point0 = tp0;
    this->point1 = tp1;
    this->point2 = tp2;
  };

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
