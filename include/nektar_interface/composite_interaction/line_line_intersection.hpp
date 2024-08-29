#ifndef __NEKTAR_INTERFACE_COMPOSITE_INTERSECTION_LINE_LINE_INTERSECTION_H_
#define __NEKTAR_INTERFACE_COMPOSITE_INTERSECTION_LINE_LINE_INTERSECTION_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include "nektar_interface/special_functions.hpp"
#include "nektar_interface/typedefs.hpp"
#include <cmath>

namespace NESO::CompositeInteraction {

/**
 * Class to determine intersections between SegGeoms and particle trajectories.
 */
class LineLineIntersection {
protected:
public:
  /// Point a, x component.
  REAL ax;
  /// Point a, y component.
  REAL ay;
  /// Point b, x component.
  REAL bx;
  /// Point b, y component.
  REAL by;
  /// Normal to SegGeom x component.
  REAL normalx;
  /// Normal to SegGeom y component.
  REAL normaly;
  /// Tolerance for intersection detection.
  REAL tol = 0.0;

  LineLineIntersection() = default;

  /**
   * Create a Line-Line intersection detection object from a SegGeom.
   *
   * @param geom SegGeom to detect particle trajectories with.
   * @param tol Optional tolerance for intersection detection.
   */
  template <typename T>
  LineLineIntersection(std::shared_ptr<T> geom, const REAL tol = 0.0)
      : tol(tol) {
    auto shape_type = geom->GetShapeType();
    NESOASSERT(shape_type == LibUtilities::eSegment,
               "LineLineIntersection not implemented for this shape type.");
    NESOASSERT(geom->GetNumVerts() == 2, "Unexpected number of vertices.");

    auto a = geom->GetVertex(0);
    NekDouble x, y, z;
    a->GetCoords(x, y, z);
    NESOASSERT(std::abs(z) < 1.0e14, "Expected z to be 0");
    this->ax = x;
    this->ay = y;
    auto b = geom->GetVertex(1);
    b->GetCoords(x, y, z);
    NESOASSERT(std::abs(z) < 1.0e14, "Expected z to be 0");
    this->bx = x;
    this->by = y;

    const REAL dx = this->bx - this->ax;
    const REAL dy = this->by - this->ay;
    const REAL n0t = -dy;
    const REAL n1t = dx;
    const REAL l = 1.0 / std::sqrt(n0t * n0t + n1t * n1t);
    this->normalx = n0t * l;
    this->normaly = n1t * l;
  };

  /**
   * Determine if an intersection occurs.
   *
   * @param[in] p00 First point, x component.
   * @param[in] p01 First point, y component.
   * @param[in] p10 Second point, x component.
   * @param[in] p11 Second point, y component.
   * @param[in, out] i0 Intersection point, x component.
   * @param[in, out] i1 Intersection point, y component.
   * @returns True if the line segment intersects the plane otherwise false.
   */
  inline bool line_line_intersection(const REAL p00, const REAL p01,
                                     const REAL p10, const REAL p11, REAL *i0,
                                     REAL *i1) const {
    REAL t0, t1, l0;
    const bool c = Particles::line_segment_intersection_2d(
        this->ax, this->ay, this->bx, this->by, p00, p01, p10, p11, t0, t1, l0,
        this->tol);
    *i0 = t0;
    *i1 = t1;
    return c;
  }
};

} // namespace NESO::CompositeInteraction

#endif
