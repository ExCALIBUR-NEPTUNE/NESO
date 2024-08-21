#ifndef __NEKTAR_INTERFACE_COMPOSITE_INTERSECTION_LINE_LINE_INTERSECTION_H_
#define __NEKTAR_INTERFACE_COMPOSITE_INTERSECTION_LINE_LINE_INTERSECTION_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;
using namespace Nektar::SpatialDomains;
using namespace Nektar::LibUtilities;

#include "nektar_interface/special_functions.hpp"
#include "nektar_interface/typedefs.hpp"
#include <cmath>

namespace NESO::CompositeInteraction {

/**
 * TODO
 */
class LineLineIntersection {
protected:
public:
  /// TODO
  REAL ax;
  REAL ay;
  REAL bx;
  REAL by;
  REAL normalx;
  REAL normaly;
  REAL tol = 0.0;

  LineLineIntersection() = default;

  /**
   * TODO
   */
  template <typename T> LineLineIntersection(std::shared_ptr<T> geom) {
    auto shape_type = geom->GetShapeType();
    NESOASSERT(shape_type == eSegment,
               "LineLineIntersection not implemented for this shape type.");
  };

  /**
   * TODO
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
