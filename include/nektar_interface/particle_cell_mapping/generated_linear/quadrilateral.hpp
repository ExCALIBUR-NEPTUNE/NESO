/**
This is a generated file. Please make non-ephemeral changes by
modifying the script which generates this file. This file was generated on git
hash

7094376e18db143be4b89da92555451f8e4a3901

by running the command

python ../../python/deformed_mappings/generate_linear_source.py
../../include/nektar_interface/particle_cell_mapping/generated_linear

*/
#ifndef __GENERATED_QUADRILATERAL_LINEAR_NEWTON_H__
#define __GENERATED_QUADRILATERAL_LINEAR_NEWTON_H__

#include <neso_particles.hpp>
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Quadrilateral {

/**
 * Perform a Newton method update step for a Newton iteration that determines
 * the local coordinates (xi) for a given set of physical coordinates. If
 * v0,v1,v2 and v3 (passed component wise) are the vertices of a linear sided
 * quadrilateral then this function performs the Newton update:
 *
 * xi_{n+1} = xi_n - J^{-1}(xi_n) * F(xi_n)
 *
 * where
 *
 * F(xi) = X(xi) - X_phys
 *
 * where X_phys are the global coordinates.
 *
 * X is defined as
 *
 *
 * X(xi) = 0.25 * v0 * (1 - xi_0) * (1 - xi_1) +
 *         0.25 * v1 * (1 + xi_0) * (1 - xi_1) +
 *         0.25 * v3 * (1 - xi_0) * (1 + xi_1) +
 *         0.25 * v2 * (1 + xi_0) * (1 + xi_1)
 *
 *
 * This is a generated function. To modify this function please edit the script
 * that generates this function. See top of file.
 *
 * @param[in] xi0 Current xi_n point, x component.
 * @param[in] xi1 Current xi_n point, y component.
 * @param[in] v00 Vertex 0, x component.
 * @param[in] v01 Vertex 0, y component.
 * @param[in] v10 Vertex 1, x component.
 * @param[in] v11 Vertex 1, y component.
 * @param[in] v20 Vertex 2, x component.
 * @param[in] v21 Vertex 2, y component.
 * @param[in] v30 Vertex 3, x component.
 * @param[in] v31 Vertex 3, y component.
 * @param[in] phys0 Target point in global space, x component.
 * @param[in] phys1 Target point in global space, y component.
 * @param[in] f0 Current f evaluation at xi, x component.
 * @param[in] f1 Current f evaluation at xi, y component.
 * @param[in, out] xin0 Output local coordinate iteration, x component.
 * @param[in, out] xin1 Output local coordinate iteration, y component.
 */
inline void
newton_step_linear_2d(const REAL xi0, const REAL xi1, const REAL v00,
                      const REAL v01, const REAL v10, const REAL v11,
                      const REAL v20, const REAL v21, const REAL v30,
                      const REAL v31, const REAL phys0, const REAL phys1,
                      const REAL f0, const REAL f1, REAL *xin0, REAL *xin1) {
  const REAL x0 = xi1 - 1;
  const REAL x1 = xi1 + 1;
  const REAL x2 = xi0 - 1;
  const REAL x3 = xi0 + 1;
  const REAL J00 =
      0.25 * v00 * x0 - 0.25 * v10 * x0 + 0.25 * v20 * x1 - 0.25 * v30 * x1;
  const REAL J01 =
      0.25 * v00 * x2 - 0.25 * v10 * x3 + 0.25 * v20 * x3 - 0.25 * v30 * x2;
  const REAL J10 =
      0.25 * v01 * x0 - 0.25 * v11 * x0 + 0.25 * v21 * x1 - 0.25 * v31 * x1;
  const REAL J11 =
      0.25 * v01 * x2 - 0.25 * v11 * x3 + 0.25 * v21 * x3 - 0.25 * v31 * x2;
  const REAL y0 = J00 * J11;
  const REAL y1 = J01 * J10;
  const REAL y2 = 1.0 / (y0 - y1);
  const REAL xin0_tmp = y2 * (J01 * f1 - J11 * f0 + xi0 * y0 - xi0 * y1);
  const REAL xin1_tmp = y2 * (-J00 * f1 + J10 * f0 + xi1 * y0 - xi1 * y1);
  *xin0 = xin0_tmp;
  *xin1 = xin1_tmp;
}

/**
 * Compute and return F evaluation where
 *
 * F(xi) = X(xi) - X_phys
 *
 * where X_phys are the global coordinates. X is defined as
 *
 *
 * X(xi) = 0.25 * v0 * (1 - xi_0) * (1 - xi_1) +
 *         0.25 * v1 * (1 + xi_0) * (1 - xi_1) +
 *         0.25 * v3 * (1 - xi_0) * (1 + xi_1) +
 *         0.25 * v2 * (1 + xi_0) * (1 + xi_1)
 *
 *
 * This is a generated function. To modify this function please edit the script
 * that generates this function. See top of file.
 *
 * @param[in] xi0 Current xi_n point, x component.
 * @param[in] xi1 Current xi_n point, y component.
 * @param[in] v00 Vertex 0, x component.
 * @param[in] v01 Vertex 0, y component.
 * @param[in] v10 Vertex 1, x component.
 * @param[in] v11 Vertex 1, y component.
 * @param[in] v20 Vertex 2, x component.
 * @param[in] v21 Vertex 2, y component.
 * @param[in] v30 Vertex 3, x component.
 * @param[in] v31 Vertex 3, y component.
 * @param[in] phys0 Target point in global space, x component.
 * @param[in] phys1 Target point in global space, y component.
 * @param[in, out] f0 Current f evaluation at xi, x component.
 * @param[in, out] f1 Current f evaluation at xi, y component.
 */
inline void newton_f_linear_2d(const REAL xi0, const REAL xi1, const REAL v00,
                               const REAL v01, const REAL v10, const REAL v11,
                               const REAL v20, const REAL v21, const REAL v30,
                               const REAL v31, const REAL phys0,
                               const REAL phys1, REAL *f0, REAL *f1) {
  const REAL x0 = xi0 - 1;
  const REAL x1 = xi1 - 1;
  const REAL x2 = xi0 + 1;
  const REAL x3 = 0.25 * x1 * x2;
  const REAL x4 = xi1 + 1;
  const REAL x5 = 0.25 * x0 * x4;
  const REAL f0_tmp = -phys0 + 0.25 * v00 * x0 * x1 - v10 * x3 +
                      0.25 * v20 * x2 * x4 - v30 * x5;
  const REAL f1_tmp = -phys1 + 0.25 * v01 * x0 * x1 - v11 * x3 +
                      0.25 * v21 * x2 * x4 - v31 * x5;
  *f0 = f0_tmp;
  *f1 = f1_tmp;
}

} // namespace Quadrilateral
} // namespace NESO

#endif
