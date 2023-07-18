/**
This is a generated file. Please make non-ephemeral changes by
modifying the script which generates this file. This file was generated on git
hash

7094376e18db143be4b89da92555451f8e4a3901

by running the command

python ../../python/deformed_mappings/generate_linear_source.py
../../include/nektar_interface/particle_cell_mapping/generated_linear

*/
#ifndef __GENERATED_PRISM_LINEAR_NEWTON_H__
#define __GENERATED_PRISM_LINEAR_NEWTON_H__

#include <neso_particles.hpp>
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Prism {

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
 * X(xi) = c_0 v_0 + c_1 v_1 + c_2 v_2 + c_3 v_3 + c_4 v_4 + c_5 v_5
 *
 * where
 *
 * eta_0 = 2 * ((1 + xi_0) / (1 - xi_2)) - 1
 * c_0 = 0.125 * (1 - eta_0) * (1 - xi_1) * (1 - xi_2)
 * c_1 = 0.125 * (1 + eta_0) * (1 - xi_1) * (1 - xi_2)
 * c_2 = 0.125 * (1 + eta_0) * (1 + xi_1) * (1 - xi_2)
 * c_3 = 0.125 * (1 - eta_0) * (1 + xi_1) * (1 - xi_2)
 * c_4 = 0.25 * (1 - xi_1) * (1 + xi_2)
 * c_5 = 0.25 * (1 + xi_1) * (1 + xi_2)
 *
 *
 * This is a generated function. To modify this function please edit the script
 * that generates this function. See top of file.
 *
 * @param[in] xi0 Current xi_n point, x component.
 * @param[in] xi1 Current xi_n point, y component.
 * @param[in] xi2 Current xi_n point, z component.
 * @param[in] v00 Vertex 0, x component.
 * @param[in] v01 Vertex 0, y component.
 * @param[in] v02 Vertex 0, z component.
 * @param[in] v10 Vertex 1, x component.
 * @param[in] v11 Vertex 1, y component.
 * @param[in] v12 Vertex 1, z component.
 * @param[in] v20 Vertex 2, x component.
 * @param[in] v21 Vertex 2, y component.
 * @param[in] v22 Vertex 2, z component.
 * @param[in] v30 Vertex 3, x component.
 * @param[in] v31 Vertex 3, y component.
 * @param[in] v32 Vertex 3, z component.
 * @param[in] v40 Vertex 4, x component.
 * @param[in] v41 Vertex 4, y component.
 * @param[in] v42 Vertex 4, z component.
 * @param[in] v50 Vertex 5, x component.
 * @param[in] v51 Vertex 5, y component.
 * @param[in] v52 Vertex 5, z component.
 * @param[in] phys0 Target point in global space, x component.
 * @param[in] phys1 Target point in global space, y component.
 * @param[in] phys2 Target point in global space, z component.
 * @param[in] f0 Current f evaluation at xi, x component.
 * @param[in] f1 Current f evaluation at xi, y component.
 * @param[in] f2 Current f evaluation at xi, z component.
 * @param[in, out] xin0 Output local coordinate iteration, x component.
 * @param[in, out] xin1 Output local coordinate iteration, y component.
 * @param[in, out] xin2 Output local coordinate iteration, z component.
 */
inline void newton_step_linear_3d(
    const REAL xi0, const REAL xi1, const REAL xi2, const REAL v00,
    const REAL v01, const REAL v02, const REAL v10, const REAL v11,
    const REAL v12, const REAL v20, const REAL v21, const REAL v22,
    const REAL v30, const REAL v31, const REAL v32, const REAL v40,
    const REAL v41, const REAL v42, const REAL v50, const REAL v51,
    const REAL v52, const REAL phys0, const REAL phys1, const REAL phys2,
    const REAL f0, const REAL f1, const REAL f2, REAL *xin0, REAL *xin1,
    REAL *xin2) {
  const REAL x0 = xi1 + 1;
  const REAL x1 = xi1 - 1;
  const REAL x2 = v00 * x1 - v30 * x0;
  const REAL x3 = xi0 + xi2;
  const REAL x4 = xi0 + 1;
  const REAL x5 = xi2 + 1;
  const REAL x6 = v01 * x1 - v31 * x0;
  const REAL x7 = v02 * x1 - v32 * x0;
  const REAL J00 = -0.25 * v10 * x1 + 0.25 * v20 * x0 + 0.25 * x2;
  const REAL J01 = 0.25 * v00 * x3 - 0.25 * v10 * x4 + 0.25 * v20 * x4 -
                   0.25 * v30 * x3 - 0.25 * v40 * x5 + 0.25 * v50 * x5;
  const REAL J02 = -0.25 * v40 * x1 + 0.25 * v50 * x0 + 0.25 * x2;
  const REAL J10 = -0.25 * v11 * x1 + 0.25 * v21 * x0 + 0.25 * x6;
  const REAL J11 = 0.25 * v01 * x3 - 0.25 * v11 * x4 + 0.25 * v21 * x4 -
                   0.25 * v31 * x3 - 0.25 * v41 * x5 + 0.25 * v51 * x5;
  const REAL J12 = -0.25 * v41 * x1 + 0.25 * v51 * x0 + 0.25 * x6;
  const REAL J20 = -0.25 * v12 * x1 + 0.25 * v22 * x0 + 0.25 * x7;
  const REAL J21 = 0.25 * v02 * x3 - 0.25 * v12 * x4 + 0.25 * v22 * x4 -
                   0.25 * v32 * x3 - 0.25 * v42 * x5 + 0.25 * v52 * x5;
  const REAL J22 = -0.25 * v42 * x1 + 0.25 * v52 * x0 + 0.25 * x7;
  const REAL y0 = J11 * J22;
  const REAL y1 = J00 * y0;
  const REAL y2 = J01 * J12;
  const REAL y3 = J20 * y2;
  const REAL y4 = J02 * J21;
  const REAL y5 = J10 * y4;
  const REAL y6 = J12 * J21;
  const REAL y7 = J00 * y6;
  const REAL y8 = J01 * J22;
  const REAL y9 = J10 * y8;
  const REAL y10 = J02 * J11;
  const REAL y11 = J20 * y10;
  const REAL y12 = 1.0 / (y1 - y11 + y3 + y5 - y7 - y9);
  const REAL y13 = J00 * f2;
  const REAL y14 = J20 * f1;
  const REAL y15 = J10 * f0;
  const REAL y16 = J00 * f1;
  const REAL y17 = J10 * f2;
  const REAL y18 = J20 * f0;
  const REAL xin0_tmp =
      y12 * (-f0 * y0 + f0 * y6 - f1 * y4 + f1 * y8 + f2 * y10 - f2 * y2 +
             xi0 * y1 - xi0 * y11 + xi0 * y3 + xi0 * y5 - xi0 * y7 - xi0 * y9);
  const REAL xin1_tmp = y12 * (J02 * y14 - J02 * y17 + J12 * y13 - J12 * y18 +
                               J22 * y15 - J22 * y16 + xi1 * y1 - xi1 * y11 +
                               xi1 * y3 + xi1 * y5 - xi1 * y7 - xi1 * y9);
  const REAL xin2_tmp = y12 * (-J01 * y14 + J01 * y17 - J11 * y13 + J11 * y18 -
                               J21 * y15 + J21 * y16 + xi2 * y1 - xi2 * y11 +
                               xi2 * y3 + xi2 * y5 - xi2 * y7 - xi2 * y9);
  *xin0 = xin0_tmp;
  *xin1 = xin1_tmp;
  *xin2 = xin2_tmp;
}

/**
 * Compute and return F evaluation where
 *
 * F(xi) = X(xi) - X_phys
 *
 * where X_phys are the global coordinates. X is defined as
 *
 *
 * X(xi) = c_0 v_0 + c_1 v_1 + c_2 v_2 + c_3 v_3 + c_4 v_4 + c_5 v_5
 *
 * where
 *
 * eta_0 = 2 * ((1 + xi_0) / (1 - xi_2)) - 1
 * c_0 = 0.125 * (1 - eta_0) * (1 - xi_1) * (1 - xi_2)
 * c_1 = 0.125 * (1 + eta_0) * (1 - xi_1) * (1 - xi_2)
 * c_2 = 0.125 * (1 + eta_0) * (1 + xi_1) * (1 - xi_2)
 * c_3 = 0.125 * (1 - eta_0) * (1 + xi_1) * (1 - xi_2)
 * c_4 = 0.25 * (1 - xi_1) * (1 + xi_2)
 * c_5 = 0.25 * (1 + xi_1) * (1 + xi_2)
 *
 *
 * This is a generated function. To modify this function please edit the script
 * that generates this function. See top of file.
 *
 * @param[in] xi0 Current xi_n point, x component.
 * @param[in] xi1 Current xi_n point, y component.
 * @param[in] xi2 Current xi_n point, z component.
 * @param[in] v00 Vertex 0, x component.
 * @param[in] v01 Vertex 0, y component.
 * @param[in] v02 Vertex 0, z component.
 * @param[in] v10 Vertex 1, x component.
 * @param[in] v11 Vertex 1, y component.
 * @param[in] v12 Vertex 1, z component.
 * @param[in] v20 Vertex 2, x component.
 * @param[in] v21 Vertex 2, y component.
 * @param[in] v22 Vertex 2, z component.
 * @param[in] v30 Vertex 3, x component.
 * @param[in] v31 Vertex 3, y component.
 * @param[in] v32 Vertex 3, z component.
 * @param[in] v40 Vertex 4, x component.
 * @param[in] v41 Vertex 4, y component.
 * @param[in] v42 Vertex 4, z component.
 * @param[in] v50 Vertex 5, x component.
 * @param[in] v51 Vertex 5, y component.
 * @param[in] v52 Vertex 5, z component.
 * @param[in] phys0 Target point in global space, x component.
 * @param[in] phys1 Target point in global space, y component.
 * @param[in] phys2 Target point in global space, z component.
 * @param[in, out] f0 Current f evaluation at xi, x component.
 * @param[in, out] f1 Current f evaluation at xi, y component.
 * @param[in, out] f2 Current f evaluation at xi, z component.
 */
inline void newton_f_linear_3d(const REAL xi0, const REAL xi1, const REAL xi2,
                               const REAL v00, const REAL v01, const REAL v02,
                               const REAL v10, const REAL v11, const REAL v12,
                               const REAL v20, const REAL v21, const REAL v22,
                               const REAL v30, const REAL v31, const REAL v32,
                               const REAL v40, const REAL v41, const REAL v42,
                               const REAL v50, const REAL v51, const REAL v52,
                               const REAL phys0, const REAL phys1,
                               const REAL phys2, REAL *f0, REAL *f1, REAL *f2) {
  const REAL x0 = xi0 + xi2;
  const REAL x1 = xi1 - 1;
  const REAL x2 = xi0 + 1;
  const REAL x3 = 0.25 * x1;
  const REAL x4 = x2 * x3;
  const REAL x5 = xi1 + 1;
  const REAL x6 = 0.25 * x0 * x5;
  const REAL x7 = xi2 + 1;
  const REAL x8 = x3 * x7;
  const REAL f0_tmp = -phys0 + 0.25 * v00 * x0 * x1 - v10 * x4 +
                      0.25 * v20 * x2 * x5 - v30 * x6 - v40 * x8 +
                      0.25 * v50 * x5 * x7;
  const REAL f1_tmp = -phys1 + 0.25 * v01 * x0 * x1 - v11 * x4 +
                      0.25 * v21 * x2 * x5 - v31 * x6 - v41 * x8 +
                      0.25 * v51 * x5 * x7;
  const REAL f2_tmp = -phys2 + 0.25 * v02 * x0 * x1 - v12 * x4 +
                      0.25 * v22 * x2 * x5 - v32 * x6 - v42 * x8 +
                      0.25 * v52 * x5 * x7;
  *f0 = f0_tmp;
  *f1 = f1_tmp;
  *f2 = f2_tmp;
}

} // namespace Prism
} // namespace NESO

#endif
