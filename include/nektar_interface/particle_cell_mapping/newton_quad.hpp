#ifndef ___NESO_PARTICLE_MAPPING_NEWTON_QUAD_H__
#define ___NESO_PARTICLE_MAPPING_NEWTON_QUAD_H__

#include "particle_cell_mapping_newton.hpp"
#include <neso_particles.hpp>

using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Newton {
namespace Quad {

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
 * This is a generated function. To modify this function please edit the script
 * that generates this function. TODO link to function.
 *
 * @param[in] xi0 Current xi_n point, x component.
 * @param[in] xi1 Current xi_n point, y component.
 * @param[in] v00 Vertex 0, x component of quadrilateral.
 * @param[in] v01 Vertex 0, y component of quadrilateral.
 * @param[in] v10 Vertex 1, x component of quadrilateral.
 * @param[in] v11 Vertex 1, y component of quadrilateral.
 * @param[in] v20 Vertex 2, x component of quadrilateral.
 * @param[in] v21 Vertex 2, y component of quadrilateral.
 * @param[in] v30 Vertex 3, x component of quadrilateral.
 * @param[in] v31 Vertex 3, y component of quadrilateral.
 * @param[in] phys0 Target point in physical space, x component.
 * @param[in] phys1 Target point in physical space, y component.
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
  const REAL x0 = v00 * v31;
  const REAL x1 = v01 * v10;
  const REAL x2 = v11 * v20;
  const REAL x3 = v21 * v30;
  const REAL x4 = v00 * v11;
  const REAL x5 = v01 * v30;
  const REAL x6 = v10 * v21;
  const REAL x7 = v20 * v31;
  const REAL x8 = v00 * v21;
  const REAL x9 = x8 * xi0;
  const REAL x10 = v01 * v20;
  const REAL x11 = x10 * xi1;
  const REAL x12 = x5 * xi0;
  const REAL x13 = v10 * v31;
  const REAL x14 = x13 * xi0;
  const REAL x15 = x3 * xi1;
  const REAL x16 = x0 * xi0;
  const REAL x17 = v11 * v30;
  const REAL x18 = x17 * xi0;
  const REAL x19 = x7 * xi1;
  const REAL x20 = x2 * xi0;
  const REAL x21 = x6 * xi0;
  const REAL x22 = x20 - x21;
  const REAL x23 = x4 * xi1;
  const REAL x24 = x1 * xi1;
  const REAL x25 = x23 - x24;
  const REAL x26 = 1 / (x0 + x1 - x10 * xi0 + x11 + x12 + x13 * xi1 + x14 +
                        x15 - x16 - x17 * xi1 - x18 - x19 + x2 + x22 + x25 +
                        x3 - x4 - x5 - x6 - x7 - x8 * xi1 + x9);
  const REAL x27 = 2.0 * f0;
  const REAL x28 = v31 * x27;
  const REAL x29 = 2.0 * f1;
  const REAL x30 = v10 * x29;
  const REAL x31 = v11 * x27;
  const REAL x32 = v30 * x29;
  const REAL x33 = xi0 * xi0;
  const REAL x34 = v01 * x27;
  const REAL x35 = v21 * x27;
  const REAL x36 = v00 * x29;
  const REAL x37 = v20 * x29;
  const REAL x38 = x14 * xi1;
  const REAL x39 = x18 * xi1;
  const REAL x40 = x11 * xi0 - x34 + x35 + x36 - x37 - x9 * xi1;
  const REAL x41 = xi1 * xi1;
  const REAL xin0_tmp =
      x26 * (-x0 * x33 + x1 * xi0 - x10 * x33 - x12 + x13 * x33 + x15 * xi0 +
             x16 - x17 * x33 - x19 * xi0 + x2 * x33 + x22 + x23 * xi0 -
             x24 * xi0 - x28 * xi0 + x28 + x3 * xi0 + x30 * xi0 + x30 -
             x31 * xi0 - x31 + x32 * xi0 - x32 + x33 * x5 - x33 * x6 +
             x33 * x8 + x34 * xi0 + x35 * xi0 - x36 * xi0 - x37 * xi0 + x38 -
             x39 - x4 * xi0 + x40 - x7 * xi0);
  const REAL xin1_tmp =
      -x26 *
      (-x0 * xi1 + x1 * x41 - x10 * x41 - x12 * xi1 - x13 * x41 - x15 +
       x16 * xi1 + x17 * x41 + x19 - x2 * xi1 - x20 * xi1 + x21 * xi1 + x25 -
       x28 * xi1 - x28 - x3 * x41 + x30 * xi1 - x30 - x31 * xi1 + x31 +
       x32 * xi1 + x32 + x34 * xi1 + x35 * xi1 - x36 * xi1 - x37 * xi1 - x38 +
       x39 - x4 * x41 + x40 + x41 * x7 + x41 * x8 + x5 * xi1 + x6 * xi1);
  *xin0 = xin0_tmp;
  *xin1 = xin1_tmp;
}

/**
 * Compute and return F evaluation for Quadrilateral where
 *
 * F(xi) = X(xi) - X_phys
 *
 * where X_phys are the global coordinates.
 *
 * This is a generated function. To modify this function please edit the script
 * that generates this function. TODO link to function.
 *
 * @param[in] xi0 Current xi_n point, x component.
 * @param[in] xi1 Current xi_n point, y component.
 * @param[in] v00 Vertex 0, x component of quadrilateral.
 * @param[in] v01 Vertex 0, y component of quadrilateral.
 * @param[in] v10 Vertex 1, x component of quadrilateral.
 * @param[in] v11 Vertex 1, y component of quadrilateral.
 * @param[in] v20 Vertex 2, x component of quadrilateral.
 * @param[in] v21 Vertex 2, y component of quadrilateral.
 * @param[in] v30 Vertex 3, x component of quadrilateral.
 * @param[in] v31 Vertex 3, y component of quadrilateral.
 * @param[in] phys0 Target point in physical space, x component.
 * @param[in] phys1 Target point in physical space, y component.
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

} // namespace Quad

struct MappingQuadLinear2D : MappingNewtonIterationBase<MappingQuadLinear2D> {

  inline void write_data_v(GeometrySharedPtr geom, void *data_host,
                           void *data_device) {

    REAL *data_device_real = static_cast<REAL *>(data_device);
    auto v0 = geom->GetVertex(0);
    auto v1 = geom->GetVertex(1);
    auto v2 = geom->GetVertex(2);
    auto v3 = geom->GetVertex(3);
    NekDouble tmp;
    NekDouble v00;
    NekDouble v01;
    NekDouble v10;
    NekDouble v11;
    NekDouble v20;
    NekDouble v21;
    NekDouble v30;
    NekDouble v31;
    v0->GetCoords(v00, v01, tmp);
    v1->GetCoords(v10, v11, tmp);
    v2->GetCoords(v20, v21, tmp);
    v3->GetCoords(v30, v31, tmp);
    data_device_real[0] = v00;
    data_device_real[1] = v01;
    data_device_real[2] = v10;
    data_device_real[3] = v11;
    data_device_real[4] = v20;
    data_device_real[5] = v21;
    data_device_real[6] = v30;
    data_device_real[7] = v31;

    // Exit tolerance scaling applied by Nektar++
    auto m_xmap = geom->GetXmap();
    auto m_geomFactors = geom->GetGeomFactors();
    Array<OneD, const NekDouble> Jac =
        m_geomFactors->GetJac(m_xmap->GetPointsKeys());
    NekDouble tol_scaling =
        Vmath::Vsum(Jac.size(), Jac, 1) / ((NekDouble)Jac.size());
    data_device_real[8] = 1.0 / tol_scaling;
  }

  inline void free_data_v(void *data_host) { return; }

  inline size_t data_size_host_v() { return 0; }

  inline size_t data_size_device_v() { return (4 * 2 + 1) * sizeof(REAL); }

  inline void newton_step_v(const void *d_data, const REAL xi0, const REAL xi1,
                            const REAL xi2, const REAL phys0, const REAL phys1,
                            const REAL phys2, const REAL f0, const REAL f1,
                            const REAL f2, REAL *xin0, REAL *xin1, REAL *xin2) {

    const REAL *data_device_real = static_cast<const REAL *>(d_data);
    const REAL v00 = data_device_real[0];
    const REAL v01 = data_device_real[1];
    const REAL v10 = data_device_real[2];
    const REAL v11 = data_device_real[3];
    const REAL v20 = data_device_real[4];
    const REAL v21 = data_device_real[5];
    const REAL v30 = data_device_real[6];
    const REAL v31 = data_device_real[7];
    Quad::newton_step_linear_2d(xi0, xi1, v00, v01, v10, v11, v20, v21, v30,
                                v31, phys0, phys1, f0, f1, xin0, xin1);
  }

  inline REAL newton_residual_v(const void *d_data, const REAL xi0,
                                const REAL xi1, const REAL xi2,
                                const REAL phys0, const REAL phys1,
                                const REAL phys2, REAL *f0, REAL *f1,
                                REAL *f2) {

    const REAL *data_device_real = static_cast<const REAL *>(d_data);
    const REAL v00 = data_device_real[0];
    const REAL v01 = data_device_real[1];
    const REAL v10 = data_device_real[2];
    const REAL v11 = data_device_real[3];
    const REAL v20 = data_device_real[4];
    const REAL v21 = data_device_real[5];
    const REAL v30 = data_device_real[6];
    const REAL v31 = data_device_real[7];

    Quad::newton_f_linear_2d(xi0, xi1, v00, v01, v10, v11, v20, v21, v30, v31,
                             phys0, phys1, f0, f1);
    *f2 = 0.0;

    // const REAL norm2 = (*f0)*(*f0) + (*f1)*(*f1);
    const REAL norm2 = MAX(ABS(*f0), ABS(*f1));
    const REAL tol_scaling = data_device_real[8];
    const REAL scaled_norm2 = norm2 * tol_scaling;
    return scaled_norm2;
  }

  inline int get_ndim_v() { return 2; }

  inline void set_initial_iteration_v(const void *d_data, REAL *xi0, REAL *xi1,
                                      REAL *xi2) {
    *xi0 = 0.0;
    *xi1 = 0.0;
    *xi2 = 0.0;
  }

  inline void loc_coord_to_loc_collapsed_v(const void *d_data, const REAL xi0,
                                           const REAL xi1, const REAL xi2,
                                           REAL *eta0, REAL *eta1, REAL *eta2) {
    *eta0 = xi0;
    *eta1 = xi1;
    *eta2 = 0.0;
  }
};

} // namespace Newton
} // namespace NESO

#endif
