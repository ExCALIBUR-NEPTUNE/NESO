#ifndef ___NESO_PARTICLE_MAPPING_NEWTON_QUAD_EMBED_3D_H__
#define ___NESO_PARTICLE_MAPPING_NEWTON_QUAD_EMBED_3D_H__

#include "generated_linear/linear_newton_implementation.hpp"
#include "particle_cell_mapping_newton.hpp"
#include <neso_particles.hpp>

using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Newton {

struct MappingQuadLinear2DEmbed3D
    : MappingNewtonIterationBase<MappingQuadLinear2DEmbed3D> {

  inline void write_data_v(GeometrySharedPtr geom, void *data_host,
                           void *data_device) {

    REAL *data_device_real = static_cast<REAL *>(data_device);
    auto v0 = geom->GetVertex(0);
    auto v1 = geom->GetVertex(1);
    auto v2 = geom->GetVertex(2);
    auto v3 = geom->GetVertex(3);

    NESOASSERT(v0->GetCoordim() == 3, "expected coordim == 3");
    NESOASSERT(v1->GetCoordim() == 3, "expected coordim == 3");
    NESOASSERT(v2->GetCoordim() == 3, "expected coordim == 3");
    NESOASSERT(v3->GetCoordim() == 3, "expected coordim == 3");

    const int num_vertices = 4;
    int ix = 0;
    for (int vx = 0; vx < num_vertices; vx++) {
      REAL xx[3];
      auto vertex = geom->GetVertex(vx);
      vertex->GetCoords(xx[0], xx[1], xx[2]);
      for (int iy = 0; iy < 3; iy++) {
        data_device_real[ix + iy] = xx[iy];
      }
      ix += 3;
    }
    NESOASSERT(ix == 12, "unexpected index");

    PointGeom p0(3, 0, 0.0, 0.0, 0.0);
    PointGeom p1(3, 1, 0.0, 0.0, 0.0);
    PointGeom cx(3, 2, 0.0, 0.0, 0.0);

    // get a vector normal to the plane containing the quad to use as a penalty
    // vector
    p0.Sub(*v1, *v0);
    p1.Sub(*v3, *v0);
    cx.Mult(p0, p1);
    NekDouble c0;
    NekDouble c1;
    NekDouble c2;
    cx.GetCoords(c0, c1, c2);
    NESOASSERT((ABS(c0) + ABS(c1) + ABS(c2)) > 0,
               "Vector normal to plane has length 0.");
    data_device_real[12] = c0;
    data_device_real[13] = c1;
    data_device_real[14] = c2;

    // Exit tolerance scaling applied by Nektar++
    auto m_xmap = geom->GetXmap();
    auto m_geomFactors = geom->GetGeomFactors();
    Array<OneD, const NekDouble> Jac =
        m_geomFactors->GetJac(m_xmap->GetPointsKeys());
    NekDouble tol_scaling =
        Vmath::Vsum(Jac.size(), Jac, 1) / ((NekDouble)Jac.size());
    data_device_real[15] = ABS(1.0 / tol_scaling);
  }

  inline void free_data_v(void *data_host) { return; }

  inline std::size_t data_size_host_v() { return 0; }

  inline std::size_t data_size_device_v() { return (5 * 3 + 1) * sizeof(REAL); }

  inline void newton_step_v(const void *d_data, const REAL xi0, const REAL xi1,
                            const REAL xi2, const REAL phys0, const REAL phys1,
                            const REAL phys2, const REAL f0, const REAL f1,
                            const REAL f2, REAL *xin0, REAL *xin1, REAL *xin2) {

    const REAL *data_device_real = static_cast<const REAL *>(d_data);
    const REAL v00 = data_device_real[0];
    const REAL v01 = data_device_real[1];
    const REAL v02 = data_device_real[2];
    const REAL v10 = data_device_real[3];
    const REAL v11 = data_device_real[4];
    const REAL v12 = data_device_real[5];
    const REAL v20 = data_device_real[6];
    const REAL v21 = data_device_real[7];
    const REAL v22 = data_device_real[8];
    const REAL v30 = data_device_real[9];
    const REAL v31 = data_device_real[10];
    const REAL v32 = data_device_real[11];
    const REAL c0 = data_device_real[12];
    const REAL c1 = data_device_real[13];
    const REAL c2 = data_device_real[14];

    QuadrilateralEmbed3D::newton_step_linear_2d(
        xi0, xi1, xi2, v00, v01, v02, v10, v11, v12, v20, v21, v22, v30, v31,
        v32, c0, c1, c2, phys0, phys1, phys2, f0, f1, f2, xin0, xin1, xin2);
  }

  inline REAL newton_residual_v(const void *d_data, const REAL xi0,
                                const REAL xi1, const REAL xi2,
                                const REAL phys0, const REAL phys1,
                                const REAL phys2, REAL *f0, REAL *f1,
                                REAL *f2) {

    const REAL *data_device_real = static_cast<const REAL *>(d_data);
    const REAL v00 = data_device_real[0];
    const REAL v01 = data_device_real[1];
    const REAL v02 = data_device_real[2];
    const REAL v10 = data_device_real[3];
    const REAL v11 = data_device_real[4];
    const REAL v12 = data_device_real[5];
    const REAL v20 = data_device_real[6];
    const REAL v21 = data_device_real[7];
    const REAL v22 = data_device_real[8];
    const REAL v30 = data_device_real[9];
    const REAL v31 = data_device_real[10];
    const REAL v32 = data_device_real[11];
    const REAL c0 = data_device_real[12];
    const REAL c1 = data_device_real[13];
    const REAL c2 = data_device_real[14];

    QuadrilateralEmbed3D::newton_f_linear_2d(
        xi0, xi1, xi2, v00, v01, v02, v10, v11, v12, v20, v21, v22, v30, v31,
        v32, c0, c1, c2, phys0, phys1, phys2, f0, f1, f2);

    const REAL norm2 = MAX(MAX(ABS(*f0), ABS(*f1)), ABS(*f1));
    const REAL tol_scaling = data_device_real[15];
    const REAL scaled_norm2 = norm2 * tol_scaling;
    return scaled_norm2;
  }

  inline int get_ndim_v() { return 3; }

  inline void set_initial_iteration_v(const void *d_data, const REAL phys0,
                                      const REAL phys1, const REAL phys2,
                                      REAL *xi0, REAL *xi1, REAL *xi2) {
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
