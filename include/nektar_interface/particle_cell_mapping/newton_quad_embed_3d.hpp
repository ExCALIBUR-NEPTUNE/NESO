#ifndef ___NESO_PARTICLE_MAPPING_NEWTON_QUAD_EMBED_3D_H__
#define ___NESO_PARTICLE_MAPPING_NEWTON_QUAD_EMBED_3D_H__

#include "generated_linear/linear_newton_implementation.hpp"
#include "particle_cell_mapping_newton.hpp"
#include <neso_particles.hpp>

using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Newton {

struct MappingQuadLinear2DEmbed3D;
template <> struct mapping_host_device_types<MappingQuadLinear2DEmbed3D> {
  struct DataDevice {
    REAL coordinates[4][3];
    REAL plane_normal[3];
    NewtonRelativeExitTolerances residual_scaling;
  };

  using DataHost = NullDataHost;
  using DataLocal = NullDataLocal;
};

struct MappingQuadLinear2DEmbed3D
    : MappingNewtonIterationBase<MappingQuadLinear2DEmbed3D> {

  inline void write_data_v([[maybe_unused]] SYCLTargetSharedPtr sycl_target,
                           GeometrySharedPtr geom, DataHost *data_host,
                           DataDevice *data_device) {

    auto v0 = geom->GetVertex(0);
    auto v1 = geom->GetVertex(1);
    auto v2 = geom->GetVertex(2);
    auto v3 = geom->GetVertex(3);

    NESOASSERT(v0->GetCoordim() == 3, "expected coordim == 3");
    NESOASSERT(v1->GetCoordim() == 3, "expected coordim == 3");
    NESOASSERT(v2->GetCoordim() == 3, "expected coordim == 3");
    NESOASSERT(v3->GetCoordim() == 3, "expected coordim == 3");

    const int num_vertices = 4;
    for (int vx = 0; vx < num_vertices; vx++) {
      REAL xx[3];
      auto vertex = geom->GetVertex(vx);
      vertex->GetCoords(xx[0], xx[1], xx[2]);
      for (int iy = 0; iy < 3; iy++) {
        data_device->coordinates[vx][iy] = xx[iy];
      }
    }

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
    data_device->plane_normal[0] = c0;
    data_device->plane_normal[1] = c1;
    data_device->plane_normal[2] = c2;

    create_newton_relative_exit_tolerances(geom,
                                           &data_device->residual_scaling);
  }

  inline void newton_step_v(const DataDevice *d_data, const REAL xi0,
                            const REAL xi1, const REAL xi2, const REAL phys0,
                            const REAL phys1, const REAL phys2, const REAL f0,
                            const REAL f1, const REAL f2, REAL *xin0,
                            REAL *xin1, REAL *xin2) {

    const REAL v00 = d_data->coordinates[0][0];
    const REAL v01 = d_data->coordinates[0][1];
    const REAL v02 = d_data->coordinates[0][2];
    const REAL v10 = d_data->coordinates[1][0];
    const REAL v11 = d_data->coordinates[1][1];
    const REAL v12 = d_data->coordinates[1][2];
    const REAL v20 = d_data->coordinates[2][0];
    const REAL v21 = d_data->coordinates[2][1];
    const REAL v22 = d_data->coordinates[2][2];
    const REAL v30 = d_data->coordinates[3][0];
    const REAL v31 = d_data->coordinates[3][1];
    const REAL v32 = d_data->coordinates[3][2];
    const REAL c0 = d_data->plane_normal[0];
    const REAL c1 = d_data->plane_normal[1];
    const REAL c2 = d_data->plane_normal[2];

    QuadrilateralEmbed3D::newton_step_linear_2d(
        xi0, xi1, xi2, v00, v01, v02, v10, v11, v12, v20, v21, v22, v30, v31,
        v32, c0, c1, c2, phys0, phys1, phys2, f0, f1, f2, xin0, xin1, xin2);
  }

  inline REAL newton_residual_v(const DataDevice *d_data, const REAL xi0,
                                const REAL xi1, const REAL xi2,
                                const REAL phys0, const REAL phys1,
                                const REAL phys2, REAL *f0, REAL *f1,
                                REAL *f2) {

    const REAL v00 = d_data->coordinates[0][0];
    const REAL v01 = d_data->coordinates[0][1];
    const REAL v02 = d_data->coordinates[0][2];
    const REAL v10 = d_data->coordinates[1][0];
    const REAL v11 = d_data->coordinates[1][1];
    const REAL v12 = d_data->coordinates[1][2];
    const REAL v20 = d_data->coordinates[2][0];
    const REAL v21 = d_data->coordinates[2][1];
    const REAL v22 = d_data->coordinates[2][2];
    const REAL v30 = d_data->coordinates[3][0];
    const REAL v31 = d_data->coordinates[3][1];
    const REAL v32 = d_data->coordinates[3][2];
    const REAL c0 = d_data->plane_normal[0];
    const REAL c1 = d_data->plane_normal[1];
    const REAL c2 = d_data->plane_normal[2];

    QuadrilateralEmbed3D::newton_f_linear_2d(
        xi0, xi1, xi2, v00, v01, v02, v10, v11, v12, v20, v21, v22, v30, v31,
        v32, c0, c1, c2, phys0, phys1, phys2, f0, f1, f2);

    return d_data->residual_scaling.get_relative_error_3d(*f0, *f1, *f2);
  }

  inline int get_ndim_v() { return 3; }

  inline void set_initial_iteration_v(const DataDevice *d_data,
                                      const REAL phys0, const REAL phys1,
                                      const REAL phys2, REAL *xi0, REAL *xi1,
                                      REAL *xi2) {
    *xi0 = 0.0;
    *xi1 = 0.0;
    *xi2 = 0.0;
  }

  inline void loc_coord_to_loc_collapsed_v(const DataDevice *d_data,
                                           const REAL xi0, const REAL xi1,
                                           const REAL xi2, REAL *eta0,
                                           REAL *eta1, REAL *eta2) {
    *eta0 = xi0;
    *eta1 = xi1;
    *eta2 = 0.0;
  }

  inline void loc_collapsed_to_loc_coord_v(const DataDevice *d_data,
                                           const REAL eta0, const REAL eta1,
                                           const REAL eta2, REAL *xi0,
                                           REAL *xi1, REAL *xi2) {
    *xi0 = eta0;
    *xi1 = eta1;
    *xi2 = 0.0;
  }
};

} // namespace Newton
} // namespace NESO

#endif
