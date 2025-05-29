#ifndef ___NESO_PARTICLE_MAPPING_NEWTON_PYR_H__
#define ___NESO_PARTICLE_MAPPING_NEWTON_PYR_H__

#include "generated_linear/linear_newton_implementation.hpp"
#include "particle_cell_mapping_newton.hpp"
#include <neso_particles.hpp>

using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Newton {

struct MappingPyrLinear3D;
template <> struct mapping_host_device_types<MappingPyrLinear3D> {
  struct DataDevice {
    REAL coordinates[5][3];
    NewtonRelativeExitTolerances residual_scaling;
  };
  using DataHost = NullDataHost;
  using DataLocal = NullDataLocal;
};

struct MappingPyrLinear3D : MappingNewtonIterationBase<MappingPyrLinear3D> {

  inline void write_data_v([[maybe_unused]] SYCLTargetSharedPtr sycl_target,
                           GeometrySharedPtr geom, DataHost *data_host,
                           DataDevice *data_device) {

    const int num_vertices = 5;
    NESOASSERT(num_vertices == geom->GetNumVerts(),
               "Unexpected number of vertices");

    for (int vx = 0; vx < num_vertices; vx++) {
      REAL xx[3];
      auto vertex = geom->GetVertex(vx);
      vertex->GetCoords(xx[0], xx[1], xx[2]);
      for (int iy = 0; iy < 3; iy++) {
        data_device->coordinates[vx][iy] = xx[iy];
      }
    }

    create_newton_relative_exit_tolerances(geom,
                                           &data_device->residual_scaling);
  }

  inline void newton_step_v(const DataDevice *d_data, const REAL xi0,
                            const REAL xi1, const REAL xi2, const REAL phys0,
                            const REAL phys1, const REAL phys2, const REAL f0,
                            const REAL f1, const REAL f2, REAL *xin0,
                            REAL *xin1, REAL *xin2) {

    // The pyramid map has a singularity at the 5th vertex v4.
    const REAL v40 = d_data->coordinates[4][0];
    const REAL v41 = d_data->coordinates[4][1];
    const REAL v42 = d_data->coordinates[4][2];
    const bool t0 = ABS(v40 - phys0) <= NekConstants::kNekZeroTol;
    const bool t1 = ABS(v41 - phys1) <= NekConstants::kNekZeroTol;
    const bool t2 = ABS(v42 - phys2) <= NekConstants::kNekZeroTol;
    const bool vt = t0 && t1 && t2;
    if (vt) {
      *xin0 = -1.0;
      *xin1 = -1.0;
      *xin2 = 1.0;
    } else {
      Pyramid::newton_step_linear_3d(
          xi0, xi1, xi2, d_data->coordinates[0][0], d_data->coordinates[0][1],
          d_data->coordinates[0][2], d_data->coordinates[1][0],
          d_data->coordinates[1][1], d_data->coordinates[1][2],
          d_data->coordinates[2][0], d_data->coordinates[2][1],
          d_data->coordinates[2][2], d_data->coordinates[3][0],
          d_data->coordinates[3][1], d_data->coordinates[3][2],
          d_data->coordinates[4][0], d_data->coordinates[4][1],
          d_data->coordinates[4][2], phys0, phys1, phys2, f0, f1, f2, xin0,
          xin1, xin2);
    }
  }

  inline REAL newton_residual_v(const DataDevice *d_data, const REAL xi0,
                                const REAL xi1, const REAL xi2,
                                const REAL phys0, const REAL phys1,
                                const REAL phys2, REAL *f0, REAL *f1,
                                REAL *f2) {

    // The pyramid map has a singularity at the 5th vertex v4.
    const REAL v40 = d_data->coordinates[4][0];
    const REAL v41 = d_data->coordinates[4][1];
    const REAL v42 = d_data->coordinates[4][2];
    const bool t0 = ABS(v40 - phys0) <= NekConstants::kNekZeroTol;
    const bool t1 = ABS(v41 - phys1) <= NekConstants::kNekZeroTol;
    const bool t2 = ABS(v42 - phys2) <= NekConstants::kNekZeroTol;
    const bool vt = t0 && t1 && t2;
    const bool x0 = ABS(xi0 - (-1.0)) <= NekConstants::kNekZeroTol;
    const bool x1 = ABS(xi1 - (-1.0)) <= NekConstants::kNekZeroTol;
    const bool x2 = ABS(xi2 - (1.0)) <= NekConstants::kNekZeroTol;
    const bool xt = x0 && x1 && x2;

    if (vt && xt) {
      return 0.0;
    } else {
      Pyramid::newton_f_linear_3d(
          xi0, xi1, xi2, d_data->coordinates[0][0], d_data->coordinates[0][1],
          d_data->coordinates[0][2], d_data->coordinates[1][0],
          d_data->coordinates[1][1], d_data->coordinates[1][2],
          d_data->coordinates[2][0], d_data->coordinates[2][1],
          d_data->coordinates[2][2], d_data->coordinates[3][0],
          d_data->coordinates[3][1], d_data->coordinates[3][2],
          d_data->coordinates[4][0], d_data->coordinates[4][1],
          d_data->coordinates[4][2], phys0, phys1, phys2, f0, f1, f2);

      return d_data->residual_scaling.get_relative_error_3d(*f0, *f1, *f2);
    }
  }

  inline int get_ndim_v() { return 3; }

  inline void set_initial_iteration_v(const DataDevice *d_data,
                                      const REAL phys0, const REAL phys1,
                                      const REAL phys2, REAL *xi0, REAL *xi1,
                                      REAL *xi2) {

    // The pyramid map has a singularity at the 5th vertex v4.
    const REAL v40 = d_data->coordinates[4][0];
    const REAL v41 = d_data->coordinates[4][1];
    const REAL v42 = d_data->coordinates[4][2];
    const bool t0 = ABS(v40 - phys0) <= NekConstants::kNekZeroTol;
    const bool t1 = ABS(v41 - phys1) <= NekConstants::kNekZeroTol;
    const bool t2 = ABS(v42 - phys2) <= NekConstants::kNekZeroTol;
    const bool vt = t0 && t1 && t2;
    if (vt) {
      *xi0 = -1.0;
      *xi1 = -1.0;
      *xi2 = 1.0;
    } else {
      *xi0 = 0.0;
      *xi1 = 0.0;
      *xi2 = 0.0;
    }
  }

  inline void loc_coord_to_loc_collapsed_v(const DataDevice *d_data,
                                           const REAL xi0, const REAL xi1,
                                           const REAL xi2, REAL *eta0,
                                           REAL *eta1, REAL *eta2) {
    GeometryInterface::Pyramid{}.loc_coord_to_loc_collapsed(xi0, xi1, xi2, eta0,
                                                            eta1, eta2);
  }

  inline void loc_collapsed_to_loc_coord_v(const DataDevice *d_data,
                                           const REAL eta0, const REAL eta1,
                                           const REAL eta2, REAL *xi0,
                                           REAL *xi1, REAL *xi2) {
    GeometryInterface::Pyramid{}.loc_collapsed_to_loc_coord(eta0, eta1, eta2,
                                                            xi0, xi1, xi2);
  }
};

} // namespace Newton
} // namespace NESO

#endif
