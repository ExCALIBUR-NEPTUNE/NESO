#ifndef ___NESO_PARTICLE_MAPPING_NEWTON_PYR_H__
#define ___NESO_PARTICLE_MAPPING_NEWTON_PYR_H__

#include "generated_linear/linear_newton_implementation.hpp"
#include "particle_cell_mapping_newton.hpp"
#include <neso_particles.hpp>

using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Newton {

struct MappingPyrLinear3D : MappingNewtonIterationBase<MappingPyrLinear3D> {

  inline void write_data_v(GeometrySharedPtr geom, void *data_host,
                           void *data_device) {

    REAL *data_device_real = static_cast<REAL *>(data_device);

    const int num_vertices = 5;
    NESOASSERT(num_vertices == geom->GetNumVerts(),
               "Unexpected number of vertices");

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

    // The pyramid map has a singularity at the 5th vertex v4.
    const REAL *data_device_real = static_cast<const REAL *>(d_data);
    const REAL v40 = data_device_real[4 * 3 + 0];
    const REAL v41 = data_device_real[4 * 3 + 1];
    const REAL v42 = data_device_real[4 * 3 + 2];
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
          xi0, xi1, xi2, data_device_real[0], data_device_real[1],
          data_device_real[2], data_device_real[3], data_device_real[4],
          data_device_real[5], data_device_real[6], data_device_real[7],
          data_device_real[8], data_device_real[9], data_device_real[10],
          data_device_real[11], data_device_real[12], data_device_real[13],
          data_device_real[14], phys0, phys1, phys2, f0, f1, f2, xin0, xin1,
          xin2);
    }
  }

  inline REAL newton_residual_v(const void *d_data, const REAL xi0,
                                const REAL xi1, const REAL xi2,
                                const REAL phys0, const REAL phys1,
                                const REAL phys2, REAL *f0, REAL *f1,
                                REAL *f2) {

    // The pyramid map has a singularity at the 5th vertex v4.
    const REAL *data_device_real = static_cast<const REAL *>(d_data);
    const REAL v40 = data_device_real[4 * 3 + 0];
    const REAL v41 = data_device_real[4 * 3 + 1];
    const REAL v42 = data_device_real[4 * 3 + 2];
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
          xi0, xi1, xi2, data_device_real[0], data_device_real[1],
          data_device_real[2], data_device_real[3], data_device_real[4],
          data_device_real[5], data_device_real[6], data_device_real[7],
          data_device_real[8], data_device_real[9], data_device_real[10],
          data_device_real[11], data_device_real[12], data_device_real[13],
          data_device_real[14], phys0, phys1, phys2, f0, f1, f2);

      const REAL norm2 = MAX(MAX(ABS(*f0), ABS(*f1)), ABS(*f2));
      const REAL tol_scaling = data_device_real[15];
      const REAL scaled_norm2 = norm2 * tol_scaling;
      return scaled_norm2;
    }
  }

  inline int get_ndim_v() { return 3; }

  inline void set_initial_iteration_v(const void *d_data, const REAL phys0,
                                      const REAL phys1, const REAL phys2,
                                      REAL *xi0, REAL *xi1, REAL *xi2) {

    // The pyramid map has a singularity at the 5th vertex v4.
    const REAL *data_device_real = static_cast<const REAL *>(d_data);
    const REAL v40 = data_device_real[4 * 3 + 0];
    const REAL v41 = data_device_real[4 * 3 + 1];
    const REAL v42 = data_device_real[4 * 3 + 2];
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

  inline void loc_coord_to_loc_collapsed_v(const void *d_data, const REAL xi0,
                                           const REAL xi1, const REAL xi2,
                                           REAL *eta0, REAL *eta1, REAL *eta2) {
    NekDouble d2 = 1.0 - xi2;
    if (fabs(d2) < NekConstants::kNekZeroTol) {
      if (d2 >= 0.) {
        d2 = NekConstants::kNekZeroTol;
      } else {
        d2 = -NekConstants::kNekZeroTol;
      }
    }
    *eta2 = xi2; // eta_z = xi_z
    *eta1 = 2.0 * (1.0 + xi1) / d2 - 1.0;
    *eta0 = 2.0 * (1.0 + xi0) / d2 - 1.0;
  }
};

} // namespace Newton
} // namespace NESO

#endif
