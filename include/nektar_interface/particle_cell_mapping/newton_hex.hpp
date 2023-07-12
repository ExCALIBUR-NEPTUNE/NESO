#ifndef ___NESO_PARTICLE_MAPPING_NEWTON_HEX_H__
#define ___NESO_PARTICLE_MAPPING_NEWTON_HEX_H__

#include "generated_linear/linear_newton_implementation.hpp"
#include "particle_cell_mapping_newton.hpp"
#include <neso_particles.hpp>

using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Newton {

struct MappingHexLinear3D : MappingNewtonIterationBase<MappingHexLinear3D> {

  inline void write_data_v(GeometrySharedPtr geom, void *data_host,
                           void *data_device) {

    REAL *data_device_real = static_cast<REAL *>(data_device);

    const int num_vertices = 8;
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
    data_device_real[24] = ABS(1.0 / tol_scaling);
  }

  inline void free_data_v(void *data_host) { return; }

  inline std::size_t data_size_host_v() { return 0; }

  inline std::size_t data_size_device_v() { return (8 * 3 + 1) * sizeof(REAL); }

  inline void newton_step_v(const void *d_data, const REAL xi0, const REAL xi1,
                            const REAL xi2, const REAL phys0, const REAL phys1,
                            const REAL phys2, const REAL f0, const REAL f1,
                            const REAL f2, REAL *xin0, REAL *xin1, REAL *xin2) {

    const REAL *data_device_real = static_cast<const REAL *>(d_data);

    Hexahedron::newton_step_linear_3d(
        xi0, xi1, xi2, data_device_real[0], data_device_real[1],
        data_device_real[2], data_device_real[3], data_device_real[4],
        data_device_real[5], data_device_real[6], data_device_real[7],
        data_device_real[8], data_device_real[9], data_device_real[10],
        data_device_real[11], data_device_real[12], data_device_real[13],
        data_device_real[14], data_device_real[15], data_device_real[16],
        data_device_real[17], data_device_real[18], data_device_real[19],
        data_device_real[20], data_device_real[21], data_device_real[22],
        data_device_real[23], phys0, phys1, phys2, f0, f1, f2, xin0, xin1,
        xin2);
  }

  inline REAL newton_residual_v(const void *d_data, const REAL xi0,
                                const REAL xi1, const REAL xi2,
                                const REAL phys0, const REAL phys1,
                                const REAL phys2, REAL *f0, REAL *f1,
                                REAL *f2) {

    const REAL *data_device_real = static_cast<const REAL *>(d_data);

    Hexahedron::newton_f_linear_3d(
        xi0, xi1, xi2, data_device_real[0], data_device_real[1],
        data_device_real[2], data_device_real[3], data_device_real[4],
        data_device_real[5], data_device_real[6], data_device_real[7],
        data_device_real[8], data_device_real[9], data_device_real[10],
        data_device_real[11], data_device_real[12], data_device_real[13],
        data_device_real[14], data_device_real[15], data_device_real[16],
        data_device_real[17], data_device_real[18], data_device_real[19],
        data_device_real[20], data_device_real[21], data_device_real[22],
        data_device_real[23], phys0, phys1, phys2, f0, f1, f2);

    const REAL norm2 = MAX(MAX(ABS(*f0), ABS(*f1)), ABS(*f2));
    const REAL tol_scaling = data_device_real[24];
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
    *eta2 = xi2;
  }
};

} // namespace Newton
} // namespace NESO

#endif
