#ifndef ___NESO_PARTICLE_MAPPING_NEWTON_QUAD_H__
#define ___NESO_PARTICLE_MAPPING_NEWTON_QUAD_H__

#include "generated_linear/linear_newton_implementation.hpp"
#include "particle_cell_mapping_newton.hpp"
#include <neso_particles.hpp>

using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Newton {

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
    data_device_real[8] = ABS(1.0 / tol_scaling);
  }

  inline void free_data_v(void *data_host) { return; }

  inline std::size_t data_size_host_v() { return 0; }

  inline std::size_t data_size_device_v() { return (4 * 2 + 1) * sizeof(REAL); }

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
    Quadrilateral::newton_step_linear_2d(xi0, xi1, v00, v01, v10, v11, v20, v21,
                                         v30, v31, phys0, phys1, f0, f1, xin0,
                                         xin1);
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

    Quadrilateral::newton_f_linear_2d(xi0, xi1, v00, v01, v10, v11, v20, v21,
                                      v30, v31, phys0, phys1, f0, f1);
    *f2 = 0.0;

    const REAL norm2 = MAX(ABS(*f0), ABS(*f1));
    const REAL tol_scaling = data_device_real[8];
    const REAL scaled_norm2 = norm2 * tol_scaling;
    return scaled_norm2;
  }

  inline int get_ndim_v() { return 2; }

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
