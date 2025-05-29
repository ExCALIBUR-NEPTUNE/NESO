#ifndef ___NESO_PARTICLE_MAPPING_NEWTON_QUAD_H__
#define ___NESO_PARTICLE_MAPPING_NEWTON_QUAD_H__

#include "generated_linear/linear_newton_implementation.hpp"
#include "particle_cell_mapping_newton.hpp"
#include <neso_particles.hpp>

using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Newton {

struct MappingQuadLinear2D;

template <> struct mapping_host_device_types<MappingQuadLinear2D> {
  struct DataDevice {
    REAL coordinates[4][2];
    NewtonRelativeExitTolerances residual_scaling;
  };
  using DataHost = NullDataHost;
  using DataLocal = NullDataLocal;
};

struct MappingQuadLinear2D : MappingNewtonIterationBase<MappingQuadLinear2D> {

  inline void write_data_v([[maybe_unused]] SYCLTargetSharedPtr sycl_target,
                           GeometrySharedPtr geom, DataHost *data_host,
                           DataDevice *data_device) {

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
    data_device->coordinates[0][0] = v00;
    data_device->coordinates[0][1] = v01;
    data_device->coordinates[1][0] = v10;
    data_device->coordinates[1][1] = v11;
    data_device->coordinates[2][0] = v20;
    data_device->coordinates[2][1] = v21;
    data_device->coordinates[3][0] = v30;
    data_device->coordinates[3][1] = v31;
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
    const REAL v10 = d_data->coordinates[1][0];
    const REAL v11 = d_data->coordinates[1][1];
    const REAL v20 = d_data->coordinates[2][0];
    const REAL v21 = d_data->coordinates[2][1];
    const REAL v30 = d_data->coordinates[3][0];
    const REAL v31 = d_data->coordinates[3][1];
    *xin2 = 0.0;
    Quadrilateral::newton_step_linear_2d(xi0, xi1, v00, v01, v10, v11, v20, v21,
                                         v30, v31, phys0, phys1, f0, f1, xin0,
                                         xin1);
  }

  inline REAL newton_residual_v(const DataDevice *d_data, const REAL xi0,
                                const REAL xi1, const REAL xi2,
                                const REAL phys0, const REAL phys1,
                                const REAL phys2, REAL *f0, REAL *f1,
                                REAL *f2) {

    const REAL v00 = d_data->coordinates[0][0];
    const REAL v01 = d_data->coordinates[0][1];
    const REAL v10 = d_data->coordinates[1][0];
    const REAL v11 = d_data->coordinates[1][1];
    const REAL v20 = d_data->coordinates[2][0];
    const REAL v21 = d_data->coordinates[2][1];
    const REAL v30 = d_data->coordinates[3][0];
    const REAL v31 = d_data->coordinates[3][1];

    Quadrilateral::newton_f_linear_2d(xi0, xi1, v00, v01, v10, v11, v20, v21,
                                      v30, v31, phys0, phys1, f0, f1);
    *f2 = 0.0;
    return d_data->residual_scaling.get_relative_error_2d(*f0, *f1);
  }

  inline int get_ndim_v() { return 2; }

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
