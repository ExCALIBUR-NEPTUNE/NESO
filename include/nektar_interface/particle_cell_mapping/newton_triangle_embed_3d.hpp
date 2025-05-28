#ifndef ___NESO_PARTICLE_MAPPING_NEWTON_TRIANGLE_EMBED_3D_H__
#define ___NESO_PARTICLE_MAPPING_NEWTON_TRIANGLE_EMBED_3D_H__

#include "../coordinate_mapping.hpp"
#include "nektar_interface/special_functions.hpp"

#include "generated_linear/linear_newton_implementation.hpp"
#include <neso_particles.hpp>

#include "mapping_newton_iteration_base.hpp"
using namespace Nektar;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Newton {

struct MappingTriangleLinear2DEmbed3D;
template <> struct mapping_host_device_types<MappingTriangleLinear2DEmbed3D> {
  struct DataDevice {
    REAL vertex_0[3];
    REAL vectors[2][3];
    NewtonRelativeExitTolerances residual_scaling;
  };
  using DataHost = NullDataHost;
  using DataLocal = NullDataLocal;
};

/**
 * Implementation for linear sided triangle within the Newton iteration
 * framework.
 */
struct MappingTriangleLinear2DEmbed3D
    : MappingNewtonIterationBase<MappingTriangleLinear2DEmbed3D> {

  inline void write_data_v([[maybe_unused]] SYCLTargetSharedPtr sycl_target,
                           GeometrySharedPtr geom, DataHost *data_host,
                           DataDevice *data_device) {

    auto v0 = geom->GetVertex(0);
    auto v1 = geom->GetVertex(1);
    auto v2 = geom->GetVertex(2);

    NESOASSERT(v0->GetCoordim() == 3, "expected coordim == 3");
    NESOASSERT(v1->GetCoordim() == 3, "expected coordim == 3");
    NESOASSERT(v2->GetCoordim() == 3, "expected coordim == 3");

    NekDouble v00;
    NekDouble v01;
    NekDouble v02;
    NekDouble v10;
    NekDouble v11;
    NekDouble v12;
    NekDouble v20;
    NekDouble v21;
    NekDouble v22;
    v0->GetCoords(v00, v01, v02);
    v1->GetCoords(v10, v11, v12);
    v2->GetCoords(v20, v21, v22);

    const REAL e10_0 = v10 - v00;
    const REAL e10_1 = v11 - v01;
    const REAL e10_2 = v12 - v02;
    const REAL e20_0 = v20 - v00;
    const REAL e20_1 = v21 - v01;
    const REAL e20_2 = v22 - v02;

    data_device->vertex_0[0] = v00;
    data_device->vertex_0[1] = v01;
    data_device->vertex_0[2] = v02;
    data_device->vectors[0][0] = e10_0;
    data_device->vectors[0][1] = e10_1;
    data_device->vectors[0][2] = e10_2;
    data_device->vectors[1][0] = e20_0;
    data_device->vectors[1][1] = e20_1;
    data_device->vectors[1][2] = e20_2;

    create_newton_relative_exit_tolerances(geom,
                                           &data_device->residual_scaling);
  }

  inline void newton_step_v(const DataDevice *d_data, const REAL xi0,
                            const REAL xi1, const REAL xi2, const REAL phys0,
                            const REAL phys1, const REAL phys2, const REAL f0,
                            const REAL f1, const REAL f2, REAL *xin0,
                            REAL *xin1, REAL *xin2) {
    // For linear sided triangles the set initial iteration method actually
    // does the entire inverse mapping.
    this->set_initial_iteration_v(d_data, phys0, phys1, phys2, xin0, xin1,
                                  xin2);
  }

  inline REAL newton_residual_v(const DataDevice *d_data, const REAL xi0,
                                const REAL xi1, const REAL xi2,
                                const REAL phys0, const REAL phys1,
                                const REAL phys2, REAL *f0, REAL *f1,
                                REAL *f2) {

    const REAL v00 = d_data->vertex_0[0];
    const REAL v01 = d_data->vertex_0[1];
    const REAL v02 = d_data->vertex_0[2];
    const REAL e10_0 = d_data->vectors[0][0];
    const REAL e10_1 = d_data->vectors[0][1];
    const REAL e10_2 = d_data->vectors[0][2];
    const REAL e20_0 = d_data->vectors[1][0];
    const REAL e20_1 = d_data->vectors[1][1];
    const REAL e20_2 = d_data->vectors[1][2];

    const REAL x0 = (xi0 + 1.0) * 0.5;
    const REAL x1 = (xi1 + 1.0) * 0.5;

    const REAL tmp0 = v00 + x0 * e10_0 + x1 * e20_0;
    const REAL tmp1 = v01 + x0 * e10_1 + x1 * e20_1;
    const REAL tmp2 = v02 + x0 * e10_2 + x1 * e20_2;

    *f0 = tmp0 - phys0;
    *f1 = tmp1 - phys1;
    *f2 = tmp2 - phys2;

    return d_data->residual_scaling.get_relative_error_3d(*f0, *f1, *f2);
  }

  inline int get_ndim_v() { return 3; }

  inline void set_initial_iteration_v(const DataDevice *d_data,
                                      const REAL phys0, const REAL phys1,
                                      const REAL phys2, REAL *xi0, REAL *xi1,
                                      REAL *xi2) {

    const REAL v00 = d_data->vertex_0[0];
    const REAL v01 = d_data->vertex_0[1];
    const REAL v02 = d_data->vertex_0[2];
    const REAL e10_0 = d_data->vectors[0][0];
    const REAL e10_1 = d_data->vectors[0][1];
    const REAL e10_2 = d_data->vectors[0][2];
    const REAL e20_0 = d_data->vectors[1][0];
    const REAL e20_1 = d_data->vectors[1][1];
    const REAL e20_2 = d_data->vectors[1][2];

    const REAL er_0 = phys0 - v00;
    const REAL er_1 = phys1 - v01;
    const REAL er_2 = phys2 - v02;

    MAPPING_CROSS_PRODUCT_3D(e10_0, e10_1, e10_2, e20_0, e20_1, e20_2,
                             const REAL norm_0, const REAL norm_1,
                             const REAL norm_2)
    MAPPING_CROSS_PRODUCT_3D(norm_0, norm_1, norm_2, e10_0, e10_1, e10_2,
                             const REAL orth1_0, const REAL orth1_1,
                             const REAL orth1_2)
    MAPPING_CROSS_PRODUCT_3D(norm_0, norm_1, norm_2, e20_0, e20_1, e20_2,
                             const REAL orth2_0, const REAL orth2_1,
                             const REAL orth2_2)

    const REAL scale0 =
        MAPPING_DOT_PRODUCT_3D(er_0, er_1, er_2, orth2_0, orth2_1, orth2_2) /
        MAPPING_DOT_PRODUCT_3D(e10_0, e10_1, e10_2, orth2_0, orth2_1, orth2_2);

    *xi0 = 2.0 * scale0 - 1.0;
    const REAL scale1 =
        MAPPING_DOT_PRODUCT_3D(er_0, er_1, er_2, orth1_0, orth1_1, orth1_2) /
        MAPPING_DOT_PRODUCT_3D(e20_0, e20_1, e20_2, orth1_0, orth1_1, orth1_2);
    *xi1 = 2.0 * scale1 - 1.0;
    *xi2 = 0.0;
  }

  inline void loc_coord_to_loc_collapsed_v(const DataDevice *d_data,
                                           const REAL xi0, const REAL xi1,
                                           const REAL xi2, REAL *eta0,
                                           REAL *eta1, REAL *eta2) {
    *eta2 = 0.0;
    GeometryInterface::Triangle{}.loc_coord_to_loc_collapsed(xi0, xi1, eta0,
                                                             eta1);
  }

  inline void loc_collapsed_to_loc_coord_v(const DataDevice *d_data,
                                           const REAL eta0, const REAL eta1,
                                           const REAL eta2, REAL *xi0,
                                           REAL *xi1, REAL *xi2) {

    *xi2 = 0.0;
    GeometryInterface::Triangle{}.loc_collapsed_to_loc_coord(eta0, eta1, xi0,
                                                             xi1);
  }
};

} // namespace Newton
} // namespace NESO

#endif
