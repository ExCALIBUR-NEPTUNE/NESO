#ifndef ___NESO_PARTICLE_MAPPING_NEWTON_TRIANGLE_EMBED_3D_H__
#define ___NESO_PARTICLE_MAPPING_NEWTON_TRIANGLE_EMBED_3D_H__

#include "generated_linear/linear_newton_implementation.hpp"
#include "nektar_interface/special_functions.hpp"
#include "particle_cell_mapping_newton.hpp"
#include <neso_particles.hpp>

using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Newton {

/**
 * Implementation for linear sided triangle within the Newton iteration
 * framework.
 */
struct MappingTriangleLinear2DEmbed3D
    : MappingNewtonIterationBase<MappingTriangleLinear2DEmbed3D> {

  inline void write_data_v([[maybe_unused]] SYCLTargetSharedPtr sycl_target,
                           GeometrySharedPtr geom, void *data_host,
                           void *data_device) {

    REAL *data_device_real = static_cast<REAL *>(data_device);
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

    data_device_real[0] = v00;
    data_device_real[1] = v01;
    data_device_real[2] = v02;
    data_device_real[3] = e10_0;
    data_device_real[4] = e10_1;
    data_device_real[5] = e10_2;
    data_device_real[6] = e20_0;
    data_device_real[7] = e20_1;
    data_device_real[8] = e20_2;

    // Exit tolerance scaling applied by Nektar++
    auto m_xmap = geom->GetXmap();
    auto m_geomFactors = geom->GetGeomFactors();
    Array<OneD, const NekDouble> Jac =
        m_geomFactors->GetJac(m_xmap->GetPointsKeys());
    NekDouble tol_scaling =
        Vmath::Vsum(Jac.size(), Jac, 1) / ((NekDouble)Jac.size());
    data_device_real[9] = ABS(1.0 / tol_scaling);
  }

  inline void free_data_v(void *data_host) { return; }

  inline std::size_t data_size_host_v() { return 0; }

  inline std::size_t data_size_device_v() { return (10) * sizeof(REAL); }

  inline void newton_step_v(const void *d_data, const REAL xi0, const REAL xi1,
                            const REAL xi2, const REAL phys0, const REAL phys1,
                            const REAL phys2, const REAL f0, const REAL f1,
                            const REAL f2, REAL *xin0, REAL *xin1, REAL *xin2) {
    // For linear sided triangles the set initial iteration method actually
    // does the entire inverse mapping.
    this->set_initial_iteration_v(d_data, phys0, phys1, phys2, xin0, xin1,
                                  xin2);
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
    const REAL e10_0 = data_device_real[3];
    const REAL e10_1 = data_device_real[4];
    const REAL e10_2 = data_device_real[5];
    const REAL e20_0 = data_device_real[6];
    const REAL e20_1 = data_device_real[7];
    const REAL e20_2 = data_device_real[8];

    const REAL x0 = (xi0 + 1.0) * 0.5;
    const REAL x1 = (xi1 + 1.0) * 0.5;

    const REAL tmp0 = v00 + x0 * e10_0 + x1 * e20_0;
    const REAL tmp1 = v01 + x0 * e10_1 + x1 * e20_1;
    const REAL tmp2 = v02 + x0 * e10_2 + x1 * e20_2;

    *f0 = tmp0 - phys0;
    *f1 = tmp1 - phys1;
    *f2 = tmp2 - phys2;

    const REAL norm2 = MAX(MAX(ABS(*f0), ABS(*f1)), ABS(*f2));
    const REAL tol_scaling = data_device_real[9];
    const REAL scaled_norm2 = norm2 * tol_scaling;
    return scaled_norm2;
  }

  inline int get_ndim_v() { return 3; }

  inline void set_initial_iteration_v(const void *d_data, const REAL phys0,
                                      const REAL phys1, const REAL phys2,
                                      REAL *xi0, REAL *xi1, REAL *xi2) {

    const REAL *data_device_real = static_cast<const REAL *>(d_data);
    const REAL v00 = data_device_real[0];
    const REAL v01 = data_device_real[1];
    const REAL v02 = data_device_real[2];
    const REAL e10_0 = data_device_real[3];
    const REAL e10_1 = data_device_real[4];
    const REAL e10_2 = data_device_real[5];
    const REAL e20_0 = data_device_real[6];
    const REAL e20_1 = data_device_real[7];
    const REAL e20_2 = data_device_real[8];

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

  inline void loc_coord_to_loc_collapsed_v(const void *d_data, const REAL xi0,
                                           const REAL xi1, const REAL xi2,
                                           REAL *eta0, REAL *eta1, REAL *eta2) {
    *eta2 = 0.0;
    GeometryInterface::Triangle{}.loc_coord_to_loc_collapsed(
        xi0, xi1, eta0, eta1);
  }
};

} // namespace Newton
} // namespace NESO

#endif
