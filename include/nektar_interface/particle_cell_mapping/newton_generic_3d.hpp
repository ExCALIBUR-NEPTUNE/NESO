#ifndef ___NESO_PARTICLE_MAPPING_NEWTON_GENERIC_3D_H__
#define ___NESO_PARTICLE_MAPPING_NEWTON_GENERIC_3D_H__

#include "../bary_interpolation/bary_evaluation.hpp"
#include "../coordinate_mapping.hpp"
#include "../utility_sycl.hpp"
#include "mapping_newton_iteration_base.hpp"
#include <neso_particles.hpp>

using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Newton {

struct MappingGeneric3D;
template <> struct mapping_host_device_types<MappingGeneric3D> {
  struct DataDevice {
    int shape_type_int;
    NewtonRelativeExitTolerances residual_scaling;
    int num_phys0;
    int num_phys1;
    int num_phys2;
    REAL *z0;
    REAL *z1;
    REAL *z2;
    REAL *bw0;
    REAL *bw1;
    REAL *bw2;
    REAL *physvals;
    REAL *physvals_deriv;
  };

  struct DataHost {
    std::size_t data_size_local;
    std::unique_ptr<BufferDevice<REAL>> d_zbw;
  };
  using DataLocal = REAL;
};

struct MappingGeneric3D : MappingNewtonIterationBase<MappingGeneric3D> {

  inline void write_data_v(SYCLTargetSharedPtr sycl_target,
                           GeometrySharedPtr geom, DataHost *data_host,
                           DataDevice *data_device) {

    auto lambda_as_vector = [](const auto &a) -> std::vector<REAL> {
      const std::size_t size = a.size();
      std::vector<REAL> v(size);
      for (int ix = 0; ix < size; ix++) {
        v.at(ix) = a[ix];
      }
      return v;
    };

    auto exp = geom->GetXmap();
    auto base = exp->GetBase();
    NESOASSERT(base.size() == 3, "Expected base of size 3.");
    const auto &z0 = lambda_as_vector(base[0]->GetZ());
    const auto &bw0 = lambda_as_vector(base[0]->GetBaryWeights());
    const auto &z1 = lambda_as_vector(base[1]->GetZ());
    const auto &bw1 = lambda_as_vector(base[1]->GetBaryWeights());
    const auto &z2 = lambda_as_vector(base[2]->GetZ());
    const auto &bw2 = lambda_as_vector(base[2]->GetBaryWeights());

    const int num_phys0 = z0.size();
    const int num_phys1 = z1.size();
    const int num_phys2 = z2.size();
    const int num_phys_total = num_phys0 + num_phys1 + num_phys2;
    const int num_physvals = num_phys0 * num_phys1 * num_phys2;
    data_device->num_phys0 = num_phys0;
    data_device->num_phys1 = num_phys1;
    data_device->num_phys2 = num_phys2;

    // push the quadrature points and weights into a device buffer
    std::vector<REAL> s_zbw;
    const int num_elements = 2 * num_phys_total + 12 * num_physvals;
    s_zbw.reserve(num_elements);
    s_zbw.insert(s_zbw.end(), z0.begin(), z0.end());
    s_zbw.insert(s_zbw.end(), z1.begin(), z1.end());
    s_zbw.insert(s_zbw.end(), z2.begin(), z2.end());
    s_zbw.insert(s_zbw.end(), bw0.begin(), bw0.end());
    s_zbw.insert(s_zbw.end(), bw1.begin(), bw1.end());
    s_zbw.insert(s_zbw.end(), bw2.begin(), bw2.end());

    NESOASSERT(exp->GetTotPoints() == num_physvals,
               "Expected these two evaluations of the number of quadrature "
               "points to match.");

    std::vector<REAL> interlaced_tmp(3 * num_physvals);
    // push the function physvals onto the vector
    Array<OneD, NekDouble> physvals0(num_physvals);
    Array<OneD, NekDouble> physvals1(num_physvals);
    Array<OneD, NekDouble> physvals2(num_physvals);
    exp->BwdTrans(geom->GetCoeffs(0), physvals0);
    exp->BwdTrans(geom->GetCoeffs(1), physvals1);
    exp->BwdTrans(geom->GetCoeffs(2), physvals2);
    for (int ix = 0; ix < num_physvals; ix++) {
      interlaced_tmp.at(3 * ix + 0) = physvals0[ix];
      interlaced_tmp.at(3 * ix + 1) = physvals1[ix];
      interlaced_tmp.at(3 * ix + 2) = physvals2[ix];
    }
    s_zbw.insert(s_zbw.end(), interlaced_tmp.begin(), interlaced_tmp.end());

    // push the function deriv phsvals onto the vector
    interlaced_tmp.resize(9 * num_physvals);

    Array<OneD, NekDouble> D0D0(num_physvals);
    Array<OneD, NekDouble> D0D1(num_physvals);
    Array<OneD, NekDouble> D0D2(num_physvals);
    Array<OneD, NekDouble> D1D0(num_physvals);
    Array<OneD, NekDouble> D1D1(num_physvals);
    Array<OneD, NekDouble> D1D2(num_physvals);
    Array<OneD, NekDouble> D2D0(num_physvals);
    Array<OneD, NekDouble> D2D1(num_physvals);
    Array<OneD, NekDouble> D2D2(num_physvals);

    // get the physvals for the derivatives
    exp->PhysDeriv(physvals0, D0D0, D0D1, D0D2);
    exp->PhysDeriv(physvals1, D1D0, D1D1, D1D2);
    exp->PhysDeriv(physvals2, D2D0, D2D1, D2D2);
    for (int ix = 0; ix < num_physvals; ix++) {
      interlaced_tmp.at(9 * ix + 0) = D0D0[ix];
      interlaced_tmp.at(9 * ix + 1) = D0D1[ix];
      interlaced_tmp.at(9 * ix + 2) = D0D2[ix];
      interlaced_tmp.at(9 * ix + 3) = D1D0[ix];
      interlaced_tmp.at(9 * ix + 4) = D1D1[ix];
      interlaced_tmp.at(9 * ix + 5) = D1D2[ix];
      interlaced_tmp.at(9 * ix + 6) = D2D0[ix];
      interlaced_tmp.at(9 * ix + 7) = D2D1[ix];
      interlaced_tmp.at(9 * ix + 8) = D2D2[ix];
    }
    s_zbw.insert(s_zbw.end(), interlaced_tmp.begin(), interlaced_tmp.end());

    // Create a device buffer with the z,bw,physvals
    data_host->d_zbw = std::make_unique<BufferDevice<REAL>>(sycl_target, s_zbw);
    // Number of REALs required for local memory
    data_host->data_size_local = num_phys0 + num_phys1 + num_phys2;

    // store the pointers into the buffer we just made in the device struct so
    // that pointer arithmetric does not have to happen in the kernel but the
    // data is all in one contiguous block
    data_device->z0 = data_host->d_zbw->ptr;
    data_device->z1 = data_device->z0 + num_phys0;
    data_device->z2 = data_device->z1 + num_phys1;
    data_device->bw0 = data_device->z2 + num_phys2;
    data_device->bw1 = data_device->bw0 + num_phys0;
    data_device->bw2 = data_device->bw1 + num_phys1;
    data_device->physvals = data_device->bw2 + num_phys2;
    data_device->physvals_deriv = data_device->physvals + 3 * num_physvals;
    NESOASSERT(data_device->physvals_deriv + 9 * num_physvals ==
                   data_host->d_zbw->ptr + num_elements,
               "Error in pointer arithmetic.");

    create_newton_relative_exit_tolerances(geom,
                                           &data_device->residual_scaling);
    data_device->shape_type_int = static_cast<int>(geom->GetShapeType());
  }

  inline std::size_t data_size_local_v(DataHost *data_host) {
    return data_host->data_size_local;
  }

  inline void newton_step_v(const DataDevice *data_device, const REAL xi0,
                            const REAL xi1, const REAL xi2, const REAL phys0,
                            const REAL phys1, const REAL phys2, const REAL f0,
                            const REAL f1, const REAL f2, REAL *xin0,
                            REAL *xin1, REAL *xin2, DataLocal *local_memory) {
    const DataDevice *d = data_device;

    REAL *div_space0 = local_memory;
    REAL *div_space1 = div_space0 + d->num_phys0;
    REAL *div_space2 = div_space1 + d->num_phys1;
    // The call to Newton step always follows a call to the residual
    // calculation so we assume that the div_space is already initialised.

    REAL J[9];
    Bary::compute_dir_210_interlaced<9>(d->num_phys0, d->num_phys1,
                                        d->num_phys2, d->physvals_deriv,
                                        div_space0, div_space1, div_space2, J);
    REAL *J0 = J;
    REAL *J1 = J + 3;
    REAL *J2 = J + 6;
    const REAL inverse_J = 1.0 / (J0[0] * (J1[1] * J2[2] - J1[2] * J2[1]) -
                                  J0[1] * (J1[0] * J2[2] - J1[2] * J2[0]) +
                                  J0[2] * (J1[0] * J2[1] - J1[1] * J2[0]));

    *xin0 = xi0 + ((J1[1] * J2[2] - J1[2] * J2[1]) * (-f0) -
                   (J0[1] * J2[2] - J0[2] * J2[1]) * (-f1) +
                   (J0[1] * J1[2] - J0[2] * J1[1]) * (-f2)) *
                      inverse_J;

    *xin1 = xi1 - ((J1[0] * J2[2] - J1[2] * J2[0]) * (-f0) -
                   (J0[0] * J2[2] - J0[2] * J2[0]) * (-f1) +
                   (J0[0] * J1[2] - J0[2] * J1[0]) * (-f2)) *
                      inverse_J;

    *xin2 = xi2 + ((J1[0] * J2[1] - J1[1] * J2[0]) * (-f0) -
                   (J0[0] * J2[1] - J0[1] * J2[0]) * (-f1) +
                   (J0[0] * J1[1] - J0[1] * J1[0]) * (-f2)) *
                      inverse_J;
  }

  inline REAL newton_residual_v(const DataDevice *data_device, const REAL xi0,
                                const REAL xi1, const REAL xi2,
                                const REAL phys0, const REAL phys1,
                                const REAL phys2, REAL *f0, REAL *f1, REAL *f2,
                                DataLocal *local_memory) {

    const DataDevice *d = data_device;

    REAL eta0, eta1, eta2;
    this->loc_coord_to_loc_collapsed(data_device, xi0, xi1, xi2, &eta0, &eta1,
                                     &eta2);

    // compute X at xi by evaluating the Bary interpolation at eta
    REAL *div_space0 = local_memory;
    REAL *div_space1 = div_space0 + d->num_phys0;
    REAL *div_space2 = div_space1 + d->num_phys1;

    Bary::preprocess_weights(d->num_phys0, eta0, d->z0, d->bw0, div_space0, 1);
    Bary::preprocess_weights(d->num_phys1, eta1, d->z1, d->bw1, div_space1, 1);
    Bary::preprocess_weights(d->num_phys2, eta2, d->z2, d->bw2, div_space2, 1);

    REAL X[3];

    Bary::compute_dir_210_interlaced<3>(d->num_phys0, d->num_phys1,
                                        d->num_phys2, d->physvals, div_space0,
                                        div_space1, div_space2, X);

    // Residual is defined as F = X(xi) - P
    *f0 = X[0] - phys0;
    *f1 = X[1] - phys1;
    *f2 = X[2] - phys2;

    return d->residual_scaling.get_relative_error_3d(*f0, *f1, *f2);
  }

  inline int get_ndim_v() { return 3; }

  inline void set_initial_iteration_v(const DataDevice *data_device,
                                      const REAL phys0, const REAL phys1,
                                      const REAL phys2, REAL *xi0, REAL *xi1,
                                      REAL *xi2) {

    /**
     * Implementations that avoid singularities:
     *   newton_pyr?
     */

    *xi0 = -0.2;
    *xi1 = -0.2;
    *xi2 = -0.2;
  }

  inline void loc_coord_to_loc_collapsed_v(const DataDevice *data_device,
                                           const REAL xi0, const REAL xi1,
                                           const REAL xi2, REAL *eta0,
                                           REAL *eta1, REAL *eta2) {

    const int shape_type = data_device->shape_type_int;

    GeometryInterface::loc_coord_to_loc_collapsed_3d(shape_type, xi0, xi1, xi2,
                                                     eta0, eta1, eta2);
  }

  inline void loc_collapsed_to_loc_coord_v(const DataDevice *data_device,
                                           const REAL eta0, const REAL eta1,
                                           const REAL eta2, REAL *xi0,
                                           REAL *xi1, REAL *xi2) {

    const int shape_type = data_device->shape_type_int;

    GeometryInterface::loc_collapsed_to_loc_coord(shape_type, eta0, eta1, eta2,
                                                  xi0, xi1, xi2);
  }
};

/**
 * This implementation requires local memory.
 */
template <> struct local_memory_required<MappingGeneric3D> {
  static bool const required = true;
};

} // namespace Newton
} // namespace NESO

#endif
