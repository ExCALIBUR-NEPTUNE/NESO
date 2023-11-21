#ifndef ___NESO_PARTICLE_MAPPING_NEWTON_GENERIC_3D_H__
#define ___NESO_PARTICLE_MAPPING_NEWTON_GENERIC_3D_H__

#include "../bary_interpolation/bary_evaluation.hpp"
#include "generated_linear/linear_newton_implementation.hpp"
#include "particle_cell_mapping_newton.hpp"
#include <neso_particles.hpp>

using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Newton {

namespace Generic3D {
struct DataDevice {
  int shape_type_int;
  REAL tol_scaling;
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
};
struct DataHost {
  std::size_t data_size_local;
  std::unique_ptr<BufferDevice<REAL>> d_zbw;
  inline void free() { this->d_zbw.reset(); }
};
} // namespace Generic3D

struct MappingGeneric3D : MappingNewtonIterationBase<MappingGeneric3D> {

  inline void write_data_v(SYCLTargetSharedPtr sycl_target,
                           GeometrySharedPtr geom, void *data_host,
                           void *data_device) {
    Generic3D::DataHost *h_data = static_cast<Generic3D::DataHost *>(data_host);
    Generic3D::DataDevice *d_data =
        static_cast<Generic3D::DataDevice *>(data_device);

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
    d_data->num_phys0 = num_phys0;
    d_data->num_phys1 = num_phys1;
    d_data->num_phys2 = num_phys2;

    // push the quadrature points and weights into a device buffer
    std::vector<REAL> s_zbw;
    const int num_elements = 2 * num_phys_total + 3 * num_physvals;
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

    // push the physvals onto the vector
    Array<OneD, NekDouble> tmp(num_physvals);
    exp->BwdTrans(geom->GetCoeffs(0), tmp);
    const auto physvals0 = lambda_as_vector(tmp);
    exp->BwdTrans(geom->GetCoeffs(1), tmp);
    const auto physvals1 = lambda_as_vector(tmp);
    exp->BwdTrans(geom->GetCoeffs(2), tmp);
    const auto physvals2 = lambda_as_vector(tmp);

    std::vector<REAL> interlaced_tmp(3 * num_physvals);
    for (int ix = 0; ix < num_physvals; ix++) {
      interlaced_tmp.at(3 * ix + 0) = physvals0.at(ix);
      interlaced_tmp.at(3 * ix + 1) = physvals1.at(ix);
      interlaced_tmp.at(3 * ix + 2) = physvals2.at(ix);
    }

    s_zbw.insert(s_zbw.end(), interlaced_tmp.begin(), interlaced_tmp.end());

    // Create a device buffer with the z,bw,physvals
    h_data->d_zbw = std::make_unique<BufferDevice<REAL>>(sycl_target, s_zbw);

    // store the pointers into the buffer we just made in the device struct so
    // that pointer arithmetric does not have to happen in the kernel but the
    // data is all in one contiguous block
    d_data->z0 = h_data->d_zbw->ptr;
    d_data->z1 = d_data->z0 + num_phys0;
    d_data->z2 = d_data->z1 + num_phys1;
    d_data->bw0 = d_data->z2 + num_phys2;
    d_data->bw1 = d_data->bw0 + num_phys0;
    d_data->bw2 = d_data->bw1 + num_phys1;
    d_data->physvals = d_data->bw2 + num_phys2;
    NESOASSERT(d_data->physvals + 3 * num_physvals ==
                   h_data->d_zbw->ptr + num_elements,
               "Error in pointer arithmetic.");

    // Exit tolerance scaling applied by Nektar++
    auto m_xmap = geom->GetXmap();
    auto m_geomFactors = geom->GetGeomFactors();
    Array<OneD, const NekDouble> Jac =
        m_geomFactors->GetJac(m_xmap->GetPointsKeys());
    NekDouble tol_scaling =
        Vmath::Vsum(Jac.size(), Jac, 1) / ((NekDouble)Jac.size());
    d_data->tol_scaling = ABS(1.0 / tol_scaling);
    d_data->shape_type_int = static_cast<int>(geom->GetShapeType());
  }

  inline void free_data_v(void *data_host) {
    Generic3D::DataHost *h_data = static_cast<Generic3D::DataHost *>(data_host);
    h_data->free();
  }

  inline std::size_t data_size_host_v() { return sizeof(Generic3D::DataHost); }

  inline std::size_t data_size_local_v(void *data_host) {
    return static_cast<Generic3D::DataHost *>(data_host)->data_size_local;
  }

  inline std::size_t data_size_device_v() {
    return sizeof(Generic3D::DataDevice);
  }

  inline void newton_step_v(const void *d_data, const REAL xi0, const REAL xi1,
                            const REAL xi2, const REAL phys0, const REAL phys1,
                            const REAL phys2, const REAL f0, const REAL f1,
                            const REAL f2, REAL *xin0, REAL *xin1, REAL *xin2,
                            void *local_memory) {
    const Generic3D::DataDevice *data =
        static_cast<const Generic3D::DataDevice *>(d_data);
  }

  inline REAL newton_residual_v(const void *d_data, const REAL xi0,
                                const REAL xi1, const REAL xi2,
                                const REAL phys0, const REAL phys1,
                                const REAL phys2, REAL *f0, REAL *f1, REAL *f2,
                                void *local_memory) {

    const Generic3D::DataDevice *d =
        static_cast<const Generic3D::DataDevice *>(d_data);
    const REAL *data_device_real = static_cast<const REAL *>(d_data);

    REAL eta0, eta1, eta2;
    this->loc_coord_to_loc_collapsed(d_data, xi0, xi1, xi2, &eta0, &eta1,
                                     &eta2);

    // compute X at xi by evaluating the Bary interpolation at eta
    REAL *div_space0 = static_cast<REAL *>(local_memory);
    REAL *div_space1 = div_space0 + d->num_phys0;
    REAL *div_space2 = div_space1 + d->num_phys1;
    Bary::preprocess_weights(d->num_phys0, eta0, d->z0, d->bw0, div_space0);
    Bary::preprocess_weights(d->num_phys1, eta1, d->z1, d->bw1, div_space1);
    Bary::preprocess_weights(d->num_phys2, eta2, d->z2, d->bw2, div_space2);

    REAL X[3];

    Bary::compute_dir_210_interlaced<3>(d->num_phys0, d->num_phys1,
                                        d->num_phys2, d->physvals, div_space0,
                                        div_space1, div_space2, X);

    // Residual is defined as F = X(xi) - P
    *f0 = X[0] - phys0;
    *f1 = X[1] - phys1;
    *f2 = X[2] - phys2;

    const REAL norm2 = MAX(MAX(ABS(*f0), ABS(*f1)), ABS(*f2));
    const REAL tol_scaling = d->tol_scaling;
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

    const Generic3D::DataDevice *data =
        static_cast<const Generic3D::DataDevice *>(d_data);
    const int shape_type = data->shape_type_int;

    constexpr int shape_type_tet =
        shape_type_to_int(LibUtilities::eTetrahedron);
    constexpr int shape_type_pyr = shape_type_to_int(LibUtilities::ePyramid);
    constexpr int shape_type_hex = shape_type_to_int(LibUtilities::eHexahedron);

    NekDouble d2 = 1.0 - xi2;
    if (fabs(d2) < NekConstants::kNekZeroTol) {
      if (d2 >= 0.) {
        d2 = NekConstants::kNekZeroTol;
      } else {
        d2 = -NekConstants::kNekZeroTol;
      }
    }
    NekDouble d12 = -xi1 - xi2;
    if (fabs(d12) < NekConstants::kNekZeroTol) {
      if (d12 >= 0.) {
        d12 = NekConstants::kNekZeroTol;
      } else {
        d12 = -NekConstants::kNekZeroTol;
      }
    }

    const REAL id2x2 = 2.0 / d2;
    const REAL a = 1.0 + xi0;
    const REAL b = (1.0 + xi1) * id2x2 - 1.0;
    const REAL c = a * id2x2 - 1.0;
    const REAL d = 2.0 * a / d12 - 1.0;

    *eta0 = (shape_type == shape_type_tet)   ? d
            : (shape_type == shape_type_hex) ? xi0
                                             : c;

    *eta1 = ((shape_type == shape_type_tet) || (shape_type == shape_type_pyr))
                ? b
                : xi1;
    *eta2 = xi2;
  }
};

template <> struct local_memory_required<MappingGeneric3D> {
  static bool const required = true;
};

} // namespace Newton
} // namespace NESO

#endif
