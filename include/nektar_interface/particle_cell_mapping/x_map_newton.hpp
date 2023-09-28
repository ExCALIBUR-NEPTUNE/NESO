#ifndef __X_MAP_NEWTON_H__
#define __X_MAP_NEWTON_H__

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace NESO::Particles;
using namespace Nektar::SpatialDomains;

#include "mapping_newton_iteration_base.hpp"

namespace NESO::Newton {
/**
 *  Utility class to evaluate X maps and their inverse.
 */
template <typename NEWTON_TYPE> class XMapNewton {
protected:
  /// Disable (implicit) copies.
  XMapNewton(const XMapNewton &st) = delete;
  /// Disable (implicit) copies.
  XMapNewton &operator=(XMapNewton const &a) = delete;

  MappingNewtonIterationBase<NEWTON_TYPE> newton_type;

  SYCLTargetSharedPtr sycl_target;
  const std::size_t num_bytes_per_map_device;
  const std::size_t num_bytes_per_map_host;

  /// The data required to perform newton iterations for each geom on the
  /// device.
  std::unique_ptr<BufferDeviceHost<char>> dh_data;

  std::unique_ptr<BufferDeviceHost<REAL>> dh_fdata;
  /// The data required to perform newton iterations for each geom on the host.
  std::vector<char> h_data;

  template <typename U> inline void write_data(U &geom) {
    if (this->num_bytes_per_map_host) {
      this->h_data = std::vector<char>(this->num_bytes_per_map_host);
    }
    if (this->num_bytes_per_map_device) {
      this->dh_data = std::make_unique<BufferDeviceHost<char>>(
          this->sycl_target, this->num_bytes_per_map_device);
    }
    auto d_data_ptr = (this->num_bytes_per_map_device)
                          ? this->dh_data->h_buffer.ptr
                          : nullptr;
    auto h_data_ptr =
        (this->num_bytes_per_map_host) ? this->h_data.data() : nullptr;

    this->newton_type.write_data(geom, h_data_ptr, d_data_ptr);
    this->dh_data->host_to_device();
  }

public:
  ~XMapNewton() {
    auto h_data_ptr =
        (this->num_bytes_per_map_host) ? this->h_data.data() : nullptr;
    this->newton_type.free_data(h_data_ptr);
  }

  /**
   *  Create new instance from Newton implementation.
   *
   *  @param sycl_target SYCLTarget to use for computation.
   *  @param geom Nektar++ geometry type that matches the Newton method
   *  implementation.
   */
  template <typename TYPE_GEOM>
  XMapNewton(SYCLTargetSharedPtr sycl_target, std::shared_ptr<TYPE_GEOM> geom)
      : newton_type(MappingNewtonIterationBase<NEWTON_TYPE>()),
        sycl_target(sycl_target),
        num_bytes_per_map_host(newton_type.data_size_host()),
        num_bytes_per_map_device(newton_type.data_size_device()) {
    this->write_data(geom);
    this->dh_fdata =
        std::make_unique<BufferDeviceHost<REAL>>(this->sycl_target, 4);
  }

  /**
   * For a reference position xi compute the global position X(xi).
   *
   * @param[in] xi0 Reference position, x component.
   * @param[in] xi1 Reference position, y component.
   * @param[in] xi2 Reference position, z component.
   * @param[in, out] xi0 Global position X(xi), x component.
   * @param[in, out] xi1 Global position X(xi), y component.
   * @param[in, out] xi2 Global position X(xi), z component.
   */
  inline void x(const REAL xi0, const REAL xi1, const REAL xi2, REAL *phys0,
                REAL *phys1, REAL *phys2) {

    auto k_map_data = this->dh_data->d_buffer.ptr;
    auto k_fdata = this->dh_fdata->d_buffer.ptr;

    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.single_task<>([=]() {
            MappingNewtonIterationBase<NEWTON_TYPE> k_newton_type{};

            REAL f0 = 0.0;
            REAL f1 = 0.0;
            REAL f2 = 0.0;
            const REAL p0 = 0.0;
            const REAL p1 = 0.0;
            const REAL p2 = 0.0;

            k_newton_type.newton_residual(k_map_data, xi0, xi1, xi2, p0, p1, p2,
                                          &f0, &f1, &f2);

            k_fdata[0] = f0;
            k_fdata[1] = f1;
            k_fdata[2] = f2;
          });
        })
        .wait_and_throw();

    this->dh_fdata->device_to_host();
    *phys0 = this->dh_fdata->h_buffer.ptr[0];
    *phys1 = this->dh_fdata->h_buffer.ptr[1];
    *phys2 = this->dh_fdata->h_buffer.ptr[2];
  }

  /**
   * For a position X(xi) compute the reference position xi via Newton
   * iteration.
   *
   * @param[in] xi0 Global position X(xi), x component.
   * @param[in] xi1 Global position X(xi), y component.
   * @param[in] xi2 Global position X(xi), z component.
   * @param[in, out] xi0 Reference position, x component.
   * @param[in, out] xi1 Reference position, y component.
   * @param[in, out] xi2 Reference position, z component.
   * @param[in] tol Optional exit tolerance for Newton iterations.
   * @returns True if inverse is found otherwise false.
   */
  inline bool x_inverse(const REAL phys0, const REAL phys1, const REAL phys2,
                        REAL *xi0, REAL *xi1, REAL *xi2,
                        const REAL tol = 1.0e-10) {

    const int k_max_iterations = 51;
    auto k_map_data = this->dh_data->d_buffer.ptr;
    auto k_fdata = this->dh_fdata->d_buffer.ptr;
    const REAL k_tol = tol;

    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.single_task<>([=]() {
            MappingNewtonIterationBase<NEWTON_TYPE> k_newton_type{};

            const REAL p0 = phys0;
            const REAL p1 = phys1;
            const REAL p2 = phys2;

            REAL k_xi0;
            REAL k_xi1;
            REAL k_xi2;
            k_newton_type.set_initial_iteration(k_map_data, p0, p1, p2, &k_xi0,
                                                &k_xi1, &k_xi2);

            // Start of Newton iteration
            REAL xin0, xin1, xin2;
            REAL f0, f1, f2;

            REAL residual = k_newton_type.newton_residual(
                k_map_data, k_xi0, k_xi1, k_xi2, p0, p1, p2, &f0, &f1, &f2);

            bool diverged = false;

            for (int stepx = 0; ((stepx < k_max_iterations) &&
                                 (residual > k_tol) && (!diverged));
                 stepx++) {
              k_newton_type.newton_step(k_map_data, k_xi0, k_xi1, k_xi2, p0, p1,
                                        p2, f0, f1, f2, &xin0, &xin1, &xin2);

              k_xi0 = xin0;
              k_xi1 = xin1;
              k_xi2 = xin2;

              residual = k_newton_type.newton_residual(
                  k_map_data, k_xi0, k_xi1, k_xi2, p0, p1, p2, &f0, &f1, &f2);

              diverged = (ABS(k_xi0) > 15.0) || (ABS(k_xi1) > 15.0) ||
                         (ABS(k_xi2) > 15.0);
            }

            k_fdata[0] = k_xi0;
            k_fdata[1] = k_xi1;
            k_fdata[2] = k_xi2;
            k_fdata[3] = (residual <= tol) ? 1 : -1;
          });
        })
        .wait_and_throw();

    this->dh_fdata->device_to_host();
    *xi0 = this->dh_fdata->h_buffer.ptr[0];
    *xi1 = this->dh_fdata->h_buffer.ptr[1];
    *xi2 = this->dh_fdata->h_buffer.ptr[2];
    return (this->dh_fdata->h_buffer.ptr[3] > 0);
  }
};

} // namespace NESO::Newton

#endif
