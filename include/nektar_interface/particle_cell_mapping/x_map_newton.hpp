#ifndef __X_MAP_NEWTON_H__
#define __X_MAP_NEWTON_H__

#include "x_map_newton_kernel.hpp"
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
  std::size_t num_bytes_local;

  // variables for higher order grids
  int num_modes;
  int num_modes_factor;
  int ndim;

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

    this->newton_type.write_data(this->sycl_target, geom, h_data_ptr,
                                 d_data_ptr);
    this->dh_data->host_to_device();
    this->num_bytes_local =
        std::max(static_cast<std::size_t>(1),
                 this->newton_type.data_size_local(h_data_ptr));
    this->num_modes = geom->GetXmap()->EvalBasisNumModesMax();
    MappingNewtonIterationBase<NEWTON_TYPE> newton_type;
    this->ndim = newton_type.get_ndim();
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
   *  @param num_modes_factor Factor to multiply the number of modes by to
   *  create the grid in reference space. Default 1, i.e a (num_modes *
   *  num_modes_factor)^(ndim)).
   */
  template <typename TYPE_GEOM>
  XMapNewton(SYCLTargetSharedPtr sycl_target, std::shared_ptr<TYPE_GEOM> geom,
             const int num_modes_factor = 1)
      : newton_type(MappingNewtonIterationBase<NEWTON_TYPE>()),
        sycl_target(sycl_target),
        num_bytes_per_map_host(newton_type.data_size_host()),
        num_bytes_per_map_device(newton_type.data_size_device()),
        num_modes_factor(num_modes_factor) {
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
   * @param[in, out] phys0 Global position X(xi), x component.
   * @param[in, out] phys1 Global position X(xi), y component.
   * @param[in, out] phys2 Global position X(xi), z component.
   */
  inline void x(const REAL xi0, const REAL xi1, const REAL xi2, REAL *phys0,
                REAL *phys1, REAL *phys2) {

    auto k_map_data = this->dh_data->d_buffer.ptr;
    auto k_fdata = this->dh_fdata->d_buffer.ptr;
    const std::size_t num_bytes_local = this->num_bytes_local;

    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          sycl::accessor<unsigned char, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>
              local_mem(sycl::range<1>(num_bytes_local), cgh);

          cgh.parallel_for<>(
              sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
              [=](auto idx) {
                MappingNewtonIterationBase<NEWTON_TYPE> k_newton_type{};

                REAL f0 = 0.0;
                REAL f1 = 0.0;
                REAL f2 = 0.0;
                const REAL p0 = 0.0;
                const REAL p1 = 0.0;
                const REAL p2 = 0.0;

                k_newton_type.newton_residual(k_map_data, xi0, xi1, xi2, p0, p1,
                                              p2, &f0, &f1, &f2, &local_mem[0]);

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
   * iteration. This method will attempt to find a root inside the reference
   * element.
   *
   * @param[in] phys0 Global position X(xi), x component.
   * @param[in] phys1 Global position X(xi), y component.
   * @param[in] phys2 Global position X(xi), z component.
   * @param[in, out] xi0 Reference position, x component.
   * @param[in, out] xi1 Reference position, y component.
   * @param[in, out] xi2 Reference position, z component.
   * @param[in] tol Optional exit tolerance for Newton iterations.
   * @param[in] contained_tol Optional tolerance for determining if a point is
   * within the reference cell.
   * @returns True if inverse is found otherwise false.
   */
  inline bool x_inverse(const REAL phys0, const REAL phys1, const REAL phys2,
                        REAL *xi0, REAL *xi1, REAL *xi2,
                        const REAL tol = 1.0e-10,
                        const REAL contained_tol = 1.0e-10) {

    const int k_max_iterations = 51;
    auto k_map_data = this->dh_data->d_buffer.ptr;
    auto k_fdata = this->dh_fdata->d_buffer.ptr;
    const REAL k_tol = tol;
    const double k_contained_tol = contained_tol;
    const std::size_t num_bytes_local = this->num_bytes_local;

    const int k_ndim = this->ndim;
    const int grid_size = this->num_modes_factor * this->num_modes;
    const int k_grid_size_x = std::max(grid_size - 1, 1);
    const int k_grid_size_y = k_ndim > 1 ? k_grid_size_x : 1;
    const int k_grid_size_z = k_ndim > 2 ? k_grid_size_x : 1;
    const REAL k_grid_width = 1.8 / (k_grid_size_x);

    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          sycl::accessor<unsigned char, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>
              local_mem(sycl::range<1>(num_bytes_local), cgh);
          cgh.parallel_for<>(
              sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
              [=](auto idx) {
                printf("NEWTON:\n");
                MappingNewtonIterationBase<NEWTON_TYPE> k_newton_type{};

                const REAL p0 = phys0;
                const REAL p1 = phys1;
                const REAL p2 = phys2;
                REAL k_xi0;
                REAL k_xi1;
                REAL k_xi2;
                REAL residual;
                bool cell_found = false;

                for (int g2 = 0; (g2 <= k_grid_size_z) && (!cell_found); g2++) {
                  for (int g1 = 0; (g1 <= k_grid_size_y) && (!cell_found);
                       g1++) {
                    for (int g0 = 0; (g0 <= k_grid_size_x) && (!cell_found);
                         g0++) {

                      k_xi0 = -0.9 + g0 * k_grid_width;
                      k_xi1 = -0.9 + g1 * k_grid_width;
                      k_xi2 = -0.9 + g2 * k_grid_width;

                      nprint("~~~~~~~~~~~~~~", g0, g1, g2, ":", k_xi0, k_xi1, k_xi2);
                      // k_newton_type.set_initial_iteration(k_map_data, p0, p1,
                      // p2,
                      //                                     &k_xi0, &k_xi1,
                      //                                     &k_xi2);

                      // Start of Newton iteration
                      REAL xin0, xin1, xin2;
                      REAL f0, f1, f2;

                      residual = k_newton_type.newton_residual(
                          k_map_data, k_xi0, k_xi1, k_xi2, p0, p1, p2, &f0, &f1,
                          &f2, &local_mem[0]);

                      bool diverged = false;
                      bool converged = false;
                      printf("residual: %f\n", residual);
                      for (int stepx = 0; ((stepx < k_max_iterations) &&
                                           (!converged) && (!diverged));
                           stepx++) {
                        printf("STEPX: %d, RES: %16.8e\n", stepx, residual);

                        k_newton_type.newton_step(
                            k_map_data, k_xi0, k_xi1, k_xi2, p0, p1, p2, f0, f1,
                            f2, &xin0, &xin1, &xin2, &local_mem[0]);

                        k_xi0 = xin0;
                        k_xi1 = xin1;
                        k_xi2 = xin2;

                        residual = k_newton_type.newton_residual(
                            k_map_data, k_xi0, k_xi1, k_xi2, p0, p1, p2, &f0,
                            &f1, &f2, &local_mem[0]);

                        diverged = (ABS(k_xi0) > 15.0) || (ABS(k_xi1) > 15.0) ||
                                   (ABS(k_xi2) > 15.0);
                        converged = (residual <= k_tol) && (!diverged);
                      }

                      REAL eta0, eta1, eta2;
                      k_newton_type.loc_coord_to_loc_collapsed(
                          k_map_data, k_xi0, k_xi1, k_xi2, &eta0, &eta1, &eta2);

                      bool contained = ((-1.0 - k_contained_tol) <= eta0) &&
                                       (eta0 <= (1.0 + k_contained_tol)) &&
                                       ((-1.0 - k_contained_tol) <= eta1) &&
                                       (eta1 <= (1.0 + k_contained_tol)) &&
                                       ((-1.0 - k_contained_tol) <= eta2) &&
                                       (eta2 <= (1.0 + k_contained_tol));
                      cell_found = contained && converged;
                    }
                  }
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
