#ifndef __X_MAP_NEWTON_H__
#define __X_MAP_NEWTON_H__

#include "../utility_sycl.hpp"
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
  using DataDevice = typename NEWTON_TYPE::DataDevice;
  using DataHost = typename NEWTON_TYPE::DataHost;
  using DataLocal = typename NEWTON_TYPE::DataLocal;

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
  std::unique_ptr<BufferDeviceHost<DataDevice>> dh_data;
  /// The data required to perform newton iterations for each geom on the host.
  std::unique_ptr<BufferHost<DataHost>> h_data;

  std::unique_ptr<BufferDeviceHost<REAL>> dh_fdata;

  std::size_t num_elements_local;

  // variables for higher order grids
  int num_modes;
  int num_modes_factor;
  int ndim;

  template <typename U> inline void write_data(U &geom) {
    if (this->num_bytes_per_map_host) {
      this->h_data =
          std::make_unique<BufferHost<DataHost>>(this->sycl_target, 1);
    }
    if (this->num_bytes_per_map_device) {
      this->dh_data =
          std::make_unique<BufferDeviceHost<DataDevice>>(this->sycl_target, 1);
    }
    auto d_data_ptr = (this->num_bytes_per_map_device)
                          ? this->dh_data->h_buffer.ptr
                          : nullptr;
    auto h_data_ptr =
        (this->num_bytes_per_map_host) ? this->h_data->ptr : nullptr;
    this->newton_type.write_data(this->sycl_target, geom, h_data_ptr,
                                 d_data_ptr);

    if (this->num_bytes_per_map_device) {
      this->dh_data->host_to_device();
    }
    this->num_elements_local =
        std::max(static_cast<std::size_t>(1),
                 this->newton_type.data_size_local(h_data_ptr));
    this->num_modes = geom->GetXmap()->EvalBasisNumModesMax();
    MappingNewtonIterationBase<NEWTON_TYPE> newton_type;
    this->ndim = newton_type.get_ndim();
  }

public:
  ~XMapNewton() {
    auto h_data_ptr =
        (this->num_bytes_per_map_host) ? this->h_data->ptr : nullptr;
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
        std::make_unique<BufferDeviceHost<REAL>>(this->sycl_target, 6);
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

    DataDevice *k_map_data = nullptr;
    if (this->dh_data) {
      k_map_data = this->dh_data->d_buffer.ptr;
    }
    NESOASSERT(this->dh_fdata != nullptr, "Bad pointer");
    auto k_fdata = this->dh_fdata->d_buffer.ptr;
    NESOASSERT(k_fdata != nullptr, "Bad pointer");
    const std::size_t num_elements_local = this->num_elements_local;

    const REAL k_xi0 = xi0;
    const REAL k_xi1 = xi1;
    const REAL k_xi2 = xi2;

    this->sycl_target->queue
        .submit([=](sycl::handler &cgh) {
          sycl::local_accessor<DataLocal, 1> local_mem(
              sycl::range<1>(num_elements_local), cgh);

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

                k_newton_type.newton_residual(k_map_data, k_xi0, k_xi1, k_xi2,
                                              p0, p1, p2, &f0, &f1, &f2,
                                              &local_mem[0]);

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
    DataDevice *k_map_data = nullptr;
    if (this->dh_data) {
      k_map_data = this->dh_data->d_buffer.ptr;
    }
    NESOASSERT(this->dh_fdata != nullptr, "Bad pointer");
    auto k_fdata = this->dh_fdata->d_buffer.ptr;
    NESOASSERT(k_fdata != nullptr, "Bad pointer");

    const REAL k_tol = tol;
    const double k_contained_tol = contained_tol;
    const std::size_t num_elements_local = this->num_elements_local;

    const int k_ndim = this->ndim;
    const int grid_size = this->num_modes_factor * this->num_modes;
    const int k_grid_size_x = std::max(grid_size - 1, 1);
    const int k_grid_size_y = k_ndim > 1 ? k_grid_size_x : 1;
    const int k_grid_size_z = k_ndim > 2 ? k_grid_size_x : 1;
    const REAL k_grid_width = 2.0 / (k_grid_size_x);

    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          sycl::local_accessor<DataLocal, 1> local_mem(
              sycl::range<1>(num_elements_local), cgh);

          cgh.parallel_for<>(
              sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
              [=](auto idx) {
                MappingNewtonIterationBase<NEWTON_TYPE> k_newton_type{};
                XMapNewtonKernel<NEWTON_TYPE> k_newton_kernel{};

                const REAL p0 = phys0;
                const REAL p1 = phys1;
                const REAL p2 = phys2;
                REAL k_xi0 = 0.0;
                REAL k_xi1 = 0.0;
                REAL k_xi2 = 0.0;
                REAL residual = 10.0;
                bool cell_found = false;

                for (int g2 = 0; (g2 <= k_grid_size_z) && (!cell_found); g2++) {
                  for (int g1 = 0; (g1 <= k_grid_size_y) && (!cell_found);
                       g1++) {
                    for (int g0 = 0; (g0 <= k_grid_size_x) && (!cell_found);
                         g0++) {

                      k_xi0 = -1.0 + g0 * k_grid_width;
                      k_xi1 = -1.0 + g1 * k_grid_width;
                      k_xi2 = -1.0 + g2 * k_grid_width;

                      bool converged = k_newton_kernel.x_inverse(
                          k_map_data, p0, p1, p2, &k_xi0, &k_xi1, &k_xi2,
                          &local_mem[0], k_max_iterations, k_tol, true);
                      REAL eta0, eta1, eta2;

                      k_newton_type.loc_coord_to_loc_collapsed(
                          k_map_data, k_xi0, k_xi1, k_xi2, &eta0, &eta1, &eta2);

                      eta0 = Kernel::min(eta0, 1.0 + k_contained_tol);
                      eta1 = Kernel::min(eta1, 1.0 + k_contained_tol);
                      eta2 = Kernel::min(eta2, 1.0 + k_contained_tol);
                      eta0 = Kernel::max(eta0, -1.0 - k_contained_tol);
                      eta1 = Kernel::max(eta1, -1.0 - k_contained_tol);
                      eta2 = Kernel::max(eta2, -1.0 - k_contained_tol);

                      k_newton_type.loc_collapsed_to_loc_coord(
                          k_map_data, eta0, eta1, eta2, &k_xi0, &k_xi1, &k_xi2);

                      residual = k_newton_type.newton_residual(
                          k_map_data, k_xi0, k_xi1, k_xi2, p0, p1, p2, &eta0,
                          &eta1, &eta2, &local_mem[0]);

                      const bool contained = residual <= k_tol;
                      cell_found = contained && converged;
                    }
                  }
                }
                k_fdata[0] = k_xi0;
                k_fdata[1] = k_xi1;
                k_fdata[2] = k_xi2;
                k_fdata[3] = (residual <= tol) && cell_found ? 1 : -1;
              });
        })
        .wait_and_throw();

    this->dh_fdata->device_to_host();
    *xi0 = this->dh_fdata->h_buffer.ptr[0];
    *xi1 = this->dh_fdata->h_buffer.ptr[1];
    *xi2 = this->dh_fdata->h_buffer.ptr[2];
    return (this->dh_fdata->h_buffer.ptr[3] > 0);
  }

  /**
   * Return a Nektar++ style bounding box for the geometry object. Padding is
   * added to each end of each dimension. i.e. a padding of 5% (pad_rel = 0.05)
   * at each end is 10% globally.
   *
   * @param grid_size Resolution of grid to use on each face of the collapsed
   * reference space. Default 32.
   * @param pad_rel Relative padding to add to computed bounding box, default
   * 0.05, i.e. 5%.
   * @param pad_abs Absolute padding to add to computed bounding box, default
   * 0.0.
   * @returns Bounding box in format [minx, miny, minz, maxx, maxy, maxz];
   */
  std::array<double, 6> get_bounding_box(std::size_t grid_size = 32,
                                         const REAL pad_rel = 0.05,
                                         const REAL pad_abs = 0.0) {
    DataDevice *k_map_data = nullptr;
    if (this->dh_data) {
      k_map_data = this->dh_data->d_buffer.ptr;
    }
    NESOASSERT(this->dh_fdata != nullptr, "Bad pointer");
    auto k_fdata = this->dh_fdata->d_buffer.ptr;
    NESOASSERT(k_fdata != nullptr, "Bad pointer");

    const std::size_t num_elements_local = this->num_elements_local;

    // Get a local size which is a power of 2.
    const std::size_t local_size =
        get_prev_power_of_two(static_cast<std::size_t>(
            std::sqrt(this->sycl_target->get_num_local_work_items(
                num_elements_local * sizeof(DataLocal),
                sycl_target->parameters
                    ->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
                    ->value))));
    // make grid_size a multiple of the local size
    grid_size = get_next_multiple(grid_size, local_size);
    sycl::range<1> range_local(local_size);
    // There are 6 faces on the collapsed reference cell
    sycl::range<1> range_global(grid_size * grid_size);
    const REAL k_width = 2.0 / static_cast<REAL>(grid_size - 1);

    constexpr REAL kc[6][3] = {
        {-1.0, 0.0, 0.0}, // x = -1
        {1.0, 0.0, 0.0},  // x =  1
        {0.0, -1.0, 0.0}, // y = -1
        {0.0, 1.0, 0.0},  // y =  1
        {0.0, 0.0, -1.0}, // z = -1
        {0.0, 0.0, 1.0}   // z =  1
    };

    constexpr REAL kcx[6][3] = {{0.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
                                {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0},
                                {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};

    constexpr REAL kcy[6][3] = {{0.0, 0.0, 1.0}, {0.0, 0.0, 1.0},
                                {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                                {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0}};

    static_assert((kc[0][0] + kcx[0][0] + kcy[0][0]) == -1);
    static_assert((kc[0][1] + kcx[0][1] + kcy[0][1]) == 1);
    static_assert((kc[0][2] + kcx[0][2] + kcy[0][2]) == 1);
    static_assert((kc[1][0] + kcx[1][0] + kcy[1][0]) == 1);
    static_assert((kc[1][1] + kcx[1][1] + kcy[1][1]) == 1);
    static_assert((kc[1][2] + kcx[1][2] + kcy[1][2]) == 1);
    static_assert((kc[2][0] + kcx[2][0] + kcy[2][0]) == 1);
    static_assert((kc[2][1] + kcx[2][1] + kcy[2][1]) == -1);
    static_assert((kc[2][2] + kcx[2][2] + kcy[2][2]) == 1);
    static_assert((kc[3][0] + kcx[3][0] + kcy[3][0]) == 1);
    static_assert((kc[3][1] + kcx[3][1] + kcy[3][1]) == 1);
    static_assert((kc[3][2] + kcx[3][2] + kcy[3][2]) == 1);
    static_assert((kc[4][0] + kcx[4][0] + kcy[4][0]) == 1);
    static_assert((kc[4][1] + kcx[4][1] + kcy[4][1]) == 1);
    static_assert((kc[4][2] + kcx[4][2] + kcy[4][2]) == -1);
    static_assert((kc[5][0] + kcx[5][0] + kcy[5][0]) == 1);
    static_assert((kc[5][1] + kcx[5][1] + kcy[5][1]) == 1);
    static_assert((kc[5][2] + kcx[5][2] + kcy[5][2]) == 1);

    for (int dx = 0; dx < 3; dx++) {
      this->dh_fdata->h_buffer.ptr[dx] = std::numeric_limits<REAL>::max();
      this->dh_fdata->h_buffer.ptr[dx + 3] =
          std::numeric_limits<REAL>::lowest();
    }
    this->dh_fdata->host_to_device();

    EventStack event_stack;

    for (std::size_t facex = 0; facex < 6; facex++) {

      event_stack.push(this->sycl_target->queue.submit([=](sycl::handler &cgh) {
        sycl::local_accessor<DataLocal, 1> local_mem(
            sycl::range<1>(num_elements_local * local_size), cgh);

        cgh.parallel_for<>(
            this->sycl_target->device_limits.validate_nd_range(
                sycl::nd_range<1>(range_global, range_local)),
            [=](auto idx) {
              MappingNewtonIterationBase<NEWTON_TYPE> k_newton_type{};

              const auto local_id = idx.get_local_linear_id();
              const std::size_t gid = idx.get_global_linear_id();

              const auto iix = gid % grid_size;
              const auto iiy = gid / grid_size;
              const REAL gx = -1.0 + iix * k_width;
              const REAL gy = -1.0 + iiy * k_width;

              const REAL eta0 =
                  kc[facex][0] + kcx[facex][0] * gx + kcy[facex][0] * gy;
              const REAL eta1 =
                  kc[facex][1] + kcx[facex][1] * gx + kcy[facex][1] * gy;
              const REAL eta2 =
                  kc[facex][2] + kcx[facex][2] * gx + kcy[facex][2] * gy;

              REAL k_xi0, k_xi1, k_xi2;
              k_newton_type.loc_collapsed_to_loc_coord(
                  k_map_data, eta0, eta1, eta2, &k_xi0, &k_xi1, &k_xi2);

              REAL f[3] = {0.0, 0.0, 0.0};
              constexpr REAL p0 = 0.0;
              constexpr REAL p1 = 0.0;
              constexpr REAL p2 = 0.0;

              k_newton_type.newton_residual(
                  k_map_data, k_xi0, k_xi1, k_xi2, p0, p1, p2, f, f + 1, f + 2,
                  &local_mem[local_id * num_elements_local]);

              //// Do the reductions, we pessimistically do not use the builtin
              //// SYCL functions as we have used all the local memory already
              //// and the SYCL reduction functions also use local memory.
              for (int dimx = 0; dimx < 3; dimx++) {
                atomic_fetch_max(&k_fdata[dimx + 3], f[dimx]);
                atomic_fetch_min(&k_fdata[dimx], f[dimx]);
              }
            });
      }));
    }
    event_stack.wait();

    this->dh_fdata->device_to_host();
    std::array<double, 6> output;
    for (int cx = 0; cx < 6; cx++) {
      output[cx] = this->dh_fdata->h_buffer.ptr[cx];
    }

    for (int dx = 0; dx < this->ndim; dx++) {
      const REAL width = output.at(dx + 3) - output.at(dx);
      const REAL padding = pad_rel * width + pad_abs;
      output.at(dx) -= padding;
      output.at(dx + 3) += padding;
    }
    return output;
  }
};

} // namespace NESO::Newton

#endif
