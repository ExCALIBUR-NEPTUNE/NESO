#ifndef __MAPPING_NEWTON_ITERATION_BASE
#define __MAPPING_NEWTON_ITERATION_BASE

#include "newton_relative_exit_tolerances.hpp"
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>
#include <type_traits>

using namespace NESO::Particles;
using namespace Nektar::SpatialDomains;

namespace NESO::Newton {

/**
 * Implementations which require local memory should define a specialisation
 * that sets this to true.
 */
template <typename T> struct local_memory_required {
  static bool const required = false;
};

/**
 * Type that indicates that no host data is required by the Newton
 * implementation.
 */
struct NullDataHost {};

/**
 * Type for implementations that do not require local memory.
 */
struct NullDataLocal {};

/**
 * Base template for mapping types to DataHost and DataDevice definitions.
 */
template <typename T> struct mapping_host_device_types;

/**
 *  Abstract base class for Newton iteration methods for binning particles into
 *  Nektar++ cells. Subclasses must be device copyable.
 */
template <typename SPECIALISATION> struct MappingNewtonIterationBase {

  using DataDevice =
      typename mapping_host_device_types<SPECIALISATION>::DataDevice;
  using DataHost = typename mapping_host_device_types<SPECIALISATION>::DataHost;
  using DataLocal =
      typename mapping_host_device_types<SPECIALISATION>::DataLocal;

  /**
   *  Specialisations should write to the pointer a struct that contains all the
   *  data which will be required to perform a Newton iteration on the SYCL
   *  device.
   *
   *  @param sycl_target A SYCLTarget instance which the mapper may use to
   *  allocate futher device memory.
   *  @param geom A geometry object which particles may be binned into.
   *  @param data_host A host pointer to a buffer which will be kept on the
   *  host.
   *  @param data_device A host pointer to a buffer which will be copied to the
   *  compute device.
   *
   */
  inline void write_data(SYCLTargetSharedPtr sycl_target,
                         GeometrySharedPtr geom, DataHost *data_host,
                         DataDevice *data_device) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    // Call the constructor on the host data type for the mapper.
    if constexpr (!std::is_same<DataHost, NullDataHost>::value) {
      new (data_host) DataHost;
    }
    underlying.write_data_v(sycl_target, geom, data_host, data_device);
  }

  /**
   *  Called at destruction to enable implementations to free additional memory
   *  which was allocated for variable length Newton iteration data (and
   *  pointed to in the data_device memory).
   */
  inline void free_data(DataHost *data_host) {
    if constexpr (!std::is_same<DataHost, NullDataHost>::value) {
      // Call the destructor on the host data type for the mapper.
      data_host->~DataHost();
    }
  }

  /**
   *  The number of bytes required to store the data required to perform a
   *  Newton iteration on the device. i.e. the write_data call is free to write
   *  this number of bytes to the passed pointer on the host (and copied to
   *  device).
   *
   *  @returns Number of bytes required to be allocated.
   */
  inline std::size_t data_size_host() {
    if constexpr (std::is_same<DataHost, NullDataHost>::value) {
      return 0;
    } else {
      return sizeof(DataHost);
    }
  }

  /**
   *  The number of bytes required to store the data required to perform a
   *  Newton iteration on the host. i.e. the write_data call is free to write
   *  this number of bytes to the passed pointer on the host.
   *
   *  @returns Number of bytes required to be allocated.
   */
  inline std::size_t data_size_device() { return sizeof(DataDevice); }

  /**
   * The number of DataLocal elements required as local memory to evaluate the
   * mapping or residual.
   *
   * @param data_host Host data region for mapper.
   * @returns Number of bytes required.
   */
  inline std::size_t data_size_local(DataHost *data_host) {
    if constexpr (local_memory_required<SPECIALISATION>::required) {
      auto &underlying = static_cast<SPECIALISATION &>(*this);
      return underlying.data_size_local_v(data_host);
    } else {
      return 0;
    }
  }

  /**
   * Perform a Newton iteration such that
   *
   * xi_{n+1} = xi_n - J^{-1}(xi_n) F(xi_n)
   *
   * where
   *
   * F(xi) = X(xi) - P
   *
   * for target physical coordinates P. All data required to perform the
   * iteration should be contained in the memory region (i.e. a struct) pointed
   * to by d_data.
   *
   * @param[in] d_data Pointer to data required to perform the Newton iteration.
   * @param[in] xi0 Current iteration of xi coordinate, x component.
   * @param[in] xi1 Current iteration of xi coordinate, y component.
   * @param[in] xi2 Current iteration of xi coordinate, z component.
   * @param[in] phys0 Target coordinate in physical space, x component.
   * @param[in] phys1 Target coordinate in physical space, y component.
   * @param[in] phys2 Target coordinate in physical space, z component.
   * @param[in] f0 F(xi), x component.
   * @param[in] f1 F(xi), y component.
   * @param[in] f2 F(xi), z component.
   * @param[in, out] xin0 Output new iteration for local coordinate xi, x
   * component.
   * @param[in, out] xin1 Output new iteration for local coordinate xi, y
   * component.
   * @param[in, out] xin2 Output new iteration for local coordinate xi, z
   * component.
   * @param[in, out] local_memory Local memory space to use for computation.
   */
  inline void newton_step(const DataDevice *d_data, const REAL xi0,
                          const REAL xi1, const REAL xi2, const REAL phys0,
                          const REAL phys1, const REAL phys2, const REAL f0,
                          const REAL f1, const REAL f2, REAL *xin0, REAL *xin1,
                          REAL *xin2, DataLocal *local_memory) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    if constexpr (local_memory_required<SPECIALISATION>::required) {
      underlying.newton_step_v(d_data, xi0, xi1, xi2, phys0, phys1, phys2, f0,
                               f1, f2, xin0, xin1, xin2, local_memory);
    } else {
      underlying.newton_step_v(d_data, xi0, xi1, xi2, phys0, phys1, phys2, f0,
                               f1, f2, xin0, xin1, xin2);
    }
  }

  /**
   * Compute residual
   *
   * ||F(xi_n)||
   *
   * where
   *
   * F(xi) = X(xi) - P
   *
   * for target physical coordinates P and sensible norm ||F||. All data
   * required to compute the residual should be contained in the memory region
   * (i.e. a struct) pointed to by d_data. Computes and returns F(xi) for the
   * passed xi. Note F should always be of the form X - P, not P - X, such that
   * the implementation of the residual can be reused as an implementation of X
   * by passing P=0.
   *
   * @param[in] d_data Pointer to data required to perform the Newton iteration.
   * @param[in] xi0 Current iteration of xi coordinate, x component.
   * @param[in] xi1 Current iteration of xi coordinate, y component.
   * @param[in] xi2 Current iteration of xi coordinate, z component.
   * @param[in] phys0 Target coordinate in physical space, x component.
   * @param[in] phys1 Target coordinate in physical space, y component.
   * @param[in] phys2 Target coordinate in physical space, z component.
   * @param[in, out] f0 F(xi), x component.
   * @param[in, out] f1 F(xi), y component.
   * @param[in, out] f2 F(xi), z component.
   * @param[in, out] local_memory Local memory space to use for computation.
   * @returns Residual.
   */
  inline REAL newton_residual(const DataDevice *d_data, const REAL xi0,
                              const REAL xi1, const REAL xi2, const REAL phys0,
                              const REAL phys1, const REAL phys2, REAL *f0,
                              REAL *f1, REAL *f2, DataLocal *local_memory) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    if constexpr (local_memory_required<SPECIALISATION>::required) {
      return underlying.newton_residual_v(d_data, xi0, xi1, xi2, phys0, phys1,
                                          phys2, f0, f1, f2, local_memory);
    } else {
      return underlying.newton_residual_v(d_data, xi0, xi1, xi2, phys0, phys1,
                                          phys2, f0, f1, f2);
    }
  }

  /**
   * Get the number of coordinate dimensions the iteration is performed in.
   * i.e. how many position components should be read from the particle and
   * reference coordinates written.
   *
   *  @returns Number of coordinate dimensions.
   */
  inline int get_ndim() {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    return underlying.get_ndim_v();
  }

  /**
   * Set the initial iteration prior to the newton iteration.
   *
   * @param[in] d_data Pointer to data required to perform the Newton iteration.
   * @param[in] phys0 Target coordinate in physical space, x component.
   * @param[in] phys1 Target coordinate in physical space, y component.
   * @param[in] phys2 Target coordinate in physical space, z component.
   * @param[in,out] xi0 Input x coordinate to reset.
   * @param[in,out] xi1 Input y coordinate to reset.
   * @param[in,out] xi2 Input z coordinate to reset.
   */
  inline void set_initial_iteration(const DataDevice *d_data, const REAL phys0,
                                    const REAL phys1, const REAL phys2,
                                    REAL *xi0, REAL *xi1, REAL *xi2) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.set_initial_iteration_v(d_data, phys0, phys1, phys2, xi0, xi1,
                                       xi2);
  }

  /**
   *  Map local coordinate (xi) to local collapsed coordinate (eta).
   *
   *
   * @param[in] d_data Pointer to data required to perform the Newton iteration.
   * @param[in] xi0 Local coordinate (xi) to be mapped to collapsed coordinate,
   * x component.
   * @param[in] xi1 Local coordinate (xi) to be mapped to collapsed coordinate,
   * y component.
   * @param[in] xi2 Local coordinate (xi) to be mapped to collapsed coordinate,
   * z component.
   * @param[in, out] eta0 Local collapsed coordinate (eta), x component.
   * @param[in, out] eta1 Local collapsed coordinate (eta), y component.
   * @param[in, out] eta2 Local collapsed coordinate (eta), z component.
   */
  inline void loc_coord_to_loc_collapsed(const DataDevice *d_data,
                                         const REAL xi0, const REAL xi1,
                                         const REAL xi2, REAL *eta0, REAL *eta1,
                                         REAL *eta2) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.loc_coord_to_loc_collapsed_v(d_data, xi0, xi1, xi2, eta0, eta1,
                                            eta2);
  }

  /**
   *  Map from local collapsed coordinate (eta) to local coordinate (xi).
   *
   * @param[in] d_data Pointer to data required to perform the Newton iteration.
   * @param[in] eta0 Local collapsed coordinate (eta), x component.
   * @param[in] eta1 Local collapsed coordinate (eta), y component.
   * @param[in] eta2 Local collapsed coordinate (eta), z component.
   * @param[in, out] xi0 Local coordinate (xi) to be mapped to collapsed
   * coordinate, x component.
   * @param[in, out] xi1 Local coordinate (xi) to be mapped to collapsed
   * coordinate, y component.
   * @param[in, out] xi2 Local coordinate (xi) to be mapped to collapsed
   * coordinate, z component.
   */
  inline void loc_collapsed_to_loc_coord(const DataDevice *d_data,
                                         const REAL eta0, const REAL eta1,
                                         const REAL eta2, REAL *xi0, REAL *xi1,
                                         REAL *xi2) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.loc_collapsed_to_loc_coord_v(d_data, eta0, eta1, eta2, xi0, xi1,
                                            xi2);
  }
};

} // namespace NESO::Newton

#endif
