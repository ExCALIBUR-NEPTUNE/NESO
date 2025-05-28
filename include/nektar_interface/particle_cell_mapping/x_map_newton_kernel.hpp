#ifndef __X_MAP_NEWTON_KERNEL_H__
#define __X_MAP_NEWTON_KERNEL_H__

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace NESO::Particles;
using namespace Nektar::SpatialDomains;

#include "mapping_newton_iteration_base.hpp"

namespace NESO::Newton {
/**
 *  Utility struct to provide the kernel functions to evaluate X maps and their
 *  inverse.
 */
template <typename NEWTON_TYPE> struct XMapNewtonKernel {
  using DataDevice =
      typename mapping_host_device_types<NEWTON_TYPE>::DataDevice;
  using DataLocal = typename mapping_host_device_types<NEWTON_TYPE>::DataLocal;

  /**
   * For a reference position xi compute the global position X(xi).
   *
   * @param[in] map_data Device data required by the Newton implementation.
   * @param[in] xi0 Reference position, x component.
   * @param[in] xi1 Reference position, y component.
   * @param[in] xi2 Reference position, z component.
   * @param[in, out] phys0 Global position X(xi), x component.
   * @param[in, out] phys1 Global position X(xi), y component.
   * @param[in, out] phys2 Global position X(xi), z component.
   * @param[in, out] local_memory Local memory as required by the Newton
   * implementation. May be modified by this function.
   */
  inline void x(const DataDevice *map_data, const REAL xi0, const REAL xi1,
                const REAL xi2, REAL *phys0, REAL *phys1, REAL *phys2,
                DataLocal *local_memory) {

    *phys0 = 0.0;
    *phys1 = 0.0;
    *phys2 = 0.0;
    const REAL p0 = 0.0;
    const REAL p1 = 0.0;
    const REAL p2 = 0.0;

    MappingNewtonIterationBase<NEWTON_TYPE> k_newton_type{};
    k_newton_type.newton_residual(map_data, xi0, xi1, xi2, p0, p1, p2, phys0,
                                  phys1, phys2, local_memory);
  }

  /**
   * For a position X(xi) compute the reference position xi via Newton
   * iteration.
   *
   * @param[in] map_data Device data required by the Newton implementation.
   * @param[in] phys0 Global position X(xi), x component.
   * @param[in] phys1 Global position X(xi), y component.
   * @param[in] phys2 Global position X(xi), z component.
   * @param[in, out] xi0 Reference position, x component.
   * @param[in, out] xi1 Reference position, y component.
   * @param[in, out] xi2 Reference position, z component.
   * @param[in, out] local_memory Local memory as required by the Newton
   * implementation. May be modified by this function.
   * @param[in] max_iterations Maximum number of Newton iterations.
   * @param[in] tol Optional exit tolerance for Newton iterations
   * (default 1.0e-10).
   * @param[in] initial_override Optional flag to override starting point with
   * the input xi values (default false).
   * @returns True if inverse is found otherwise false.
   */
  inline bool x_inverse(const DataDevice *map_data, const REAL phys0,
                        const REAL phys1, const REAL phys2, REAL *xi0,
                        REAL *xi1, REAL *xi2, DataLocal *local_memory,
                        const INT max_iterations, const REAL tol = 1.0e-10,
                        const bool initial_override = false) {

    MappingNewtonIterationBase<NEWTON_TYPE> k_newton_type{};

    const REAL p0 = phys0;
    const REAL p1 = phys1;
    const REAL p2 = phys2;

    REAL k_xi0 = 0.0;
    REAL k_xi1 = 0.0;
    REAL k_xi2 = 0.0;
    k_newton_type.set_initial_iteration(map_data, p0, p1, p2, &k_xi0, &k_xi1,
                                        &k_xi2);

    k_xi0 = initial_override ? *xi0 : k_xi0;
    k_xi1 = initial_override ? *xi1 : k_xi1;
    k_xi2 = initial_override ? *xi2 : k_xi2;

    // Start of Newton iteration
    REAL xin0 = 0.0;
    REAL xin1 = 0.0;
    REAL xin2 = 0.0;
    REAL f0 = 0.0;
    REAL f1 = 0.0;
    REAL f2 = 0.0;

    REAL residual = k_newton_type.newton_residual(
        map_data, k_xi0, k_xi1, k_xi2, p0, p1, p2, &f0, &f1, &f2, local_memory);

    bool diverged = false;

    for (int stepx = 0;
         ((stepx < max_iterations) && (residual > tol) && (!diverged));
         stepx++) {
      k_newton_type.newton_step(map_data, k_xi0, k_xi1, k_xi2, p0, p1, p2, f0,
                                f1, f2, &xin0, &xin1, &xin2, local_memory);

      k_xi0 = xin0;
      k_xi1 = xin1;
      k_xi2 = xin2;

      residual =
          k_newton_type.newton_residual(map_data, k_xi0, k_xi1, k_xi2, p0, p1,
                                        p2, &f0, &f1, &f2, local_memory);

      diverged = (ABS(k_xi0) > 15.0) || (ABS(k_xi1) > 15.0) ||
                 (ABS(k_xi2) > 15.0) || (!sycl::isfinite(residual));
    }
    *xi0 = k_xi0;
    *xi1 = k_xi1;
    *xi2 = k_xi2;
    return (residual <= tol) ? true : false;
  }
};

} // namespace NESO::Newton

#endif
