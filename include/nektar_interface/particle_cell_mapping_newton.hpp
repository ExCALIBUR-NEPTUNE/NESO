#ifndef __PARTICLE_CELL_MAPPING_NEWTON_H__
#define __PARTICLE_CELL_MAPPING_NEWTON_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

#include "candidate_cell_mapping.hpp"
#include "particle_cell_mapping_common.hpp"
#include "particle_mesh_interface.hpp"

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Newton {

/**
 *  Abstract base class for Newton iteration methods for binning particles into
 *  Nektar++ cells.
 */
template <typename SPECIALISATION> struct MappingNewtonIterationBase {

  /**
   *  Specialisations should write to the pointer a struct that contains all the
   *  data which will be required to perform a Newton iteration on the SYCL
   *  device.
   *
   *  @param data_host A host pointer to a buffer which will be kept on the
   *  host.
   *  @param data_device A host pointer to a buffer which will be copied to the
   *  compute device.
   *
   */
  inline void write_data(void *data_host, void *data_device) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.write_data(data_host, data_device);
  }

  /**
   *  Called at destruction to enable implementations to free additional memory
   *  which was allocated for variable length Newton iteration data (and
   *  pointed to in the data_device memory).
   */
  inline void free_data(void *data_host) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.free_data(data_host);
  }

  /**
   *  The number of bytes required to store the data required to perform a
   *  Newton iteration on the device. i.e. the write_data call is free to write
   *  this number of bytes to the passed pointer on the host (and copied to
   *  device).
   *
   *  @returns Number of bytes required to be allocated.
   */
  inline size_t data_size_host() {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.data_size_host();
  }

  /**
   *  The number of bytes required to store the data required to perform a
   *  Newton iteration on the host. i.e. the write_data call is free to write
   *  this number of bytes to the passed pointer on the host.
   *
   *  @returns Number of bytes required to be allocated.
   */
  inline size_t data_size_device() {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.data_size_device();
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
   * @param[in, out] xin0 Output new iteration for local coordinate xi, x
   * component.
   * @param[in, out] xin1 Output new iteration for local coordinate xi, y
   * component.
   * @param[in, out] xin2 Output new iteration for local coordinate xi, z
   * component.
   */
  inline size_t newton_step(const void *d_data, const REAL xi0, const REAL xi1,
                            const REAL xi2, const REAL phys0, const REAL phys1,
                            const REAL phys2, REAL *xin0, REAL *xin1,
                            REAL *xin2) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.newton_step(d_data, xi0, xi1, xi2, phys0, phys1, phys2, xin0,
                           xin1, xin2);
  }
};

} // namespace Newton
} // namespace NESO

#endif
