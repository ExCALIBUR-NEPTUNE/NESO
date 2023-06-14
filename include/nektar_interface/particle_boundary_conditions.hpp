#ifndef __PARTICLE_BOUNDARY_CONDITIONS_H__
#define __PARTICLE_BOUNDARY_CONDITIONS_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <stack>
#include <vector>

#include <mpi.h>

#include "bounding_box_intersection.hpp"
#include "global_bounding_box.hpp"
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

/**
 * Periodic boundary conditions implementation designed to work with a
 * CartesianHMesh.
 */
class NektarCartesianPeriodic {
private:
  BufferDevice<double> d_origin;
  BufferDevice<double> d_extents;
  SYCLTargetSharedPtr sycl_target;
  ParticleDatSharedPtr<REAL> position_dat;
  std::shared_ptr<GlobalBoundingBox> global_bounding_box;
  const int ndim;

public:
  ~NektarCartesianPeriodic(){};

  /**
   * Construct instance to apply periodic boundary conditions to particles
   * within the passed ParticleDat.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param graph Nektar++ MeshGraph on which particles move.
   * @param position_dat ParticleDat containing particle positions.
   */
  NektarCartesianPeriodic(SYCLTargetSharedPtr sycl_target,
                          Nektar::SpatialDomains::MeshGraphSharedPtr graph,
                          ParticleDatSharedPtr<REAL> position_dat) :
    NektarCartesianPeriodic(sycl_target, graph, position_dat,
        std::make_shared<GlobalBoundingBox>(sycl_target, graph)) {};

  /**
   * Construct instance to apply periodic boundary conditions to particles
   * within the passed ParticleDat.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param graph Nektar++ MeshGraph on which particles move.
   * @param position_dat ParticleDat containing particle positions.
   * @param global_bounding_box std::shared_ptr<GlobalBoundingBox> useful info about domain.
   */
  NektarCartesianPeriodic(SYCLTargetSharedPtr sycl_target,
                          Nektar::SpatialDomains::MeshGraphSharedPtr graph,
                          ParticleDatSharedPtr<REAL> position_dat,
                          std::shared_ptr<GlobalBoundingBox> global_bounding_box)
      : sycl_target(sycl_target), ndim(graph->GetMeshDimension()),
        position_dat(position_dat), global_bounding_box(global_bounding_box),
        d_extents(sycl_target, 3), d_origin(sycl_target, 3) {

    NESOASSERT(this->ndim <= 3, "bad mesh ndim");

    sycl_target->queue
        .memcpy(this->d_extents.ptr, this->global_bounding_box->global_extent(),
                this->ndim * sizeof(double))
        .wait_and_throw();

    sycl_target->queue
        .memcpy(this->d_origin.ptr, this->global_bounding_box->global_origin(),
                this->ndim * sizeof(double))
        .wait_and_throw();
  };

  /**
   * Apply periodic boundary conditions to the particle positions in the
   * ParticleDat this instance was created with.
   */
  inline void execute() {

    auto t0 = profile_timestamp();
    auto pl_iter_range = this->position_dat->get_particle_loop_iter_range();
    auto pl_stride = this->position_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = this->position_dat->get_particle_loop_npart_cell();
    const int k_ndim = this->ndim;

    NESOASSERT(((k_ndim > 0) && (k_ndim < 4)), "Bad number of dimensions");
    const auto k_origin = this->d_origin.ptr;
    const auto k_extents = this->d_extents.ptr;
    auto k_positions_dat = this->position_dat->cell_dat.device_ptr();

    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                for (int dimx = 0; dimx < k_ndim; dimx++) {
                  const double pos =
                      k_positions_dat[cellx][dimx][layerx] - k_origin[dimx];
                  // offset the position in the current dimension to be
                  // positive by adding a value times the extent
                  const double n_extent_offset_real = ABS(pos);
                  const double tmp_extent = k_extents[dimx];
                  const INT n_extent_offset_int = n_extent_offset_real + 2.0;
                  const double pos_fmod =
                      fmod(pos + n_extent_offset_int * tmp_extent, tmp_extent);
                  k_positions_dat[cellx][dimx][layerx] =
                      pos_fmod + k_origin[dimx];
                }
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    sycl_target->profile_map.inc("NektarCartesianPeriodic", "execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));
  }
};

} // namespace NESO
#endif
