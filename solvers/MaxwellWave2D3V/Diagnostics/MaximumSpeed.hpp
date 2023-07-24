#ifndef __MAXIMUM_SPEED_H_
#define __MAXIMUM_SPEED_H_

#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>

using namespace NESO::Particles;

/**
 * Compute the maximum speed of particles in a ParticleGroup.
 */
class MaximumSpeed {
private:
  BufferDeviceHost<double> dh_max_speed;

public:
  /// ParticleGroup of interest.
  ParticleGroupSharedPtr particle_group;
  /// The MPI communicator used by this instance.
  MPI_Comm comm;
  /// The maximum speed of the particles
  double max_speed;

  /*
   *  Create new instance.
   *
   *  @parm particle_group ParticleGroup to compute the maximum speed of.
   *  @param comm MPI communicator (default MPI_COMM_WORLD).
   */
  MaximumSpeed(ParticleGroupSharedPtr particle_group,
               MPI_Comm comm = MPI_COMM_WORLD)
      : particle_group(particle_group),
        comm(comm), dh_max_speed(particle_group->sycl_target, 1) {

    int flag;
    MPICHK(MPI_Initialized(&flag));
    ASSERTL1(flag, "MPI is not initialised");
  }

  /**
   *  Compute the current kinetic energy of the ParticleGroup.
   */
  inline double compute() {

    auto t0 = profile_timestamp();
    auto sycl_target = this->particle_group->sycl_target;
    auto k_V = (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();
    const auto k_ndim_velocity = (*this->particle_group)[Sym<REAL>("V")]->ncomp;

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    this->dh_max_speed.h_buffer.ptr[0] = 0.0;
    this->dh_max_speed.host_to_device();

    auto k_max_speed = this->dh_max_speed.d_buffer.ptr;

    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                double vv = 0.0;
                for (int vdimx = 0; vdimx < k_ndim_velocity; vdimx++) {
                  const double V_vdimx = k_V[cellx][vdimx][layerx];
                  vv += (V_vdimx * V_vdimx);
                }

                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    max_speed_atomic(k_max_speed[0]);
                max_speed_atomic.fetch_max(std::sqrt(vv));

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target->profile_map.inc("MaximumSpeed", "Execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));

    this->dh_max_speed.device_to_host();
    const double kernel_max_speed = this->dh_max_speed.h_buffer.ptr[0];

    MPICHK(MPI_Allreduce(&kernel_max_speed, &(this->max_speed), 1, MPI_DOUBLE,
                         MPI_SUM, this->comm));

    return this->max_speed;
  }
};

#endif
