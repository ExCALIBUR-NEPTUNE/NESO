#ifndef __KINETIC_ENERGY_H_
#define __KINETIC_ENERGY_H_

#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>

using namespace NESO::Particles;

/**
 * Compute the kinetic energy of particles in a ParticleGroup.
 */
class KineticEnergy {
private:
  BufferDeviceHost<double> dh_kinetic_energy;

public:
  /// ParticleGroup of interest.
  ParticleGroupSharedPtr particle_group;
  /// The MPI communicator used by this instance.
  MPI_Comm comm;
  /// The last kinetic energy that was computed on call to write.
  double energy;
  /// The mass of the particles.
  const double particle_mass;

  /*
   *  Create new instance.
   *
   *  @parm particle_group ParticleGroup to compute kinetic energy of.
   *  @param particle_mass Mass of each particle.
   *  @param comm MPI communicator (default MPI_COMM_WORLD).
   */
  KineticEnergy(ParticleGroupSharedPtr particle_group,
                const double particle_mass, MPI_Comm comm = MPI_COMM_WORLD)
      : particle_group(particle_group), particle_mass(particle_mass),
        comm(comm), dh_kinetic_energy(particle_group->sycl_target, 1) {

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
    const double k_half_particle_mass = 0.5 * this->particle_mass;
    auto k_W = (*this->particle_group)[Sym<REAL>("W")]
                   ->cell_dat.device_ptr(); // weight
    auto k_V = (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();
    const auto k_ndim_velocity = (*this->particle_group)[Sym<REAL>("V")]->ncomp;

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    this->dh_kinetic_energy.h_buffer.ptr[0] = 0.0;
    this->dh_kinetic_energy.host_to_device();

    auto k_kinetic_energy = this->dh_kinetic_energy.d_buffer.ptr;

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
                // 0.5 mass * velocity^2 * weight
                double half_wmvv =
                    vv * k_W[cellx][0][layerx] * k_half_particle_mass;

                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    kinetic_energy_atomic(k_kinetic_energy[0]);
                kinetic_energy_atomic.fetch_add(half_wmvv);

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target->profile_map.inc("KineticEnergy", "Execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));
    this->dh_kinetic_energy.device_to_host();
    const double kernel_kinetic_energy =
        this->dh_kinetic_energy.h_buffer.ptr[0];

    MPICHK(MPI_Allreduce(&kernel_kinetic_energy, &(this->energy), 1, MPI_DOUBLE,
                         MPI_SUM, this->comm));

    return this->energy;
  }
};

#endif
