#ifndef __NESOSOLVERS_ELECTROSTATIC2D3V_KINETICENERGY_HPP__
#define __NESOSOLVERS_ELECTROSTATIC2D3V_KINETICENERGY_HPP__

#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>

using namespace NESO::Particles;

/**
 * Compute the kinetic energy of particles in a ParticleGroup.
 */
class KineticEnergy {
private:
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
        comm(comm) {

    int flag;
    MPICHK(MPI_Initialized(&flag));
    NESOASSERT(flag, "MPI is not initialised");
  }

  /**
   *  Compute the current kinetic energy of the ParticleGroup.
   */
  inline double compute() {
    auto t0 = profile_timestamp();
    const REAL k_half_particle_mass = 0.5 * this->particle_mass;
    const auto k_ndim_velocity =
        this->particle_group->get_dat(Sym<REAL>("V"))->ncomp;

    auto ga_kinetic_energy = std::make_shared<GlobalArray<REAL>>(
        this->particle_group->sycl_target, 1, 0.0);

    particle_loop(
        "KineticEnergy::compute", this->particle_group,
        [=](auto k_V, auto k_kinetic_energy) {
          REAL half_mvv = 0.0;
          for (int vdimx = 0; vdimx < k_ndim_velocity; vdimx++) {
            const REAL V_vdimx = k_V.at(vdimx);
            half_mvv += (V_vdimx * V_vdimx);
          }
          half_mvv *= k_half_particle_mass;
          k_kinetic_energy.add(0, half_mvv);
        },
        Access::read(Sym<REAL>("V")), Access::add(ga_kinetic_energy))
        ->execute();

    this->energy = ga_kinetic_energy->get().at(0);
    return this->energy;
  }
};

#endif // __NESOSOLVERS_ELECTROSTATIC2D3V_KINETICENERGY_HPP__
