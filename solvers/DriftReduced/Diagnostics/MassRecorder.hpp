#ifndef __NESOSOLVERS_DRIFTREDUCED_MASSRECORDER_HPP__
#define __NESOSOLVERS_DRIFTREDUCED_MASSRECORDER_HPP__

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>

#include "../ParticleSystems/NeutralParticleSystem.hpp"

namespace LU = Nektar::LibUtilities;

namespace NESO::Solvers::DriftReduced {
/**
 * @brief Class to manage recording of masses in a coupled fluid-particle
 * system.
 */
template <typename T> class MassRecorder {
protected:
  /// File handle for recording output
  std::ofstream fh;
  /// Flag to track whether initial fluid mass has been computed
  bool initial_fluid_mass_computed;
  /// The initial fluid mass
  double initial_mass_fluid;
  /// Pointer to number density field
  std::shared_ptr<T> n;
  /// Pointer to particle system
  std::shared_ptr<NeutralParticleSystem> particle_sys;
  /// MPI rank
  int rank;
  /// Sets recording frequency (value of 0 disables recording)
  int recording_step;
  /// Pointer to session object
  const LU::SessionReaderSharedPtr session;
  /// Pointer to sycl target
  NP::SYCLTargetSharedPtr sycl_target;

public:
  MassRecorder(const LU::SessionReaderSharedPtr session,
               std::shared_ptr<NeutralParticleSystem> particle_sys,
               std::shared_ptr<T> n)
      : session(session), particle_sys(particle_sys), n(n),
        sycl_target(particle_sys->sycl_target),
        initial_fluid_mass_computed(false) {

    this->session->LoadParameter("mass_recording_step", this->recording_step,
                                 0);
    this->rank = this->sycl_target->comm_pair.rank_parent;
    if ((this->rank == 0) && (this->recording_step > 0)) {
      this->fh.open("mass_recording.csv");
      this->fh << "step,relative_error,mass_particles,mass_fluid\n";
    }
  };

  ~MassRecorder() {
    if ((this->rank == 0) && (this->recording_step > 0)) {
      this->fh.close();
    }
  }

  /**
   * Integrate the Nektar number density field and convert the result to SI
   */
  inline double compute_fluid_mass() {
    return this->n->Integral(this->n->GetPhys()) * this->particle_sys->n_to_SI;
  }

  /**
   * Compute and store the integral of the initial number density field.
   */
  inline void compute_initial_fluid_mass() {
    if (this->recording_step > 0) {
      if (!this->initial_fluid_mass_computed) {
        this->initial_mass_fluid = compute_fluid_mass();
        this->initial_fluid_mass_computed = true;
      }
    }
  }

  /**
   * Get the initial fluid mass
   */
  inline double get_initial_mass() {
    NESOASSERT(this->initial_fluid_mass_computed == true,
               "initial fluid mass not computed");
    return this->initial_mass_fluid;
  }

  /**
   * Compute total mass (computational weight) in the particle system across all
   * MPI tasks.
   */
  inline double compute_particle_mass() {
    auto ga_total_weight = std::make_shared<NP::GlobalArray<REAL>>(
        this->particle_sys->particle_group->sycl_target, 1, 0.0);

    NP::particle_loop(
        "MassRecorder::compute_particle_mass",
        this->particle_sys->particle_group,
        [=](auto k_W, auto k_ga_total_weight) {
          k_ga_total_weight.add(0, k_W.at(0));
        },
        NP::Access::read(NP::Sym<NP::REAL>("COMPUTATIONAL_WEIGHT")),
        NP::Access::add(ga_total_weight))
        ->execute();

    return ga_total_weight->get().at(0);
  }

  /**
   * Compute the total mass that has been added to the particle system (ignoring
   * subsequent changes to the computational weights)
   */
  inline double compute_total_added_mass() {
    // N.B. in this case, total_num_particles_added already accounts for all MPI
    // ranks - no need for an Allreduce
    double added_mass =
        ((double)this->particle_sys->total_num_particles_added) *
        this->particle_sys->particle_init_weight;
    return added_mass;
  }

  /***
   * Compute the masses of the fluid and particle systems, and the fractional
   * error in the total mass relative to that expected. Output to file and to
   * stdout in Debug mode.
   */
  inline void compute(int step) {
    if (this->recording_step > 0) {
      if (step % this->recording_step == 0) {
        const double mass_particles = compute_particle_mass();
        const double mass_fluid = compute_fluid_mass();
        const double mass_total = mass_particles + mass_fluid;
        const double mass_added = compute_total_added_mass();
        const double correct_total = mass_added + this->initial_mass_fluid;

        // Write values to file
        if (this->rank == 0) {
          NP::nprint(step, ",",
                     abs(correct_total - mass_total) / abs(correct_total), ",",
                     mass_particles, ",", mass_fluid, ",");
          this->fh << step << ","
                   << abs(correct_total - mass_total) / abs(correct_total)
                   << "," << mass_particles << "," << mass_fluid << "\n";
        }
      }
    }
  };
};
} // namespace NESO::Solvers::DriftReduced

#endif // __NESOSOLVERS_DRIFTREDUCED_MASSRECORDER_HPP__
