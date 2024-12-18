#ifndef H3LAPD_MASS_RECORDER_H
#define H3LAPD_MASS_RECORDER_H

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>

#include "../ParticleSystems/NeutralParticleSystem.hpp"

namespace LU = Nektar::LibUtilities;

namespace NESO::Solvers::H3LAPD {
/**
 * @brief Class to manage recording of masses in a coupled fluid-particle
 * system.
 */
template <typename T> class MassRecorder {
protected:
  /// File handle for recording output
  std::ofstream m_fh;
  /// Flag to track whether initial fluid mass has been computed
  bool m_initial_fluid_mass_computed;
  /// The initial fluid mass
  double m_initial_mass_fluid;
  /// Pointer to number density field
  std::shared_ptr<T> m_n;
  /// Pointer to particle system
  std::shared_ptr<NeutralParticleSystem> m_particle_sys;
  /// MPI rank
  int m_rank;
  /// Sets recording frequency (value of 0 disables recording)
  int m_recording_step;
  /// Pointer to session object
  const LU::SessionReaderSharedPtr m_session;
  /// Pointer to sycl target
  SYCLTargetSharedPtr m_sycl_target;

public:
  MassRecorder(const LU::SessionReaderSharedPtr session,
               std::shared_ptr<NeutralParticleSystem> particle_sys,
               std::shared_ptr<T> n)
      : m_session(session), m_particle_sys(particle_sys), m_n(n),
        m_sycl_target(particle_sys->m_sycl_target),
        m_initial_fluid_mass_computed(false) {

    m_session->LoadParameter("mass_recording_step", m_recording_step, 0);
    m_rank = m_sycl_target->comm_pair.rank_parent;
    if ((m_rank == 0) && (m_recording_step > 0)) {
      m_fh.open("mass_recording.csv");
      m_fh << "step,relative_error,mass_particles,mass_fluid\n";
    }
  };

  ~MassRecorder() {
    if ((m_rank == 0) && (m_recording_step > 0)) {
      m_fh.close();
    }
  }

  /**
   * Integrate the Nektar number density field and convert the result to SI
   */
  inline double compute_fluid_mass() {
    return m_n->Integral(m_n->GetPhys()) * m_particle_sys->m_n_to_SI;
  }

  /**
   * Compute and store the integral of the initial number density field.
   */
  inline void compute_initial_fluid_mass() {
    if (m_recording_step > 0) {
      if (!m_initial_fluid_mass_computed) {
        m_initial_mass_fluid = compute_fluid_mass();
        m_initial_fluid_mass_computed = true;
      }
    }
  }

  /**
   * Get the initial fluid mass
   */
  inline double get_initial_mass() {
    NESOASSERT(m_initial_fluid_mass_computed == true,
               "initial fluid mass not computed");
    return m_initial_mass_fluid;
  }

  /**
   * Compute total mass (computational weight) in the particle system across all
   * MPI tasks.
   */
  inline double compute_particle_mass() {
    auto ga_total_weight = std::make_shared<GlobalArray<REAL>>(
        this->m_particle_sys->m_particle_group->sycl_target, 1, 0.0);

    particle_loop(
        "MassRecorder::compute_particle_mass",
        this->m_particle_sys->m_particle_group,
        [=](auto k_W, auto k_ga_total_weight) {
          k_ga_total_weight.add(0, k_W.at(0));
        },
        Access::read(Sym<REAL>("COMPUTATIONAL_WEIGHT")),
        Access::add(ga_total_weight))
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
    double added_mass = ((double)m_particle_sys->m_total_num_particles_added) *
                        m_particle_sys->m_particle_init_weight;
    return added_mass;
  }

  /***
   * Compute the masses of the fluid and particle systems, and the fractional
   * error in the total mass relative to that expected. Output to file and to
   * stdout in Debug mode.
   */
  inline void compute(int step) {
    if (m_recording_step > 0) {
      if (step % m_recording_step == 0) {
        const double mass_particles = compute_particle_mass();
        const double mass_fluid = compute_fluid_mass();
        const double mass_total = mass_particles + mass_fluid;
        const double mass_added = compute_total_added_mass();
        const double correct_total = mass_added + m_initial_mass_fluid;

        // Write values to file
        if (m_rank == 0) {
          nprint(step, ",",
                 abs(correct_total - mass_total) / abs(correct_total), ",",
                 mass_particles, ",", mass_fluid, ",");
          m_fh << step << ","
               << abs(correct_total - mass_total) / abs(correct_total) << ","
               << mass_particles << "," << mass_fluid << "\n";
        }
      }
    }
  };
};
} // namespace NESO::Solvers::H3LAPD

#endif
