#ifndef __NESOSOLVERS_SIMPLESOL_MASSCONSERVATION_HPP__
#define __NESOSOLVERS_SIMPLESOL_MASSCONSERVATION_HPP__

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>

#include "../ParticleSystems/NeutralParticleSystem.hpp"

#include <fstream>
#include <iostream>

namespace LU = Nektar::LibUtilities;
namespace NESO::Solvers::SimpleSOL {
template <typename T> class MassRecording {
protected:
  const LU::SessionReaderSharedPtr session;
  std::shared_ptr<NeutralParticleSystem> particle_sys;
  std::shared_ptr<T> rho;

  NP::SYCLTargetSharedPtr sycl_target;
  bool initial_mass_computed;
  double initial_mass_fluid;
  int mass_recording_step;
  int rank;
  std::ofstream fh;

public:
  MassRecording(const LU::SessionReaderSharedPtr session,
                std::shared_ptr<NeutralParticleSystem> particle_sys,
                std::shared_ptr<T> rho)
      : session(session), particle_sys(particle_sys), rho(rho),
        sycl_target(particle_sys->sycl_target), initial_mass_computed(false) {

    session->LoadParameter("mass_recording_step", mass_recording_step, 0);
    rank = sycl_target->comm_pair.rank_parent;
    if ((rank == 0) && (mass_recording_step > 0)) {
      fh.open("mass_recording.csv");
      fh << "step,relative_error,mass_particles,mass_fluid\n";
    }
  };

  ~MassRecording() {
    if ((rank == 0) && (mass_recording_step > 0)) {
      fh.close();
    }
  }

  inline double compute_particle_mass() {
    auto ga_total_weight =
        std::make_shared<NP::GlobalArray<NP::REAL>>(this->sycl_target, 1, 0.0);

    NP::particle_loop(
        "MassRecording::compute_particle_mass",
        this->particle_sys->particle_group,
        [=](auto k_W, auto k_ga_total_weight) {
          k_ga_total_weight.add(0, k_W.at(0));
        },
        NP::Access::read(NP::Sym<NP::REAL>("COMPUTATIONAL_WEIGHT")),
        NP::Access::add(ga_total_weight))
        ->execute();

    return ga_total_weight->get().at(0);
  }

  inline double compute_fluid_mass() {
    return this->rho->Integral(this->rho->GetPhys()) *
           this->particle_sys->n_to_SI;
  }

  inline double compute_total_added_mass() {
    const uint64_t num_particles_added =
        this->particle_sys->total_num_particles_added;
    uint64_t global_num_added;
    MPICHK(MPI_Allreduce(&num_particles_added, &global_num_added, 1,
                         MPI_UINT64_T, MPI_SUM,
                         sycl_target->comm_pair.comm_parent));

    double added_mass =
        ((double)global_num_added) * this->particle_sys->particle_weight;

    return added_mass;
  }

  inline void compute_initial_fluid_mass() {
    if (!this->initial_mass_computed) {
      this->initial_mass_fluid = this->compute_fluid_mass();
      this->initial_mass_computed = true;
    }
  }

  inline double get_initial_mass() {
    NESOASSERT(this->initial_mass_computed == true,
               "initial mass not computed");
    return this->initial_mass_fluid;
  }

  inline void compute(int step) {
    if (mass_recording_step > 0) {
      if (step % mass_recording_step == 0) {
        const double mass_particles = this->compute_particle_mass();
        const double mass_fluid = this->compute_fluid_mass();
        const double mass_total = mass_particles + mass_fluid;
        const double mass_added = this->compute_total_added_mass();
        const double correct_total = mass_added + this->initial_mass_fluid;

        // Write values to file
        if (rank == 0) {
          NP::nprint(step, ",",
                     abs(correct_total - mass_total) / abs(correct_total), ",",
                     mass_particles, ",", mass_fluid, ",");
          fh << step << ","
             << abs(correct_total - mass_total) / abs(correct_total) << ","
             << mass_particles << "," << mass_fluid << "\n";
        }
      }
    }
  };
};
} // namespace NESO::Solvers::SimpleSOL
#endif // __NESOSOLVERS_SIMPLESOL_MASSCONSERVATION_HPP__
