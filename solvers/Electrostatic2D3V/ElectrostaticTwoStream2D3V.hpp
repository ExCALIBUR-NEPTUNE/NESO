#ifndef __ELECTROSTATIC_TWO_STREAM_2D3V_H_
#define __ELECTROSTATIC_TWO_STREAM_2D3V_H_

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Timer.h>

#include <SolverUtils/Driver.h>
#include <SolverUtils/EquationSystem.h>

#include "Diagnostics/field_energy.hpp"
#include "Diagnostics/kinetic_energy.hpp"
#include "Diagnostics/potential_energy.hpp"
#include "ParticleSystems/charged_particles.hpp"
#include "ParticleSystems/poisson_particle_coupling.hpp"

#include <memory>

using namespace Nektar;
using namespace Nektar::SolverUtils;

template <typename T> class ElectrostaticTwoStream2D3V {
private:
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  DriverSharedPtr drv;

  int num_time_steps;
  int num_write_particle_steps;
  int num_write_field_steps;
  int num_write_field_energy_steps;
  int num_print_steps;

public:
  std::shared_ptr<ChargedParticles> charged_particles;
  std::shared_ptr<PoissonParticleCoupling<T>> poisson_particle_coupling;
  std::shared_ptr<FieldEnergy<T>> field_energy;
  std::shared_ptr<KineticEnergy> kinetic_energy;
  std::shared_ptr<PotentialEnergy<T>> potential_energy;

  ElectrostaticTwoStream2D3V(LibUtilities::SessionReaderSharedPtr session,
                             SpatialDomains::MeshGraphSharedPtr graph,
                             DriverSharedPtr drv)
      : session(session), graph(graph), drv(drv) {

    this->charged_particles =
        std::make_shared<ChargedParticles>(session, graph);
    this->poisson_particle_coupling =
        std::make_shared<PoissonParticleCoupling<T>>(session, graph, drv,
                                                     this->charged_particles);

    this->session->LoadParameter("particle_num_time_steps",
                                 this->num_time_steps);
    this->session->LoadParameter("particle_num_write_particle_steps",
                                 this->num_write_particle_steps);
    this->session->LoadParameter("particle_num_write_field_steps",
                                 this->num_write_field_steps);
    this->session->LoadParameter("particle_num_write_field_energy_steps",
                                 this->num_write_field_energy_steps);
    this->session->LoadParameter("particle_num_print_steps",
                                 this->num_print_steps);

    if (this->num_write_field_energy_steps > 0) {
      this->field_energy = std::make_shared<FieldEnergy<T>>(
          this->poisson_particle_coupling->potential_function,
          "field_energy.h5");
      this->kinetic_energy = std::make_shared<KineticEnergy>(
          this->charged_particles->particle_group, "kinetic_energy.h5",
          this->charged_particles->particle_mass);
      this->potential_energy = std::make_shared<PotentialEnergy<T>>(
          this->poisson_particle_coupling->potential_function,
          this->charged_particles->particle_group,
          this->charged_particles->cell_id_translation, "potential_energy.h5");
    }
  };

  inline void run() {

    if (this->num_print_steps > 0) {
      if (this->charged_particles->sycl_target->comm_pair.rank_parent == 0) {
        nprint("Particle count  :", this->charged_particles->num_particles);
        nprint("Particle Weight :", this->charged_particles->particle_weight);
      }
    }

    auto t0 = profile_timestamp();
    for (int stepx = 0; stepx < this->num_time_steps; stepx++) {

      this->charged_particles->velocity_verlet_1();
      this->poisson_particle_coupling->compute_field();
      this->charged_particles->velocity_verlet_2();

      // writes trajectory
      if (this->num_write_particle_steps > 0) {
        if ((stepx % this->num_write_particle_steps) == 0) {
          this->charged_particles->write();
        }
      }
      if (this->num_write_field_steps > 0) {
        if ((stepx % this->num_write_field_steps) == 0) {
          this->poisson_particle_coupling->write_forcing(stepx);
          this->poisson_particle_coupling->write_potential(stepx);
        }
      }
      if (this->num_write_field_energy_steps > 0) {
        if ((stepx % this->num_write_field_energy_steps) == 0) {
          this->field_energy->write(stepx);
          this->kinetic_energy->write(stepx);
          this->potential_energy->write(stepx);
        }
      }
      if (this->num_print_steps > 0) {
        if ((stepx % this->num_print_steps) == 0) {
          if (this->charged_particles->sycl_target->comm_pair.rank_parent ==
              0) {
            const double fe = this->field_energy->energy;
            const double ke = this->kinetic_energy->energy;
            const double pe = this->potential_energy->energy;
            const double te = 0.5 * pe + ke;
            nprint("step:", stepx,
                   profile_elapsed(t0, profile_timestamp()) / (stepx + 1),
                   "fe:", fe, "pe:", pe, "ke:", ke, "te:", te);
          }
        }
      }
    }

    if (this->num_print_steps > 0) {
      if (this->charged_particles->sycl_target->comm_pair.rank_parent == 0) {
        const double time_taken = profile_elapsed(t0, profile_timestamp());
        const double time_taken_per_step = time_taken / this->num_time_steps;
        nprint("Time taken:", time_taken);
        nprint("Time taken per step:", time_taken_per_step);
      }
    }
  }

  inline void finalise() {
    this->field_energy->close();
    this->kinetic_energy->close();
    this->potential_energy->close();
    this->charged_particles->free();
  }
};

#endif
