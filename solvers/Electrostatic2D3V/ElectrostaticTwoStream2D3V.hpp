#ifndef __ELECTROSTATIC_TWO_STREAM_2D3V_H_
#define __ELECTROSTATIC_TWO_STREAM_2D3V_H_

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Timer.h>

#include <SolverUtils/Driver.h>
#include <SolverUtils/EquationSystem.h>

#include "Diagnostics/field_energy.hpp"
#include "Diagnostics/generic_hdf5_writer.hpp"
#include "Diagnostics/kinetic_energy.hpp"
#include "Diagnostics/potential_energy.hpp"
#include "ParticleSystems/charged_particles.hpp"
#include "ParticleSystems/poisson_particle_coupling.hpp"

#include <memory>

using namespace Nektar;
using namespace Nektar::SolverUtils;

/**
 *  This is the class that sets up the components for an electrostatic PIC
 *  simulation (Two Stream) and contains the main loop.
 *
 */
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
  bool global_hdf5_write;
  int rank;

public:
  /// This is the object that contains the particles.
  std::shared_ptr<ChargedParticles> charged_particles;
  /// Couples the particles to the Nektar++ fields.
  std::shared_ptr<PoissonParticleCoupling<T>> poisson_particle_coupling;
  /// Helper class to compute and write to HDF5 the energy of the potential
  /// evaluated as the L2 norm.
  std::shared_ptr<FieldEnergy<T>> field_energy;
  /// Helper class that computes the total kinetic energy of the particle
  /// system.
  std::shared_ptr<KineticEnergy> kinetic_energy;
  /// Helper class to compute the potential energy measured particle-wise using
  /// the potential field.
  std::shared_ptr<PotentialEnergy<T>> potential_energy;
  /// Class to write simulation details to HDF5 file
  std::shared_ptr<GenericHDF5Writer> generic_hdf5_writer;

  /**
   *  Create new simulation instance using a nektar++ session. The parameters
   *  for the simulation are read from the nektar+ input file.
   *
   *  @param session Nektar++ session object.
   *  @param graph Nektar++ MeshGraph instance.
   *  @param drv Nektar++ Driver instance.
   */
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

    this->rank = this->charged_particles->sycl_target->comm_pair.rank_parent;
    if ((this->rank == 0) && ((this->num_write_field_energy_steps > 0) ||
                              (this->num_write_particle_steps > 0))) {
      this->global_hdf5_write = true;
    } else {
      this->global_hdf5_write = false;
    }

    if (this->global_hdf5_write) {
      this->generic_hdf5_writer =
          std::make_shared<GenericHDF5Writer>("electrostatic_two_stream.h5");
    }

    if (this->num_write_field_energy_steps > 0) {
      this->field_energy = std::make_shared<FieldEnergy<T>>(
          this->poisson_particle_coupling->potential_function);
      this->kinetic_energy = std::make_shared<KineticEnergy>(
          this->charged_particles->particle_group,
          this->charged_particles->particle_mass);
      this->potential_energy = std::make_shared<PotentialEnergy<T>>(
          this->poisson_particle_coupling->potential_function,
          this->charged_particles->particle_group,
          this->charged_particles->cell_id_translation);
    }
  };

  /**
   *  Run the simulation.
   */
  inline void run() {

    if (this->num_print_steps > 0) {
      if (this->rank == 0) {
        nprint("Particle count  :", this->charged_particles->num_particles);
        nprint("Particle Weight :", this->charged_particles->particle_weight);
      }
    }

    auto t0 = profile_timestamp();
    // MAIN LOOP START
    for (int stepx = 0; stepx < this->num_time_steps; stepx++) {

      // These 3 lines perform the simulation timestep.
      this->charged_particles->velocity_verlet_1();
      this->poisson_particle_coupling->compute_field();
      this->charged_particles->velocity_verlet_2();

      // Below this line are the diagnostic calls for the timestep.
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
          this->field_energy->compute();
          this->kinetic_energy->compute();
          this->potential_energy->compute();
          if (this->global_hdf5_write) {

            this->generic_hdf5_writer->step_start(stepx);
            this->generic_hdf5_writer->write_value_step(
                "field_energy", this->field_energy->energy);
            this->generic_hdf5_writer->write_value_step(
                "kinetic_energy", this->kinetic_energy->energy);
            this->generic_hdf5_writer->write_value_step(
                "potential_energy", this->potential_energy->energy);
            this->generic_hdf5_writer->step_end();
          }
        }
      }
      if (this->num_print_steps > 0) {
        if ((stepx % this->num_print_steps) == 0) {

          if (this->rank == 0) {

            if (this->num_write_field_energy_steps > 0) {
              const double fe = this->field_energy->energy;
              const double ke = this->kinetic_energy->energy;
              const double pe = this->potential_energy->energy;
              const double te = 0.5 * pe + ke;
              nprint("step:", stepx,
                     profile_elapsed(t0, profile_timestamp()) / (stepx + 1),
                     "fe:", fe, "pe:", pe, "ke:", ke, "te:", te);
            } else {
              nprint("step:", stepx, profile_elapsed(t0, profile_timestamp()));
            }
          }
        }
      }
    } // MAIN LOOP END

    if (this->num_print_steps > 0) {
      if (this->rank == 0) {
        const double time_taken = profile_elapsed(t0, profile_timestamp());
        const double time_taken_per_step = time_taken / this->num_time_steps;
        nprint("Time taken:", time_taken);
        nprint("Time taken per step:", time_taken_per_step);
      }
    }
  }

  /**
   * Finalise the simulation, i.e. close output files and free objects.
   */
  inline void finalise() {

    this->charged_particles->free();

    if (this->global_hdf5_write) {
      this->generic_hdf5_writer->close();
    }
  }
};

#endif
