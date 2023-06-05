#ifndef __ELECTROSTATIC_TWO_STREAM_2D3V_H_
#define __ELECTROSTATIC_TWO_STREAM_2D3V_H_

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Timer.h>

#include <SolverUtils/Driver.h>
#include <SolverUtils/EquationSystem.h>

#include "Diagnostics/field_energy.hpp"
#include "Diagnostics/generic_hdf5_writer.hpp"
#include "Diagnostics/kinetic_energy.hpp"
#include "Diagnostics/line_field_evaluations.hpp"
#include "Diagnostics/potential_energy.hpp"
#include "ParticleSystems/charged_particles.hpp"
#include "ParticleSystems/maxwell_wave_particle_coupling.hpp"

#include <functional>
#include <memory>
#include <vector>

using namespace Nektar;
using namespace Nektar::SolverUtils;

/// Forward declaration
template <typename T> class RingBeam2D3V;

/**
 *  This is the class that sets up the components for an electrostatic PIC
 *  simulation (Two Stream) and contains the main loop.
 *
 */
template <typename T> class RingBeam2D3V {
private:
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  DriverSharedPtr drv;

  int num_write_particle_steps;
  int num_write_field_steps;
  int num_write_field_energy_steps;
  int num_print_steps;
  bool global_hdf5_write;
  int rank;
  std::vector<std::function<void(RingBeam2D3V<T> *)>> callbacks;

  bool line_field_deriv_evaluations_flag;
  int line_field_deriv_evaluations_step;
  std::shared_ptr<LineFieldEvaluations<T>> line_field_evaluations;
  std::shared_ptr<LineFieldEvaluations<T>> line_field_deriv_evaluations;

public:
  /// the number of particle species
  int num_particle_species;
  /// The number of time steps in the main loop.
  int num_time_steps;
  /// The current time step of the simulation.
  int time_step;
  /// This is the object that contains the particles.
  std::shared_ptr<ChargedParticles> charged_particles;
  /// Couples the particles to the Nektar++ fields.
  std::shared_ptr<MaxwellWaveParticleCoupling<T>>
      maxwell_wave_particle_coupling;
  /// Helper class to compute and write to HDF5 the energy of the potential
  /// evaluated as the L2 norm.
  std::shared_ptr<FieldEnergy<T>> field_energy;
  /// Helper class that computes the total kinetic energy of the particle
  /// system.
  std::vector<std::shared_ptr<KineticEnergy>> kinetic_energies;
  /// Helper class to compute the potential energy measured particle-wise using
  /// the potential field.
  std::vector<std::shared_ptr<PotentialEnergy<T>>> potential_energies;
  /// Class to write simulation details to HDF5 file
  std::shared_ptr<GenericHDF5Writer> generic_hdf5_writer;
  /// offset magnetic field
  std::tuple<double, double, double> m_Bxyz;

  /**
   *  Create new simulation instance using a nektar++ session. The parameters
   *  for the simulation are read from the nektar+ input file.
   *
   *  @param session Nektar++ session object.
   *  @param graph Nektar++ MeshGraph instance.
   *  @param drv Nektar++ Driver instance.
   */
  RingBeam2D3V(LibUtilities::SessionReaderSharedPtr session,
               SpatialDomains::MeshGraphSharedPtr graph, DriverSharedPtr drv)
      : session(session), graph(graph), drv(drv) {

    this->session->LoadParameter("number_of_particle_species",
                                 this->num_particle_species);
    this->charged_particles = std::make_shared<ChargedParticles>(session, graph);
    this->maxwell_wave_particle_coupling =
        std::make_shared<MaxwellWaveParticleCoupling<T>>(
            session, graph, drv, this->charged_particles);

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

    this->field_energy = std::make_shared<FieldEnergy<T>>(
        this->maxwell_wave_particle_coupling->phi_function);

    for (uint32_t i = 0; i < this->charged_particles->num_species; ++i) {
      auto mass = this->charged_particles->particle_initial_conditions[i].mass;
      auto charge = this->charged_particles->particle_initial_conditions[i].charge;
      this->kinetic_energies.emplace_back(
          std::make_shared<KineticEnergy>(this->charged_particles->particle_groups[i],
                                          mass));
      // TODO fix potential energy
      this->potential_energies.emplace_back(std::make_shared<PotentialEnergy<T>>(
          this->maxwell_wave_particle_coupling->phi_function,
          this->maxwell_wave_particle_coupling->ax_function,
          this->maxwell_wave_particle_coupling->ay_function,
          this->maxwell_wave_particle_coupling->az_function,
          this->charged_particles->particle_groups[i],
          this->charged_particles->cell_id_translation));
    }

    // extract the B field z magnitude from the config file

    double B_x = 0.0;
    double B_z = 0.0;
    double B_y = 0.0;
    std::string B_z_magnitude_name = "B_z_magnitude";
    if (this->session->DefinesParameter(B_z_magnitude_name)) {
      this->session->LoadParameter(B_z_magnitude_name, B_z);
    }
    // extract the B field y magnitude from the config file
    std::string B_y_magnitude_name = "B_y_magnitude";
    if (this->session->DefinesParameter(B_y_magnitude_name)) {
      this->session->LoadParameter(B_y_magnitude_name, B_y);
    }
    // extract the B field x magnitude from the config file
    std::string B_x_magnitude_name = "B_x_magnitude";
    if (this->session->DefinesParameter(B_x_magnitude_name)) {
      this->session->LoadParameter(B_x_magnitude_name, B_x);
    }
    m_Bxyz = std::make_tuple(B_x, B_y, B_z);
    //this->charged_particles->set_B_field(B_x, B_y, B_z);

    // Rescaling factor for E field.
    //std::string particle_E_rescale_name = "particle_E_rescale";
    //double particle_E_rescale = 1.0;
    //if (this->session->DefinesParameter(particle_E_rescale_name)) {
    //  this->session->LoadParameter(particle_E_rescale_name, particle_E_rescale);
    //}
    //this->charged_particles = species->set_E_coefficent(particle_E_rescale);

    if (this->global_hdf5_write) {
      this->generic_hdf5_writer = std::make_shared<GenericHDF5Writer>(
          "MaxwellWave2D3V_field_trajectory.h5");

      this->generic_hdf5_writer->write_value_global(
          "L_x",
          this->charged_particles->boundary_condition->global_extent[0]);
      this->generic_hdf5_writer->write_value_global(
          "L_y",
          this->charged_particles->boundary_condition->global_extent[1]);
      uint32_t counter = 0;
      for (auto pic : this->charged_particles->particle_initial_conditions) {
        this->generic_hdf5_writer->write_value_global("q_"+std::to_string(counter),
          pic.charge);
        this->generic_hdf5_writer->write_value_global("m_"+std::to_string(counter),
          pic.mass);
        this->generic_hdf5_writer->write_value_global("w_"+std::to_string(counter),
          pic.weight);
        counter += 1;
      }
      this->generic_hdf5_writer->write_value_global("B_z", B_z);
//      this->generic_hdf5_writer->write_value_global("particle_E_rescale",
//                                                    particle_E_rescale);
    }

    std::string line_field_deriv_evalutions_name =
        "line_field_deriv_evaluations_step";
    this->line_field_deriv_evaluations_flag =
        this->session->DefinesParameter(line_field_deriv_evalutions_name);

    int eval_nx = -1;
    int eval_ny = -1;
    if (this->line_field_deriv_evaluations_flag) {
      this->session->LoadParameter(line_field_deriv_evalutions_name,
                                   this->line_field_deriv_evaluations_step);
      this->session->LoadParameter("line_field_deriv_evaluations_numx",
                                   eval_nx);
      this->session->LoadParameter("line_field_deriv_evaluations_numy",
                                   eval_ny);
    }
    this->line_field_deriv_evaluations_flag &=
        (this->line_field_deriv_evaluations_step > 0);

    if (this->line_field_deriv_evaluations_flag) {
      this->line_field_evaluations = std::make_shared<LineFieldEvaluations<T>>(
          this->maxwell_wave_particle_coupling->phi_function,
          this->charged_particles, eval_nx, eval_ny, false, true);
      this->line_field_deriv_evaluations =
          std::make_shared<LineFieldEvaluations<T>>(
              this->maxwell_wave_particle_coupling->phi_function,
              this->charged_particles, eval_nx, eval_ny, true);
    }
  };

  /**
   *  Run the simulation.
   */
  inline void run() {

    if (this->num_print_steps > 0) {
      if (this->rank == 0) {
        nprint(" Number of species ",
          this->charged_particles->num_species);
      }
    }
    const auto timestep_original = this->charged_particles->dt;
    NESOASSERT(timestep_original > 0, "The time step must be > 0");

    auto t0 = profile_timestamp();
    auto t0_benchmark = profile_timestamp();
    // MAIN LOOP START
    auto startuptimestep = 1.0e-16;
    for (int stepx = 0; stepx < this->num_time_steps; stepx++) {

      // use timestep_multiplieriplier to warm up the field solver
      const double dtMultiplier = std::min(1.0, std::pow(10.0, std::max(-1, stepx - 16)));

      if (dtMultiplier < 1) {
        stepx = 0;
      }

      this->time_step = stepx;

      // These 3 lines perform the simulation timestep.
      this->charged_particles->accelerate(dtMultiplier);
      this->charged_particles->advect(0.5 * dtMultiplier);
      this->charged_particles->communicate();// maybe only need the other one?
      this->maxwell_wave_particle_coupling->deposit_current();
      this->charged_particles->advect(0.5 * dtMultiplier);
      this->charged_particles->communicate();
      this->maxwell_wave_particle_coupling->deposit_charge();
      this->maxwell_wave_particle_coupling->integrate_fields(dtMultiplier);

      if (stepx == 99) {
        t0_benchmark = profile_timestamp();
      }

      // Below this line are the diagnostic calls for the timestep.
      if (this->num_write_particle_steps > 0) {
        if ((stepx % this->num_write_particle_steps) == 0) {
          this->charged_particles->write();
        }
      }
      if (this->num_write_field_steps > 0) {
        if ((stepx % this->num_write_field_steps) == 0) {
          this->maxwell_wave_particle_coupling->write_sources(stepx);
          this->maxwell_wave_particle_coupling->write_potentials(stepx);
        }
      }

      if (this->num_write_field_energy_steps > 0) {
        if ((stepx % this->num_write_field_energy_steps) == 0) {
          this->field_energy->compute();
          for (auto ke : this->kinetic_energies) { ke->compute(); }
          for (auto pe : this->potential_energies) { pe->compute(); }
          if (this->global_hdf5_write) {

            this->generic_hdf5_writer->step_start(stepx);
            this->generic_hdf5_writer->write_value_step(
                "field_energy", this->field_energy->energy);
            uint32_t counter = 0;
            for (auto ke : this->kinetic_energies) {
              this->generic_hdf5_writer->write_value_step(
                  "kinetic_energy_" + std::to_string(counter),
                   ke->energy);
              counter += 1;
            }
            counter = 0;
            for (auto pe : this->potential_energies) {
              this->generic_hdf5_writer->write_value_step(
                  "potential_energy_" + std::to_string(counter), pe->energy);
              counter += 1;
            }
            this->generic_hdf5_writer->step_end();
          }
        }
      }

      if (this->line_field_deriv_evaluations_flag &&
          (stepx % this->line_field_deriv_evaluations_step == 0)) {
        this->line_field_evaluations->write(stepx);
        this->line_field_deriv_evaluations->write(stepx);
      }

      if (this->num_print_steps > 0) {
        if ((stepx % this->num_print_steps) == 0) {

          if (this->rank == 0) {

            if (this->num_write_field_energy_steps > 0) {
              double ke = 0.0; // total
              for (auto i : this->kinetic_energies) { ke += i->energy; }
              double pe = 0.0; // total
              for (auto i : this->potential_energies) { pe += i->energy; }
              const double fe = this->field_energy->energy;
              const double te = pe + ke;
              nprint("step:", stepx,
                     profile_elapsed(t0, profile_timestamp()) / (stepx + 1),
                     "fe:", fe, "pe:", pe, "ke:", ke, "te:", te);
            } else {
              nprint("step:", stepx,
                     profile_elapsed(t0, profile_timestamp()) / (stepx + 1));
            }
          }
        }
      }

      // call each callback with this object
      for (auto &cx : this->callbacks) {
        cx(this);
      }

    } // MAIN LOOP END

    if (this->num_print_steps > 0) {
      if (this->rank == 0) {
        const double time_taken = profile_elapsed(t0, profile_timestamp());
        const double time_taken_per_step = time_taken / this->num_time_steps;
        const double bench_time_taken =
            profile_elapsed(t0_benchmark, profile_timestamp());
        const double bench_time_taken_per_step =
            bench_time_taken / (this->num_time_steps - 100);
        nprint("Time taken:", time_taken);
        nprint("Time taken per step:", time_taken_per_step);
        nprint("BENCHMARK Time taken:", bench_time_taken);
        nprint("BENCHMARK Time taken per step:", bench_time_taken_per_step);
      }
    }
  }

  /**
   * Finalise the simulation, i.e. close output files and free objects.
   */
  inline void finalise() {

    if (this->line_field_deriv_evaluations_flag) {
      this->line_field_evaluations->close();
      this->line_field_deriv_evaluations->close();
    }
    this->charged_particles->free();

    if (this->global_hdf5_write) {
      this->generic_hdf5_writer->close();
    }
  }

  /**
   * Push a callback onto the set of functions to call at each time step.
   *
   * @param func Callback to add.
   */
  inline void push_callback(std::function<void(RingBeam2D3V<T> *)> &func) {
    this->callbacks.emplace_back(func);
  }
};

#endif
