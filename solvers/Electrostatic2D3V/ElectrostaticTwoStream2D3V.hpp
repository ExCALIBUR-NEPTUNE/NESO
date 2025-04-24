#ifndef __NESOSOLVERS_ELECTROSTATIC2D3V_ELECTROSTATICTWOSTREAM2D3V_HPP__
#define __NESOSOLVERS_ELECTROSTATIC2D3V_ELECTROSTATICTWOSTREAM2D3V_HPP__

#include <functional>
#include <vector>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <io/generic_hdf5_writer.hpp>

#include "Diagnostics/FieldEnergy.hpp"
#include "Diagnostics/KineticEnergy.hpp"
#include "Diagnostics/LineFieldEvaluations.hpp"
#include "Diagnostics/PotentialEnergy.hpp"
#include "ParticleSystems/ChargedParticles.hpp"
#include "ParticleSystems/PoissonParticleCoupling.hpp"

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::Electrostatic2D3V {

/// Forward declaration
template <typename T> class ElectrostaticTwoStream2D3V;

/**
 *  This is the class that sets up the components for an electrostatic PIC
 *  simulation (Two Stream) and contains the main loop.
 *
 */
template <typename T> class ElectrostaticTwoStream2D3V {
private:
  LU::SessionReaderSharedPtr session;
  SD::MeshGraphSharedPtr graph;
  SU::DriverSharedPtr drv;

  int num_write_particle_steps;
  int num_write_field_steps;
  int num_write_field_energy_steps;
  int num_print_steps;
  bool global_hdf5_write;
  int rank;
  std::vector<std::function<void(ElectrostaticTwoStream2D3V<T> *)>> callbacks;

  bool line_field_deriv_evaluations_flag;
  int line_field_deriv_evaluations_step;
  std::shared_ptr<LineFieldEvaluations<T>> line_field_evaluations;
  std::shared_ptr<LineFieldEvaluations<T>> line_field_deriv_evaluations;

  /// Integrator type: 0 -> velocity verlet, 1 -> Boris.
  int particle_integrator_type = 1;

  inline void integrator_1() {
    if (this->particle_integrator_type == 0) {
      this->charged_particles->velocity_verlet_1();
    } else {
      this->charged_particles->boris_1();
    }
  }

  inline void integrator_2() {
    if (this->particle_integrator_type == 0) {
      this->charged_particles->velocity_verlet_2();
    } else {
      this->charged_particles->boris_2();
    }
  }

public:
  /// The number of time steps in the main loop.
  int num_time_steps;
  /// The current time step of the simulation.
  int time_step;
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
  std::shared_ptr<NESO::IO::GenericHDF5Writer> generic_hdf5_writer;

  /**
   *  Create new simulation instance using a nektar++ session. The parameters
   *  for the simulation are read from the nektar+ input file.
   *
   *  @param session Nektar++ session object.
   *  @param graph Nektar++ MeshGraph instance.
   *  @param drv Nektar++ Driver instance.
   */
  ElectrostaticTwoStream2D3V(LU::SessionReaderSharedPtr session,
                             SD::MeshGraphSharedPtr graph,
                             SU::DriverSharedPtr drv)
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

    this->field_energy = std::make_shared<FieldEnergy<T>>(
        this->poisson_particle_coupling->potential_function);
    this->kinetic_energy =
        std::make_shared<KineticEnergy>(this->charged_particles->particle_group,
                                        this->charged_particles->particle_mass);
    this->potential_energy = std::make_shared<PotentialEnergy<T>>(
        this->poisson_particle_coupling->potential_function,
        this->charged_particles->particle_group,
        this->charged_particles->cell_id_translation);

    // No B field set -> use velocity verlet
    this->particle_integrator_type = 0;
    // extract the B field z magnitude from the config file

    double B_x = 0.0;
    double B_z = 0.0;
    double B_y = 0.0;
    std::string particle_B_z_magnitude_name = "particle_B_z_magnitude";
    if (this->session->DefinesParameter(particle_B_z_magnitude_name)) {
      this->session->LoadParameter(particle_B_z_magnitude_name, B_z);
      // set boris as the integrator type
      this->particle_integrator_type = 1;
    }
    // extract the B field y magnitude from the config file
    std::string particle_B_y_magnitude_name = "particle_B_y_magnitude";
    if (this->session->DefinesParameter(particle_B_y_magnitude_name)) {
      this->session->LoadParameter(particle_B_y_magnitude_name, B_y);
      // set boris as the integrator type
      this->particle_integrator_type = 1;
    }
    // extract the B field x magnitude from the config file
    std::string particle_B_x_magnitude_name = "particle_B_x_magnitude";
    if (this->session->DefinesParameter(particle_B_x_magnitude_name)) {
      this->session->LoadParameter(particle_B_x_magnitude_name, B_x);
      // set boris as the integrator type
      this->particle_integrator_type = 1;
    }
    this->charged_particles->set_B_field(B_x, B_y, B_z);

    // Override deduced integrator type with what the user requested.
    std::string particle_integrator_type_name = "particle_integrator_type";
    if (this->session->DefinesParameter(particle_integrator_type_name)) {
      this->session->LoadParameter(particle_integrator_type_name,
                                   this->particle_integrator_type);
    }

    // Rescaling factor for E field.
    std::string particle_E_rescale_name = "particle_E_rescale";
    double particle_E_rescale = 1.0;
    if (this->session->DefinesParameter(particle_E_rescale_name)) {
      this->session->LoadParameter(particle_E_rescale_name, particle_E_rescale);
    }
    this->charged_particles->set_E_coefficent(particle_E_rescale);

    NESOASSERT(((this->particle_integrator_type >= 0) ||
                (this->particle_integrator_type <= 1)),
               "Bad particle integrator type.");

    if (this->global_hdf5_write) {
      this->generic_hdf5_writer = std::make_shared<NESO::IO::GenericHDF5Writer>(
          "Electrostatic2D3V_field_trajectory.h5");

      this->generic_hdf5_writer->write_value_global(
          "L_x",
          this->charged_particles->boundary_conditions->global_extent[0]);
      this->generic_hdf5_writer->write_value_global(
          "L_y",
          this->charged_particles->boundary_conditions->global_extent[1]);
      this->generic_hdf5_writer->write_value_global(
          "q", this->charged_particles->particle_charge);
      this->generic_hdf5_writer->write_value_global(
          "m", this->charged_particles->particle_mass);
      this->generic_hdf5_writer->write_value_global(
          "w", this->charged_particles->particle_weight);
      this->generic_hdf5_writer->write_value_global("B_z", B_z);
      this->generic_hdf5_writer->write_value_global(
          "particle_integrator_type", this->particle_integrator_type);
      this->generic_hdf5_writer->write_value_global("particle_E_rescale",
                                                    particle_E_rescale);
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
          this->poisson_particle_coupling->potential_function,
          this->charged_particles, eval_nx, eval_ny, false, true);
      this->line_field_deriv_evaluations =
          std::make_shared<LineFieldEvaluations<T>>(
              this->poisson_particle_coupling->potential_function,
              this->charged_particles, eval_nx, eval_ny, true);
    }
  };

  /**
   *  Run the simulation.
   */
  inline void run() {

    if (this->num_print_steps > 0) {
      if (this->rank == 0) {
        NP::nprint("Particle count  :", this->charged_particles->num_particles);
        NP::nprint("Particle Weight :",
                   this->charged_particles->particle_weight);
      }
    }

    auto t0 = NP::profile_timestamp();
    auto t0_benchmark = NP::profile_timestamp();
    // MAIN LOOP START
    for (int stepx = 0; stepx < this->num_time_steps; stepx++) {
      this->time_step = stepx;
      // These 3 lines perform the simulation timestep.
      this->integrator_1();
      this->poisson_particle_coupling->compute_field();
      this->integrator_2();

      if (stepx == 99) {
        t0_benchmark = NP::profile_timestamp();
      }

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

      if (this->line_field_deriv_evaluations_flag &&
          (stepx % this->line_field_deriv_evaluations_step == 0)) {
        this->line_field_evaluations->write(stepx);
        this->line_field_deriv_evaluations->write(stepx);
      }

      if (this->num_print_steps > 0) {
        if ((stepx % this->num_print_steps) == 0) {

          if (this->rank == 0) {

            if (this->num_write_field_energy_steps > 0) {
              const double fe = this->field_energy->energy;
              const double ke = this->kinetic_energy->energy;
              const double pe = this->potential_energy->energy;
              const double te = pe + ke;
              NP::nprint("step:", stepx,
                         NP::profile_elapsed(t0, NP::profile_timestamp()) /
                             (stepx + 1),
                         "fe:", fe, "pe:", pe, "ke:", ke, "te:", te);
            } else {
              NP::nprint("step:", stepx,
                         NP::profile_elapsed(t0, NP::profile_timestamp()) /
                             (stepx + 1));
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
        const double time_taken =
            NP::profile_elapsed(t0, NP::profile_timestamp());
        const double time_taken_per_step = time_taken / this->num_time_steps;
        const double bench_time_taken =
            NP::profile_elapsed(t0_benchmark, NP::profile_timestamp());
        const double bench_time_taken_per_step =
            bench_time_taken / (this->num_time_steps - 100);
        NP::nprint("Time taken:", time_taken);
        NP::nprint("Time taken per step:", time_taken_per_step);
        NP::nprint("BENCHMARK Time taken:", bench_time_taken);
        NP::nprint("BENCHMARK Time taken per step:", bench_time_taken_per_step);
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
  inline void
  push_callback(std::function<void(ElectrostaticTwoStream2D3V<T> *)> &func) {
    this->callbacks.push_back(func);
  }
};

} // namespace NESO::Solvers::Electrostatic2D3V

#endif // __NESOSOLVERS_ELECTROSTATIC2D3V_ELECTROSTATICTWOSTREAM2D3V_HPP__
