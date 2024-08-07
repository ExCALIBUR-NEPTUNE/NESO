#ifndef __MAXWELLWAVE2D3VDRIVER_H_
#define __MAXWELLWAVE2D3VDRIVER_H_

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Timer.h>

#include <SolverUtils/Driver.h>
#include <SolverUtils/EquationSystem.h>

#include "Diagnostics/FieldEnergy.hpp"
#include "Diagnostics/FieldMean.hpp"
#include "Diagnostics/GenericHDF5Writer.hpp"
#include "Diagnostics/KineticEnergy.hpp"
#include "Diagnostics/GridFieldEvaluations.hpp"
#include "Diagnostics/PotentialEnergy.hpp"
#include "ParticleSystems/ChargedParticles.hpp"
#include "ParticleSystems/MaxwellWaveParticleCoupling.hpp"

#include <functional>
#include <memory>
#include <vector>

using namespace Nektar;
using namespace Nektar::SolverUtils;

/// Forward declaration
template <typename T> class MaxwellWave2D3VDriver;

/**
 *  This is the class that sets up the 2D3V EM PIC
 *  simulation and contains the main loop.
 *
 */
template <typename T> class MaxwellWave2D3VDriver {
private:
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;

  int num_write_particle_steps;
  int num_write_field_steps;
  int num_write_field_energy_steps;
  int num_print_steps;
  bool global_hdf5_write;
  int rank;
  std::vector<std::function<void(MaxwellWave2D3VDriver<T> *)>> callbacks;

  bool grid_field_evaluations_flag;
  int grid_field_evaluations_step;
  std::vector<std::shared_ptr<GridFieldEvaluations<T>>> grid_field_evaluations;

public:
  /// the number of particle species
  int num_particle_species;
  /// The number of time steps in the main loop.
  int num_time_steps;
  /// The current time step of the simulation.
  int time_step;
  /// The parameter that controls implicitness (0 = explicit, 1 = implicit)
  double theta;
  /// A flag to switch off the deposition of particle quantities to the grid
  int m_do_deposition;
  /// A flag to switch off the use of div j = -drho / dt and deposit charge instead
  int m_use_charge_conservation;
  /// This is the object that contains the particles.
  std::shared_ptr<ChargedParticles> m_chargedParticles;
  /// Couples the particles to the Nektar++ fields.
  std::shared_ptr<MaxwellWaveParticleCoupling<T>>
      m_maxwellWaveParticleCoupling;
  /// Helper class to compute and write to HDF5 the energy of the fields
  /// evaluated as the L2 norm.
  std::shared_ptr<FieldEnergy<T>> m_fieldEnergy;
  /// Helper class that computes the total kinetic energy of the particle
  /// system.
  std::vector<std::shared_ptr<KineticEnergy>> kinetic_energies;
  /// Helper class to compute the potential energy measured particle-wise using
  /// the potential field.
  std::vector<std::shared_ptr<PotentialEnergy<T>>> potential_energies;
  /// Class to write simulation details to HDF5 file
  std::shared_ptr<GenericHDF5Writer> generic_hdf5_writer;

  /**
   *  Create new simulation instance using a nektar++ session. The parameters
   *  for the simulation are read from the nektar+ input file.
   *
   *  @param session Nektar++ session object.
   *  @param graph Nektar++ MeshGraph instance.
   */
  MaxwellWave2D3VDriver(LibUtilities::SessionReaderSharedPtr session,
              SpatialDomains::MeshGraphSharedPtr graph)
      : session(session), graph(graph) {

    this->session->LoadParameter("number_of_particle_species",
                                 this->num_particle_species);

    this->session->LoadParameter("theta", this->theta, 0.0);
    this->session->LoadParameter("perform_deposition", this->m_do_deposition, 1);
    NESOASSERT(this->m_do_deposition == 0 || this->m_do_deposition == 1,
      "perform_deposition must be 0 (for false) or 1 (for true)");
    this->session->LoadParameter("use_charge_conservation",
        this->m_use_charge_conservation, 1);
    NESOASSERT(this->m_use_charge_conservation == 0 || this->m_use_charge_conservation == 1,
      "use_charge_conservation must be 0 (for false) or 1 (for true)");
    this->m_chargedParticles =
        std::make_shared<ChargedParticles>(session, graph);
    this->m_maxwellWaveParticleCoupling =
        std::make_shared<MaxwellWaveParticleCoupling<T>>(
            session, graph, this->m_chargedParticles);

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

    this->rank = this->m_chargedParticles->sycl_target->comm_pair.rank_parent;
    if ((this->rank == 0) && ((this->num_write_field_energy_steps > 0) ||
                              (this->num_write_particle_steps > 0))) {
      this->global_hdf5_write = true;
    } else {
      this->global_hdf5_write = false;
    }

    if (rank == 0) {
      std::cout << "The simulation will run for " << this->num_time_steps <<
        " time steps" << std::endl;
    }

    this->m_fieldEnergy = std::make_shared<FieldEnergy<T>>();

    for (uint32_t i = 0; i < this->m_chargedParticles->num_species; ++i) {
      auto mass = this->m_chargedParticles->particle_initial_conditions[i].mass;
      auto charge =
          this->m_chargedParticles->particle_initial_conditions[i].charge;
      this->kinetic_energies.emplace_back(std::make_shared<KineticEnergy>(
          this->m_chargedParticles->particle_groups[i], mass));
      // TODO fix potential energy
      this->potential_energies.emplace_back(
          std::make_shared<PotentialEnergy<T>>(
              this->m_maxwellWaveParticleCoupling->phi_field,
              this->m_maxwellWaveParticleCoupling->ax_field,
              this->m_maxwellWaveParticleCoupling->ay_field,
              this->m_maxwellWaveParticleCoupling->az_field,
              this->m_chargedParticles->particle_groups[i],
              this->m_chargedParticles->cell_id_translations[i]));
    }

    // extract the B field z magnitude from the config file

    double B0x = 0.0;
    double B0y = 0.0;
    double B0z = 0.0;
    // extract the B field x magnitude from the config file
    if (this->session->DefinesParameter("B0x")) {
      this->session->LoadParameter("B0x", B0x);
    }
    // extract the B field y magnitude from the config file
    if (this->session->DefinesParameter("B0y")) {
      this->session->LoadParameter("B0y", B0y);
    }
    // extract the B field z magnitude from the config file
    if (this->session->DefinesParameter("B0z")) {
      this->session->LoadParameter("B0z", B0z);
    }

    if (this->global_hdf5_write) {
      this->generic_hdf5_writer = std::make_shared<GenericHDF5Writer>(
          "MaxwellWave2D3V_field_trajectory.h5");

      this->generic_hdf5_writer->write_value_global(
          "L_x", this->m_chargedParticles->boundary_conditions[0]->global_extent[0]);
      this->generic_hdf5_writer->write_value_global(
          "L_y", this->m_chargedParticles->boundary_conditions[0]->global_extent[1]);
      uint32_t counter = 0;
      for (auto pic : this->m_chargedParticles->particle_initial_conditions) {
        this->generic_hdf5_writer->write_value_global(
            "q_" + std::to_string(counter), pic.charge);
        this->generic_hdf5_writer->write_value_global(
            "m_" + std::to_string(counter), pic.mass);
        this->generic_hdf5_writer->write_value_global(
            "w_" + std::to_string(counter), pic.weight);
        counter += 1;
      }
      this->generic_hdf5_writer->write_value_global("B0x", B0x);
      this->generic_hdf5_writer->write_value_global("B0y", B0y);
      this->generic_hdf5_writer->write_value_global("B0z", B0z);
      //      this->generic_hdf5_writer->write_value_global("particle_E_rescale",
      //                                                    particle_E_rescale);
    }


    this->session->LoadParameter("grid_field_evaluations_step",
                                 this->grid_field_evaluations_step, 0);

    this->grid_field_evaluations_flag =
        (this->grid_field_evaluations_step > 0);

    int eval_nx = -1;
    int eval_ny = -1;
    if (this->grid_field_evaluations_flag) {
      this->session->LoadParameter("grid_field_evaluations_numx",
                                   eval_nx);
      this->session->LoadParameter("grid_field_evaluations_numy",
                                   eval_ny);
    }

    if (this->grid_field_evaluations_flag) {
      this->grid_field_evaluations.push_back(std::make_shared<GridFieldEvaluations<T>>(
          this->m_maxwellWaveParticleCoupling->phi_field,
          this->m_chargedParticles, eval_nx, eval_ny, "phi_grid.h5part"));
      this->grid_field_evaluations.push_back(std::make_shared<GridFieldEvaluations<T>>(
          this->m_maxwellWaveParticleCoupling->ax_field,
          this->m_chargedParticles, eval_nx, eval_ny, "ax_grid.h5part"));
      this->grid_field_evaluations.push_back(std::make_shared<GridFieldEvaluations<T>>(
          this->m_maxwellWaveParticleCoupling->ay_field,
          this->m_chargedParticles, eval_nx, eval_ny, "ay_grid.h5part"));
      this->grid_field_evaluations.push_back(std::make_shared<GridFieldEvaluations<T>>(
          this->m_maxwellWaveParticleCoupling->az_field,
          this->m_chargedParticles, eval_nx, eval_ny, "az_grid.h5part"));
      this->grid_field_evaluations.push_back(std::make_shared<GridFieldEvaluations<T>>(
          this->m_maxwellWaveParticleCoupling->rho_field,
          this->m_chargedParticles, eval_nx, eval_ny, "rho_grid.h5part"));
      this->grid_field_evaluations.push_back(std::make_shared<GridFieldEvaluations<T>>(
          this->m_maxwellWaveParticleCoupling->jx_field,
          this->m_chargedParticles, eval_nx, eval_ny, "jx_grid.h5part"));
      this->grid_field_evaluations.push_back(std::make_shared<GridFieldEvaluations<T>>(
          this->m_maxwellWaveParticleCoupling->jy_field,
          this->m_chargedParticles, eval_nx, eval_ny, "jy_grid.h5part"));
      this->grid_field_evaluations.push_back(std::make_shared<GridFieldEvaluations<T>>(
          this->m_maxwellWaveParticleCoupling->jz_field,
          this->m_chargedParticles, eval_nx, eval_ny, "jz_grid.h5part"));
      this->grid_field_evaluations.push_back(std::make_shared<GridFieldEvaluations<T>>(
          this->m_maxwellWaveParticleCoupling->ex_field,
          this->m_chargedParticles, eval_nx, eval_ny, "ex_grid.h5part"));
      this->grid_field_evaluations.push_back(std::make_shared<GridFieldEvaluations<T>>(
          this->m_maxwellWaveParticleCoupling->ey_field,
          this->m_chargedParticles, eval_nx, eval_ny, "ey_grid.h5part"));
      this->grid_field_evaluations.push_back(std::make_shared<GridFieldEvaluations<T>>(
          this->m_maxwellWaveParticleCoupling->ez_field,
          this->m_chargedParticles, eval_nx, eval_ny, "ez_grid.h5part"));
      this->grid_field_evaluations.push_back(std::make_shared<GridFieldEvaluations<T>>(
          this->m_maxwellWaveParticleCoupling->bx_field,
          this->m_chargedParticles, eval_nx, eval_ny, "bx_grid.h5part"));
      this->grid_field_evaluations.push_back(std::make_shared<GridFieldEvaluations<T>>(
          this->m_maxwellWaveParticleCoupling->by_field,
          this->m_chargedParticles, eval_nx, eval_ny, "by_grid.h5part"));
      this->grid_field_evaluations.push_back(std::make_shared<GridFieldEvaluations<T>>(
          this->m_maxwellWaveParticleCoupling->bz_field,
          this->m_chargedParticles, eval_nx, eval_ny, "bz_grid.h5part"));
    }
  };

  /**
   *  Run the simulation.
   */
  inline void run() {

    if (this->num_print_steps > 0) {
      if (this->rank == 0) {
        nprint(" Number of species ", this->m_chargedParticles->num_species);
      }
    }
    const auto timestep_original = this->m_chargedParticles->dt;
    NESOASSERT(timestep_original > 0, "The time step must be > 0");

    auto t0 = profile_timestamp();
    auto t0_benchmark = profile_timestamp();
    // MAIN LOOP START
    double initialBenergy = 0.0;

    const double dtMultiplier = 1.0;
    //if (this->m_do_deposition == 1) {
    //  this->m_chargedParticles->advect(-dtMultiplier / 2);
    //  this->m_chargedParticles->communicate();
    //  this->m_maxwellWaveParticleCoupling->deposit_charge();
    //  this->m_chargedParticles->advect(dtMultiplier / 2);
    //  this->m_chargedParticles->communicate();
    //}
    for (int stepx = 0; stepx < this->num_time_steps; stepx++) {
      this->time_step = stepx;
      // These 5 lines perform the simulation timestep.
      // integrate_fields first does divJ to get drho/dt
      this->m_maxwellWaveParticleCoupling->integrate_fields(this->theta,
                                                            dtMultiplier,
                                                            this->m_use_charge_conservation);
      this->m_chargedParticles->accelerate(dtMultiplier);
      this->m_chargedParticles->advect(dtMultiplier / 2);
      this->m_chargedParticles->communicate();
      if (this->m_do_deposition == 1) {
        this->m_maxwellWaveParticleCoupling->deposit_current();
      }
      this->m_chargedParticles->advect(dtMultiplier / 2);
      this->m_chargedParticles->communicate();
      if (this->m_use_charge_conservation == 0) {
        this->m_maxwellWaveParticleCoupling->deposit_charge();
      }

      if (stepx == 99) {
        t0_benchmark = profile_timestamp();
      }

      // Below this line are the diagnostic calls for the timestep.
      bool cond = (stepx == 0) || ((this->num_write_particle_steps > 0) &&
        ((stepx % this->num_write_particle_steps) == 0));
      if (cond) {
        this->m_chargedParticles->write();
      }

      cond = (stepx == 0) || ((this->num_write_field_steps > 0) &&
        ((stepx % this->num_write_field_steps) == 0));

      if (cond) {
          this->m_maxwellWaveParticleCoupling->write_sources(stepx);
          this->m_maxwellWaveParticleCoupling->write_potentials(stepx);
          this->m_maxwellWaveParticleCoupling->write_fields(stepx);
      }

      cond = (stepx == 0) || ((this->num_write_field_energy_steps > 0) &&
        ((stepx % this->num_write_field_energy_steps) == 0));

      if (cond) {
        double bx_energy = this->m_fieldEnergy->compute(
          this->m_maxwellWaveParticleCoupling->bx_field);
        double by_energy = this->m_fieldEnergy->compute(
          this->m_maxwellWaveParticleCoupling->by_field);
        double bz_energy = this->m_fieldEnergy->compute(
          this->m_maxwellWaveParticleCoupling->bz_field);
        double ex_energy = this->m_fieldEnergy->compute(
          this->m_maxwellWaveParticleCoupling->ex_field);
        double ey_energy = this->m_fieldEnergy->compute(
          this->m_maxwellWaveParticleCoupling->ey_field);
        double ez_energy = this->m_fieldEnergy->compute(
          this->m_maxwellWaveParticleCoupling->ez_field);
        for (auto ke : this->kinetic_energies) { ke->compute(); }
        for (auto pe : this->potential_energies) { pe->compute(); }

        if (this->global_hdf5_write) {

          this->generic_hdf5_writer->step_start(stepx);
          this->generic_hdf5_writer->write_value_step(
              "field_energy_bx", bx_energy);
          this->generic_hdf5_writer->write_value_step(
              "field_energy_by", by_energy);
          this->generic_hdf5_writer->write_value_step(
              "field_energy_bz", bz_energy);
          this->generic_hdf5_writer->write_value_step(
              "field_energy_ex", ex_energy);
          this->generic_hdf5_writer->write_value_step(
              "field_energy_ey", ey_energy);
          this->generic_hdf5_writer->write_value_step(
              "field_energy_ez", ez_energy);
          uint32_t counter = 0;
          for (auto ke : this->kinetic_energies) {
            this->generic_hdf5_writer->write_value_step(
                "kinetic_energy_" + std::to_string(counter), ke->energy);
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
        if (this->rank == 0) {
          if ((stepx % this->num_print_steps) == 0) {
            double ke = 0.0; // total
            for (auto i : this->kinetic_energies) {
              ke += i->energy;
            }
            double pe = 0.0; // total
            for (auto i : this->potential_energies) {
              pe += i->energy;
            }
            double be = bx_energy + by_energy + bz_energy;
            double ee = ex_energy + ey_energy + ez_energy;
            const double te = ke + be + ee;
            nprint("step:", stepx,
                   "sim time:", stepx * timestep_original * dtMultiplier,
                   "t/step:", profile_elapsed(t0, profile_timestamp()) / (stepx + 1),
                   "pe:", pe, "ke:", ke, "ee:", ee,
                   "be:", be - initialBenergy, "te:", te - initialBenergy);
          } else {
            nprint("step:", stepx,
                   profile_elapsed(t0, profile_timestamp()) / (stepx + 1));
          }
        }
      }

      cond = (this->grid_field_evaluations_flag) &&
        ((stepx % this->grid_field_evaluations_step) == 0);
      if (cond) {
        for (auto gfe : this->grid_field_evaluations) {
          gfe->write(stepx);
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

    if (this->grid_field_evaluations_flag) {
      for (auto gfe : this->grid_field_evaluations) {
        gfe->close();
      }
    }

    this->m_chargedParticles->free();

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
  push_callback(std::function<void(MaxwellWave2D3VDriver<T> *)> &func) {
    this->callbacks.emplace_back(func);
  }
};

#endif
