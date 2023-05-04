#ifndef __CHARGED_PARTICLES_H_
#define __CHARGED_PARTICLES_H_

#include <nektar_interface/function_evaluation.hpp>
#include <nektar_interface/function_projection.hpp>
#include <nektar_interface/particle_interface.hpp>
#include <neso_particles.hpp>

#include <particle_utility/position_distribution.hpp>

#include <LibUtilities/BasicUtils/SessionReader.h>

#include <boost/math/special_functions/erf.hpp>

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <random>

#include "boris_integrator.hpp"
#include "parallel_initialisation.hpp"
#include "particle_initial_condition.hpp"

using namespace Nektar;
using namespace NESO;
using namespace NESO::Particles;

#ifndef PIC_2D3V_CROSS_PRODUCT_3D
#define PIC_2D3V_CROSS_PRODUCT_3D(a1, a2, a3, b1, b2, b3, c1, c2, c3)     \
  (c1) = ((a2) * (b3)) - ((a3) * (b2));                                        \
  (c2) = ((a3) * (b1)) - ((a1) * (b3));                                        \
  (c3) = ((a1) * (b2)) - ((a2) * (b1));
#endif

/**
 * Helper function to get values from the session file.
 *
 * @param session Session object.
 * @param name Name of the parameter.
 * @param output Reference to the output variable.
 * @param default Default value if name not found in the session file.
 */
template <typename T>
inline void
pic2d3v_get_from_session(LibUtilities::SessionReaderSharedPtr session,
                          std::string name, T &output, T default_value) {
  if (session->DefinesParameter(name)) {
    session->LoadParameter(name, output);
  } else {
    output = default_value;
  }
}
// make a base class that handles the domain, particle bcs
// handle graphmap, sycl target, particle mesh interface
// nektar graph local mapper
// cell id are per group
// boundary conditions are per group
class ChargedParticles {
private:
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  MPI_Comm comm;
  const double tol;
  const int ndim = 2;
  double charge_density;
  bool h5part_exists;

//  REAL B0_offset;
//  REAL B1_offset;
//  REAL B2_offset;
  REAL particle_E_coefficient;

  std::shared_ptr<IntegratorBoris> integrator_boris;

  inline void add_particles() {//const std::vector<ParticleInitialConditions> & particle_ics) {

    long rstart, rend;
    const long size = this->sycl_target->comm_pair.size_parent;
    const long rank = this->sycl_target->comm_pair.rank_parent;

    get_decomp_1d(size, (long)this->num_particles, rank, &rstart, &rend);
    const long N = rend - rstart;
    const int cell_count = this->domain->mesh->get_cell_count();

    // get seed from file
    std::srand(std::time(nullptr));
    int seed;
    pic2d3v_get_from_session(this->session, "particle_position_seed", seed,
                              std::rand());

    std::mt19937 rng_phasespace(seed + rank);
    std::bernoulli_distribution coin_toss(0.5);

    std::uniform_real_distribution<double> uniform01(0, 1);

    int distribution_function = -1;
    session->LoadParameter("particle_distribution_function",
                           distribution_function);
    NESOASSERT(distribution_function > -1, "Bad particle distribution key.");
    NESOASSERT(distribution_function < 6, "Bad particle distribution key.");

    if (N > 0) {
      ParticleSet initial_distribution(
          N, this->particle_group->get_particle_spec());

      // Get the requested particle distribution type from the config file
      int particle_distribution_type = 0;
      pic2d3v_get_from_session(session, "particle_distribution_type",
                                particle_distribution_type, 0);

      NESOASSERT(particle_distribution_type >= 0,
                 "Bad particle distribution type.");
      NESOASSERT(particle_distribution_type <= 1,
                 "Bad particle distribution type.");

      std::vector<std::vector<double>> positions;

      // create the requested particle position distribution
      if (particle_distribution_type == 0) {
        positions = sobol_within_extents(
            N, ndim, this->boundary_conditions->global_extent, rstart,
            (unsigned int)seed);
      } else {
        positions = uniform_within_extents(
            N, ndim, this->boundary_conditions->global_extent, rng_phasespace);
      }

      if (distribution_function == 0) {
        double initial_velocity;
        session->LoadParameter("particle_initial_velocity", initial_velocity);
        // square in lower left

        for (int px = 0; px < N; px++) {
          for (int dimx = 0; dimx < ndim; dimx++) {
            const double pos_orig =
                positions[dimx][px] +
                this->boundary_conditions->global_origin[dimx];
            initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig * 0.25;
          }

          initial_distribution[Sym<REAL>("V")][px][0] = initial_velocity;
          initial_distribution[Sym<REAL>("V")][px][1] = 0.0;
          initial_distribution[Sym<REAL>("Q")][px][0] = this->particle_charge;
          initial_distribution[Sym<REAL>("M")][px][0] = this->particle_mass;
        }
      } else if (distribution_function == 1) {
        double initial_velocity;
        session->LoadParameter("particle_initial_velocity", initial_velocity);
        // two stream - as two streams....

        for (int px = 0; px < N; px++) {

          // x position
          const double pos_orig_0 =
              positions[0][px] + this->boundary_conditions->global_origin[0];
          initial_distribution[Sym<REAL>("P")][px][0] = pos_orig_0;

          const bool species = coin_toss(rng_phasespace);

          // y position
          double pos_orig_1 = (species) ? 0.25 : 0.75;
          pos_orig_1 =
              this->boundary_conditions->global_extent[1] * pos_orig_1 +
              this->boundary_conditions->global_origin[1];

          // add some uniform random variation
          const double shift_1 =
              0.01 * (positions[1][px] /
                      this->boundary_conditions->global_extent[1]) -
              0.005;

          initial_distribution[Sym<REAL>("P")][px][1] = pos_orig_1 + shift_1;

          initial_distribution[Sym<REAL>("V")][px][0] =
              (species) ? initial_velocity : -initial_velocity;
          ;
          initial_distribution[Sym<REAL>("V")][px][1] = 0.0;
          initial_distribution[Sym<REAL>("Q")][px][0] = this->particle_charge;
          initial_distribution[Sym<REAL>("M")][px][0] = this->particle_mass;
        }
      } else if (distribution_function == 2) {
        double initial_velocity;
        session->LoadParameter("particle_initial_velocity", initial_velocity);
        // two stream - as standard two stream
        for (int px = 0; px < N; px++) {

          const bool species = coin_toss(rng_phasespace);
          // x position
          const double pos_orig_0 =
              positions[0][px] + this->boundary_conditions->global_origin[0];
          initial_distribution[Sym<REAL>("P")][px][0] = pos_orig_0;

          // y position
          const double pos_orig_1 =
              positions[1][px] + this->boundary_conditions->global_origin[1];
          initial_distribution[Sym<REAL>("P")][px][1] = pos_orig_1;

          initial_distribution[Sym<REAL>("V")][px][0] =
              (species) ? initial_velocity : -1.0 * initial_velocity;
          // initial_distribution[Sym<REAL>("V")][px][1] =
          //     (species) ? initial_velocity : -1.0 * initial_velocity;
          initial_distribution[Sym<REAL>("V")][px][1] = 0.0;
          initial_distribution[Sym<REAL>("Q")][px][0] = this->particle_charge;
          initial_distribution[Sym<REAL>("M")][px][0] = this->particle_mass;
        }
      } else if (distribution_function == 3) {
        // 3V Maxwellian
        auto positions = uniform_within_extents(
            N, ndim, this->boundary_conditions->global_extent, rng_phasespace);

        double thermal_velocity;
        session->LoadParameter("particle_thermal_velocity", thermal_velocity);

        for (int px = 0; px < N; px++) {

          // x position
          const double pos_orig_0 =
              positions[0][px] + this->boundary_conditions->global_origin[0];
          initial_distribution[Sym<REAL>("P")][px][0] = pos_orig_0;

          // y position
          const double pos_orig_1 =
              positions[1][px] + this->boundary_conditions->global_origin[1];
          initial_distribution[Sym<REAL>("P")][px][1] = pos_orig_1;

          // vx, vy, vz thermally distributed velocities
          auto rvx = boost::math::erf_inv(2 * uniform01(rng_phasespace) - 1);
          auto rvy = boost::math::erf_inv(2 * uniform01(rng_phasespace) - 1);
          auto rvz = boost::math::erf_inv(2 * uniform01(rng_phasespace) - 1);

          const auto isthermal = uniform01(rng_phasespace) > 0.1;

          if (!isthermal) {
            const auto theta = 2 * boost::math::constants::pi<double>() *
                               uniform01(rng_phasespace);
            rvx = 0;
            rvy = 4 * thermal_velocity * std::sin(theta);
            rvz = 4 * thermal_velocity * std::cos(theta);
          }

          initial_distribution[Sym<REAL>("V")][px][0] = thermal_velocity * rvx;
          initial_distribution[Sym<REAL>("V")][px][1] = thermal_velocity * rvy;
          initial_distribution[Sym<REAL>("V")][px][2] = thermal_velocity * rvz;

          initial_distribution[Sym<REAL>("Q")][px][0] = this->particle_charge;
          initial_distribution[Sym<REAL>("M")][px][0] = this->particle_mass;
        }
      } else if (distribution_function == 4) {
        double initial_velocity;
        session->LoadParameter("particle_initial_velocity", initial_velocity);
        // two stream - as standard two stream
        for (int px = 0; px < N; px++) {
          // x position
          const double pos_orig_0 =
              positions[0][px] + this->boundary_conditions->global_origin[0];
          initial_distribution[Sym<REAL>("P")][px][0] = pos_orig_0;

          // y position
          const double pos_orig_1 =
              positions[1][px] + this->boundary_conditions->global_origin[1];
          initial_distribution[Sym<REAL>("P")][px][1] = pos_orig_1;

          initial_distribution[Sym<REAL>("V")][px][0] = initial_velocity;
          initial_distribution[Sym<REAL>("V")][px][1] =
              1.0 + uniform01(rng_phasespace);
          initial_distribution[Sym<REAL>("V")][px][2] = 0.0;
          initial_distribution[Sym<REAL>("Q")][px][0] = this->particle_charge;
          initial_distribution[Sym<REAL>("M")][px][0] = this->particle_mass;
        }
      }

      for (int px = 0; px < N; px++) {
        initial_distribution[Sym<REAL>("E")][px][0] = 0.0;
        initial_distribution[Sym<REAL>("E")][px][1] = 0.0;
        initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
        initial_distribution[Sym<INT>("PARTICLE_ID")][px][0] = px + rstart;
      }
      this->particle_group->add_particles_local(initial_distribution);
    }

    NESO::parallel_advection_initialisation(this->particle_group);
    NESO::parallel_advection_store(this->particle_group);

    // auto h5part_local = std::make_shared<H5Part>(
    //       "foo.h5part", this->particle_group,
    //       Sym<REAL>("P"), Sym<REAL>("ORIG_POS"), Sym<INT>("NESO_MPI_RANK"),
    //       Sym<INT>("PARTICLE_ID"), Sym<REAL>("NESO_REFERENCE_POSITIONS"));
    const int num_steps = 20;
    for (int stepx = 0; stepx < num_steps; stepx++) {
      NESO::parallel_advection_step(this->particle_group, num_steps, stepx);
      this->transfer_particles();
      // h5part_local->write();
    }
    NESO::parallel_advection_restore(this->particle_group);
    // h5part_local->write();
    // h5part_local->close();

    // Move particles to the owning ranks and correct cells.
    this->transfer_particles();
  }

public:
  /// Disable (implicit) copies.
  ChargedParticles(const ChargedParticles &st) = delete;
  /// Disable (implicit) copies.
  ChargedParticles &operator=(ChargedParticles const &a) = delete;

  /// Global number of particles in the simulation.
  int64_t num_particles;
//  /// Global number of particles in the simulation for each species
//  std::vector<int64_t> num_particles_per_species;
  /// Average number of particles per cell (element) in the simulation.
  int64_t num_particles_per_cell;
  /// Time step size used for particles
  double dt;
  /// Number of species
  int nspecies;
  /// Mass of particles
  const double particle_mass = 1.0;
  /// Charge of particles
  double particle_charge = 1.0;
  /// Scaling coefficient for RHS
  double particle_weight;
  /// Number density in simulation domain (per species)
  double number_density;
  /// HMesh instance that allows particles to move over nektar++ meshes.
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  /// Compute target.
  SYCLTargetSharedPtr sycl_target;
  /// Mapping instance to map particles into nektar++ elements.
  std::shared_ptr<NektarGraphLocalMapperT> nektar_graph_local_mapper;
  /// NESO-Particles domain.
  DomainSharedPtr domain;
  /// NESO-Particles ParticleGroup containing charged particles.
  ParticleGroupSharedPtr particle_group;
  /// Method to apply particle boundary conditions.
  std::shared_ptr<NektarCartesianPeriodic> boundary_conditions;
  /// Method to map to/from nektar geometry ids to 0,N-1 used by NESO-Particles
  std::shared_ptr<CellIDTranslation> cell_id_translation;
  /// Trajectory writer for particles.
  std::shared_ptr<H5Part> h5part;

  /**
   *  Set a scaling coefficient x such that the effect of the electric field is
   *  xqE instead of qE.
   *
   *  @param x New scaling coefficient.
   */
  inline void set_E_coefficent(const REAL x) {
    this->particle_E_coefficient = x;
    this->integrator_boris->set_E_coefficent(x);
  }

//  /**
//   *  Set the constant and uniform magnetic field over the entire domain.
//   *
//   *  @param Bx Magnetic field B in x direction.
//   *  @param By Magnetic field B in y direction.
//   *  @param Bz Magnetic field B in z direction.
//   */
//  inline void set_B_field(const REAL Bx = 0.0, const REAL By = 0.0,
//                          const REAL Bz = 0.0) {
//    this->B0_offset = Bx;
//    this->B1_offset = By;
//    this->B2_offset = Bz;
//    this->integrator_boris->set_B_field(B0, B1, B2);
//  }

  /**
   *  Create a new instance.
   *
   *  @param session Nektar++ session to use for parameters and simulation
   * specification.
   *  @param graph Nektar++ MeshGraph on which particles exist.
   *  @param comm (optional) MPI communicator to use - default MPI_COMM_WORLD.
   *
   */
  ChargedParticles(LibUtilities::SessionReaderSharedPtr session,
                   SpatialDomains::MeshGraphSharedPtr graph,
                   MPI_Comm comm = MPI_COMM_WORLD)
                   //const std::vector<ParticleInitialConditions> & particle_ics)
      : session(session), graph(graph), comm(comm), tol(1.0e-8),
        h5part_exists(false) {

    this->particle_E_coefficient = 1.0;

    // Read the number of requested particles per cell.
    int tmp_int;
    this->session->LoadParameter("num_particles_per_cell", tmp_int);
    this->num_particles_per_cell = tmp_int;

    this->session->LoadParameter("particle_time_step", this->dt);


    this->session->LoadParameter("nspecies",
                                 this->nspecies);
    //for (std::size_t i=0; i < this->nspecies; ++i) {
    //    double number_density;
    //    this->session->LoadParameter("number_density_" + std::to_string(i),
    //                                 number_density);
    //    this->initial_number_densities.emplace_back(number_density);
    //    double temperature_ev;
    //    this->session->LoadParameter("temperature_ev_" + std::to_string(i),
    //                                 temperature_ev);
    //    this->initial_temperatures_ev.emplace_back(temperature_ev);
    //    double drift_ev;
    //    this->session->LoadParameter("drift_ev_" + std::to_string(i),
    //                                 drift_ev);
    //    this->initial_drifts_ev.emplace_back(drift_ev);
    //    double pitch;
    //    this->session->LoadParameter("pitch_" + std::to_string(i),
    //                                 pitch);
    //    this->initial_pitches.emplace_back(pitch);
    //}

    // Reduce the global number of elements
    const int num_elements_local = this->graph->GetNumElements();
    int num_elements_global;
    MPICHK(MPI_Allreduce(&num_elements_local, &num_elements_global, 1, MPI_INT,
                         MPI_SUM, this->comm));

    // compute the global number of particles
    this->num_particles =
        ((int64_t)num_elements_global) * this->num_particles_per_cell;

    this->session->LoadParameter("num_particles_total", tmp_int);
    if (tmp_int > -1) {
      this->num_particles = tmp_int;
    }

    //int total_count = 0;
    //for (int i=0; i < this->nspecies - 1; ++i) {
    //    this->num_particles_per_species.emplace_back(this->num_particles / this->nspecies);
    //    total_count += this->num_particles_species[i];
    //}
    //this->num_particles_species.emplace_back(this->num_particles - total_count);

    // Create interface between particles and nektar++
    this->particle_mesh_interface =
        std::make_shared<ParticleMeshInterface>(graph, 0, this->comm);
    this->sycl_target =
        std::make_shared<SYCLTarget>(0, particle_mesh_interface->get_comm());
    this->nektar_graph_local_mapper = std::make_shared<NektarGraphLocalMapperT>(
        this->sycl_target, this->particle_mesh_interface, this->tol);
    this->domain = std::make_shared<Domain>(this->particle_mesh_interface,
                                            this->nektar_graph_local_mapper);

    // Create ParticleGroup
    ParticleSpec particle_spec{
      ParticleProp(Sym<REAL>("P"), 2, true), // poition
      ParticleProp(Sym<INT>("CELL_ID"), 1, true),
      ParticleProp(Sym<INT>("PARTICLE_ID"), 1),
      ParticleProp(Sym<REAL>("Q"), 1), // charge
      ParticleProp(Sym<REAL>("M"), 1), // mass
      ParticleProp(Sym<REAL>("W"), 1), // weight
      ParticleProp(Sym<REAL>("V"), 3), // velocity
      ParticleProp(Sym<REAL>("B"), 3), // B field
      ParticleProp(Sym<REAL>("E"), 3), // E field
      ParticleProp(Sym<REAL>("WQ"), 1), // weight * charge
      ParticleProp(Sym<REAL>("WQV"), 3) // weight * charge * velocity
    };

    this->particle_group = std::make_shared<ParticleGroup>(
        this->domain, particle_spec, this->sycl_target);

    // Setup PBC boundary conditions.
    this->boundary_conditions = std::make_shared<NektarCartesianPeriodic>(
        this->sycl_target, this->graph, this->particle_group->position_dat);

    // Setup map between cell indices
    this->cell_id_translation = std::make_shared<CellIDTranslation>(
        this->sycl_target, this->particle_group->cell_id_dat,
        this->particle_mesh_interface);

    const double volume = this->boundary_conditions->global_extent[0] *
                          this->boundary_conditions->global_extent[1];

    // read or deduce a number density from the configuration file
    this->session->LoadParameter("number_density",
                                 this->number_density);
//    for (int i=0; i<this->nspecies; ++i) {
//      const auto number_physical_particles =
//          this->initial_number_densities[i] * volume;
//      this->particle_weights.emplace_back(
//          number_physical_particles / this->num_particles_per_species[i]);
//    }

    //this->charge_density = -this->number_density;
    //this->session->LoadParameter("fast_ion_charge", this->fast_ion_charge);

    // Add particle to the particle group
    this->add_particles(); //particle_ics);

    // create a Boris integrator
    this->integrator_boris = std::make_shared<IntegratorBoris>(
        this->particle_group, this->dt,
        this->particle_E_coefficient);
  };

  /**
   *  Write current particle state to trajectory.
   */
  inline void write() {
    if (!this->h5part_exists) {
      // Create instance to write particle data to h5 file
      this->h5part = std::make_shared<H5Part>(
          "MaxwellWave2D3V_particle_trajectory.h5part", this->particle_group,
          Sym<REAL>("P"), Sym<INT>("CELL_ID"), Sym<REAL>("V"), Sym<REAL>("E"),
          Sym<REAL>("Q"), Sym<INT>("M"), Sym<REAL>("B"),
          Sym<INT>("NESO_MPI_RANK"), Sym<INT>("PARTICLE_ID"),
          Sym<REAL>("NESO_REFERENCE_POSITIONS"));
      this->h5part_exists = true;
    }

    this->h5part->write();
  }

  /**
   *  Apply boundary conditions and transfer particles between MPI ranks.
   */
  inline void transfer_particles() {
    auto t0 = profile_timestamp();
    this->boundary_conditions->execute();
    this->particle_group->hybrid_move();
    this->cell_id_translation->execute();
    this->particle_group->cell_move();
    this->sycl_target->profile_map.inc(
        "ChargedParticles", "transfer_particles", 1,
        profile_elapsed(t0, profile_timestamp()));
  }

  /**
   *  Free the object before MPI_Finalize is called.
   */
  inline void free() {
    if (this->h5part_exists) {
      this->h5part->close();
    }
    this->particle_group->free();
    this->particle_mesh_interface->free();
    this->sycl_target->free();
  };

  /**
   * Boris
   */
  inline void accelerate() {
    this->integrator_boris->accelerate();
  }

  inline void advect(const double fraction_dt) {
    this->integrator_boris->advect(fraction_dt);
  }

  inline void communicate() {
    // positions were written so we apply boundary conditions and move
    // particles between ranks
    this->transfer_particles();
  }
  /**
   *  Get the Sym object for the ParticleDat holding particle charge multiplied by its weight
   *  required for projection to charge density rho
   */
  inline Sym<REAL> get_rho_sym() { return Sym<REAL>("WQ"); }

  /**
   *  Get the Sym object for the ParticleDat holding w * q * v = j
   */
  inline Sym<REAL> get_current_sym() { return Sym<REAL>("WQV"); }

  /**
   *  Get the Sym object for the ParticleDat to hold the potential gradient.
   */
  inline Sym<REAL> get_potential_gradient_sym() { return Sym<REAL>("E"); }

  /**
   *  Get the charge density of the system.
   */
  inline double get_charge_density() { return this->charge_density; }
};

#endif
