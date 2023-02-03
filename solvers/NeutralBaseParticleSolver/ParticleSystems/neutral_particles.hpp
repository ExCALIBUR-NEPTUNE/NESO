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

using namespace Nektar;
using namespace NESO;
using namespace NESO::Particles;

class NeutralParticleSystem {
private:
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  MPI_Comm comm;
  const double tol;
  const int ndim = 2;
  bool h5part_exists;

  /**
   * Helper function to get values from the session file.
   *
   * @param session Session object.
   * @param name Name of the parameter.
   * @param output Reference to the output variable.
   * @param default Default value if name not found in the session file.
   */
  template <typename T>
  inline void get_from_session(LibUtilities::SessionReaderSharedPtr session,
                               std::string name, T &output, T default_value) {
    if (session->DefinesParameter(name)) {
      session->LoadParameter(name, output);
    } else {
      output = default_value;
    }
  }

  inline void add_particles() {

    long rstart, rend;
    const long size = this->sycl_target->comm_pair.size_parent;
    const long rank = this->sycl_target->comm_pair.rank_parent;

    get_decomp_1d(size, (long)this->num_particles, rank, &rstart, &rend);
    const long N = rend - rstart;
    const int cell_count = this->domain->mesh->get_cell_count();

    // get seed from file
    std::srand(std::time(nullptr));
    int seed;
    get_from_session(this->session, "particle_position_seed", seed,
                     std::rand());

    std::mt19937 rng_phasespace(seed + rank);

    NESOASSERT(session->DefinesParameter("particle_distribution_position"),
               "Position distribution configuration not found in session.");
    NESOASSERT(session->DefinesParameter("particle_distribution_velocity"),
               "Velocity distribution configuration not found in session.");

    int distribution_position;
    get_from_session(this->session, "particle_distribution_position",
                     distribution_position, -1);
    NESOASSERT(distribution_position > -1,
               "Bad particle position distribution key.");
    NESOASSERT(distribution_position < 1,
               "Bad particle position distribution key.");

    int distribution_velocity;
    get_from_session(this->session, "particle_distribution_velocity",
                     distribution_position, -1);
    session->LoadParameter("particle_distribution_velocity",
                           distribution_position);
    NESOASSERT(distribution_position > -1,
               "Bad particle velocity distribution key.");
    NESOASSERT(distribution_position < 1,
               "Bad particle velocity distribution key.");

    if (N > 0) {
      ParticleSet initial_distribution(
          N, this->particle_group->get_particle_spec());

      // Get the requested particle distribution type from the config file
      int particle_distribution_type = 0;
      get_from_session(session, "particle_distribution_type",
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
        //} else if (particle_distribution_type == 4) {
        //  positions = rsequence_within_extents(
        //      N, ndim, this->boundary_conditions->global_extent);
      } else {
        positions = uniform_within_extents(
            N, ndim, this->boundary_conditions->global_extent, rng_phasespace);
      }

      if (distribution_position == 0) {
        // TODO
        for (int px = 0; px < N; px++) {
          initial_distribution[Sym<REAL>("P")][px][0] = 0.0;
          initial_distribution[Sym<REAL>("P")][px][1] = 0.0;
        }
      }

      if (distribution_velocity == 0) {
        double thermal_velocity;
        NESOASSERT(this->session->DefinesParameter("particle_thermal_velocity"),
                   "particle_thermal_velocity not found in config");
        session->LoadParameter("particle_thermal_velocity", thermal_velocity);

        for (int px = 0; px < N; px++) {
          // Maybe we want some drift velocities?
          // TODO Set units of thermal_velocity?
          std::normal_distribution<> velocity_normal_distribution{0, thermal_velocity};

          const double vx = velocity_normal_distribution(rng_phasespace);
          const double vy = velocity_normal_distribution(rng_phasespace);
          const double vz = velocity_normal_distribution(rng_phasespace);

          initial_distribution[Sym<REAL>("V")][px][0] = vx;
          initial_distribution[Sym<REAL>("V")][px][1] = vy;
          initial_distribution[Sym<REAL>("V")][px][2] = vz;
        }
      }

      // Initialise the remaining particle properties
      for (int px = 0; px < N; px++) {
        initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
        initial_distribution[Sym<INT>("PARTICLE_ID")][px][0] = rank;
        initial_distribution[Sym<INT>("PARTICLE_ID")][px][1] = px;
        initial_distribution[Sym<REAL>("M")][px][0] = this->particle_mass;
      }
      this->particle_group->add_particles_local(initial_distribution);
    }

    NESO::Particles::parallel_advection_initialisation(this->particle_group);
    // Move particles to the owning ranks and correct cells.
    this->transfer_particles();
  }

public:
  /// Disable (implicit) copies.
  NeutralParticleSystem(const NeutralParticleSystem &st) = delete;
  /// Disable (implicit) copies.
  NeutralParticleSystem &operator=(NeutralParticleSystem const &a) = delete;

  /// Global number of particles in the simulation.
  int64_t num_particles;
  /// Average number of particles per cell (element) in the simulation.
  int64_t num_particles_per_cell;
  /// Time step size used for particles
  double dt;
  /// Mass of particles
  const double particle_mass = 1.0;
  /// Initial particle weight.
  double particle_weight;
  /// Number density in simulation domain (per specicies)
  double particle_number_density;
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
   *  Create a new instance.
   *
   *  @param session Nektar++ session to use for parameters and simulation
   * specification.
   *  @param graph Nektar++ MeshGraph on which particles exist.
   *  @param comm (optional) MPI communicator to use - default MPI_COMM_WORLD.
   *
   */
  NeutralParticleSystem(LibUtilities::SessionReaderSharedPtr session,
                        SpatialDomains::MeshGraphSharedPtr graph,
                        MPI_Comm comm = MPI_COMM_WORLD)
      : session(session), graph(graph), comm(comm), tol(1.0e-8),
        h5part_exists(false) {

    // Read the number of requested particles per cell.
    int tmp_int;
    this->session->LoadParameter("num_particles_per_cell", tmp_int);
    this->num_particles_per_cell = tmp_int;

    this->session->LoadParameter("particle_time_step", this->dt);

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
    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), 2, true),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<INT>("PARTICLE_ID"), 2),
                               ParticleProp(Sym<REAL>("M"), 1),
                               ParticleProp(Sym<REAL>("V"), 3)};

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
    this->session->LoadParameter("particle_number_density",
                                 this->particle_number_density);
    if (this->particle_number_density < 0.0) {
      this->particle_weight = 1.0;
      this->particle_number_density = this->num_particles / volume;
    } else {
      const double number_physical_particles =
          this->particle_number_density * volume;
      this->particle_weight = number_physical_particles / this->num_particles;
    }

    // Add particle to the particle group
    this->add_particles();
  };

  /**
   *  Write current particle state to trajectory.
   */
  inline void write() {
    if (!this->h5part_exists) {
      // Create instance to write particle data to h5 file
      this->h5part = std::make_shared<H5Part>(
          "Electrostatic2D3V_particle_trajectory.h5part", this->particle_group,
          Sym<REAL>("P"), Sym<INT>("CELL_ID"), Sym<REAL>("V"),
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
        "NeutralParticleSystem", "transfer_particles", 1,
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
   * Apply Forward-Euler, which with no forces is trivial.
   */
  inline void forward_euler() {

    const double k_dt = this->dt;

    auto t0 = profile_timestamp();

    auto k_P = (*this->particle_group)[Sym<REAL>("P")]->cell_dat.device_ptr();
    auto k_V = (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    sycl_target->profile_map.inc("NeutralParticleSystem",
                                 "ForwardEuler_Prepare", 1,
                                 profile_elapsed(t0, profile_timestamp()));

    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                k_P[cellx][0][layerx] += k_dt * k_V[cellx][0][layerx];
                k_P[cellx][1][layerx] += k_dt * k_V[cellx][1][layerx];

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target->profile_map.inc("NeutralParticleSystem",
                                 "ForwardEuler_Execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));

    // positions were written so we apply boundary conditions and move
    // particles between ranks
    this->transfer_particles();
  }
};

#endif
