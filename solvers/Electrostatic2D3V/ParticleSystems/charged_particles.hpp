#ifndef __CHARGED_PARTICLES_H_
#define __CHARGED_PARTICLES_H_

#include <nektar_interface/function_evaluation.hpp>
#include <nektar_interface/function_projection.hpp>
#include <nektar_interface/particle_interface.hpp>
#include <neso_particles.hpp>

#include <LibUtilities/BasicUtils/SessionReader.h>

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <random>

using namespace Nektar;
using namespace NESO::Particles;

class ChargedParticles {
private:
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  MPI_Comm comm;
  const double tol;
  const int ndim = 2;
  double charge_density;

  inline void add_particles() {

    long rstart, rend;
    const long size = this->sycl_target->comm_pair.size_parent;
    const long rank = this->sycl_target->comm_pair.rank_parent;

    get_decomp_1d(size, (long)this->num_particles, rank, &rstart, &rend);
    const long N = rend - rstart;
    const int cell_count = this->domain->mesh->get_cell_count();

    std::srand(std::time(nullptr));
    std::mt19937 rng_pos(std::rand() + rank);

    if (N > 0) {
      ParticleSet initial_distribution(
          N, this->particle_group->get_particle_spec());

      auto positions = uniform_within_extents(
          N, ndim, this->boundary_conditions->global_extent, rng_pos);

      for (int px = 0; px < N; px++) {
        for (int dimx = 0; dimx < ndim; dimx++) {
          const double pos_orig =
              positions[dimx][px] +
              this->boundary_conditions->global_origin[dimx];
          initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig * 0.25;
          initial_distribution[Sym<REAL>("E")][px][dimx] = 0.0;
        }

        initial_distribution[Sym<REAL>("V")][px][0] = 0.1;
        initial_distribution[Sym<REAL>("V")][px][1] = 0.0;
        initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
        initial_distribution[Sym<REAL>("Q")][px][0] = this->particle_charge;
      }

      this->particle_group->add_particles_local(initial_distribution);
    }

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
  /// Average number of particles per cell (element) in the simulation.
  int64_t num_particles_per_cell;
  /// Time step size used for particles
  double dt;
  /// Mass of particles
  const double particle_mass = 1.0;
  /// Charge of particles
  double particle_charge = 1.0;
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
   *  Create a new instance. TODO
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
      : session(session), graph(graph), comm(comm), tol(1.0e-8) {

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
                               ParticleProp(Sym<REAL>("Q"), 1),
                               ParticleProp(Sym<REAL>("V"), 3),
                               ParticleProp(Sym<REAL>("E"), 2)};

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

    // create a charge density of 1.0
    this->particle_charge = 8.0 * volume / this->num_particles;
    this->charge_density = this->particle_charge * this->num_particles / volume;

    // Add particle to the particle group
    this->add_particles();

    // Create instance to write particle data to h5 file
    this->h5part = std::make_shared<H5Part>(
        "Electrostatic2D3V.h5part", this->particle_group, Sym<REAL>("P"),
        Sym<INT>("CELL_ID"), Sym<REAL>("V"), Sym<REAL>("E"),
        Sym<INT>("NESO_MPI_RANK"), Sym<REAL>("NESO_REFERENCE_POSITIONS"));
  };

  /**
   *  Write current particle state to trajectory.
   */
  inline void write() { this->h5part->write(); }

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
    this->h5part->close();
    this->particle_group->free();
    this->particle_mesh_interface->free();
    this->sycl_target->free();
  };

  /**
   * Velocity Verlet - First step.
   */
  inline void velocity_verlet_1() {

    const double k_dt = this->dt;
    const double k_dht = this->dt * 0.5;
    const double k_dht_inverse_particle_mass = k_dht / this->particle_mass;

    auto t0 = profile_timestamp();

    auto k_P = (*this->particle_group)[Sym<REAL>("P")]->cell_dat.device_ptr();
    auto k_V = (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();
    const auto k_E =
        (*this->particle_group)[Sym<REAL>("E")]->cell_dat.device_ptr();
    const auto k_Q =
        (*this->particle_group)[Sym<REAL>("Q")]->cell_dat.device_ptr();

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    sycl_target->profile_map.inc("ChargedParticles", "VelocityVerlet_1_Prepare",
                                 1, profile_elapsed(t0, profile_timestamp()));
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const double Q = k_Q[cellx][0][layerx];

                k_V[cellx][0][layerx] -=
                    k_E[cellx][0][layerx] * k_dht_inverse_particle_mass * Q;
                k_V[cellx][1][layerx] -=
                    k_E[cellx][1][layerx] * k_dht_inverse_particle_mass * Q;

                k_P[cellx][0][layerx] += k_dt * k_V[cellx][0][layerx];
                k_P[cellx][1][layerx] += k_dt * k_V[cellx][1][layerx];

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target->profile_map.inc("ChargedParticles", "VelocityVerlet_1_Execute",
                                 1, profile_elapsed(t0, profile_timestamp()));

    // positions were written so we apply boundary conditions and move
    // particles between ranks
    this->transfer_particles();
  }

  /**
   * Velocity Verlet - Second step.
   */
  inline void velocity_verlet_2() {
    const double k_dht = this->dt * 0.5;
    const double k_dht_inverse_particle_mass = k_dht / this->particle_mass;

    auto t0 = profile_timestamp();

    auto k_V = (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();
    const auto k_E =
        (*this->particle_group)[Sym<REAL>("E")]->cell_dat.device_ptr();
    const auto k_Q =
        (*this->particle_group)[Sym<REAL>("Q")]->cell_dat.device_ptr();

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    sycl_target->profile_map.inc("ChargedParticles", "VelocityVerlet_2_Prepare",
                                 1, profile_elapsed(t0, profile_timestamp()));
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const double Q = k_Q[cellx][0][layerx];

                k_V[cellx][0][layerx] -=
                    k_E[cellx][0][layerx] * k_dht_inverse_particle_mass * Q;
                k_V[cellx][1][layerx] -=
                    k_E[cellx][1][layerx] * k_dht_inverse_particle_mass * Q;

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target->profile_map.inc("ChargedParticles", "VelocityVerlet_2_Execute",
                                 1, profile_elapsed(t0, profile_timestamp()));
  }

  /**
   *  Get the Sym object for the ParticleDat holding particle charge.
   */
  inline Sym<REAL> get_charge_sym() { return Sym<REAL>("Q"); }

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
