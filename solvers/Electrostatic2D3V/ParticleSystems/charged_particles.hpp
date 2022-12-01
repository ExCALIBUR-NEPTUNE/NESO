#ifndef __CHARGED_PARTICLES_H_
#define __CHARGED_PARTICLES_H_

#include <nektar_interface/function_evaluation.hpp>
#include <nektar_interface/function_projection.hpp>
#include <nektar_interface/particle_interface.hpp>
#include <neso_particles.hpp>

#include <LibUtilities/BasicUtils/SessionReader.h>

#include <boost/math/special_functions/erf.hpp>

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <random>

using namespace Nektar;
using namespace NESO::Particles;

#ifndef ELEC_PIC_2D3V_CROSS_PRODUCT_3D
#define ELEC_PIC_2D3V_CROSS_PRODUCT_3D(a1, a2, a3, b1, b2, b3, c1, c2, c3)     \
  (c1) = ((a2) * (b3)) - ((a3) * (b2));                                        \
  (c2) = ((a3) * (b1)) - ((a1) * (b3));                                        \
  (c3) = ((a1) * (b2)) - ((a2) * (b1));
#endif

class ChargedParticles {
private:
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  MPI_Comm comm;
  const double tol;
  const int ndim = 2;
  double charge_density;
  bool h5part_exists;

  REAL B_0;
  REAL B_1;
  REAL B_2;
  REAL particle_E_coefficient;

  inline void add_particles() {

    long rstart, rend;
    const long size = this->sycl_target->comm_pair.size_parent;
    const long rank = this->sycl_target->comm_pair.rank_parent;

    get_decomp_1d(size, (long)this->num_particles, rank, &rstart, &rend);
    const long N = rend - rstart;
    const int cell_count = this->domain->mesh->get_cell_count();

    std::srand(std::time(nullptr));
    std::mt19937 rng_pos(std::rand() + rank);
    std::mt19937 rng_vel(std::rand() + rank);
    std::uniform_real_distribution<double> uniform01(0, 1);

    int distribution_position = -1;
    session->LoadParameter("particle_distribution_position",
                           distribution_position);
    NESOASSERT(distribution_position > -1, "Bad particle distribution key.");
    NESOASSERT(distribution_position < 5, "Bad particle distribution key.");


    if (N > 0) {
      ParticleSet initial_distribution(
          N, this->particle_group->get_particle_spec());

      if (distribution_position == 0) {
        double initial_velocity;
        session->LoadParameter("particle_initial_velocity", initial_velocity);
        // square in lower left
        auto positions = uniform_within_extents(
            N, ndim, this->boundary_conditions->global_extent, rng_pos);
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
      } else if (distribution_position == 1) {
        double initial_velocity;
        session->LoadParameter("particle_initial_velocity", initial_velocity);
        // two stream - as two streams....
        auto positions = uniform_within_extents(
            N, ndim, this->boundary_conditions->global_extent, rng_pos);
        for (int px = 0; px < N; px++) {

          // x position
          const double pos_orig_0 =
              positions[0][px] + this->boundary_conditions->global_origin[0];
          initial_distribution[Sym<REAL>("P")][px][0] = pos_orig_0;

          // y position
          double pos_orig_1 = (px % 2 == 0) ? 0.25 : 0.75;
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
              (px % 2 == 0) ? initial_velocity : -1.0 * initial_velocity;
          ;
          initial_distribution[Sym<REAL>("V")][px][1] = 0.0;
          initial_distribution[Sym<REAL>("Q")][px][0] = this->particle_charge;
          initial_distribution[Sym<REAL>("M")][px][0] = this->particle_mass;
        }
      } else if (distribution_position == 2) {
        double initial_velocity;
        session->LoadParameter("particle_initial_velocity", initial_velocity);
        // two stream - as standard two stream
        auto positions = uniform_within_extents(
            N, ndim, this->boundary_conditions->global_extent, rng_pos);
        for (int px = 0; px < N; px++) {

          // x position
          const double pos_orig_0 =
              positions[0][px] + this->boundary_conditions->global_origin[0];
          initial_distribution[Sym<REAL>("P")][px][0] = pos_orig_0;

          // y position
          const double pos_orig_1 =
              positions[1][px] + this->boundary_conditions->global_origin[1];
          initial_distribution[Sym<REAL>("P")][px][1] = pos_orig_1;

          initial_distribution[Sym<REAL>("V")][px][0] =
              (px % 2 == 0) ? initial_velocity : -1.0 * initial_velocity;
          // initial_distribution[Sym<REAL>("V")][px][1] =
          //     (px % 2 == 0) ? initial_velocity : -1.0 * initial_velocity;
          initial_distribution[Sym<REAL>("V")][px][1] = 0.0;
          initial_distribution[Sym<REAL>("Q")][px][0] = this->particle_charge;
          initial_distribution[Sym<REAL>("M")][px][0] = this->particle_mass;
        }
      } else if (distribution_position == 3) {
        double initial_velocity;
        session->LoadParameter("particle_initial_velocity", initial_velocity);
        // two stream - as standard two stream
        auto positions = uniform_within_extents(
            N, ndim, this->boundary_conditions->global_extent, rng_pos);
        for (int px = 0; px < N; px++) {

          // x position
          const double pos_orig_0 =
              positions[0][px] + this->boundary_conditions->global_origin[0];
          initial_distribution[Sym<REAL>("P")][px][0] = pos_orig_0;

          // y position
          const double pos_orig_1 =
              positions[1][px] + this->boundary_conditions->global_origin[1];
          initial_distribution[Sym<REAL>("P")][px][1] = pos_orig_1;

          initial_distribution[Sym<REAL>("V")][px][0] =
              (px % 2 == 0) ? 0.0 : initial_velocity;
          ;
          initial_distribution[Sym<REAL>("V")][px][1] = 0.0;
          initial_distribution[Sym<REAL>("Q")][px][0] =
              (px % 2 == 0) ? this->particle_charge
                            : -1.0 * this->particle_charge;
          initial_distribution[Sym<REAL>("M")][px][0] =
              (px % 2 == 0) ? this->particle_mass * 1000000
                            : this->particle_mass;
        }
      } else if (distribution_position == 4) {
        // 3V Maxwellian
        auto positions = uniform_within_extents(
            N, ndim, this->boundary_conditions->global_extent, rng_pos);

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
          const auto rvx = boost::math::erf_inv(2 * uniform01(rng_vel) - 1);
          const auto rvy = boost::math::erf_inv(2 * uniform01(rng_vel) - 1);
          const auto rvz = boost::math::erf_inv(2 * uniform01(rng_vel) - 1);
          initial_distribution[Sym<REAL>("V")][px][0] = thermal_velocity * rvx;
          initial_distribution[Sym<REAL>("V")][px][1] = thermal_velocity * rvy;
          initial_distribution[Sym<REAL>("V")][px][2] = thermal_velocity * rvz;

          initial_distribution[Sym<REAL>("Q")][px][0] = this->particle_charge;
          initial_distribution[Sym<REAL>("M")][px][0] = this->particle_mass;
        }
      }


      for (int px = 0; px < N; px++) {
        initial_distribution[Sym<REAL>("E")][px][0] = 0.0;
        initial_distribution[Sym<REAL>("E")][px][1] = 0.0;
        initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
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
  /// Scaling coefficient for RHS of poisson equation.
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
   *  Set the constant and uniform magnetic field over the entire domain.
   *
   *  @param B0 Magnetic fiel B in x direction.
   *  @param B1 Magnetic fiel B in y direction.
   *  @param B2 Magnetic fiel B in z direction.
   */
  inline void set_B_field(const REAL B0 = 0.0, const REAL B1 = 0.0,
                          const REAL B2 = 0.0) {
    this->B_0 = B0;
    this->B_1 = B1;
    this->B_2 = B2;
  }

  /**
   *  Set a scaling coefficient x such that the effect of the electric field is
   *  xqE instead of qE.
   *
   *  @param x New scaling coefficient.
   */
  inline void set_E_coefficent(const REAL x) {
    this->particle_E_coefficient = x;
  }

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
      : session(session), graph(graph), comm(comm), tol(1.0e-8),
        h5part_exists(false) {

    this->set_B_field(0.0, 0.0, 0.0);
    this->set_E_coefficent(1.0);

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
                               ParticleProp(Sym<REAL>("Q"), 1),
                               ParticleProp(Sym<REAL>("M"), 1),
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

    // read or deduce a number density from the configuration file
    this->session->LoadParameter("particle_number_density",
                                 this->particle_number_density);
    if (this->particle_number_density < 0.0) {
      this->particle_weight = 1.0;
      this->particle_number_density = this->num_particles / volume;
    } else {
      const double number_mass = 2.0 * this->particle_number_density * volume;
      this->particle_weight = number_mass / this->num_particles;
    }

    if (this->session->DefinesParameter("particle_charge_density")) {
      this->session->LoadParameter("particle_charge_density",
                                   this->charge_density);

      const double number_mass = 2.0 * this->particle_number_density * volume;

      // determine the charge per physical particle
      this->particle_charge = 2.0 * this->charge_density * volume / number_mass;
    } else if (this->session->DefinesParameter("particle_charge")) {
      this->session->LoadParameter("particle_charge",
                                   this->particle_charge);
      this->charge_density = this->particle_number_density * this->particle_charge;
    } else {
      // error, not enough information
      // TODO throw!
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
          Sym<REAL>("P"), Sym<INT>("CELL_ID"), Sym<REAL>("V"), Sym<REAL>("E"),
          Sym<INT>("NESO_MPI_RANK"), Sym<REAL>("NESO_REFERENCE_POSITIONS"));
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
   * Velocity Verlet - First step.
   */
  inline void velocity_verlet_1() {

    const double k_dt = this->dt;
    const double k_dht = this->dt * 0.5;

    auto t0 = profile_timestamp();

    auto k_P = (*this->particle_group)[Sym<REAL>("P")]->cell_dat.device_ptr();
    auto k_V = (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();
    auto k_M = (*this->particle_group)[Sym<REAL>("M")]->cell_dat.device_ptr();
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

    const REAL k_E_coefficient = this->particle_E_coefficient;

    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const double Q = k_Q[cellx][0][layerx];
                const double dht_inverse_particle_mass =
                    k_E_coefficient * k_dht * Q / k_M[cellx][0][layerx];
                k_V[cellx][0][layerx] -=
                    k_E[cellx][0][layerx] * dht_inverse_particle_mass;
                k_V[cellx][1][layerx] -=
                    k_E[cellx][1][layerx] * dht_inverse_particle_mass;

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

    auto t0 = profile_timestamp();

    auto k_V = (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();
    const auto k_E =
        (*this->particle_group)[Sym<REAL>("E")]->cell_dat.device_ptr();
    const auto k_Q =
        (*this->particle_group)[Sym<REAL>("Q")]->cell_dat.device_ptr();
    auto k_M = (*this->particle_group)[Sym<REAL>("M")]->cell_dat.device_ptr();

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    const REAL k_E_coefficient = this->particle_E_coefficient;

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
                const double dht_inverse_particle_mass =
                    k_E_coefficient * k_dht * Q / k_M[cellx][0][layerx];
                k_V[cellx][0][layerx] -=
                    k_E[cellx][0][layerx] * dht_inverse_particle_mass;
                k_V[cellx][1][layerx] -=
                    k_E[cellx][1][layerx] * dht_inverse_particle_mass;

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target->profile_map.inc("ChargedParticles", "VelocityVerlet_2_Execute",
                                 1, profile_elapsed(t0, profile_timestamp()));
  }

  /**
   * Boris - First step.
   */
  inline void boris_1() {
    /*
    auto t0 = profile_timestamp();

    auto k_P = (*this->particle_group)[Sym<REAL>("P")]->cell_dat.device_ptr();
    auto k_V = (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    const double k_dht = this->dt * 0.5;

    sycl_target->profile_map.inc("ChargedParticles", "VelocityVerlet_2_Prepare",
                                 1, profile_elapsed(t0, profile_timestamp()));
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                // half update of position
                k_P[cellx][0][layerx] += k_dht * k_V[cellx][0][layerx];
                k_P[cellx][1][layerx] += k_dht * k_V[cellx][1][layerx];

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    sycl_target->profile_map.inc("ChargedParticles", "Boris_1_Execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));

    // positions were written so we apply boundary conditions and move
    // particles between ranks
    this->transfer_particles();
    */
  }

  /**
   * Boris - Second step.
   */
  inline void boris_2() {
    auto t0 = profile_timestamp();

    auto k_P = (*this->particle_group)[Sym<REAL>("P")]->cell_dat.device_ptr();
    auto k_V = (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();
    const auto k_M =
        (*this->particle_group)[Sym<REAL>("M")]->cell_dat.device_ptr();
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

    const double k_dt = this->dt;
    const double k_dht = this->dt * 0.5;
    const double k_B_0 = this->B_0;
    const double k_B_1 = this->B_1;
    const double k_B_2 = this->B_2;
    const REAL k_E_coefficient = this->particle_E_coefficient;

    sycl_target->profile_map.inc("ChargedParticles", "VelocityVerlet_2_Prepare",
                                 1, profile_elapsed(t0, profile_timestamp()));
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const REAL Q = k_Q[cellx][0][layerx];
                const REAL M = k_M[cellx][0][layerx];
                const REAL QoM = Q / M;

                const REAL scaling_t = QoM * k_dht;
                const REAL t_0 = k_B_0 * scaling_t;
                const REAL t_1 = k_B_1 * scaling_t;
                const REAL t_2 = k_B_2 * scaling_t;

                const REAL tmagsq = t_0 * t_0 + t_1 * t_1 + t_2 * t_2;
                const REAL scaling_s = 2.0 / (1.0 + tmagsq);

                const REAL s_0 = scaling_s * t_0;
                const REAL s_1 = scaling_s * t_1;
                const REAL s_2 = scaling_s * t_2;

                const REAL V_0 = k_V[cellx][0][layerx];
                const REAL V_1 = k_V[cellx][1][layerx];
                const REAL V_2 = k_V[cellx][2][layerx];

                // The E dat contains d(phi)/dx not E -> multiply by -1.
                const REAL v_minus_0 = V_0 + (-1.0 * k_E[cellx][0][layerx]) *
                                                 scaling_t * k_E_coefficient;
                const REAL v_minus_1 = V_1 + (-1.0 * k_E[cellx][1][layerx]) *
                                                 scaling_t * k_E_coefficient;
                // E is zero in the z direction
                const REAL v_minus_2 = V_2;

                REAL v_prime_0, v_prime_1, v_prime_2;
                ELEC_PIC_2D3V_CROSS_PRODUCT_3D(v_minus_0, v_minus_1, v_minus_2,
                                               t_0, t_1, t_2, v_prime_0,
                                               v_prime_1, v_prime_2)

                v_prime_0 += v_minus_0;
                v_prime_1 += v_minus_1;
                v_prime_2 += v_minus_2;

                REAL v_plus_0, v_plus_1, v_plus_2;
                ELEC_PIC_2D3V_CROSS_PRODUCT_3D(v_prime_0, v_prime_1, v_prime_2,
                                               s_0, s_1, s_2, v_plus_0,
                                               v_plus_1, v_plus_2)

                v_plus_0 += v_minus_0;
                v_plus_1 += v_minus_1;
                v_plus_2 += v_minus_2;

                // The E dat contains d(phi)/dx not E -> multiply by -1.
                k_V[cellx][0][layerx] =
                    v_plus_0 + scaling_t * (-1.0 * k_E[cellx][0][layerx]) *
                                   k_E_coefficient;
                k_V[cellx][1][layerx] =
                    v_plus_1 + scaling_t * (-1.0 * k_E[cellx][1][layerx]) *
                                   k_E_coefficient;
                // E is zero in the z direction
                k_V[cellx][2][layerx] = v_plus_2;

                // update of position to next time step
                k_P[cellx][0][layerx] += k_dt * k_V[cellx][0][layerx];
                k_P[cellx][1][layerx] += k_dt * k_V[cellx][1][layerx];

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    sycl_target->profile_map.inc("ChargedParticles", "Boris_2_Execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));

    // positions were written so we apply boundary conditions and move
    // particles between ranks
    this->transfer_particles();
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
