#ifndef __NESOSOLVERS_ELECTROSTATIC2D3V_CHARGEDPARTICLES_HPP__
#define __NESOSOLVERS_ELECTROSTATIC2D3V_CHARGEDPARTICLES_HPP__

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <random>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <boost/math/special_functions/erf.hpp>
#include <nektar_interface/function_evaluation.hpp>
#include <nektar_interface/function_projection.hpp>
#include <nektar_interface/geometry_transport/halo_extension.hpp>
#include <nektar_interface/particle_interface.hpp>
#include <neso_particles.hpp>
#include <particle_utility/position_distribution.hpp>

#include "IntegratorBorisUniformB.hpp"

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NP = NESO::Particles;

#ifndef ELEC_PIC_2D3V_CROSS_PRODUCT_3D
#define ELEC_PIC_2D3V_CROSS_PRODUCT_3D(a1, a2, a3, b1, b2, b3, c1, c2, c3)     \
  (c1) = ((a2) * (b3)) - ((a3) * (b2));                                        \
  (c2) = ((a3) * (b1)) - ((a1) * (b3));                                        \
  (c3) = ((a1) * (b2)) - ((a2) * (b1));
#endif

namespace NESO::Solvers::Electrostatic2D3V {

/**
 * Helper function to get values from the session file.
 *
 * @param session Session object.
 * @param name Name of the parameter.
 * @param output Reference to the output variable.
 * @param default Default value if name not found in the session file.
 */
template <typename T>
inline void elec2d3v_get_from_session(LU::SessionReaderSharedPtr session,
                                      std::string name, T &output,
                                      T default_value) {
  if (session->DefinesParameter(name)) {
    session->LoadParameter(name, output);
  } else {
    output = default_value;
  }
}

class ChargedParticles {
private:
  LU::SessionReaderSharedPtr session;
  SD::MeshGraphSharedPtr graph;
  MPI_Comm comm;
  const double tol;
  const int ndim = 2;
  double charge_density;
  bool h5part_exists;

  NP::REAL B_0;
  NP::REAL B_1;
  NP::REAL B_2;
  NP::REAL particle_E_coefficient;

  std::shared_ptr<IntegratorBorisUniformB> integrator_boris;

  inline void add_particles() {

    long rstart, rend;
    const long size = this->sycl_target->comm_pair.size_parent;
    const long rank = this->sycl_target->comm_pair.rank_parent;

    NP::get_decomp_1d(size, (long)this->num_particles, rank, &rstart, &rend);
    const long N = rend - rstart;
    const int cell_count = this->domain->mesh->get_cell_count();

    // get seed from file
    std::srand(std::time(nullptr));
    int seed;
    elec2d3v_get_from_session(this->session, "particle_position_seed", seed,
                              std::rand());

    std::mt19937 rng_phasespace(seed + rank);
    std::bernoulli_distribution coin_toss(0.5);

    std::uniform_real_distribution<double> uniform01(0, 1);

    int distribution_position = -1;
    session->LoadParameter("particle_distribution_position",
                           distribution_position);
    NESOASSERT(distribution_position > -1, "Bad particle distribution key.");
    NESOASSERT(distribution_position < 6, "Bad particle distribution key.");

    if (N > 0) {
      NP::ParticleSet initial_distribution(
          N, this->particle_group->get_particle_spec());

      // Get the requested particle distribution type from the config file
      int particle_distribution_type = 0;
      elec2d3v_get_from_session(session, "particle_distribution_type",
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
        positions = NP::uniform_within_extents(
            N, ndim, this->boundary_conditions->global_extent, rng_phasespace);
      }

      if (distribution_position == 0) {
        double initial_velocity;
        session->LoadParameter("particle_initial_velocity", initial_velocity);
        // square in lower left

        for (int px = 0; px < N; px++) {
          for (int dimx = 0; dimx < ndim; dimx++) {
            const double pos_orig =
                positions[dimx][px] +
                this->boundary_conditions->global_origin[dimx];
            initial_distribution[NP::Sym<NP::REAL>("P")][px][dimx] =
                pos_orig * 0.25;
          }

          initial_distribution[NP::Sym<NP::REAL>("V")][px][0] =
              initial_velocity;
          initial_distribution[NP::Sym<NP::REAL>("V")][px][1] = 0.0;
          initial_distribution[NP::Sym<NP::REAL>("Q")][px][0] =
              this->particle_charge;
          initial_distribution[NP::Sym<NP::REAL>("M")][px][0] =
              this->particle_mass;
        }
      } else if (distribution_position == 1) {
        double initial_velocity;
        session->LoadParameter("particle_initial_velocity", initial_velocity);
        // two stream - as two streams....

        for (int px = 0; px < N; px++) {

          // x position
          const double pos_orig_0 =
              positions[0][px] + this->boundary_conditions->global_origin[0];
          initial_distribution[NP::Sym<NP::REAL>("P")][px][0] = pos_orig_0;

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

          initial_distribution[NP::Sym<NP::REAL>("P")][px][1] =
              pos_orig_1 + shift_1;

          initial_distribution[NP::Sym<NP::REAL>("V")][px][0] =
              (species) ? initial_velocity : -1.0 * initial_velocity;
          ;
          initial_distribution[NP::Sym<NP::REAL>("V")][px][1] = 0.0;
          initial_distribution[NP::Sym<NP::REAL>("Q")][px][0] =
              this->particle_charge;
          initial_distribution[NP::Sym<NP::REAL>("M")][px][0] =
              this->particle_mass;
        }
      } else if (distribution_position == 2) {
        double initial_velocity;
        session->LoadParameter("particle_initial_velocity", initial_velocity);
        // two stream - as standard two stream
        for (int px = 0; px < N; px++) {

          const bool species = coin_toss(rng_phasespace);
          // x position
          const double pos_orig_0 =
              positions[0][px] + this->boundary_conditions->global_origin[0];
          initial_distribution[NP::Sym<NP::REAL>("P")][px][0] = pos_orig_0;

          // y position
          const double pos_orig_1 =
              positions[1][px] + this->boundary_conditions->global_origin[1];
          initial_distribution[NP::Sym<NP::REAL>("P")][px][1] = pos_orig_1;

          initial_distribution[NP::Sym<NP::REAL>("V")][px][0] =
              (species) ? initial_velocity : -1.0 * initial_velocity;
          // initial_distribution[NP::Sym<NP::REAL>("V")][px][1] =
          //     (species) ? initial_velocity : -1.0 * initial_velocity;
          initial_distribution[NP::Sym<NP::REAL>("V")][px][1] = 0.0;
          initial_distribution[NP::Sym<NP::REAL>("Q")][px][0] =
              this->particle_charge;
          initial_distribution[NP::Sym<NP::REAL>("M")][px][0] =
              this->particle_mass;
        }
      } else if (distribution_position == 3) {
        double initial_velocity;
        session->LoadParameter("particle_initial_velocity", initial_velocity);
        // two stream - with one species 1000000x the mass
        for (int px = 0; px < N; px++) {

          const bool species = coin_toss(rng_phasespace);
          // x position
          const double pos_orig_0 =
              positions[0][px] + this->boundary_conditions->global_origin[0];
          initial_distribution[NP::Sym<NP::REAL>("P")][px][0] = pos_orig_0;

          // y position
          const double pos_orig_1 =
              positions[1][px] + this->boundary_conditions->global_origin[1];
          initial_distribution[NP::Sym<NP::REAL>("P")][px][1] = pos_orig_1;

          initial_distribution[NP::Sym<NP::REAL>("V")][px][0] =
              (species) ? 0.0 : initial_velocity;
          ;
          initial_distribution[NP::Sym<NP::REAL>("V")][px][1] = 0.0;
          initial_distribution[NP::Sym<NP::REAL>("Q")][px][0] =
              (species) ? this->particle_charge : -1.0 * this->particle_charge;
          initial_distribution[NP::Sym<NP::REAL>("M")][px][0] =
              (species) ? this->particle_mass * 1000000 : this->particle_mass;
        }
      } else if (distribution_position == 4) {
        // 3V Maxwellian
        auto positions = NP::uniform_within_extents(
            N, ndim, this->boundary_conditions->global_extent, rng_phasespace);

        double thermal_velocity;
        session->LoadParameter("particle_thermal_velocity", thermal_velocity);

        for (int px = 0; px < N; px++) {

          // x position
          const double pos_orig_0 =
              positions[0][px] + this->boundary_conditions->global_origin[0];
          initial_distribution[NP::Sym<NP::REAL>("P")][px][0] = pos_orig_0;

          // y position
          const double pos_orig_1 =
              positions[1][px] + this->boundary_conditions->global_origin[1];
          initial_distribution[NP::Sym<NP::REAL>("P")][px][1] = pos_orig_1;

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

          initial_distribution[NP::Sym<NP::REAL>("V")][px][0] =
              thermal_velocity * rvx;
          initial_distribution[NP::Sym<NP::REAL>("V")][px][1] =
              thermal_velocity * rvy;
          initial_distribution[NP::Sym<NP::REAL>("V")][px][2] =
              thermal_velocity * rvz;

          initial_distribution[NP::Sym<NP::REAL>("Q")][px][0] =
              this->particle_charge;
          initial_distribution[NP::Sym<NP::REAL>("M")][px][0] =
              this->particle_mass;
        }
      } else if (distribution_position == 5) {
        double initial_velocity;
        session->LoadParameter("particle_initial_velocity", initial_velocity);
        // two stream - as standard two stream
        for (int px = 0; px < N; px++) {
          // x position
          const double pos_orig_0 =
              positions[0][px] + this->boundary_conditions->global_origin[0];
          initial_distribution[NP::Sym<NP::REAL>("P")][px][0] = pos_orig_0;

          // y position
          const double pos_orig_1 =
              positions[1][px] + this->boundary_conditions->global_origin[1];
          initial_distribution[NP::Sym<NP::REAL>("P")][px][1] = pos_orig_1;

          initial_distribution[NP::Sym<NP::REAL>("V")][px][0] =
              initial_velocity;
          initial_distribution[NP::Sym<NP::REAL>("V")][px][1] =
              1.0 + uniform01(rng_phasespace);
          initial_distribution[NP::Sym<NP::REAL>("V")][px][2] = 0.0;
          initial_distribution[NP::Sym<NP::REAL>("Q")][px][0] =
              this->particle_charge;
          initial_distribution[NP::Sym<NP::REAL>("M")][px][0] =
              this->particle_mass;
        }
      }

      for (int px = 0; px < N; px++) {
        initial_distribution[NP::Sym<NP::REAL>("E")][px][0] = 0.0;
        initial_distribution[NP::Sym<NP::REAL>("E")][px][1] = 0.0;
        initial_distribution[NP::Sym<NP::INT>("CELL_ID")][px][0] =
            px % cell_count;
        initial_distribution[NP::Sym<NP::INT>("PARTICLE_ID")][px][0] =
            px + rstart;
      }
      this->particle_group->add_particles_local(initial_distribution);
    }

    NP::parallel_advection_initialisation(this->particle_group);
    NP::parallel_advection_store(this->particle_group);

    // auto h5part_local = std::make_shared<NP::H5Part>(
    //       "foo.h5part", this->particle_group,
    //       NP::Sym<NP::REAL>("P"), NP::Sym<NP::REAL>("ORIG_POS"),
    //       NP::Sym<NP::INT>("NESO_MPI_RANK"), NP::Sym<NP::INT>("PARTICLE_ID"),
    //       NP::Sym<NP::REAL>("NESO_REFERENCE_POSITIONS"));
    const int num_steps = 20;
    for (int stepx = 0; stepx < num_steps; stepx++) {
      NP::parallel_advection_step(this->particle_group, num_steps, stepx);
      this->transfer_particles();
      // h5part_local->write();
    }
    NP::parallel_advection_restore(this->particle_group);
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
  NP::SYCLTargetSharedPtr sycl_target;
  /// Mapping instance to map particles into nektar++ elements.
  std::shared_ptr<NektarGraphLocalMapper> nektar_graph_local_mapper;
  /// NESO-Particles domain.
  NP::DomainSharedPtr domain;
  /// NESO-Particles ParticleGroup containing charged particles.
  NP::ParticleGroupSharedPtr particle_group;
  /// Method to apply particle boundary conditions.
  std::shared_ptr<NektarCartesianPeriodic> boundary_conditions;
  /// Method to map to/from nektar geometry ids to 0,N-1 used by NESO-Particles
  std::shared_ptr<CellIDTranslation> cell_id_translation;
  /// Trajectory writer for particles.
  std::shared_ptr<NP::H5Part> h5part;

  /**
   *  Set the constant and uniform magnetic field over the entire domain.
   *
   *  @param B0 Magnetic fiel B in x direction.
   *  @param B1 Magnetic fiel B in y direction.
   *  @param B2 Magnetic fiel B in z direction.
   */
  inline void set_B_field(const NP::REAL B0 = 0.0, const NP::REAL B1 = 0.0,
                          const NP::REAL B2 = 0.0) {
    this->B_0 = B0;
    this->B_1 = B1;
    this->B_2 = B2;
    this->integrator_boris->set_B_field(B0, B1, B2);
  }

  /**
   *  Set a scaling coefficient x such that the effect of the electric field is
   *  xqE instead of qE.
   *
   *  @param x New scaling coefficient.
   */
  inline void set_E_coefficent(const NP::REAL x) {
    this->particle_E_coefficient = x;
    this->integrator_boris->set_E_coefficent(x);
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
  ChargedParticles(LU::SessionReaderSharedPtr session,
                   SD::MeshGraphSharedPtr graph, MPI_Comm comm = MPI_COMM_WORLD)
      : session(session), graph(graph), comm(comm), tol(1.0e-8),
        h5part_exists(false) {

    this->B_0 = 0.0;
    this->B_1 = 0.0;
    this->B_2 = 0.0;
    this->particle_E_coefficient = 1.0;

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

    extend_halos_fixed_offset(0, particle_mesh_interface);

    this->sycl_target = std::make_shared<NP::SYCLTarget>(
        0, particle_mesh_interface->get_comm());
    this->nektar_graph_local_mapper = std::make_shared<NektarGraphLocalMapper>(
        this->sycl_target, this->particle_mesh_interface);
    this->domain = std::make_shared<NP::Domain>(
        this->particle_mesh_interface, this->nektar_graph_local_mapper);

    // Create ParticleGroup
    ParticleSpec particle_spec{
        NP::ParticleProp(NP::Sym<NP::REAL>("P"), 2, true),
        NP::ParticleProp(NP::Sym<NP::INT>("CELL_ID"), 1, true),
        NP::ParticleProp(NP::Sym<NP::INT>("PARTICLE_ID"), 1),
        NP::ParticleProp(NP::Sym<NP::REAL>("Q"), 1),
        NP::ParticleProp(NP::Sym<NP::REAL>("M"), 1),
        NP::ParticleProp(NP::Sym<NP::REAL>("V"), 3),
        NP::ParticleProp(NP::Sym<NP::REAL>("E"), 2)};

    this->particle_group = std::make_shared<NP::ParticleGroup>(
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

    if (this->session->DefinesParameter("particle_charge_density")) {
      this->session->LoadParameter("particle_charge_density",
                                   this->charge_density);

      const double number_physical_particles =
          this->particle_number_density * volume;

      // determine the charge per physical particle
      this->particle_charge =
          this->charge_density * volume / number_physical_particles;
    } else if (this->session->DefinesParameter("particle_charge")) {
      this->session->LoadParameter("particle_charge", this->particle_charge);
      this->charge_density =
          this->particle_number_density * this->particle_charge;
    } else {
      // error, not enough information
      // TODO throw!
    }

    // Add particle to the particle group
    this->add_particles();

    // create a Boris integrator
    this->integrator_boris = std::make_shared<IntegratorBorisUniformB>(
        this->particle_group, this->dt, this->B_0, this->B_1, this->B_2,
        this->particle_E_coefficient);
  };

  /**
   *  Write current particle state to trajectory.
   */
  inline void write() {
    if (!this->h5part_exists) {
      // Create instance to write particle data to h5 file
      this->h5part = std::make_shared<NP::H5Part>(
          "Electrostatic2D3V_particle_trajectory.h5part", this->particle_group,
          NP::Sym<NP::REAL>("P"), NP::Sym<NP::INT>("CELL_ID"),
          NP::Sym<NP::REAL>("V"), NP::Sym<NP::REAL>("E"),
          NP::Sym<NP::INT>("NESO_MPI_RANK"), NP::Sym<NP::INT>("PARTICLE_ID"),
          NP::Sym<NP::REAL>("NESO_REFERENCE_POSITIONS"));
      this->h5part_exists = true;
    }

    this->h5part->write();
  }

  /**
   *  Apply boundary conditions and transfer particles between MPI ranks.
   */
  inline void transfer_particles() {
    auto t0 = NP::profile_timestamp();
    this->boundary_conditions->execute();
    this->particle_group->hybrid_move();
    this->cell_id_translation->execute();
    this->particle_group->cell_move();
    this->sycl_target->profile_map.inc(
        "ChargedParticles", "transfer_particles", 1,
        NP::profile_elapsed(t0, NP::profile_timestamp()));
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
    auto t0 = NP::profile_timestamp();
    const double k_dt = this->dt;
    const double k_dht = this->dt * 0.5;
    const NP::REAL k_E_coefficient = this->particle_E_coefficient;

    NP::particle_loop(
        "ChargedParticles::velocity_verlet_1", this->particle_group,
        [=](auto k_Q, auto k_E, auto k_M, auto k_V, auto k_P) {
          const double Q = k_Q.at(0);
          const double dht_inverse_particle_mass =
              k_E_coefficient * k_dht * Q / k_M.at(0);
          k_V.at(0) -= k_E.at(0) * dht_inverse_particle_mass;
          k_V.at(1) -= k_E.at(1) * dht_inverse_particle_mass;

          k_P.at(0) += k_dt * k_V.at(0);
          k_P.at(1) += k_dt * k_V.at(1);
        },
        NP::Access::read(NP::Sym<NP::REAL>("Q")),
        NP::Access::read(NP::Sym<NP::REAL>("E")),
        NP::Access::read(NP::Sym<NP::REAL>("M")),
        NP::Access::write(NP::Sym<NP::REAL>("V")),
        NP::Access::write(NP::Sym<NP::REAL>("P")))
        ->execute();

    sycl_target->profile_map.inc(
        "ChargedParticles", "VelocityVerlet_1_Execute", 1,
        NP::profile_elapsed(t0, NP::profile_timestamp()));

    // positions were written so we apply boundary conditions and move
    // particles between ranks
    this->transfer_particles();
  }

  /**
   * Velocity Verlet - Second step.
   */
  inline void velocity_verlet_2() {
    auto t0 = NP::profile_timestamp();
    const double k_dht = this->dt * 0.5;
    const NP::REAL k_E_coefficient = this->particle_E_coefficient;

    NP::particle_loop(
        "ChargedParticles::velocity_verlet_2", this->particle_group,
        [=](auto k_Q, auto k_E, auto k_M, auto k_V) {
          const double Q = k_Q.at(0);
          const double dht_inverse_particle_mass =
              k_E_coefficient * k_dht * Q / k_M.at(0);
          k_V.at(0) -= k_E.at(0) * dht_inverse_particle_mass;
          k_V.at(1) -= k_E.at(1) * dht_inverse_particle_mass;
        },
        NP::Access::read(NP::Sym<NP::REAL>("Q")),
        NP::Access::read(NP::Sym<NP::REAL>("E")),
        NP::Access::read(NP::Sym<NP::REAL>("M")),
        NP::Access::write(NP::Sym<NP::REAL>("V")))
        ->execute();

    sycl_target->profile_map.inc(
        "ChargedParticles", "VelocityVerlet_2_Execute", 1,
        NP::profile_elapsed(t0, NP::profile_timestamp()));
  }

  /**
   * Boris - First step.
   */
  inline void boris_1() { this->integrator_boris->boris_1(); }

  /**
   * Boris - Second step.
   */
  inline void boris_2() {
    this->integrator_boris->boris_2();
    // positions were written so we apply boundary conditions and move
    // particles between ranks
    this->transfer_particles();
  }

  /**
   *  Get the NP::Sym object for the ParticleDat holding particle charge.
   */
  inline NP::Sym<NP::REAL> get_charge_sym() { return NP::Sym<NP::REAL>("Q"); }

  /**
   *  Get the NP::Sym object for the ParticleDat to hold the potential gradient.
   */
  inline NP::Sym<NP::REAL> get_potential_gradient_sym() {
    return NP::Sym<NP::REAL>("E");
  }

  /**
   *  Get the charge density of the system.
   */
  inline double get_charge_density() { return this->charge_density; }
};

} // namespace NESO::Solvers::Electrostatic2D3V

#endif // __NESOSOLVERS_ELECTROSTATIC2D3V_CHARGEDPARTICLES_HPP__
