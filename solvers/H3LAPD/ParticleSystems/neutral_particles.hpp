#ifndef __H3LAPD_NEUTRAL_PARTICLES_H_
#define __H3LAPD_NEUTRAL_PARTICLES_H_

#include <nektar_interface/function_evaluation.hpp>
#include <nektar_interface/function_projection.hpp>
#include <nektar_interface/particle_interface.hpp>
#include <nektar_interface/utilities.hpp>
#include <neso_particles.hpp>

#include <particle_utility/particle_initialisation_line.hpp>
#include <particle_utility/position_distribution.hpp>

#include <FieldUtils/Interpolator.h>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <boost/math/special_functions/erf.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <random>

using namespace Nektar;
using namespace NESO;
using namespace NESO::Particles;
using namespace Nektar::SpatialDomains;

// TODO move this to the correct place
/**
 * Evaluate the Barry et al approximation to the exponential integral function
 * https://en.wikipedia.org/wiki/Exponential_integral E_1(x)
 */
inline double expint_barry_approx(const double x) {
  constexpr double gamma_Euler_Mascheroni = 0.5772156649015329;
  const double G = std::exp(-gamma_Euler_Mascheroni);
  const double b = std::sqrt(2 * (1 - G) / G / (2 - G));
  const double h_inf = (1 - G) * (std::pow(G, 2) - 6 * G + 12) /
                       (3 * G * std::pow(2 - G, 2) * b);
  const double q = 20.0 / 47.0 * std::pow(x, std::sqrt(31.0 / 26.0));
  const double h = 1 / (1 + x * std::sqrt(x)) + h_inf * q / (1 + q);
  const double logfactor =
      std::log(1 + G / x - (1 - G) / std::pow(h + b * x, 2));
  return std::exp(-x) / (G + (1 - G) * std::exp(-(x / (1 - G)))) * logfactor;
}

class NeutralParticleSystem {
protected:
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  MPI_Comm comm;
  const double tol;
  const int ndim;
  bool h5part_exists;
  double simulation_time;
  int debug_step;
  std::shared_ptr<ErrorPropagate> ep_ionisation;
  bool low_order_project;

  /**
   *  Returns true if all boundary conditions on the density fields are
   *  periodic.
   */
  inline bool is_fully_periodic() {
    NESOASSERT(this->fields.count("ne") == 1, "ne field not found in fields.");
    auto bcs = this->fields["ne"]->GetBndConditions();
    bool is_pbc = true;
    for (auto &bc : bcs) {
      is_pbc &= (bc->GetBoundaryConditionType() == ePeriodic);
    }
    return is_pbc;
  }

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

  std::mt19937 rng_phasespace;
  std::shared_ptr<ParticleRemover> particle_remover;

  // Project object to project onto number density and momentum fields
  std::shared_ptr<FieldProject<DisContField>> field_project;
  // Evaluate object to evaluate number density field
  std::shared_ptr<FieldEvaluate<DisContField>> field_evaluate_ne;

  int debug_write_fields_count;
  std::map<std::string, std::shared_ptr<DisContField>> fields;

public:
  /// Disable (implicit) copies.
  NeutralParticleSystem(const NeutralParticleSystem &st) = delete;
  /// Disable (implicit) copies.
  NeutralParticleSystem &operator=(NeutralParticleSystem const &a) = delete;

  /// Global number of particles in the simulation.
  int64_t num_particles;
  /// Average number of particles per cell (element) in the simulation.
  int64_t num_particles_per_cell;
  /// Total number of particles added on this MPI rank.
  uint64_t total_num_particles_added;
  /// Integer lable for initial particle distribution
  int particle_distribution_type;
  /// Mass of particles
  const double particle_mass = 1.0;
  /// Initial particle velocity.
  double particle_init_vel;
  /// Initial particle weight.
  double particle_init_weight;
  /// Number density in simulation domain (per species)
  double particle_number_density;
  // PARTICLE_ID value used to flag particles for removal from the simulation
  const int particle_remove_key = -1;
  /// Particle thermal velocity
  double particle_thermal_velocity;
  /// Particle drift velocity
  double particle_drift_velocity;
  // Random seed used in particle initialisation
  int seed;
  /// HMesh instance that allows particles to move over nektar++ meshes.
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  /// Compute target.
  SYCLTargetSharedPtr sycl_target;
  /// Mapping instance to map particles into nektar++ elements.
  std::shared_ptr<NektarGraphLocalMapper> nektar_graph_local_mapper;
  /// NESO-Particles domain.
  DomainSharedPtr domain;
  /// NESO-Particles ParticleGroup containing charged particles.
  ParticleGroupSharedPtr particle_group;
  /// Method to apply particle boundary conditions.
  std::shared_ptr<NektarCartesianPeriodic> periodic_bc;
  /// Method to map to/from nektar geometry ids to 0,N-1 used by NESO-Particles
  std::shared_ptr<CellIDTranslation> cell_id_translation;
  /// Trajectory writer for particles.
  std::shared_ptr<H5Part> h5part;

  // Temperature assumed for ionisation rate, read from session
  double TeV;

  // Background density, read from session
  double n_bg_SI;

  // Factors to convert nektar units to units required by ionisation calc
  double t_to_SI;
  double n_to_SI;

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
      : session(session), graph(graph), comm(comm),
        ndim(graph->GetSpaceDimension()), tol(1.0e-8), h5part_exists(false),
        simulation_time(0.0), total_num_particles_added(0) {

    this->debug_write_fields_count = 0;
    this->debug_step = 0;

    // Set plasma temperature from session param
    get_from_session(this->session, "Te_eV", this->TeV, 10.0);
    // Set background density from session param
    get_from_session(this->session, "n_bg_SI", this->n_bg_SI, 1e18);

    // Read the number of requested particles per cell.
    int tmp_int;
    this->session->LoadParameter("num_particles_per_cell", tmp_int);
    this->num_particles_per_cell = tmp_int;

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
    this->nektar_graph_local_mapper = std::make_shared<NektarGraphLocalMapper>(
        this->sycl_target, this->particle_mesh_interface);
    this->domain = std::make_shared<Domain>(this->particle_mesh_interface,
                                            this->nektar_graph_local_mapper);

    // SI scaling factors required by ionise()
    this->session->LoadParameter("n_to_SI", this->n_to_SI, 1e17);
    this->session->LoadParameter("t_to_SI", this->t_to_SI, 1e-3);

    // Create ParticleGroup
    ParticleSpec particle_spec{
        ParticleProp(Sym<REAL>("POSITION"), 3, true),
        ParticleProp(Sym<INT>("CELL_ID"), 1, true),
        ParticleProp(Sym<INT>("PARTICLE_ID"), 1),
        ParticleProp(Sym<REAL>("COMPUTATIONAL_WEIGHT"), 1),
        ParticleProp(Sym<REAL>("SOURCE_DENSITY"), 1),
        ParticleProp(Sym<REAL>("ELECTRON_DENSITY"), 1),
        ParticleProp(Sym<REAL>("MASS"), 1),
        ParticleProp(Sym<REAL>("VELOCITY"), 3)};

    this->particle_group = std::make_shared<ParticleGroup>(
        this->domain, particle_spec, this->sycl_target);

    this->particle_remover =
        std::make_shared<ParticleRemover>(this->sycl_target);

    // Set up periodic boundary conditions.
    this->periodic_bc = std::make_shared<NektarCartesianPeriodic>(
        this->sycl_target, this->graph, this->particle_group->position_dat);

    // Set up map between cell indices
    this->cell_id_translation = std::make_shared<CellIDTranslation>(
        this->sycl_target, this->particle_group->cell_id_dat,
        this->particle_mesh_interface);

    // Set properties that affect the behaviour of add_particles()
    get_from_session(this->session, "particle_thermal_velocity",
                     this->particle_thermal_velocity, 1.0);
    get_from_session(this->session, "particle_drift_velocity",
                     this->particle_drift_velocity, 0.0);

    // Set particle region = domain volume for now
    double particle_region_volume = this->periodic_bc->global_extent[0];
    for (auto idim = 1; idim < this->ndim; idim++) {
      particle_region_volume *= this->periodic_bc->global_extent[idim];
    }

    // read or deduce a number density from the configuration file
    get_from_session(this->session, "particle_number_density",
                     this->particle_number_density, -1.0);
    if (this->particle_number_density < 0.0) {
      this->particle_init_weight = 1.0;
      this->particle_number_density =
          this->num_particles / particle_region_volume;
    } else {
      const double num_phys_particles =
          this->particle_number_density * particle_region_volume;
      this->particle_init_weight =
          (this->num_particles == 0) ? 0.0
                                     : num_phys_particles / this->num_particles;
    }

    // get seed from file
    std::srand(std::time(nullptr));

    get_from_session(this->session, "particle_position_seed", this->seed,
                     std::rand());

    const long rank = this->sycl_target->comm_pair.rank_parent;
    this->rng_phasespace = std::mt19937(this->seed + rank);

    this->ep_ionisation = std::make_shared<ErrorPropagate>(this->sycl_target);
  };

  /**
   * Setup the projection object
   *
   * @param ne_src Nektar++ field to project particle source terms onto.
   */
  inline void setup_project(std::shared_ptr<DisContField> ne_src) {
    std::vector<std::shared_ptr<DisContField>> fields = {ne_src};
    this->field_project = std::make_shared<FieldProject<DisContField>>(
        fields, this->particle_group, this->cell_id_translation);

    // Add to local map
    this->fields["ne_src"] = ne_src;
    this->low_order_project = false;
  }

  /**
   * Setup the projection object. Project onto the first argument then
   * interpolate that onto the second argument.
   *
   * @param ne_src_interp Nektar++ field to project particle source terms onto.
   * @param ne_src Nektar++ field to interpolate the projected source terms
   * onto.
   */
  inline void setup_project(std::shared_ptr<DisContField> ne_src_interp,
                            std::shared_ptr<DisContField> ne_src) {
    std::vector<std::shared_ptr<DisContField>> fields = {ne_src_interp};
    this->field_project = std::make_shared<FieldProject<DisContField>>(
        fields, this->particle_group, this->cell_id_translation);

    // Add to local map
    this->fields["ne_src_interp"] = ne_src_interp;
    this->fields["ne_src"] = ne_src;
    this->low_order_project = true;
  }

  /**
   * Setup the evaluation of a number density field.
   *
   * @param n Nektar++ field storing plasma number density.
   */
  inline void setup_evaluate_ne(std::shared_ptr<DisContField> n) {
    this->field_evaluate_ne = std::make_shared<FieldEvaluate<DisContField>>(
        n, this->particle_group, this->cell_id_translation);
    this->fields["ne"] = n;
  }

  /**
   *  Project particle source terms onto nektar fields.
   */
  inline void project_source_terms() {
    NESOASSERT(this->field_project != nullptr,
               "Field project object is null. Was setup_project called?");

    // this->particle_group->print(Sym<REAL>("SOURCE_DENSITY"));

    std::vector<Sym<REAL>> syms = {Sym<REAL>("SOURCE_DENSITY")};
    std::vector<int> components = {0};
    this->field_project->project(syms, components);
    if (this->low_order_project) {
      nprint("interpolating");
      FieldUtils::Interpolator interpolator{};
      std::vector<MultiRegions::ExpListSharedPtr> in_exp = {
          this->fields["ne_src_interp"]};
      std::vector<MultiRegions::ExpListSharedPtr> out_exp = {
          this->fields["ne_src"]};
      interpolator.Interpolate(in_exp, out_exp);
    }
    // remove fully ionised particles from the simulation
    remove_marked_particles();
  }

  /**
   * Add particles to the simulation.
   *
   * @param add_proportion Specifies the proportion of the number of particles
   * added in a time step.
   */
  inline void add_particles(const double add_proportion) {
    long rstart, rend;
    const long size = this->sycl_target->comm_pair.size_parent;
    const long rank = this->sycl_target->comm_pair.rank_parent;

    const long num_particles_to_add =
        std::round(add_proportion * ((double)this->num_particles));
    nprint("num_particles_to_add:", num_particles_to_add);

    get_decomp_1d(size, num_particles_to_add, rank, &rstart, &rend);
    const long N = rend - rstart;
    const int cell_count = this->domain->mesh->get_cell_count();

    // // Read the particle distribution type and position from the session
    // int particle_distribution_type;
    // get_from_session(session, "particle_distribution_type",
    //                  particle_distribution_type, 0);
    // NESOASSERT(particle_distribution_type == 0,
    //            "Bad particle distribution type.");
    // int distribution_position;
    // get_from_session(session, "particle_distribution_position",
    //                  distribution_position, -1);
    // NESOASSERT(distribution_position == 0,
    //            "Bad particle distribution position.");

    if (N > 0) {
      // Generate N particles
      ParticleSet initial_distribution(
          N, this->particle_group->get_particle_spec());

      // Generate particle positions and velocities
      std::vector<std::vector<double>> positions, velocities;

      // Positions are Gaussian with same width in all dims
      double mu = 0.0;
      double sigma = 0.2;
      positions = NESO::Particles::normal_distribution(N, this->ndim, mu, sigma,
                                                       this->rng_phasespace);
      // Centre of distribution
      std::vector<double> offsets = {0.0, 0.0,
                                     (this->periodic_bc->global_extent[2] -
                                      this->periodic_bc->global_origin[2]) /
                                         2};

      velocities = NESO::Particles::normal_distribution(
          N, this->ndim, this->particle_drift_velocity,
          this->particle_thermal_velocity, this->rng_phasespace);

      // Set positions, velocities
      for (int ipart = 0; ipart < N; ipart++) {
        for (int idim = 0; idim < this->ndim; idim++) {
          initial_distribution[Sym<REAL>("POSITION")][ipart][idim] =
              positions[idim][ipart] + offsets[idim];
          initial_distribution[Sym<REAL>("VELOCITY")][ipart][idim] =
              velocities[idim][ipart];
        }
      }

      // Set remaining properties
      for (int ipart = 0; ipart < N; ipart++) {
        initial_distribution[Sym<INT>("CELL_ID")][ipart][0] =
            ipart % cell_count;
        initial_distribution[Sym<REAL>("COMPUTATIONAL_WEIGHT")][ipart][0] =
            this->particle_init_weight;
        initial_distribution[Sym<REAL>("MASS")][ipart][0] = this->particle_mass;
        initial_distribution[Sym<INT>("PARTICLE_ID")][ipart][0] =
            ipart + rstart + this->total_num_particles_added;
      }
      this->particle_group->add_particles_local(initial_distribution);
    }
    this->total_num_particles_added += num_particles_to_add;

    parallel_advection_initialisation(this->particle_group);
    parallel_advection_store(this->particle_group);

    // auto h5part_local = std::make_shared<H5Part>(
    //       "foo.h5part", this->particle_group,
    //       Sym<REAL>("P"), Sym<REAL>("ORIG_POS"), Sym<INT>("NESO_MPI_RANK"),
    //       Sym<INT>("PARTICLE_ID"), Sym<REAL>("NESO_REFERENCE_POSITIONS"));
    const int num_steps = 20;
    for (int stepx = 0; stepx < num_steps; stepx++) {
      parallel_advection_step(this->particle_group, num_steps, stepx);
      this->transfer_particles();
      // h5part_local->write();
    }
    parallel_advection_restore(this->particle_group);
    // h5part_local->write();
    // h5part_local->close();

    // Move particles to the owning ranks and correct cells.
    this->transfer_particles();
  }

  /**
   *  Write current particle state to trajectory.
   *
   *  @param step Time step number.
   */
  inline void write(const int step) {

    if (this->sycl_target->comm_pair.rank_parent == 0) {
      nprint("Writing particle trajectories at step", step);
    }

    if (!this->h5part_exists) {
      // Create instance to write particle data to h5 file
      this->h5part = std::make_shared<H5Part>(
          "particle_trajectory.h5part", this->particle_group,
          Sym<REAL>("POSITION"), Sym<INT>("CELL_ID"),
          Sym<REAL>("COMPUTATIONAL_WEIGHT"), Sym<REAL>("VELOCITY"),
          Sym<INT>("PARTICLE_ID"));
      this->h5part_exists = true;
    }

    this->h5part->write();
  }

  /**
   *  Write the projection fields to vtu for debugging.
   */
  inline void write_source_fields() {
    for (auto entry : this->fields) {
      std::string filename = "debug_" + entry.first + "_" +
                             std::to_string(this->debug_write_fields_count++) +
                             ".vtu";
      write_vtu(entry.second, filename, entry.first);
    }
  }

  /**
   * Apply boundary conditions to particles that have left the domain.
   */
  inline void wall_boundary_conditions() {
    NESOASSERT(false, "wall_boundary_conditions not implemented");
    // // Find particles that have travelled outside the domain in the x
    // direction.auto k_P =
    //     (*this->particle_group)[Sym<REAL>("POSITION")]->cell_dat.device_ptr();
    // // reuse this dat for remove flags
    // auto k_PARTICLE_ID =
    //     (*this->particle_group)[Sym<INT>("PARTICLE_ID")]->cell_dat.device_ptr();

    // const auto pl_iter_range =
    //     this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    // const auto pl_stride =
    //     this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    // const auto pl_npart_cell =
    //     this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    // const REAL k_lower_bound = 0.0;
    // const REAL k_upper_bound = k_lower_bound + this->unrotated_x_max;

    // const INT k_remove_key = this->particle_remove_key;

    // sycl_target->queue
    //     .submit([&](sycl::handler &cgh) {
    //       cgh.parallel_for<>(
    //           sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
    //             NESO_PARTICLES_KERNEL_START
    //             const INT cellx = NESO_PARTICLES_KERNEL_CELL;
    //             const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
    //             const REAL px = k_P[cellx][0][layerx];
    //             if ((px < k_lower_bound) || (px > k_upper_bound)) {
    //               // mark the particle as removed
    //               k_PARTICLE_ID[cellx][0][layerx] = k_remove_key;
    //             }
    //             NESO_PARTICLES_KERNEL_END
    //           });
    //     })
    //     .wait_and_throw();

    // // remove particles marked to remove by the boundary conditions
    // remove_marked_particles();
  }

  inline void remove_marked_particles() {
    this->particle_remover->remove(
        this->particle_group, (*this->particle_group)[Sym<INT>("PARTICLE_ID")],
        this->particle_remove_key);
    nprint("Remaining particles:", this->particle_group->get_npart_local());
  }

  /**
   *  Apply the boundary conditions to the particle system.
   */
  inline void boundary_conditions() {
    if (!this->is_fully_periodic()) {
      this->wall_boundary_conditions();
    }
    this->periodic_bc->execute();
  }

  /**
   *  Apply boundary conditions and transfer particles between MPI ranks.
   */
  inline void transfer_particles() {
    auto t0 = profile_timestamp();
    this->boundary_conditions();
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
   *  Integrate the particle system forward in time to the requested time using
   *  at most the requested time step.
   *
   *  @param time_end Target time to integrate to.
   *  @param dt Time step size.
   */
  inline void integrate(const double time_end, const double dt) {

    // Get the current simulation time.
    NESOASSERT(time_end >= this->simulation_time,
               "Cannot integrate backwards in time.");
    if (time_end == this->simulation_time) {
      return;
    }
    nprint(time_end, dt);
    if (this->total_num_particles_added == 0) {
      this->add_particles(1.0);
      nprint("added particles");
    }

    double time_tmp = this->simulation_time;
    while (time_tmp < time_end) {
      const double dt_inner = std::min(dt, time_end - time_tmp);
      // this->add_particles(dt_inner / dt);
      this->forward_euler(dt_inner);
      this->ionise(dt_inner);
      time_tmp += dt_inner;
    }

    this->simulation_time = time_end;

    this->debug_step++;
  }

  /**
   * Apply Forward-Euler, which with no forces is trivial.
   *
   * @param dt Time step size.
   */
  inline void forward_euler(const double dt) {

    const double k_dt = dt;

    auto t0 = profile_timestamp();

    auto k_P =
        (*this->particle_group)[Sym<REAL>("POSITION")]->cell_dat.device_ptr();
    auto k_V =
        (*this->particle_group)[Sym<REAL>("VELOCITY")]->cell_dat.device_ptr();

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
                k_P[cellx][2][layerx] += k_dt * k_V[cellx][2][layerx];

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

  /**
   *  Evaluate fields at the particle locations.
   */
  inline void evaluate_fields() {

    NESOASSERT(this->field_evaluate_ne != nullptr,
               "FieldEvaluate object is null. Was setup_evaluate_ne called?");

    this->field_evaluate_ne->evaluate(Sym<REAL>("ELECTRON_DENSITY"));

    // Particle property to update
    auto k_n = (*this->particle_group)[Sym<REAL>("ELECTRON_DENSITY")]
                   ->cell_dat.device_ptr();

    auto k_n_bg_SI = this->n_bg_SI;

    // Unit conversion factors
    double k_n_to_SI = this->n_to_SI;

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                k_n[cellx][0][layerx] =
                    k_n_bg_SI + k_n[cellx][0][layerx] * k_n_to_SI;
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
  }

  /**
   * Apply ionisation
   *
   * @param dt Time step size.
   */
  inline void ionise(const double dt) {

    // Evaluate the density and temperature fields at the particle locations
    this->evaluate_fields();

    const double k_dt = dt;
    const double k_dt_SI = dt * this->t_to_SI;
    const double k_n_scale = 1 / this->n_to_SI;

    const double k_a_i = 4.0e-14; // a_i constant for hydrogen (a_1)
    const double k_b_i = 0.6;     // b_i constant for hydrogen (b_1)
    const double k_c_i = 0.56;    // c_i constant for hydrogen (c_1)
    const double k_E_i =
        13.6; // E_i binding energy for most bound electron in hydrogen (E_1)
    const double k_q_i = 1.0; // Number of electrons in inner shell for hydrogen
    const double k_b_i_expc_i =
        k_b_i * std::exp(k_c_i); // exp(c_i) constant for hydrogen (c_1)

    const double k_rate_factor =
        -k_q_i * 6.7e7 * k_a_i * 1e-6; // 1e-6 to go from cm^3 to m^3

    const INT k_remove_key = this->particle_remove_key;

    auto t0 = profile_timestamp();

    auto k_ID =
        (*this->particle_group)[Sym<INT>("PARTICLE_ID")]->cell_dat.device_ptr();
    auto k_TeV = this->TeV;
    auto k_n = (*this->particle_group)[Sym<REAL>("ELECTRON_DENSITY")]
                   ->cell_dat.device_ptr();
    auto k_SD = (*this->particle_group)[Sym<REAL>("SOURCE_DENSITY")]
                    ->cell_dat.device_ptr();

    auto k_V =
        (*this->particle_group)[Sym<REAL>("VELOCITY")]->cell_dat.device_ptr();
    auto k_W = (*this->particle_group)[Sym<REAL>("COMPUTATIONAL_WEIGHT")]
                   ->cell_dat.device_ptr();

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    sycl_target->profile_map.inc("NeutralParticleSystem", "Ionisation_Prepare",
                                 1, profile_elapsed(t0, profile_timestamp()));

    const int k_debug_step = this->debug_step;

    auto k_ep = this->ep_ionisation->device_ptr();

    const REAL invratio = k_E_i / TeV;
    const REAL rate = -k_rate_factor / (TeV * std::sqrt(TeV)) *
                      (expint_barry_approx(invratio) / invratio +
                       (k_b_i_expc_i / (invratio + k_c_i)) *
                           expint_barry_approx(invratio + k_c_i));

    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                // get the temperature in eV. TODO: ensure not unit conversion
                // is required
                const REAL TeV = k_TeV;
                const REAL n_SI = k_n[cellx][0][layerx];

                /*
                const REAL invratio = k_E_i / TeV;
                const REAL rate = -k_rate_factor / (TeV * std::sqrt(TeV)) *
                                  (expint_barry_approx(invratio) / invratio +
                                   (k_b_i_expc_i / (invratio + k_c_i)) *
                                       expint_barry_approx(invratio + k_c_i));
                */

                NESO_KERNEL_ASSERT(std::isfinite(rate), k_ep);

                const REAL weight = k_W[cellx][0][layerx];
                // note that the rate will be a positive number, so minus sign
                // here
                REAL deltaweight = -rate * weight * k_dt_SI * n_SI;

                /* Check whether weight is about to drop below zero.
                   If so, flag particle for removal and adjust deltaweight.
                   These particles are removed after the project call.
                */
                // TODO unbreak
                if ((weight + deltaweight) <= 0) {
                  k_ID[cellx][0][layerx] = k_remove_key;
                  // printf("R %4.3e, %4.3e, %4.3e\n", rate, weight, n_SI);
                  deltaweight = -weight;
                }

                // Mutate the weight on the particle
                k_W[cellx][0][layerx] += deltaweight;
                // Set value for fluid density source (num / Nektar unit time)
                k_SD[cellx][0][layerx] = -deltaweight * k_n_scale / k_dt;

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    this->ep_ionisation->check_and_throw("Ionisation rate is not finite.");

    sycl_target->profile_map.inc("NeutralParticleSystem", "Ionisation_Execute",
                                 1, profile_elapsed(t0, profile_timestamp()));
  }
};

#endif
