#ifndef __CHARGED_PARTICLES_H_
#define __CHARGED_PARTICLES_H_

#include <nektar_interface/function_evaluation.hpp>
#include <nektar_interface/function_projection.hpp>
#include <nektar_interface/particle_interface.hpp>
#include <nektar_interface/utilities.hpp>
#include <neso_particles.hpp>

#include <particle_utility/particle_initialisation_line.hpp>
#include <particle_utility/position_distribution.hpp>

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
  const int ndim = 2;
  bool h5part_exists;
  double simulation_time;

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

  int source_line_count;
  int source_line_bin_count;
  double source_line_offset;
  double particle_thermal_velocity;
  double theta;
  std::mt19937 rng_phasespace;
  std::normal_distribution<> velocity_normal_distribution;
  std::vector<std::shared_ptr<ParticleInitialisationLine>> source_lines;
  std::vector<
      std::shared_ptr<SimpleUniformPointSampler<ParticleInitialisationLine>>>
      source_samplers;
  std::shared_ptr<ParticleRemover> particle_remover;

  // Project object to project onto density and momentum fields
  std::shared_ptr<FieldProject<DisContField>> field_project;
  // Evaluate object to evaluate density field
  std::shared_ptr<FieldEvaluate<DisContField>> field_evaluate_rho;
  // Evaluate object to evaluate temperature field
  std::shared_ptr<FieldEvaluate<DisContField>> field_evaluate_T;

  int debug_write_fields_count;
  std::map<std::string, std::shared_ptr<DisContField>> debug_write_fields;

public:
  /// Disable (implicit) copies.
  NeutralParticleSystem(const NeutralParticleSystem &st) = delete;
  /// Disable (implicit) copies.
  NeutralParticleSystem &operator=(NeutralParticleSystem const &a) = delete;

  /// Global number of particles in the simulation.
  int64_t num_particles;
  /// Average number of particles per cell (element) in the simulation.
  int64_t num_particles_per_cell;
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
  std::shared_ptr<NektarCartesianPeriodic> periodic_bc;
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
        h5part_exists(false), simulation_time(0.0) {

    this->debug_write_fields_count = 0;

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
    this->nektar_graph_local_mapper = std::make_shared<NektarGraphLocalMapperT>(
        this->sycl_target, this->particle_mesh_interface, this->tol);
    this->domain = std::make_shared<Domain>(this->particle_mesh_interface,
                                            this->nektar_graph_local_mapper);

    // Create ParticleGroup
    ParticleSpec particle_spec{
        ParticleProp(Sym<REAL>("POSITION"), 2, true),
        ParticleProp(Sym<INT>("CELL_ID"), 1, true),
        ParticleProp(Sym<INT>("PARTICLE_ID"), 2),
        ParticleProp(Sym<REAL>("COMPUTATIONAL_WEIGHT"), 1),
        ParticleProp(Sym<REAL>("SOURCE_DENSITY"), 1),
        ParticleProp(Sym<REAL>("SOURCE_ENERGY"), 1),
        ParticleProp(Sym<REAL>("SOURCE_MOMENTUM"), 2),
        ParticleProp(Sym<REAL>("ELECTRON_DENSITY"), 1),
        ParticleProp(Sym<REAL>("ELECTRON_TEMPERATURE"), 1),
        ParticleProp(Sym<REAL>("MASS"), 1),
        ParticleProp(Sym<REAL>("VELOCITY"), 3)};

    this->particle_group = std::make_shared<ParticleGroup>(
        this->domain, particle_spec, this->sycl_target);

    this->particle_remover =
        std::make_shared<ParticleRemover>(this->sycl_target);

    // Setup PBC boundary conditions.
    this->periodic_bc = std::make_shared<NektarCartesianPeriodic>(
        this->sycl_target, this->graph, this->particle_group->position_dat);

    // Setup map between cell indices
    this->cell_id_translation = std::make_shared<CellIDTranslation>(
        this->sycl_target, this->particle_group->cell_id_dat,
        this->particle_mesh_interface);

    const double volume = this->periodic_bc->global_extent[0] *
                          this->periodic_bc->global_extent[1];

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

    // setup how particles are added to the domain each time add_particles is
    // called
    get_from_session(this->session, "particle_source_line_count",
                     this->source_line_count, 2);
    get_from_session(this->session, "particle_source_line_bin_count",
                     this->source_line_bin_count, 4000);
    get_from_session(this->session, "particle_source_line_offset",
                     this->source_line_offset, 0.2);
    get_from_session(this->session, "particle_thermal_velocity",
                     this->particle_thermal_velocity, 1.0);
    get_from_session(this->session, "theta", this->theta, 0.0);

    // get seed from file
    std::srand(std::time(nullptr));
    int seed;
    get_from_session(this->session, "particle_position_seed", seed,
                     std::rand());

    const long rank = this->sycl_target->comm_pair.rank_parent;
    this->rng_phasespace = std::mt19937(seed + rank);
    this->velocity_normal_distribution =
        std::normal_distribution<>{0, this->particle_thermal_velocity};

    if (this->source_line_count == 1) {

      // TODO move to an end
      const double mid_point_x = this->periodic_bc->global_origin[0] +
                                 0.5 * this->periodic_bc->global_extent[0];

      std::vector<double> line_start = {mid_point_x,
                                        this->periodic_bc->global_origin[1]};
      std::vector<double> line_end = {mid_point_x,
                                      this->periodic_bc->global_origin[1] +
                                          this->periodic_bc->global_extent[1]};

      auto tmp_init = std::make_shared<ParticleInitialisationLine>(
          this->domain, this->sycl_target, line_start, line_end,
          this->source_line_bin_count);
      this->source_lines.push_back(tmp_init);
      this->source_samplers.push_back(
          std::make_shared<
              SimpleUniformPointSampler<ParticleInitialisationLine>>(
              this->sycl_target, tmp_init));

    } else if (this->source_line_count == 2) {

      const double lower_x =
          this->periodic_bc->global_origin[0] +
          this->source_line_offset * this->periodic_bc->global_extent[0];
      const double upper_x = this->periodic_bc->global_origin[0] +
                             (1.0 - this->source_line_offset) *
                                 this->periodic_bc->global_extent[0];

      // lower line
      std::vector<double> line_start0 = {lower_x,
                                         this->periodic_bc->global_origin[1]};
      std::vector<double> line_end0 = {lower_x,
                                       this->periodic_bc->global_origin[1] +
                                           this->periodic_bc->global_extent[1]};

      auto tmp_init0 = std::make_shared<ParticleInitialisationLine>(
          this->domain, this->sycl_target, line_start0, line_end0,
          this->source_line_bin_count);
      this->source_lines.push_back(tmp_init0);
      this->source_samplers.push_back(
          std::make_shared<
              SimpleUniformPointSampler<ParticleInitialisationLine>>(
              this->sycl_target, tmp_init0));

      // upper line
      std::vector<double> line_start1 = {upper_x,
                                         this->periodic_bc->global_origin[1]};
      std::vector<double> line_end1 = {upper_x,
                                       this->periodic_bc->global_origin[1] +
                                           this->periodic_bc->global_extent[1]};

      auto tmp_init1 = std::make_shared<ParticleInitialisationLine>(
          this->domain, this->sycl_target, line_start1, line_end1,
          this->source_line_bin_count);
      this->source_lines.push_back(tmp_init1);
      this->source_samplers.push_back(
          std::make_shared<
              SimpleUniformPointSampler<ParticleInitialisationLine>>(
              this->sycl_target, tmp_init1));

    } else {
      NESOASSERT(false, "Error creating particle source lines.");
    }
  };

  /**
   * Setup the projection object to use the following fields.
   *
   * @param rho_src Nektar++ fields to project ionised particle data onto.
   */
  inline void setup_project(std::shared_ptr<DisContField> rho_src,
                            std::shared_ptr<DisContField> rhou_src,
                            std::shared_ptr<DisContField> rhov_src,
                            std::shared_ptr<DisContField> E_src) {
    std::vector<std::shared_ptr<DisContField>> fields = {rho_src, rhou_src,
                                                         rhov_src, E_src};
    this->field_project = std::make_shared<FieldProject<DisContField>>(
        fields, this->particle_group, this->cell_id_translation);

    // Setup debugging output for each field
    this->debug_write_fields["rho_src"] = rho_src;
    this->debug_write_fields["rhou_src"] = rhou_src;
    this->debug_write_fields["rhov_src"] = rhov_src;
    this->debug_write_fields["E_src"] = E_src;
  }

  /**
   * Setup the evaluation of a density field.
   *
   * @param rho Nektar++ field storing plasma density.
   */
  inline void setup_evaluate_rho(std::shared_ptr<DisContField> rho) {
    this->field_evaluate_rho = std::make_shared<FieldEvaluate<DisContField>>(
        rho, this->particle_group, this->cell_id_translation);
  }

  /**
   * Setup the evaluation of a temperature field.
   *
   * @param E Nektar++ field storing plasma energy.
   */
  inline void setup_evaluate_T(std::shared_ptr<DisContField> E) {
    this->field_evaluate_T = std::make_shared<FieldEvaluate<DisContField>>(
        E, this->particle_group, this->cell_id_translation);
  }

  /**
   *  Project the plasma source and momentum contributions from particle data
   *  onto field data.
   */
  inline void project_source_terms() {
    NESOASSERT(this->field_project != nullptr,
               "Field project object is null. Was setup_project called?");

    std::vector<Sym<REAL>> syms = {
        Sym<REAL>("SOURCE_DENSITY"), Sym<REAL>("SOURCE_MOMENTUM"),
        Sym<REAL>("SOURCE_MOMENTUM"), Sym<REAL>("SOURCE_ENERGY")};
    std::vector<int> components = {0, 0, 1, 0};
    this->field_project->project(syms, components);
  }

  /**
   * Add particles to the simulation.
   *
   * @param add_proportion Specifies the proportion of the number of particles
   * added in a time step.
   */
  inline void add_particles(const double add_proportion) {

    const int num_particles_per_line =
        add_proportion *
        (((double)this->num_particles / ((double)this->source_line_count)));

    const long rank = this->sycl_target->comm_pair.rank_parent;

    std::list<int> point_indices;
    for (int linex = 0; linex < this->source_line_count; linex++) {
      const int N = this->source_samplers[linex]->get_samples(
          num_particles_per_line, point_indices);

      if (N > 0) {
        ParticleSet line_distribution(
            N, this->particle_group->get_particle_spec());
        auto src_line = this->source_lines[linex];
        for (int px = 0; px < N; px++) {

          // Get the source point information
          const int point_index = point_indices.back();
          point_indices.pop_back();
          for (int dimx = 0; dimx < 2; dimx++) {
            line_distribution[Sym<REAL>("POSITION")][px][dimx] =
                src_line->point_phys_positions[dimx][point_index];
            line_distribution[Sym<REAL>("NESO_REFERENCE_POSITIONS")][px][dimx] =
                src_line->point_ref_positions[dimx][point_index];
          }
          line_distribution[Sym<INT>("CELL_ID")][px][0] =
              src_line->point_neso_cells[point_index];

          // sample/set the remaining particle properties
          for (int dimx = 0; dimx < 3; dimx++) {
            const double vx =
                velocity_normal_distribution(this->rng_phasespace);
            line_distribution[Sym<REAL>("VELOCITY")][px][dimx] = vx;
          }

          line_distribution[Sym<INT>("PARTICLE_ID")][px][0] = rank;
          line_distribution[Sym<INT>("PARTICLE_ID")][px][1] = px;
          line_distribution[Sym<REAL>("MASS")][px][0] = this->particle_mass;
          line_distribution[Sym<REAL>("COMPUTATIONAL_WEIGHT")][px][0] =
              this->particle_weight;
        }

        this->particle_group->add_particles_local(line_distribution);
      }
    }
  }

  /**
   *  Write current particle state to trajectory.
   *
   *  @param step Time step number.
   */
  inline void write(const int step) {

    if (this->sycl_target->comm_pair.rank_parent == 0) {
      nprint("Writing particle trajectory:", step);
    }

    if (!this->h5part_exists) {
      // Create instance to write particle data to h5 file
      this->h5part = std::make_shared<H5Part>(
          "SimpleSOL_particle_trajectory.h5part", this->particle_group,
          Sym<REAL>("POSITION"), Sym<INT>("CELL_ID"), Sym<REAL>("VELOCITY"),
          Sym<INT>("NESO_MPI_RANK"), Sym<INT>("PARTICLE_ID"),
          Sym<REAL>("NESO_REFERENCE_POSITIONS"));
      this->h5part_exists = true;
    }

    this->h5part->write();
  }

  /**
   *  Write the projection fields to vtu for debugging.
   */
  inline void write_source_fields() {
    for (auto entry : this->debug_write_fields) {
      std::string filename = "debug_" + entry.first + "_" +
                             std::to_string(this->debug_write_fields_count++) +
                             ".vtu";
      write_vtu(entry.second, filename);
    }
  }

  /**
   * Apply boundary conditions to particles that have travelled over the x
   * extents.
   */
  inline void wall_boundary_conditions() {

    // Find particles that have travelled outside the domain in the x direction.
    auto k_P =
        (*this->particle_group)[Sym<REAL>("POSITION")]->cell_dat.device_ptr();
    // reuse this dat for remove flags
    auto k_PARTICLE_ID =
        (*this->particle_group)[Sym<INT>("PARTICLE_ID")]->cell_dat.device_ptr();

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    const REAL k_lower_bound = this->periodic_bc->global_origin[0];
    const REAL k_upper_bound =
        k_lower_bound + this->periodic_bc->global_extent[0];

    // Particles that are to be removed are marked with this value in the
    // PARTICLE_ID dat.
    const INT k_remove_key = -1;

    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                const REAL px = k_P[cellx][0][layerx];
                if ((px < k_lower_bound) || (px > k_upper_bound)) {
                  // mark the particle as removed
                  k_PARTICLE_ID[cellx][0][layerx] = k_remove_key;
                }
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    // remove the departing particles from the simulation
    this->particle_remover->remove(
        this->particle_group, (*this->particle_group)[Sym<INT>("PARTICLE_ID")],
        k_remove_key);
  }

  /**
   *  Apply the boundary conditions to the particle system.
   */
  inline void boundary_conditions() {
    this->wall_boundary_conditions();
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
    double time_tmp = this->simulation_time;
    while (time_tmp < time_end) {
      const double dt_inner = std::min(dt, time_end - time_tmp);
      this->add_particles(dt_inner / dt);
      this->forward_euler(dt_inner);
      this->ionise(dt_inner);
      time_tmp += dt_inner;
    }

    this->simulation_time = time_end;
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
   *  Get the Sym object for the ParticleDat holding the source for density.
   */
  inline Sym<REAL> get_source_density_sym() {
    return Sym<REAL>("SOURCE_DENSITY");
  }

  /**
   *  Evaluate the density and temperature fields at the particle locations.
   * Values are placed in ELECTRON_DENSITY and ELECTRON_TEMPERATURE
   * respectively.
   *
   *  TODO unit conversion.
   */
  inline void evaluate_fields() {

    NESOASSERT(this->field_evaluate_rho != nullptr,
               "FieldEvaluate object is null. Was setup_evaluate_rho called?");
    NESOASSERT(this->field_evaluate_T != nullptr,
               "FieldEvaluate object is null. Was setup_evaluate_T called?");

    this->field_evaluate_rho->evaluate(Sym<REAL>("ELECTRON_DENSITY"));
    this->field_evaluate_T->evaluate(Sym<REAL>("ELECTRON_TEMPERATURE"));
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

    auto k_cos_theta = std::cos(this->theta);
    auto k_sin_theta = std::sin(this->theta);

    auto t0 = profile_timestamp();

    auto k_TeV = (*this->particle_group)[Sym<REAL>("ELECTRON_TEMPERATURE")]
                     ->cell_dat.device_ptr();
    auto k_rho = (*this->particle_group)[Sym<REAL>("ELECTRON_DENSITY")]
                     ->cell_dat.device_ptr();
    auto k_SD = (*this->particle_group)[Sym<REAL>("SOURCE_DENSITY")]
                    ->cell_dat.device_ptr();
    auto k_SE = (*this->particle_group)[Sym<REAL>("SOURCE_ENERGY")]
                    ->cell_dat.device_ptr();
    auto k_SM = (*this->particle_group)[Sym<REAL>("SOURCE_MOMENTUM")]
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

    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                // get the temperatue in eV. TODO: ensure not unit conversion is
                // required
                const REAL TeV = k_TeV[cellx][0][layerx];
                const REAL rho = k_rho[cellx][0][layerx];
                const REAL invratio = k_E_i / TeV;
                const REAL rate = -k_rate_factor / (TeV * std::sqrt(TeV)) *
                                  (expint_barry_approx(invratio) / invratio +
                                   (k_b_i_expc_i / (invratio + k_c_i)) *
                                       expint_barry_approx(invratio + k_c_i));
                const REAL weight = k_W[cellx][0][layerx];
                // note that the rate will be a positive number, so minus sign
                // here
                const REAL deltaweight = -rate * k_dt * rho;
                k_W[cellx][0][layerx] += deltaweight;

                // Set value for fluid density source
                k_SD[cellx][0][layerx] = -deltaweight;

                // Compute velocity along the SimpleSOL problem axis.
                // (No momentum coupling in orthogonal dimensions)
                const REAL v_s = k_V[cellx][0][layerx] * k_cos_theta +
                                 k_V[cellx][1][layerx] * k_sin_theta;
                // Set value for fluid momentum density source
                k_SM[cellx][0][layerx] =
                    k_SD[cellx][0][layerx] * v_s * k_cos_theta;
                k_SM[cellx][1][layerx] =
                    k_SD[cellx][0][layerx] * v_s * k_sin_theta;
                // Set value for fluid energy source
                k_SE[cellx][1][layerx] = k_SD[cellx][0][layerx] * v_s * v_s / 2;
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target->profile_map.inc("NeutralParticleSystem", "Ionisation_Execute",
                                 1, profile_elapsed(t0, profile_timestamp()));
  }
};

#endif
