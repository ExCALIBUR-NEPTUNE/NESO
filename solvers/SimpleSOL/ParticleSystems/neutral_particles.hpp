#ifndef __CHARGED_PARTICLES_H_
#define __CHARGED_PARTICLES_H_

#include <hipSYCL/sycl/handler.hpp>
#include <hipSYCL/sycl/usm.hpp>
#include <nektar_interface/function_evaluation.hpp>
#include <nektar_interface/function_projection.hpp>
#include <nektar_interface/particle_interface.hpp>
#include <nektar_interface/utilities.hpp>
#include <neso_particles.hpp>

// #include <NEC_abstract_reaction/reaction_kernel.hpp>
#include <NEC_abstract_reaction/reaction_controller.hpp>
#include <new>
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
using namespace Nektar::SpatialDomains;

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
   *  Returns true if all boundary conditions on the density fields are
   *  periodic.
   */
  inline bool is_fully_periodic() {
    NESOASSERT(this->fields.count("rho") == 1,
               "Density field not found in fields.");
    auto bcs = this->fields["rho"]->GetBndConditions();
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

  int source_region_count;
  double source_region_offset;
  int source_line_bin_count;
  double particle_source_region_gaussian_width;
  int particle_source_lines_per_gaussian;
  double particle_thermal_velocity;
  double unrotated_x_max;
  double unrotated_y_max;
  std::mt19937 rng_phasespace;
  std::normal_distribution<> velocity_normal_distribution;
  std::vector<std::shared_ptr<ParticleInitialisationLine>> source_lines;
  std::vector<
      std::shared_ptr<SimpleUniformPointSampler<ParticleInitialisationLine>>>
      source_samplers;
  std::shared_ptr<ParticleRemover> particle_remover;

  // Project object to project onto number density and momentum fields
  std::shared_ptr<FieldProject<DisContField>> field_project;
  // Evaluate object to evaluate number density field
  std::shared_ptr<FieldEvaluate<DisContField>> field_evaluate_n;
  // Evaluate object to evaluate temperature field
  std::shared_ptr<FieldEvaluate<DisContField>> field_evaluate_T;

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
  /// Mass of particles
  const double particle_mass = 1.0;
  /// Initial particle weight.
  double particle_weight;
  /// Number density in simulation domain (per specicies)
  double particle_number_density;
  // PARTICLE_ID value used to flag particles for removal from the simulation
  const int particle_remove_key = -1;
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

  double theta;

  // Factors to convert nektar units to units required by ionisation calc
  double t_to_SI;
  double T_to_eV;
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
      : session(session), graph(graph), comm(comm), tol(1.0e-8),
        h5part_exists(false), simulation_time(0.0) {

    this->total_num_particles_added = 0;
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

    // Load scaling parameters from session
    double Rs, pInf, rhoInf, uInf;
    get_from_session(this->session, "GasConstant", Rs, 1.0);
    get_from_session(this->session, "rhoInf", rhoInf, 1.0);
    get_from_session(this->session, "uInf", uInf, 1.0);

    // Ions are Deuterium
    constexpr int nucleons_per_ion = 2;

    // Constants from https://physics.nist.gov
    constexpr double mp_kg = 1.67e-27;
    constexpr double kB_eV_per_K = 8.617333262e-5;
    constexpr double kB_J_per_K = 1.380649e-23;

    // Typical SOL properties
    constexpr double SOL_num_density_SI = 3e18;
    constexpr double SOL_sound_speed_SI = 3e4;

    // Mean molecular mass in kg
    constexpr double mu_SI = nucleons_per_ion * mp_kg;
    // Specific gas constant in J/K/kg
    constexpr double Rs_SI = kB_J_per_K / mu_SI;

    // SI scaling factors
    const double Rs_to_SI = Rs_SI / Rs;
    const double vel_to_SI = SOL_sound_speed_SI / uInf;
    const double T_to_K = vel_to_SI * vel_to_SI / Rs_to_SI;

    // Scaling factors for units required by ionise()
    this->n_to_SI = SOL_num_density_SI / rhoInf;
    this->T_to_eV = T_to_K * kB_eV_per_K;
    // nektar length unit already in m
    double L_to_SI = 1;
    this->t_to_SI = L_to_SI / vel_to_SI;

    // Create ParticleGroup
    ParticleSpec particle_spec{
        ParticleProp(Sym<REAL>("POSITION"), 2, true),
        ParticleProp(Sym<INT>("CELL_ID"), 1, true),
        ParticleProp(Sym<INT>("PARTICLE_ID"), 2),
        ParticleProp(Sym<REAL>("COMPUTATIONAL_WEIGHT"), 1),
        ParticleProp(Sym<INT>("INTERNAL_STATE"), 1),
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

    // setup how particles are added to the domain each time add_particles is
    // called
    get_from_session(this->session, "particle_source_region_count",
                     this->source_region_count, 2);
    get_from_session(this->session, "particle_source_region_offset",
                     this->source_region_offset, 0.2);
    get_from_session(this->session, "particle_source_line_bin_count",
                     this->source_line_bin_count, 4000);
    get_from_session(this->session, "particle_thermal_velocity",
                     this->particle_thermal_velocity, 1.0);
    get_from_session(this->session, "particle_source_region_gaussian_width",
                     this->particle_source_region_gaussian_width, 0.001);
    get_from_session(this->session, "particle_source_lines_per_gaussian",
                     this->particle_source_lines_per_gaussian, 3);
    get_from_session(this->session, "theta", this->theta, 0.0);
    get_from_session(this->session, "unrotated_x_max", this->unrotated_x_max,
                     110.0);
    get_from_session(this->session, "unrotated_y_max", this->unrotated_y_max,
                     1.0);

    const double particle_region_volume =
        particle_source_region_gaussian_width * std::pow(L_to_SI, 3) *
        this->unrotated_x_max * this->unrotated_y_max;

    // read or deduce a number density from the configuration file
    this->session->LoadParameter("particle_number_density",
                                 this->particle_number_density);
    if (this->particle_number_density < 0.0) {
      this->particle_weight = 1.0;
      this->particle_number_density =
          this->num_particles / particle_region_volume;
    } else {
      const double number_physical_particles =
          this->particle_number_density * particle_region_volume;
      this->particle_weight =
          (this->num_particles == 0)
              ? 0.0
              : number_physical_particles / this->num_particles;
    }

    // get seed from file
    std::srand(std::time(nullptr));
    int seed;
    get_from_session(this->session, "particle_position_seed", seed,
                     std::rand());

    const long rank = this->sycl_target->comm_pair.rank_parent;
    this->rng_phasespace = std::mt19937(seed + rank);
    this->velocity_normal_distribution =
        std::normal_distribution<>{0, this->particle_thermal_velocity};

    std::vector<std::pair<std::vector<double>, std::vector<double>>>
        region_lines;

    if (this->source_region_count == 1) {

      // TODO move to an end
      const double mid_point_x = 0.5 * this->unrotated_x_max;

      std::vector<double> line_start = {mid_point_x, 0.0};

      std::vector<double> line_end = {mid_point_x, this->unrotated_y_max};

      region_lines.push_back(std::make_pair(line_start, line_end));
    } else if (this->source_region_count == 2) {

      const double lower_x = this->source_region_offset * this->unrotated_x_max;
      const double upper_x =
          (1.0 - this->source_region_offset) * this->unrotated_x_max;
      // lower line
      std::vector<double> line_start0 = {lower_x, 0.0};
      std::vector<double> line_end0 = {lower_x, this->unrotated_y_max};
      region_lines.push_back(std::make_pair(line_start0, line_end0));
      // upper line
      std::vector<double> line_start1 = {upper_x, 0.0};
      std::vector<double> line_end1 = {upper_x, this->unrotated_y_max};

      region_lines.push_back(std::make_pair(line_start1, line_end1));
    } else {
      NESOASSERT(false, "Error creating particle source region lines.");
    }
    // now generate all the region_lines
    const auto theta = this->theta; // make it easier to capture in the lambda
    auto rotate = [theta](auto xy) {
      const auto x = xy[0];
      const auto y = xy[1];
      const auto xt = x * std::cos(theta) - y * std::sin(theta);
      const auto yt = x * std::sin(theta) + y * std::cos(theta);
      xy[0] = xt;
      xy[1] = yt;
      return xy;
    };

    for (auto region_line : region_lines) {
      double sigma =
          this->particle_source_region_gaussian_width * this->unrotated_x_max;
      double pslpg = (double)this->particle_source_lines_per_gaussian;
      for (int line_counter = 0; line_counter < pslpg; ++line_counter) {
        auto line_start = region_line.first;
        auto line_end = region_line.second;
        // i * 2/N - 1 + 1/N
        const auto expx = line_counter * 2 / pslpg - 1.0 + 1.0 / pslpg;
        line_start[0] += boost::math::erf_inv(expx) * 3 * sigma;
        line_end[0] += boost::math::erf_inv(expx) * 3 * sigma;
        // rotate the lines in accordance with the orientation of the flow
        auto rotated_line_start = rotate(line_start);
        auto rotated_line_end = rotate(line_end);

        auto tmp_init = std::make_shared<ParticleInitialisationLine>(
            this->domain, this->sycl_target, rotated_line_start,
            rotated_line_end, this->source_line_bin_count);
        this->source_lines.push_back(tmp_init);
        this->source_samplers.push_back(
            std::make_shared<
                SimpleUniformPointSampler<ParticleInitialisationLine>>(
                this->sycl_target, tmp_init));
      }
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
    this->fields["rho_src"] = rho_src;
    this->fields["rhou_src"] = rhou_src;
    this->fields["rhov_src"] = rhov_src;
    this->fields["E_src"] = E_src;
  }

  /**
   * Setup the evaluation of a number density field.
   *
   * @param n Nektar++ field storing plasma number density.
   */
  inline void setup_evaluate_n(std::shared_ptr<DisContField> n) {
    this->field_evaluate_n = std::make_shared<FieldEvaluate<DisContField>>(
        n, this->particle_group, this->cell_id_translation);
    this->fields["rho"] = n;
  }

  /**
   * Setup the evaluation of a temperature field.
   *
   * @param T Nektar++ field storing plasma energy.
   */
  inline void setup_evaluate_T(std::shared_ptr<DisContField> T) {
    this->field_evaluate_T = std::make_shared<FieldEvaluate<DisContField>>(
        T, this->particle_group, this->cell_id_translation);
    this->fields["T"] = T;
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
    const int total_lines = this->source_lines.size();

    const int num_particles_per_line =
        add_proportion *
        (((double)this->num_particles / ((double)total_lines)));

    const long rank = this->sycl_target->comm_pair.rank_parent;

    std::list<int> point_indices;
    for (int linex = 0; linex < total_lines; linex++) {
      const int N = this->source_samplers[linex]->get_samples(
          num_particles_per_line, point_indices);

      if (N > 0) {
        this->total_num_particles_added += static_cast<uint64_t>(N);

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
          line_distribution[Sym<INT>("INTERNAL_STATE")][px][0] = 0;
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
          Sym<REAL>("POSITION"), Sym<INT>("CELL_ID"),
          Sym<REAL>("COMPUTATIONAL_WEIGHT"), Sym<INT>("INTERNAL_STATE"),
          Sym<REAL>("VELOCITY"), Sym<INT>("NESO_MPI_RANK"),
          Sym<INT>("PARTICLE_ID"), Sym<REAL>("NESO_REFERENCE_POSITIONS"));
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

    const REAL k_lower_bound = 0.0;
    const REAL k_upper_bound = k_lower_bound + this->unrotated_x_max;

    const INT k_remove_key = this->particle_remove_key;

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

    // remove particles marked to remove by the boundary conditions
    remove_marked_particles();
  }

  inline void remove_marked_particles() {
    this->particle_remover->remove(
        this->particle_group, (*this->particle_group)[Sym<INT>("PARTICLE_ID")],
        this->particle_remove_key);
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
    double time_tmp = this->simulation_time;

    std::vector<INT> in_states{1, 0};
    std::vector<INT> out_states{-1, 0, 0};

    while (time_tmp < time_end) {
      const double dt_inner = std::min(dt, time_end - time_tmp);
      this->add_particles(dt_inner / dt);
      this->forward_euler(dt_inner);

      ionise_reaction reactionKernel(
        in_states,
        out_states,
        dt_inner,
        this->t_to_SI,
        this->n_to_SI,
        this->theta,
        this->particle_group
      );

      // this->ionise(dt_inner);
      ReactionController<ionise_reaction> reaction_controller(
        this->particle_group,
        reactionKernel,
        sycl_target
      );

      this->evaluate_fields();
      reaction_controller.apply();

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
   */
  inline void evaluate_fields() {

    NESOASSERT(this->field_evaluate_n != nullptr,
               "FieldEvaluate object is null. Was setup_evaluate_n called?");
    NESOASSERT(this->field_evaluate_T != nullptr,
               "FieldEvaluate object is null. Was setup_evaluate_T called?");

    this->field_evaluate_n->evaluate(Sym<REAL>("ELECTRON_DENSITY"));
    this->field_evaluate_T->evaluate(Sym<REAL>("ELECTRON_TEMPERATURE"));

    // Unit conversion
    auto k_TeV = (*this->particle_group)[Sym<REAL>("ELECTRON_TEMPERATURE")]
                     ->cell_dat.device_ptr();
    auto k_n = (*this->particle_group)[Sym<REAL>("ELECTRON_DENSITY")]
                   ->cell_dat.device_ptr();

    // Unit conversion factors
    double k_T_to_eV = this->T_to_eV;
    double k_n_scale_fac = this->n_to_SI;

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(pl_iter_range),
                             [=](sycl::id<1> idx) {
                               NESO_PARTICLES_KERNEL_START
                               const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                               const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                               k_TeV[cellx][0][layerx] *= k_T_to_eV;
                               k_n[cellx][0][layerx] *= k_n_scale_fac;
                               NESO_PARTICLES_KERNEL_END
                             });
        })
        .wait_and_throw();
  }
};

#endif
