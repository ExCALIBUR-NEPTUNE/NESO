#ifndef __NESOSOLVERS_SIMPLESOL_NEUTRALPARTICLES_HPP__
#define __NESOSOLVERS_SIMPLESOL_NEUTRALPARTICLES_HPP__

#include <algorithm>
#include <cmath>
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
#include <nektar_interface/particle_interface.hpp>
#include <nektar_interface/solver_base/partsys_base.hpp>
#include <nektar_interface/utilities.hpp>
#include <neso_particles.hpp>
#include <particle_utility/particle_initialisation_line.hpp>
#include <particle_utility/position_distribution.hpp>

#include "../../common/solver_utils.hpp"

namespace MR = Nektar::MultiRegions;
namespace NP = NESO::Particles;
namespace SD = Nektar::SpatialDomains;

namespace NESO::Solvers::SimpleSOL {
class NeutralParticleSystem : public NP::PartSysBase {

public:
  static std::string class_name;
  /**
   * @brief Create an instance of this class and initialise it.
   */
  static ParticleSystemSharedPtr
  create(const NP::ParticleReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    ParticleSystemSharedPtr p =
        Nektar::MemoryManager<NeutralParticleSystem>::AllocateSharedPtr(session,
                                                                        graph);
    return p;
  }

protected:
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
   * @param config ParticleReader object.
   * @param name Name of the parameter.
   * @param output Reference to the output variable.
   * @param default Default value if name not found in the session file.
   */
  template <typename T>
  inline void get_from_session(NP::ParticleReaderSharedPtr config,
                               std::string name, T &output, T default_value) {
    if (config->defines_parameter(name)) {
      config->load_parameter(name, output);
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
  double theta;
  double unrotated_x_max;
  double unrotated_y_max;
  std::mt19937 rng_phasespace;
  std::normal_distribution<> velocity_normal_distribution;
  std::vector<std::shared_ptr<ParticleInitialisationLine>> source_lines;
  std::vector<
      std::shared_ptr<SimpleUniformPointSampler<ParticleInitialisationLine>>>
      source_samplers;
  std::shared_ptr<NP::ParticleRemover> particle_remover;

  // Project object to project onto number density and momentum fields
  std::shared_ptr<FieldProject<MR::DisContField>> field_project;
  // Evaluate object to evaluate number density field
  std::shared_ptr<FieldEvaluate<MR::DisContField>> field_evaluate_n;
  // Evaluate object to evaluate temperature field
  std::shared_ptr<FieldEvaluate<MR::DisContField>> field_evaluate_T;

  int debug_write_fields_count;
  std::map<std::string, std::shared_ptr<MR::DisContField>> fields;

public:
  /// Disable (implicit) copies.
  NeutralParticleSystem(const NeutralParticleSystem &st) = delete;
  /// Disable (implicit) copies.
  NeutralParticleSystem &operator=(NeutralParticleSystem const &a) = delete;

  ~NeutralParticleSystem() override = default;
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
  /// Method to apply particle boundary conditions.
  std::shared_ptr<NektarCartesianPeriodic> periodic_bc;
  /// Method to map to/from nektar geometry ids to 0,N-1 used by
  /// NESO-Particles

  // Factors to convert nektar units to units required by ionisation calc
  double t_to_SI;
  double T_to_eV;
  double n_to_SI;

  virtual void init_spec() override;

  /**
   *  Create a new instance.
   *
   *  @param config ParticleReader to use for parameters and simulation
   * specification.
   *  @param graph Nektar++ MeshGraph on which particles exist.
   *  @param comm (optional) MPI communicator to use - default MPI_COMM_WORLD.
   *
   */
  NeutralParticleSystem(NP::ParticleReaderSharedPtr config,
                        SD::MeshGraphSharedPtr graph,
                        MPI_Comm comm = MPI_COMM_WORLD)
      : NP::PartSysBase(config, graph, comm), simulation_time(0.0){};

  /**
   * Setup the projection object to use the following fields.
   *
   * @param rho_src Nektar++ fields to project ionised particle data onto.
   */
  inline void setup_project(std::shared_ptr<MR::DisContField> rho_src,
                            std::shared_ptr<MR::DisContField> rhou_src,
                            std::shared_ptr<MR::DisContField> rhov_src,
                            std::shared_ptr<MR::DisContField> E_src) {
    std::vector<std::shared_ptr<MR::DisContField>> fields = {rho_src, rhou_src,
                                                             rhov_src, E_src};
    this->field_project = std::make_shared<FieldProject<MR::DisContField>>(
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
  inline void setup_evaluate_n(std::shared_ptr<MR::DisContField> n) {
    this->field_evaluate_n = std::make_shared<FieldEvaluate<MR::DisContField>>(
        n, this->particle_group, this->cell_id_translation);
    this->fields["rho"] = n;
  }

  /**
   * Setup the evaluation of a temperature field.
   *
   * @param T Nektar++ field storing plasma energy.
   */
  inline void setup_evaluate_T(std::shared_ptr<MR::DisContField> T) {
    this->field_evaluate_T = std::make_shared<FieldEvaluate<MR::DisContField>>(
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

    std::vector<NP::Sym<NP::REAL>> syms = {NP::Sym<NP::REAL>("SOURCE_DENSITY"),
                                           NP::Sym<NP::REAL>("SOURCE_MOMENTUM"),
                                           NP::Sym<NP::REAL>("SOURCE_MOMENTUM"),
                                           NP::Sym<NP::REAL>("SOURCE_ENERGY")};
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
        (((double)this->num_parts_tot / ((double)total_lines)));

    const long rank = this->sycl_target->comm_pair.rank_parent;

    std::list<int> point_indices;
    for (int linex = 0; linex < total_lines; linex++) {
      const int N = this->source_samplers[linex]->get_samples(
          num_particles_per_line, point_indices);

      if (N > 0) {
        this->total_num_particles_added += static_cast<uint64_t>(N);

        NP::ParticleSet line_distribution(
            N, this->particle_group->get_particle_spec());
        auto src_line = this->source_lines[linex];
        for (int px = 0; px < N; px++) {

          // Get the source point information
          const int point_index = point_indices.back();
          point_indices.pop_back();
          for (int dimx = 0; dimx < 2; dimx++) {
            line_distribution[NP::Sym<NP::REAL>("POSITION")][px][dimx] =
                src_line->point_phys_positions[dimx][point_index];
            line_distribution[NP::Sym<NP::REAL>(
                "NESO_REFERENCE_POSITIONS")][px][dimx] =
                src_line->point_ref_positions[dimx][point_index];
          }
          line_distribution[NP::Sym<NP::INT>("CELL_ID")][px][0] =
              src_line->point_neso_cells[point_index];

          // sample/set the remaining particle properties
          for (int dimx = 0; dimx < 3; dimx++) {
            const double vx =
                velocity_normal_distribution(this->rng_phasespace);
            line_distribution[NP::Sym<NP::REAL>("VELOCITY")][px][dimx] = vx;
          }

          line_distribution[NP::Sym<NP::INT>("PARTICLE_ID")][px][0] = rank;
          line_distribution[NP::Sym<NP::INT>("PARTICLE_ID")][px][1] = px;
          line_distribution[NP::Sym<NP::REAL>("MASS")][px][0] =
              this->particle_mass;
          line_distribution[NP::Sym<NP::REAL>("COMPUTATIONAL_WEIGHT")][px][0] =
              this->particle_weight;
        }

        this->particle_group->add_particles_local(line_distribution);
      }
    }
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

    // Find particles that have travelled outside the domain in the x
    // direction.
    const NP::REAL k_lower_bound = 0.0;
    const NP::REAL k_upper_bound = k_lower_bound + this->unrotated_x_max;
    const INT k_remove_key = this->particle_remove_key;

    NP::particle_loop(
        "NeutralParticleSystem::wall_boundary_conditions", this->particle_group,
        [=](auto k_P, auto k_PARTICLE_ID) {
          const NP::REAL px = k_P.at(0);
          if ((px < k_lower_bound) || (px > k_upper_bound)) {
            // mark the particle as removed
            k_PARTICLE_ID.at(0) = k_remove_key;
          }
        },
        NP::Access::read(NP::Sym<NP::REAL>("POSITION")),
        NP::Access::write(NP::Sym<NP::INT>("PARTICLE_ID")))
        ->execute();

    // remove particles marked to remove by the boundary conditions
    remove_marked_particles();
  }

  inline void remove_marked_particles() {
    this->particle_remover->remove(
        this->particle_group,
        (*this->particle_group)[NP::Sym<NP::INT>("PARTICLE_ID")],
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
    auto t0 = NP::profile_timestamp();
    this->boundary_conditions();
    this->particle_group->hybrid_move();
    this->cell_id_translation->execute();
    this->particle_group->cell_move();
    this->sycl_target->profile_map.inc(
        "NeutralParticleSystem", "transfer_particles", 1,
        NP::profile_elapsed(t0, NP::profile_timestamp()));
  }

  /**
   *  Integrate the particle system forward in time to the requested time
   * using at most the requested time step.
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
    auto t0 = NP::profile_timestamp();
    const double k_dt = dt;
    NP::particle_loop(
        "NeutralParticleSystem::forward_euler", this->particle_group,
        [=](auto k_P, auto k_V) {
          k_P.at(0) += k_dt * k_V.at(0);
          k_P.at(1) += k_dt * k_V.at(1);
        },
        NP::Access::write(NP::Sym<NP::REAL>("POSITION")),
        NP::Access::read(NP::Sym<NP::REAL>("VELOCITY")))
        ->execute();
    sycl_target->profile_map.inc(
        "NeutralParticleSystem", "ForwardEuler_Execute", 1,
        NP::profile_elapsed(t0, NP::profile_timestamp()));
    // positions were written so we apply boundary conditions and move
    // particles between ranks
    this->transfer_particles();
  }

  /**
   *  Get the Sym object for the ParticleDat holding the source for density.
   */
  inline NP::Sym<NP::REAL> get_source_density_sym() {
    return NP::Sym<NP::REAL>("SOURCE_DENSITY");
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

    this->field_evaluate_n->evaluate(NP::Sym<NP::REAL>("ELECTRON_DENSITY"));
    this->field_evaluate_T->evaluate(NP::Sym<NP::REAL>("ELECTRON_TEMPERATURE"));

    // Unit conversion factors
    const double k_T_to_eV = this->T_to_eV;
    const double k_n_scale_fac = this->n_to_SI;

    // Unit conversion
    NP::particle_loop(
        "NeutralParticleSystem::evaluate_fields", this->particle_group,
        [=](auto k_TeV, auto k_n) {
          k_TeV.at(0) *= k_T_to_eV;
          k_n.at(0) *= k_n_scale_fac;
        },
        NP::Access::write(NP::Sym<NP::REAL>("ELECTRON_TEMPERATURE")),
        NP::Access::write(NP::Sym<NP::REAL>("ELECTRON_DENSITY")))
        ->execute();
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

    auto k_cos_theta = std::cos(this->theta);
    auto k_sin_theta = std::sin(this->theta);

    const INT k_remove_key = this->particle_remove_key;

    auto t0 = NP::profile_timestamp();

    NP::particle_loop(
        "NeutralParticleSystem::ionise", this->particle_group,
        [=](auto k_ID, auto k_TeV, auto k_n, auto k_SD, auto k_SE, auto k_SM,
            auto k_V, auto k_W) {
          // get the temperatue in eV. TODO: ensure not unit conversion is
          // required
          const NP::REAL TeV = k_TeV.at(0);
          const NP::REAL n_SI = k_n.at(0);
          const NP::REAL invratio = k_E_i / TeV;
          const NP::REAL rate = -k_rate_factor / (TeV * sycl::sqrt(TeV)) *
                                (expint_barry_approx(invratio) / invratio +
                                 (k_b_i_expc_i / (invratio + k_c_i)) *
                                     expint_barry_approx(invratio + k_c_i));
          const NP::REAL weight = k_W.at(0);
          // note that the rate will be a positive number, so minus sign
          // here
          NP::REAL deltaweight = -rate * weight * k_dt_SI * n_SI;

          /* Check whether weight is about to drop below zero.
             If so, flag particle for removal and adjust deltaweight.
             These particles are removed after the project call.
          */
          if ((weight + deltaweight) <= 0) {
            k_ID.at(0) = k_remove_key;
            deltaweight = -weight;
          }

          // Mutate the weight on the particle
          k_W.at(0) += deltaweight;
          // Set value for fluid density source (num / Nektar unit time)
          k_SD.at(0) = -deltaweight * k_n_scale / k_dt;

          // Compute velocity along the SimpleSOL problem axis.
          // (No momentum coupling in orthogonal dimensions)
          const NP::REAL v_s =
              k_V.at(0) * k_cos_theta + k_V.at(1) * k_sin_theta;

          // Set value for fluid momentum density source
          k_SM.at(0) = k_SD.at(0) * v_s * k_cos_theta;
          k_SM.at(1) = k_SD.at(0) * v_s * k_sin_theta;

          // Set value for fluid energy source
          k_SE.at(0) = k_SD.at(0) * v_s * v_s * 0.5;
        },
        NP::Access::write(NP::Sym<NP::INT>("PARTICLE_ID")),
        NP::Access::read(NP::Sym<NP::REAL>("ELECTRON_TEMPERATURE")),
        NP::Access::read(NP::Sym<NP::REAL>("ELECTRON_DENSITY")),
        NP::Access::write(NP::Sym<NP::REAL>("SOURCE_DENSITY")),
        NP::Access::write(NP::Sym<NP::REAL>("SOURCE_ENERGY")),
        NP::Access::write(NP::Sym<NP::REAL>("SOURCE_MOMENTUM")),
        NP::Access::read(NP::Sym<NP::REAL>("VELOCITY")),
        NP::Access::write(NP::Sym<NP::REAL>("COMPUTATIONAL_WEIGHT")))
        ->execute();

    sycl_target->profile_map.inc(
        "NeutralParticleSystem", "Ionisation_Execute", 1,
        NP::profile_elapsed(t0, NP::profile_timestamp()));
  }

  virtual void set_up_particles() override {
    NP::PartSysBase::set_up_particles();
    this->total_num_particles_added = 0;
    this->debug_write_fields_count = 0;

    // Load scaling parameters from session
    double Rs, pInf, rhoInf, uInf;
    get_from_session(this->config, "GasConstant", Rs, 1.0);
    get_from_session(this->config, "rhoInf", rhoInf, 1.0);
    get_from_session(this->config, "uInf", uInf, 1.0);

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

    this->particle_remover =
        std::make_shared<NP::ParticleRemover>(this->sycl_target);

    this->periodic_bc = std::make_shared<NektarCartesianPeriodic>(
        this->sycl_target, this->graph, this->particle_group->position_dat);
    // setup how particles are added to the domain each time add_particles is
    // called
    get_from_session(this->config, "particle_source_region_count",
                     this->source_region_count, 2);
    get_from_session(this->config, "particle_source_region_offset",
                     this->source_region_offset, 0.2);
    get_from_session(this->config, "particle_source_line_bin_count",
                     this->source_line_bin_count, 4000);
    get_from_session(this->config, "particle_thermal_velocity",
                     this->particle_thermal_velocity, 1.0);
    get_from_session(this->config, "particle_source_region_gaussian_width",
                     this->particle_source_region_gaussian_width, 0.001);
    get_from_session(this->config, "particle_source_lines_per_gaussian",
                     this->particle_source_lines_per_gaussian, 3);
    get_from_session(this->config, "theta", this->theta, 0.0);
    get_from_session(this->config, "unrotated_x_max", this->unrotated_x_max,
                     110.0);
    get_from_session(this->config, "unrotated_y_max", this->unrotated_y_max,
                     1.0);

    const double particle_region_volume =
        particle_source_region_gaussian_width * std::pow(L_to_SI, 3) *
        this->unrotated_x_max * this->unrotated_y_max;

    // read or deduce a number density from the configuration file
    this->config->load_parameter("particle_number_density",
                                 this->particle_number_density);
    if (this->particle_number_density < 0.0) {
      this->particle_weight = 1.0;
      this->particle_number_density =
          this->num_parts_tot / particle_region_volume;
    } else {
      const double number_physical_particles =
          this->particle_number_density * particle_region_volume;
      this->particle_weight =
          (this->num_parts_tot == 0)
              ? 0.0
              : number_physical_particles / this->num_parts_tot;
    }

    // get seed from file
    std::srand(std::time(nullptr));
    int seed;
    get_from_session(this->config, "particle_position_seed", seed, std::rand());

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

    report_param("Num particles added per step per rank (set via " +
                     NP::PartSysBase::NUM_PARTS_TOT_STR + "!)",
                 this->num_parts_tot);
    report_param("Number of (Gaussian) particle source regions",
                 this->source_region_count);
    report_param("Separation between each source and the domain edge (in "
                 "domain lengths)",
                 this->source_region_offset);
    report_param("Width of source regions (in domain lengths)",
                 particle_source_region_gaussian_width);
    report_param("Number of sampling lines per (Gaussian) source",
                 this->source_line_bin_count);
    report_param("Thermal velocity", this->particle_thermal_velocity);

    // Setup particle output
    init_output(
        "SimpleSOL_particle_trajectory.h5part", NP::Sym<NP::REAL>("POSITION"),
        NP::Sym<NP::INT>("CELL_ID"), NP::Sym<NP::REAL>("COMPUTATIONAL_WEIGHT"),
        NP::Sym<NP::REAL>("VELOCITY"), NP::Sym<NP::INT>("NESO_MPI_RANK"),
        NP::Sym<NP::INT>("PARTICLE_ID"),
        NP::Sym<NP::REAL>("NESO_REFERENCE_POSITIONS"));
  }
};
} // namespace NESO::Solvers::SimpleSOL
#endif // __NESOSOLVERS_SIMPLESOL_NEUTRALPARTICLES_HPP__
