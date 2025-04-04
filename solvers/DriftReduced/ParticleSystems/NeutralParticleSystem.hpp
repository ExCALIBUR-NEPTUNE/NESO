#ifndef __NESOSOLVERS_DRIFTREDUCED_NEUTRALPARTICLESYSTEM_HPP__
#define __NESOSOLVERS_DRIFTREDUCED_NEUTRALPARTICLESYSTEM_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <random>

#include <FieldUtils/Interpolator.h>
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

namespace FU = Nektar::FieldUtils;
namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace NP = NESO::Particles;
namespace SD = Nektar::SpatialDomains;

namespace NESO::Solvers::DriftReduced {

constexpr int particle_remove_key = -1;

/**
 * @brief System of Neutral particles that can be coupled to equation systems
 * inheriting from NESO::Solvers::DriftReduced::DriftReducedSystem.
 */
class NeutralParticleSystem : public NP::PartSysBase {

public:
  static std::string class_name;
  /**
   * @brief Create an instance of this class and initialise it.
   */
  static ParticleSystemSharedPtr
  create(const NP::ParticleReaderSharedPtr &config,
         const SD::MeshGraphSharedPtr &graph) {
    ParticleSystemSharedPtr p =
        Nektar::MemoryManager<NeutralParticleSystem>::AllocateSharedPtr(config,
                                                                        graph);
    return p;
  }
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
      : NP::PartSysBase(config, graph, comm){};

  /// Disable (implicit) copies.
  NeutralParticleSystem(const NeutralParticleSystem &st) = delete;
  /// Disable (implicit) copies.
  NeutralParticleSystem &operator=(NeutralParticleSystem const &a) = delete;

  virtual void set_up_particles() override {
    NP::PartSysBase::set_up_particles();
    this->debug_write_fields_count = 0;

    // Set plasma temperature from session param
    get_from_session(this->config, "Te_eV", this->TeV, 10.0);
    // Set background density from session param
    get_from_session(this->config, "n_bg_SI", this->n_bg_SI, 1e18);

    // SI scaling factors required by ionise()
    this->config->load_parameter("n_to_SI", this->n_to_SI, 1e17);
    this->config->load_parameter("t_to_SI", this->t_to_SI, 1e-3);

    this->particle_remover =
        std::make_shared<NP::ParticleRemover>(this->sycl_target);

    // Set up periodic boundary conditions.
    this->periodic_bc = std::make_shared<NektarCartesianPeriodic>(
        this->sycl_target, this->graph, this->particle_group->position_dat);

    // Set properties that affect the behaviour of add_particles()
    get_from_session(this->config, "particle_thermal_velocity",
                     this->particle_thermal_velocity, 1.0);
    get_from_session(this->config, "particle_drift_velocity",
                     this->particle_drift_velocity, 0.0);

    // Set particle region = domain volume for now
    double particle_region_volume = this->periodic_bc->global_extent[0];
    for (auto idim = 1; idim < this->ndim; idim++) {
      particle_region_volume *= this->periodic_bc->global_extent[idim];
    }

    // read or deduce a number density from the configuration file
    get_from_session(this->config, "particle_number_density",
                     this->particle_number_density, -1.0);
    if (this->particle_number_density < 0.0) {
      this->particle_init_weight = 1.0;
      this->particle_number_density =
          this->num_parts_tot / particle_region_volume;
    } else {
      const double num_phys_particles =
          this->particle_number_density * particle_region_volume;
      this->particle_init_weight =
          (this->num_parts_tot == 0) ? 0.0
                                     : num_phys_particles / this->num_parts_tot;
    }

    // get seed from file
    std::srand(std::time(nullptr));

    get_from_session(this->config, "particle_position_seed", this->random_seed,
                     std::rand());

    const long rank = this->sycl_target->comm_pair.rank_parent;
    this->rng_phasespace = std::mt19937(this->random_seed + rank);

    // Set up per-step output
    init_output("DriftReduced_particle_trajectory.h5part",
                NP::Sym<NP::REAL>("POSITION"), NP::Sym<NP::INT>("CELL_ID"),
                NP::Sym<NP::REAL>("COMPUTATIONAL_WEIGHT"),
                NP::Sym<NP::REAL>("VELOCITY"), NP::Sym<NP::INT>("PARTICLE_ID"));
  }
  /// Factor to convert nektar density units to SI (required by ionisation calc)
  double n_to_SI;
  /// Initial particle weight.
  double particle_init_weight;
  /// Total number of particles added on this MPI rank.
  uint64_t total_num_particles_added = 0;

  virtual void init_spec() override;

  /**
   *  Integrate the particle system forward to the requested time using
   *  (at most) the requested time step.
   *
   *  @param time_end Target time to integrate to.
   *  @param dt Time step size.
   */
  inline void integrate(const double time_end, const double dt) {

    // Get the current simulation time.
    NESOASSERT(time_end >= this->simulation_time,
               "Cannot integrate backwards in time.");
    if (time_end == this->simulation_time || this->num_parts_tot == 0) {
      return;
    }
    if (this->total_num_particles_added == 0) {
      this->add_particles(1.0);
    }

    double time_tmp = this->simulation_time;
    while (time_tmp < time_end) {
      const double dt_inner = std::min(dt, time_end - time_tmp);
      this->forward_euler(dt_inner);
      this->ionise(dt_inner);
      time_tmp += dt_inner;
    }

    this->simulation_time = time_end;
  }

  /**
   *  Project particle source terms onto nektar fields.
   */
  inline void project_source_terms() {
    NESOASSERT(this->field_project != nullptr,
               "Field project object is null. Was setup_project called?");

    std::vector<NP::Sym<NP::REAL>> syms = {NP::Sym<NP::REAL>("SOURCE_DENSITY")};
    std::vector<int> components = {0};
    this->field_project->project(syms, components);
    if (this->low_order_project) {
      FU::Interpolator<std::vector<MR::ExpListSharedPtr>> interpolator{};
      std::vector<MR::ExpListSharedPtr> in_exp = {
          this->discont_fields["ne_src_interp"]};
      std::vector<MR::ExpListSharedPtr> out_exp = {
          this->discont_fields["ne_src"]};
      interpolator.Interpolate(in_exp, out_exp);
    }
    // remove fully ionised particles from the simulation
    remove_marked_particles();
  }

  /**
   * Set up the evaluation of a number density field.
   *
   * @param n Nektar++ field storing fluid number density.
   */
  inline void setup_evaluate_ne(std::shared_ptr<MR::DisContField> n) {
    this->field_evaluate_ne = std::make_shared<FieldEvaluate<MR::DisContField>>(
        n, this->particle_group, this->cell_id_translation);
    this->discont_fields["ne"] = n;
  }

  /**
   * Set up the projection object
   *
   * @param ne_src Nektar++ field to project particle source terms onto.
   */
  inline void setup_project(std::shared_ptr<MR::DisContField> ne_src) {
    std::vector<std::shared_ptr<MR::DisContField>> fields = {ne_src};
    this->field_project = std::make_shared<FieldProject<MR::DisContField>>(
        fields, this->particle_group, this->cell_id_translation);

    // Add to local map
    this->discont_fields["ne_src"] = ne_src;
    this->low_order_project = false;
  }

  /**
   * Alternative projection set up which first projects number density onto
   * @p ne_src_interp then interpolates onto @p ne_src.
   *
   * @param ne_src_interp Nektar++ field to project particle source terms onto.
   * @param ne_src Nektar++ field to interpolate the projected source terms
   * onto.
   */
  inline void setup_project(std::shared_ptr<MR::DisContField> ne_src_interp,
                            std::shared_ptr<MR::DisContField> ne_src) {
    std::vector<std::shared_ptr<MR::DisContField>> fields = {ne_src_interp};
    this->field_project = std::make_shared<FieldProject<MR::DisContField>>(
        fields, this->particle_group, this->cell_id_translation);

    // Add to local map
    this->discont_fields["ne_src_interp"] = ne_src_interp;
    this->discont_fields["ne_src"] = ne_src;
    this->low_order_project = true;
  }

  /**
   *  Write the projection fields to vtu for debugging.
   */
  inline void write_source_fields() {
    for (auto entry : this->discont_fields) {
      std::string filename = "debug_" + entry.first + "_" +
                             std::to_string(this->debug_write_fields_count++) +
                             ".vtu";
      write_vtu(entry.second, filename, entry.first);
    }
  }

protected:
  /// Assumed background density in SI units, read from session
  double n_bg_SI;
  /// Particle drift velocity
  double particle_drift_velocity;
  /// Mass of particles
  const double particle_mass = 1.0;
  /// Number density in simulation domain
  double particle_number_density;
  /// PARTICLE_ID value used to flag particles for removal from the simulation
  /// Particle thermal velocity
  double particle_thermal_velocity;
  /// Object used to apply particle boundary conditions
  std::shared_ptr<NektarCartesianPeriodic> periodic_bc;
  // Random seed used in particle initialisation
  int random_seed;
  /// Factor to convert nektar time units to SI (required by ionisation calc)
  double t_to_SI;
  /// Temperature assumed for ionisation rate, read from session
  double TeV;

  /// Counter used to name debugging output files
  int debug_write_fields_count;
  /// Map of pointers to Nektar fields coupled via evaluation and/or projection
  std::map<std::string, std::shared_ptr<MR::DisContField>> discont_fields;
  /// Object used to evaluate Nektar number density field
  std::shared_ptr<FieldEvaluate<MR::DisContField>> field_evaluate_ne;
  /// Object used to project onto Nektar number density field
  std::shared_ptr<FieldProject<MR::DisContField>> field_project;
  /// Variable to toggle use of low order projection
  bool low_order_project;
  /// Object to handle particle removal
  std::shared_ptr<NP::ParticleRemover> particle_remover;
  /// Random number generator
  std::mt19937 rng_phasespace;
  /// Simulation time
  double simulation_time = 0.0;

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
        std::round(add_proportion * ((double)this->num_parts_tot));

    get_decomp_1d(size, num_particles_to_add, rank, &rstart, &rend);
    const long N = rend - rstart;
    const int cell_count = this->domain->mesh->get_cell_count();

    if (N > 0) {
      // Generate N particles
      NP::ParticleSet initial_distribution(
          N, this->particle_group->get_particle_spec());

      // Generate particle positions and velocities
      std::vector<std::vector<double>> positions, velocities;

      // Positions are Gaussian, centred at origin, same width in all dims
      double mu = 0.0;
      double sigma;
      get_from_session(this->config, "particle_source_width", sigma, 0.5);
      positions = NP::normal_distribution(N, this->ndim, mu, sigma,
                                          this->rng_phasespace);
      // Centre of distribution
      std::vector<double> offsets = {0.0, 0.0,
                                     (this->periodic_bc->global_extent[2] -
                                      this->periodic_bc->global_origin[2]) /
                                         2};

      velocities = NP::normal_distribution(
          N, this->ndim, this->particle_drift_velocity,
          this->particle_thermal_velocity, this->rng_phasespace);

      // Set positions, velocities
      for (int ipart = 0; ipart < N; ipart++) {
        for (int idim = 0; idim < this->ndim; idim++) {
          initial_distribution[NP::Sym<NP::REAL>("POSITION")][ipart][idim] =
              positions[idim][ipart] + offsets[idim];
          initial_distribution[NP::Sym<NP::REAL>("VELOCITY")][ipart][idim] =
              velocities[idim][ipart];
        }
      }

      // Set remaining properties
      for (int ipart = 0; ipart < N; ipart++) {
        initial_distribution[NP::Sym<NP::INT>("CELL_ID")][ipart][0] =
            ipart % cell_count;
        initial_distribution[NP::Sym<NP::REAL>("COMPUTATIONAL_WEIGHT")][ipart]
                            [0] = this->particle_init_weight;
        initial_distribution[NP::Sym<NP::REAL>("MASS")][ipart][0] =
            this->particle_mass;
        initial_distribution[NP::Sym<NP::INT>("PARTICLE_ID")][ipart][0] =
            ipart + rstart + this->total_num_particles_added;
      }
      this->particle_group->add_particles_local(initial_distribution);
    }
    this->total_num_particles_added += num_particles_to_add;

    NP::parallel_advection_initialisation(this->particle_group);
    NP::parallel_advection_store(this->particle_group);

    const int num_steps = 20;
    for (int stepx = 0; stepx < num_steps; stepx++) {
      parallel_advection_step(this->particle_group, num_steps, stepx);
      this->transfer_particles();
    }
    NP::parallel_advection_restore(this->particle_group);

    // Move particles to the owning ranks and correct cells.
    this->transfer_particles();
  }

  /**
   *  Apply the boundary conditions to the particle system.
   */
  inline void boundary_conditions() {
    NESOASSERT(this->is_fully_periodic(),
               "NeutralParticleSystem: Only fully periodic BCs are supported.");
    this->periodic_bc->execute();
  }

  /**
   *  Evaluate fields at the particle locations.
   */
  inline void evaluate_fields() {
    NESOASSERT(this->field_evaluate_ne != nullptr,
               "FieldEvaluate object is null. Was setup_evaluate_ne called?");

    this->field_evaluate_ne->evaluate(NP::Sym<NP::REAL>("ELECTRON_DENSITY"));

    // Unit conversion factors
    const double k_n_to_SI = this->n_to_SI;
    const auto k_n_bg_SI = this->n_bg_SI;

    NP::particle_loop(
        "NeutralParticleSystem::evaluate_fields", this->particle_group,
        [=](auto k_n) { k_n.at(0) = k_n_bg_SI + k_n.at(0) * k_n_to_SI; },
        NP::Access::write(NP::Sym<NP::REAL>("ELECTRON_DENSITY")))
        ->execute();
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
          k_P.at(2) += k_dt * k_V.at(2);
        },
        NP::Access::write(NP::Sym<NP::REAL>("POSITION")),
        NP::Access::read(NP::Sym<NP::REAL>("VELOCITY")))
        ->execute();

    this->sycl_target->profile_map.inc(
        "NeutralParticleSystem", "ForwardEuler_Execute", 1,
        NP::profile_elapsed(t0, NP::profile_timestamp()));

    // positions were written so we apply boundary conditions and move
    // particles between ranks
    this->transfer_particles();
  }

  /**
   * Helper function to get values from the session file.
   *
   * @param session Session object.
   * @param name Name of the parameter.
   * @param[out] output Reference to the output variable.
   * @param default_value Default value if name not found in the session file.
   */
  template <typename T>
  inline void get_from_session(NP::ParticleReaderSharedPtr session,
                               std::string name, T &output, T default_value) {
    if (session->defines_parameter(name)) {
      session->load_parameter(name, output);
    } else {
      output = default_value;
    }
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

    const NP::REAL invratio = k_E_i / this->TeV;
    const NP::REAL rate = -k_rate_factor / (this->TeV * std::sqrt(this->TeV)) *
                          (expint_barry_approx(invratio) / invratio +
                           (k_b_i_expc_i / (invratio + k_c_i)) *
                               expint_barry_approx(invratio + k_c_i));
    const NP::INT k_remove_key = particle_remove_key;

    auto t0 = NP::profile_timestamp();

    NP::particle_loop(
        "NeutralParticleSystem::ionise", this->particle_group,
        [=](auto k_ID, auto k_n, auto k_SD, auto k_W) {
          const NP::REAL n_SI = k_n.at(0);
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
        },
        NP::Access::write(NP::Sym<NP::INT>("PARTICLE_ID")),
        NP::Access::read(NP::Sym<NP::REAL>("ELECTRON_DENSITY")),
        NP::Access::write(NP::Sym<NP::REAL>("SOURCE_DENSITY")),
        NP::Access::write(NP::Sym<NP::REAL>("COMPUTATIONAL_WEIGHT")))
        ->execute();

    this->sycl_target->profile_map.inc(
        "NeutralParticleSystem", "Ionisation_Execute", 1,
        NP::profile_elapsed(t0, NP::profile_timestamp()));
  }

  /**
   *  Returns true if all boundary conditions on the density field are
   *  periodic.
   */
  inline bool is_fully_periodic() {
    NESOASSERT(this->discont_fields.count("ne") == 1,
               "ne field not found in fields.");
    auto bcs = this->discont_fields["ne"]->GetBndConditions();
    bool is_pbc = true;
    for (auto &bc : bcs) {
      is_pbc &= (bc->GetBoundaryConditionType() == ePeriodic);
    }
    return is_pbc;
  }

  /**
   * Remove particles from the system whose ID has been flagged with a
   * particular key
   */
  inline void remove_marked_particles() {
    this->particle_remover->remove(
        this->particle_group,
        (*this->particle_group)[NP::Sym<NP::INT>("PARTICLE_ID")],
        particle_remove_key);
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
};
} // namespace NESO::Solvers::DriftReduced
#endif // __NESOSOLVERS_DRIFTREDUCED_NEUTRALPARTICLESYSTEM_HPP__
