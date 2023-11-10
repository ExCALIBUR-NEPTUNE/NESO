#ifndef H3LAPD_NEUTRAL_PARTICLE_SYSTEM_H
#define H3LAPD_NEUTRAL_PARTICLE_SYSTEM_H

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

namespace LU = Nektar::LibUtilities;
namespace NP = NESO::Particles;
namespace SD = Nektar::SpatialDomains;

namespace NESO::Solvers::H3LAPD {

// TODO move this to the correct place
/**
 * @brief Evaluate the Barry et al approximation to the exponential integral
 * function https://en.wikipedia.org/wiki/Exponential_integral E_1(x)
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

/**
 * @brief System of Neutral particles that can be coupled to equation systems
 * inheriting from NESO::Solvers::H3LAPD::LAPDSystem.
 */
class NeutralParticleSystem {
public:
  /**
   *  Create a new instance.
   *
   *  @param session Nektar++ session to use for parameters and simulation
   * specification.
   *  @param graph Nektar++ MeshGraph on which particles exist.
   *  @param comm (optional) MPI communicator to use - default MPI_COMM_WORLD.
   *
   */
  NeutralParticleSystem(LU::SessionReaderSharedPtr session,
                        SD::MeshGraphSharedPtr graph,
                        MPI_Comm comm = MPI_COMM_WORLD)
      : m_session(session), m_graph(graph), m_comm(comm),
        m_ndim(graph->GetSpaceDimension()), m_h5part_exists(false),
        m_simulation_time(0.0), m_total_num_particles_added(0) {

    m_debug_write_fields_count = 0;

    // Set plasma temperature from session param
    get_from_session(m_session, "Te_eV", m_TeV, 10.0);
    // Set background density from session param
    get_from_session(m_session, "n_bg_SI", m_n_bg_SI, 1e18);

    // Read the number of particles per cell / total number of particles
    int tmp_int;
    m_session->LoadParameter("num_particles_per_cell", tmp_int, -1);
    m_num_particles_per_cell = tmp_int;
    m_session->LoadParameter("num_particles_total", tmp_int, -1);
    m_num_particles = tmp_int;

    if (m_num_particles > 0) {
      if (m_num_particles_per_cell > 0) {
        nprint("Ignoring value of 'num_particles_per_cell' because  "
               "'num_particles_total' was specified.");
        m_num_particles_per_cell = -1;
      }
    } else {
      if (m_num_particles_per_cell > 0) {
        // Reduce the global number of elements
        const int num_elements_local = m_graph->GetNumElements();
        int num_elements_global;
        MPICHK(MPI_Allreduce(&num_elements_local, &num_elements_global, 1,
                             MPI_INT, MPI_SUM, m_comm));

        // compute the global number of particles
        m_num_particles =
            ((int64_t)num_elements_global) * m_num_particles_per_cell;
      } else {
        nprint("Neutral particles disabled (Neither 'num_particles_total' or "
               "'num_particles_per_cell' are set)");
      }
    }

    // Create interface between particles and nektar++
    m_particle_mesh_interface =
        std::make_shared<ParticleMeshInterface>(m_graph, 0, m_comm);
    m_sycl_target =
        std::make_shared<SYCLTarget>(0, m_particle_mesh_interface->get_comm());
    m_nektar_graph_local_mapper = std::make_shared<NektarGraphLocalMapper>(
        m_sycl_target, m_particle_mesh_interface);
    m_domain = std::make_shared<Domain>(m_particle_mesh_interface,
                                        m_nektar_graph_local_mapper);

    // SI scaling factors required by ionise()
    m_session->LoadParameter("n_to_SI", m_n_to_SI, 1e17);
    m_session->LoadParameter("t_to_SI", m_t_to_SI, 1e-3);

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

    m_particle_group =
        std::make_shared<ParticleGroup>(m_domain, particle_spec, m_sycl_target);

    m_particle_remover = std::make_shared<ParticleRemover>(m_sycl_target);

    // Set up periodic boundary conditions.
    m_periodic_bc = std::make_shared<NektarCartesianPeriodic>(
        m_sycl_target, m_graph, m_particle_group->position_dat);

    // Set up map between cell indices
    m_cell_id_translation = std::make_shared<CellIDTranslation>(
        m_sycl_target, m_particle_group->cell_id_dat,
        m_particle_mesh_interface);

    // Set properties that affect the behaviour of add_particles()
    get_from_session(m_session, "particle_thermal_velocity",
                     m_particle_thermal_velocity, 1.0);
    get_from_session(m_session, "particle_drift_velocity",
                     m_particle_drift_velocity, 0.0);

    // Set particle region = domain volume for now
    double particle_region_volume = m_periodic_bc->global_extent[0];
    for (auto idim = 1; idim < m_ndim; idim++) {
      particle_region_volume *= m_periodic_bc->global_extent[idim];
    }

    // read or deduce a number density from the configuration file
    get_from_session(m_session, "particle_number_density",
                     m_particle_number_density, -1.0);
    if (m_particle_number_density < 0.0) {
      m_particle_init_weight = 1.0;
      m_particle_number_density = m_num_particles / particle_region_volume;
    } else {
      const double num_phys_particles =
          m_particle_number_density * particle_region_volume;
      m_particle_init_weight =
          (m_num_particles == 0) ? 0.0 : num_phys_particles / m_num_particles;
    }

    // get seed from file
    std::srand(std::time(nullptr));

    get_from_session(m_session, "particle_position_seed", m_seed, std::rand());

    const long rank = m_sycl_target->comm_pair.rank_parent;
    m_rng_phasespace = std::mt19937(m_seed + rank);
  };

  /// Disable (implicit) copies.
  NeutralParticleSystem(const NeutralParticleSystem &st) = delete;
  /// Disable (implicit) copies.
  NeutralParticleSystem &operator=(NeutralParticleSystem const &a) = delete;

  /// Factor to convert nektar density units to SI (required by ionisation calc)
  double m_n_to_SI;
  /// Global number of particles in the simulation.
  int64_t m_num_particles;
  /// NESO-Particles ParticleGroup containing charged particles.
  ParticleGroupSharedPtr m_particle_group;
  /// Initial particle weight.
  double m_particle_init_weight;
  /// Compute target.
  SYCLTargetSharedPtr m_sycl_target;
  /// Total number of particles added on this MPI rank.
  uint64_t m_total_num_particles_added;

  /**
   *  Free the object before MPI_Finalize is called.
   */
  inline void free() {
    if (m_h5part_exists) {
      m_h5part->close();
    }
    m_particle_group->free();
    m_particle_mesh_interface->free();
    m_sycl_target->free();
  };

  /**
   *  Integrate the particle system forward to the requested time using
   *  (at most) the requested time step.
   *
   *  @param time_end Target time to integrate to.
   *  @param dt Time step size.
   */
  inline void integrate(const double time_end, const double dt) {

    // Get the current simulation time.
    NESOASSERT(time_end >= m_simulation_time,
               "Cannot integrate backwards in time.");
    if (time_end == m_simulation_time || m_num_particles == 0) {
      return;
    }
    if (m_total_num_particles_added == 0) {
      this->add_particles(1.0);
    }

    double time_tmp = m_simulation_time;
    while (time_tmp < time_end) {
      const double dt_inner = std::min(dt, time_end - time_tmp);
      this->forward_euler(dt_inner);
      this->ionise(dt_inner);
      time_tmp += dt_inner;
    }

    m_simulation_time = time_end;
  }

  /**
   *  Project particle source terms onto nektar fields.
   */
  inline void project_source_terms() {
    NESOASSERT(m_field_project != nullptr,
               "Field project object is null. Was setup_project called?");

    std::vector<Sym<REAL>> syms = {Sym<REAL>("SOURCE_DENSITY")};
    std::vector<int> components = {0};
    m_field_project->project(syms, components);
    if (m_low_order_project) {
      FieldUtils::Interpolator interpolator{};
      std::vector<MultiRegions::ExpListSharedPtr> in_exp = {
          m_fields["ne_src_interp"]};
      std::vector<MultiRegions::ExpListSharedPtr> out_exp = {
          m_fields["ne_src"]};
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
  inline void setup_evaluate_ne(std::shared_ptr<DisContField> n) {
    m_field_evaluate_ne = std::make_shared<FieldEvaluate<DisContField>>(
        n, m_particle_group, m_cell_id_translation);
    m_fields["ne"] = n;
  }

  /**
   * Set up the projection object
   *
   * @param ne_src Nektar++ field to project particle source terms onto.
   */
  inline void setup_project(std::shared_ptr<DisContField> ne_src) {
    std::vector<std::shared_ptr<DisContField>> fields = {ne_src};
    m_field_project = std::make_shared<FieldProject<DisContField>>(
        fields, m_particle_group, m_cell_id_translation);

    // Add to local map
    m_fields["ne_src"] = ne_src;
    m_low_order_project = false;
  }

  /**
   * Alternative projection set up which first projects number density onto
   * @p ne_src_interp then interpolates onto @p ne_src.
   *
   * @param ne_src_interp Nektar++ field to project particle source terms onto.
   * @param ne_src Nektar++ field to interpolate the projected source terms
   * onto.
   */
  inline void setup_project(std::shared_ptr<DisContField> ne_src_interp,
                            std::shared_ptr<DisContField> ne_src) {
    std::vector<std::shared_ptr<DisContField>> fields = {ne_src_interp};
    m_field_project = std::make_shared<FieldProject<DisContField>>(
        fields, m_particle_group, m_cell_id_translation);

    // Add to local map
    m_fields["ne_src_interp"] = ne_src_interp;
    m_fields["ne_src"] = ne_src;
    m_low_order_project = true;
  }

  /**
   *  Write current particle state to trajectory output file.
   *
   *  @param step Time step number.
   */
  inline void write(const int step) {

    if (m_sycl_target->comm_pair.rank_parent == 0) {
      nprint("Writing particle trajectories at step", step);
    }

    if (!m_h5part_exists) {
      // Create instance to write particle data to h5 file
      m_h5part = std::make_shared<H5Part>(
          "particle_trajectory.h5part", m_particle_group, Sym<REAL>("POSITION"),
          Sym<INT>("CELL_ID"), Sym<REAL>("COMPUTATIONAL_WEIGHT"),
          Sym<REAL>("VELOCITY"), Sym<INT>("PARTICLE_ID"));
      m_h5part_exists = true;
    }

    m_h5part->write();
  }

  /**
   *  Write the projection fields to vtu for debugging.
   */
  inline void write_source_fields() {
    for (auto entry : m_fields) {
      std::string filename = "debug_" + entry.first + "_" +
                             std::to_string(m_debug_write_fields_count++) +
                             ".vtu";
      write_vtu(entry.second, filename, entry.first);
    }
  }

protected:
  /// Object used to map to/from nektar geometry ids to 0,N-1
  std::shared_ptr<CellIDTranslation> m_cell_id_translation;
  /// NESO-Particles domain.
  DomainSharedPtr m_domain;
  /// Trajectory writer for particles
  std::shared_ptr<H5Part> m_h5part;
  /// Assumed background density in SI units, read from session
  double m_n_bg_SI;
  /// Mapping instance to map particles into nektar++ elements.
  std::shared_ptr<NektarGraphLocalMapper> m_nektar_graph_local_mapper;
  /// Average number of particles per cell (element) in the simulation.
  int64_t m_num_particles_per_cell;
  /// Particle drift velocity
  double m_particle_drift_velocity;
  /// Initial particle velocity.
  double m_particle_init_vel;
  /// Mass of particles
  const double m_particle_mass = 1.0;
  /// HMesh instance that allows particles to move over nektar++ meshes.
  ParticleMeshInterfaceSharedPtr m_particle_mesh_interface;
  /// Number density in simulation domain
  double m_particle_number_density;
  /// PARTICLE_ID value used to flag particles for removal from the simulation
  const int m_particle_remove_key = -1;
  /// Particle thermal velocity
  double m_particle_thermal_velocity;
  /// Object used to apply particle boundary conditions
  std::shared_ptr<NektarCartesianPeriodic> m_periodic_bc;
  // Random seed used in particle initialisation
  int m_seed;
  /// Factor to convert nektar time units to SI (required by ionisation calc)
  double m_t_to_SI;
  /// Temperature assumed for ionisation rate, read from session
  double m_TeV;

  /// MPI communicator
  MPI_Comm m_comm;
  /// Counter used to name debugging output files
  int m_debug_write_fields_count;
  /// Object used to evaluate Nektar number density field
  std::shared_ptr<FieldEvaluate<DisContField>> m_field_evaluate_ne;
  /// Object used to project onto Nektar number density field
  std::shared_ptr<FieldProject<DisContField>> m_field_project;
  /// Map of pointers to Nektar fields coupled via evaluation and/or projection
  std::map<std::string, std::shared_ptr<DisContField>> m_fields;
  /// Pointer to Nektar Meshgraph object
  SD::MeshGraphSharedPtr m_graph;
  /// Variable to track existence of output data file
  bool m_h5part_exists;
  /// Variable to toggle use of low order projection
  bool m_low_order_project;
  /// Number of spatial dimensions being used
  const int m_ndim;
  /// Object to handle particle removal
  std::shared_ptr<ParticleRemover> m_particle_remover;
  /// Random number generator
  std::mt19937 m_rng_phasespace;
  /// Pointer to Session object
  LU::SessionReaderSharedPtr m_session;
  /// Simulation time
  double m_simulation_time;

  /**
   * Add particles to the simulation.
   *
   * @param add_proportion Specifies the proportion of the number of particles
   * added in a time step.
   */
  inline void add_particles(const double add_proportion) {
    long rstart, rend;
    const long size = m_sycl_target->comm_pair.size_parent;
    const long rank = m_sycl_target->comm_pair.rank_parent;

    const long num_particles_to_add =
        std::round(add_proportion * ((double)m_num_particles));

    get_decomp_1d(size, num_particles_to_add, rank, &rstart, &rend);
    const long N = rend - rstart;
    const int cell_count = m_domain->mesh->get_cell_count();

    if (N > 0) {
      // Generate N particles
      ParticleSet initial_distribution(N,
                                       m_particle_group->get_particle_spec());

      // Generate particle positions and velocities
      std::vector<std::vector<double>> positions, velocities;

      // Positions are Gaussian, centred at origin, same width in all dims
      double mu = 0.0;
      double sigma;
      get_from_session(m_session, "particle_source_width", sigma, 0.5);
      positions = NESO::Particles::normal_distribution(N, m_ndim, mu, sigma,
                                                       m_rng_phasespace);
      // Centre of distribution
      std::vector<double> offsets = {
          0.0, 0.0,
          (m_periodic_bc->global_extent[2] - m_periodic_bc->global_origin[2]) /
              2};

      velocities = NESO::Particles::normal_distribution(
          N, m_ndim, m_particle_drift_velocity, m_particle_thermal_velocity,
          m_rng_phasespace);

      // Set positions, velocities
      for (int ipart = 0; ipart < N; ipart++) {
        for (int idim = 0; idim < m_ndim; idim++) {
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
            m_particle_init_weight;
        initial_distribution[Sym<REAL>("MASS")][ipart][0] = m_particle_mass;
        initial_distribution[Sym<INT>("PARTICLE_ID")][ipart][0] =
            ipart + rstart + m_total_num_particles_added;
      }
      m_particle_group->add_particles_local(initial_distribution);
    }
    m_total_num_particles_added += num_particles_to_add;

    parallel_advection_initialisation(m_particle_group);
    parallel_advection_store(m_particle_group);

    const int num_steps = 20;
    for (int stepx = 0; stepx < num_steps; stepx++) {
      parallel_advection_step(m_particle_group, num_steps, stepx);
      this->transfer_particles();
    }
    parallel_advection_restore(m_particle_group);

    // Move particles to the owning ranks and correct cells.
    this->transfer_particles();
  }

  /**
   *  Apply the boundary conditions to the particle system.
   */
  inline void boundary_conditions() {
    NESOASSERT(this->is_fully_periodic(),
               "NeutralParticleSystem: Only fully periodic BCs are supported.");
    m_periodic_bc->execute();
  }

  /**
   *  Evaluate fields at the particle locations.
   */
  inline void evaluate_fields() {

    NESOASSERT(m_field_evaluate_ne != nullptr,
               "FieldEvaluate object is null. Was setup_evaluate_ne called?");

    m_field_evaluate_ne->evaluate(Sym<REAL>("ELECTRON_DENSITY"));

    // Particle property to update
    auto k_n = (*m_particle_group)[Sym<REAL>("ELECTRON_DENSITY")]
                   ->cell_dat.device_ptr();

    auto k_n_bg_SI = m_n_bg_SI;

    // Unit conversion factors
    double k_n_to_SI = m_n_to_SI;

    const auto pl_iter_range =
        m_particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        m_particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        m_particle_group->mpi_rank_dat->get_particle_loop_npart_cell();
    m_sycl_target->queue
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
   * Apply Forward-Euler, which with no forces is trivial.
   *
   * @param dt Time step size.
   */
  inline void forward_euler(const double dt) {

    const double k_dt = dt;

    auto t0 = profile_timestamp();

    auto k_P =
        (*m_particle_group)[Sym<REAL>("POSITION")]->cell_dat.device_ptr();
    auto k_V =
        (*m_particle_group)[Sym<REAL>("VELOCITY")]->cell_dat.device_ptr();

    const auto pl_iter_range =
        m_particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        m_particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        m_particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    m_sycl_target->profile_map.inc("NeutralParticleSystem",
                                   "ForwardEuler_Prepare", 1,
                                   profile_elapsed(t0, profile_timestamp()));

    m_sycl_target->queue
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
    m_sycl_target->profile_map.inc("NeutralParticleSystem",
                                   "ForwardEuler_Execute", 1,
                                   profile_elapsed(t0, profile_timestamp()));

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
  inline void get_from_session(LU::SessionReaderSharedPtr session,
                               std::string name, T &output, T default_value) {
    if (session->DefinesParameter(name)) {
      session->LoadParameter(name, output);
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
    const double k_dt_SI = dt * m_t_to_SI;
    const double k_n_scale = 1 / m_n_to_SI;

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

    const INT k_remove_key = m_particle_remove_key;

    auto t0 = profile_timestamp();

    auto k_ID =
        (*m_particle_group)[Sym<INT>("PARTICLE_ID")]->cell_dat.device_ptr();
    auto k_n = (*m_particle_group)[Sym<REAL>("ELECTRON_DENSITY")]
                   ->cell_dat.device_ptr();
    auto k_SD =
        (*m_particle_group)[Sym<REAL>("SOURCE_DENSITY")]->cell_dat.device_ptr();

    auto k_V =
        (*m_particle_group)[Sym<REAL>("VELOCITY")]->cell_dat.device_ptr();
    auto k_W = (*m_particle_group)[Sym<REAL>("COMPUTATIONAL_WEIGHT")]
                   ->cell_dat.device_ptr();

    const auto pl_iter_range =
        m_particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        m_particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        m_particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    m_sycl_target->profile_map.inc("NeutralParticleSystem",
                                   "Ionisation_Prepare", 1,
                                   profile_elapsed(t0, profile_timestamp()));

    const REAL invratio = k_E_i / m_TeV;
    const REAL rate = -k_rate_factor / (m_TeV * std::sqrt(m_TeV)) *
                      (expint_barry_approx(invratio) / invratio +
                       (k_b_i_expc_i / (invratio + k_c_i)) *
                           expint_barry_approx(invratio + k_c_i));

    m_sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                const REAL n_SI = k_n[cellx][0][layerx];

                const REAL weight = k_W[cellx][0][layerx];
                // note that the rate will be a positive number, so minus sign
                // here
                REAL deltaweight = -rate * weight * k_dt_SI * n_SI;

                /* Check whether weight is about to drop below zero.
                   If so, flag particle for removal and adjust deltaweight.
                   These particles are removed after the project call.
                */
                if ((weight + deltaweight) <= 0) {
                  k_ID[cellx][0][layerx] = k_remove_key;
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

    m_sycl_target->profile_map.inc("NeutralParticleSystem",
                                   "Ionisation_Execute", 1,
                                   profile_elapsed(t0, profile_timestamp()));
  }

  /**
   *  Returns true if all boundary conditions on the density field are
   *  periodic.
   */
  inline bool is_fully_periodic() {
    NESOASSERT(m_fields.count("ne") == 1, "ne field not found in fields.");
    auto bcs = m_fields["ne"]->GetBndConditions();
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
    m_particle_remover->remove(m_particle_group,
                               (*m_particle_group)[Sym<INT>("PARTICLE_ID")],
                               m_particle_remove_key);
  }

  /**
   *  Apply boundary conditions and transfer particles between MPI ranks.
   */
  inline void transfer_particles() {
    auto t0 = profile_timestamp();
    this->boundary_conditions();
    m_particle_group->hybrid_move();
    m_cell_id_translation->execute();
    m_particle_group->cell_move();
    m_sycl_target->profile_map.inc("NeutralParticleSystem",
                                   "transfer_particles", 1,
                                   profile_elapsed(t0, profile_timestamp()));
  }
};
} // namespace NESO::Solvers::H3LAPD
#endif // H3LAPD_NEUTRAL_PARTICLE_SYSTEM_H
