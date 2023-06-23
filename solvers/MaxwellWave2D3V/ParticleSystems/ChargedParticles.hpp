#ifndef __CHARGED_PARTICLES_H_
#define __CHARGED_PARTICLES_H_

#include <cmath>
#include <nektar_interface/function_evaluation.hpp>
#include <nektar_interface/function_projection.hpp>
#include <nektar_interface/particle_interface.hpp>
#include <neso_particles.hpp>

#include <particle_utility/position_distribution.hpp>
#include <utilities.hpp>

#include <LibUtilities/BasicUtils/SessionReader.h>

#include <boost/math/special_functions/erf.hpp>

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <random>

#include "../EquationSystems/UnitConverter.hpp"
#include "IntegratorBoris.hpp"
#include "ParallelInitialisation.hpp"
#include "ParticleInitialCondition.hpp"

using namespace Nektar;
using namespace NESO;
using namespace NESO::Particles;

#ifndef PIC_2D3V_CROSS_PRODUCT_3D
#define PIC_2D3V_CROSS_PRODUCT_3D(a1, a2, a3, b1, b2, b3, c1, c2, c3)          \
  (c1) = ((a2) * (b3)) - ((a3) * (b2));                                        \
  (c2) = ((a3) * (b1)) - ((a1) * (b3));                                        \
  (c3) = ((a1) * (b2)) - ((a2) * (b1));
#endif

/**
 * Helper function to get values from the session file.
 *
 * @param session Session object.
 * @param name Name of the parameter.
 * @param output Reference to the output variable.
 * @param default Default value if name not found in the session file.
 */
template <typename T>
inline void
pic2d3v_get_from_session(LibUtilities::SessionReaderSharedPtr session,
                         std::string name, T &output, T default_value) {
  if (session->DefinesParameter(name)) {
    session->LoadParameter(name, output);
  } else {
    output = default_value;
  }
}
// make a base class that handles the domain, particle bcs
// handle graphmap, sycl target, particle mesh interface
// nektar graph local mapper
// cell id are per group
// boundary conditions are per group
class ChargedParticles {
private:
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  MPI_Comm comm;
  const double graph_mapper_tol;
  const int ndim = 2;

  std::vector<std::shared_ptr<IntegratorBoris>> boris_integrators;

  inline void add_particles() {

    long rstart, rend;
    const long size = this->sycl_target->comm_pair.size_parent;
    const long rank = this->sycl_target->comm_pair.rank_parent;

    get_decomp_1d(size, (long)this->num_particles_per_species, rank, &rstart,
                  &rend);
    const long N = rend - rstart;
    const int cell_count = this->domain->mesh->get_cell_count();

    // get seed from file
    std::srand(std::time(nullptr));
    int seed;
    pic2d3v_get_from_session(this->session, "particle_position_seed", seed,
                             std::rand());

    std::mt19937 rng_phasespace(seed + rank);
    std::bernoulli_distribution coin_toss(0.5);

    std::uniform_real_distribution<double> uniform01(0, 1);

    double B0x = 0.0;
    double B0y = 0.0;
    double B0z = 0.0;
    session->LoadParameter("B0x", B0x);
    session->LoadParameter("B0y", B0y);
    session->LoadParameter("B0z", B0z);

    B0x = m_unitConverter->si_magneticfield_to_sim(B0x);
    B0y = m_unitConverter->si_magneticfield_to_sim(B0y);
    B0z = m_unitConverter->si_magneticfield_to_sim(B0z);
    if (rank == 0) {
      std::cout << "B0x in dimensionless units is " << B0x << std::endl;
      std::cout << "B0y in dimensionless units is " << B0y << std::endl;
      std::cout << "B0z in dimensionless units is " << B0z << std::endl;
    }

    double B0 = std::sqrt(B0x * B0x + B0y * B0y + B0z * B0z);

    std::vector<double> B0vector = {B0x, B0y, B0z};
    if (rank == 0) {
      std::cout << *(this->m_unitConverter) << std::endl;
      std::cout << "The resolution light crossing time (L / c) / dt = "
                << 1.0 / this->dt
                << std::endl; // L = c = 1 in dimensionless units
      std::cout << "The resolution of the electron cyclotron period is "
                << 2 * M_PI / B0 / this->dt << std::endl;
    }

    double bpitch = B0z / B0;
    double B0_tmp = B0;
    if (B0_tmp == 0.0) {
      B0_tmp = 1.0;
      bpitch = 1.0; // then the rotation matrix is diagonal
    }
    double sinacosbpitch = std::sin(std::acos(bpitch));
    double cx = -B0y / B0_tmp;
    double cy = B0x / B0_tmp;
    // rotation matrix
    double r00 = cx * cx * (1 - bpitch) + bpitch;
    double r01 = cx * cy * (1 - bpitch);
    double r02 = cy * sinacosbpitch;
    double r10 = r01;
    double r11 = cy * cy * (1 - bpitch) + bpitch;
    double r12 = -cx * sinacosbpitch;
    double r20 = -r02;
    double r21 = -r12;
    double r22 = bpitch;
    NESOASSERT(
        std::fabs(r00 * (r11 * r22 - r12 * r21) +
                  r01 * (r12 * r20 - r10 * r22) +
                  r02 * (r10 * r21 - r11 * r20) - 1) < 1e-8,
        "The magnetic field rotation matrix must have a unit determinant");

    if (N > 0) {
      for (uint32_t s = 0; s < num_species; ++s) {
        const auto ics = this->particle_initial_conditions[s];
        auto sstr = std::to_string(s);

        int distribution_function = -1;
        session->LoadParameter("particle_distribution_function_species_" + sstr,
                               distribution_function);
        NESOASSERT(distribution_function > -1,
                   "Bad particle distribution key.");
        NESOASSERT(distribution_function < 1, "Bad particle distribution key.");

        ParticleSet initial_distribution(
            N, this->particle_groups[s]->get_particle_spec());

        // Get the requested particle distribution type from the config file
        int distribution_type = 0;
        pic2d3v_get_from_session(session, "distribution_type_" + sstr,
                                 distribution_type, 0);

        auto positions = uniform_within_extents(
            N, ndim, this->boundary_condition->global_extent, rng_phasespace);

        if (distribution_function == 0) {
          // 3V Maxwellian
          double charge = ics.charge;
          double mass = ics.mass;
          double thermal_velocity = std::sqrt(2 * ics.temperature / mass);
          double drift_velocity = std::sqrt(2 * ics.drift / mass);
          double drift_para = drift_velocity * ics.pitch;
          double drift_perp =
              drift_velocity * std::sqrt(1 - std::pow(ics.pitch, 2));

          double vperp_min = std::max(0.0, drift_perp - 6.0 * thermal_velocity);
          double vperp_peak = (drift_perp +
              std::sqrt(std::pow(drift_perp, 2) +
              2 * std::pow(thermal_velocity, 2))) / 2;

          for (int p = 0; p < N; p++) {
            // x position
            initial_distribution[Sym<REAL>("X")][p][0] =
                positions[0][p] + this->boundary_condition->global_origin[0];

            // y position
            initial_distribution[Sym<REAL>("X")][p][1] =
                positions[1][p] + this->boundary_condition->global_origin[1];

            // vpara, vperp thermally distributed velocities
            auto rvpara =
                boost::math::erf_inv(2 * uniform01(rng_phasespace) - 1) *
                std::sqrt(2.0);
            auto vpara = thermal_velocity * rvpara + drift_para;
            // in the case of a delta function i.e. thermal_velocity == 0.0;
            double vperp = drift_perp;
            // accept reject
            if (thermal_velocity > 0) {
              while (true) {
                vperp = vperp_min + uniform01(rng_phasespace) * 2.0 * 6.0 *
                                        thermal_velocity;
                double vf_eval =
                    vperp / vperp_peak *
                    std::exp(
                        -std::pow((vperp - drift_perp) / thermal_velocity, 2));
                NESOASSERT(vf_eval <= 1,
                           "Error in the accept reject algorithm, f > 1");
                if (uniform01(rng_phasespace) < vf_eval) {
                  break;
                }
              }
            }
            auto gyroangle = uniform01(rng_phasespace) * 2 *
                             boost::math::constants::pi<double>();
            double vperp0 = vperp * std::cos(gyroangle);
            double vperp1 = vperp * std::sin(gyroangle);
            double vx = r00 * vperp0 + r01 * vperp1 + r02 * vpara;
            double vy = r10 * vperp0 + r11 * vperp1 + r12 * vpara;
            double vz = r20 * vperp0 + r21 * vperp1 + r22 * vpara;

            initial_distribution[Sym<REAL>("V")][p][0] = vx;
            initial_distribution[Sym<REAL>("V")][p][1] = vy;
            initial_distribution[Sym<REAL>("V")][p][2] = vz;

            initial_distribution[Sym<INT>("Q")][p][0] = charge;
            initial_distribution[Sym<INT>("M")][p][0] = mass;
            initial_distribution[Sym<REAL>("W")][p][0] = ics.weight;
            initial_distribution[Sym<REAL>("WQ")][p][0] = ics.weight * charge;
          }
        }

        for (int p = 0; p < N; p++) {
          initial_distribution[Sym<REAL>("phi")][p][0] = 0.0;
          for (int d = 0; d < 3; ++d) {
            initial_distribution[Sym<REAL>("A")][p][d] = 0.0;
            initial_distribution[Sym<REAL>("B")][p][d] = B0vector[d];
            initial_distribution[Sym<REAL>("E")][p][d] = 0.0;
          }
          initial_distribution[Sym<INT>("CELL_ID")][p][0] = p % cell_count;
          initial_distribution[Sym<INT>("PARTICLE_ID")][p][0] = p + rstart;
        }
        this->particle_groups[s]->add_particles_local(initial_distribution);

      } // for loop over species
    }

    for (auto pg : this->particle_groups) {
      NESO::parallel_advection_initialisation(pg);
      NESO::parallel_advection_store(pg);
    }
    const int num_steps = 20;
    for (int stepx = 0; stepx < num_steps; stepx++) {
      for (auto pg : this->particle_groups) {
        NESO::parallel_advection_step(pg, num_steps, stepx);
        this->transfer_particles();
      }
    }
    for (auto pg : this->particle_groups) {
      NESO::parallel_advection_restore(pg);
    }

    // Move particles to the owning ranks and correct cells.
    this->transfer_particles();
  }

public:
  /// Disable (implicit) copies.
  ChargedParticles(const ChargedParticles &st) = delete;
  /// Disable (implicit) copies.
  ChargedParticles &operator=(ChargedParticles const &a) = delete;

  /// The lengthscale of the system, which ultimately defines the units
  double m_lengthScale;
  /// Global number of particles per species in the simulation.
  int num_particles_per_species;
  /// Time step size used for particles
  double dt;
  /// Number of species
  int num_species;
  /// An num_species long vector of structs with the initial conditions for each
  /// particlegroup
  std::vector<ParticleInitialConditions> particle_initial_conditions;
  /// HMesh instance that allows particles to move over nektar++ meshes.
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  /// Compute target.
  SYCLTargetSharedPtr sycl_target;
  /// Mapping instance to map particles into nektar++ elements.
  std::shared_ptr<NektarGraphLocalMapperT> nektar_graph_local_mapper;
  /// NESO-Particles domain.
  DomainSharedPtr domain;
  /// NESO-Particles ParticleGroup containing charged particles: one per species
  std::vector<ParticleGroupSharedPtr> particle_groups;
  /// Method to apply particle boundary conditions.
  std::shared_ptr<NektarCartesianPeriodic> boundary_condition;
  /// Method to map to/from nektar geometry ids to 0,N-1 used by NESO-Particles
  std::shared_ptr<CellIDTranslation> cell_id_translation;
  /// Trajectory writer for particles.
  std::vector<std::shared_ptr<H5Part>> h5parts;
  /// A helper class to convert SI units to simulation units and back
  std::shared_ptr<UnitConverter> m_unitConverter;

  DomainSharedPtr domain_shptr() const { return this->domain; };

  //  /**
  //   *  Set the constant and uniform magnetic field over the entire domain.
  //   *
  //   *  @param Bx Magnetic field B in x direction.
  //   *  @param By Magnetic field B in y direction.
  //   *  @param Bz Magnetic field B in z direction.
  //   */
  //  inline void set_B_field(const double Bx = 0.0, const double By = 0.0,
  //                          const double Bz = 0.0) {
  //    this->B0x = Bx;
  //    this->B0y = By;
  //    this->B0z = Bz;
  //  }

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
      // const std::vector<ParticleInitialConditions> & particle_ics)
      : session(session), graph(graph), comm(comm), graph_mapper_tol(1.0e-8) {

    // Reduce the global number of elements
    const int num_elements_local = this->graph->GetNumElements();
    int num_elements_global;
    MPICHK(MPI_Allreduce(&num_elements_local, &num_elements_global, 1, MPI_INT,
                         MPI_SUM, this->comm));

    this->session->LoadParameter("length_scale", this->m_lengthScale);

    // compute the global number of particles
    this->session->LoadParameter("num_particles_per_species",
                                 this->num_particles_per_species);

    this->session->LoadParameter("TimeStep", this->dt);

    this->m_unitConverter = std::make_shared<UnitConverter>(m_lengthScale);

    // Create interface between particles and nektar++
    this->particle_mesh_interface =
        std::make_shared<ParticleMeshInterface>(graph, 0, this->comm);
    this->sycl_target =
        std::make_shared<SYCLTarget>(0, particle_mesh_interface->get_comm());
    this->nektar_graph_local_mapper = std::make_shared<NektarGraphLocalMapperT>(
        this->sycl_target, this->particle_mesh_interface, this->graph_mapper_tol);
    this->domain = std::make_shared<Domain>(this->particle_mesh_interface,
                                            this->nektar_graph_local_mapper);

    const long rank = this->sycl_target->comm_pair.rank_parent;

    // Create ParticleSpec
    ParticleSpec particle_spec{
        ParticleProp(Sym<REAL>("X"), 2, true), // position
        ParticleProp(Sym<INT>("CELL_ID"), 1, true),
        ParticleProp(Sym<INT>("PARTICLE_ID"), 1),
        ParticleProp(Sym<INT>("Q"), 1),   // charge
        ParticleProp(Sym<INT>("M"), 1),   // mass
        ParticleProp(Sym<REAL>("W"), 1),   // weight
        ParticleProp(Sym<REAL>("V"), 3),   // velocity
        ParticleProp(Sym<REAL>("phi"), 1), // phi field
        ParticleProp(Sym<REAL>("A"), 3),   // A field
        ParticleProp(Sym<REAL>("B"), 3),   // B field
        ParticleProp(Sym<REAL>("E"), 3),   // E field
        ParticleProp(Sym<REAL>("WQ"), 1),  // weight * charge
        ParticleProp(Sym<REAL>("WQV"), 3)  // weight * charge * velocity
    };

    this->session->LoadParameter("number_of_particle_species",
                                 this->num_species);

    this->particle_groups.reserve(this->num_species);
    for (uint32_t i = 0; i < this->num_species; ++i) {
      // create a particle group per species
      auto pg = std::make_shared<ParticleGroup>(this->domain, particle_spec,
                                                this->sycl_target);
      this->particle_groups.push_back(pg);
    }

    // Setup PBC boundary conditions.
    this->boundary_condition = std::make_shared<NektarCartesianPeriodic>(
        this->sycl_target, this->graph); // should come from ParticleSpec

    // Setup map between cell indices. Assume all groups have same cell_id_dat
    this->cell_id_translation = std::make_shared<CellIDTranslation>(
        this->sycl_target, this->particle_mesh_interface);

    const double volume_nounits = this->boundary_condition->global_extent[0] *
                                  this->boundary_condition->global_extent[1];

    if (rank == 0) {
      std::cout << "The volume of the mesh in dimensionless units = "
                << volume_nounits << std::endl;
    }

    double totalChargeDensity = 0.0;
    double totalDensity = 0.0;
    double totalParallelCurrent = 0.0;

    for (std::size_t s = 0; s < this->num_species; ++s) {
      std::string species_string = std::to_string(s);

      double charge;
      this->session->LoadParameter("charge_" + species_string, charge);

      double mass;
      this->session->LoadParameter("mass_" + species_string, mass);

      double number_density;
      this->session->LoadParameter("number_density_" + species_string,
                                   number_density);

      double temperature;
      this->session->LoadParameter("temperature_" + species_string,
                                   temperature);

      double drift = 0.0;
      this->session->LoadParameter("drift_" + species_string, drift);
      double pitch = 1.0;
      this->session->LoadParameter("pitch_" + species_string, pitch);

      temperature = m_unitConverter->si_temperature_ev_to_sim(temperature);
      drift = m_unitConverter->si_temperature_ev_to_sim(drift);
      number_density = m_unitConverter->si_numberdensity_to_sim(number_density);

      double weight =
          number_density * volume_nounits / num_particles_per_species;

      if (rank == 0) {
        std::cout << "The number density in dimensionless units is "
                  << number_density << std::endl;
        std::cout << "The temperature in dimensionless units is " << temperature
                  << std::endl;
        std::cout << "The drift in dimensionless units is " << drift
                  << std::endl;
        std::cout << "Weight from nondim units " << weight << std::endl;
      }

      ParticleInitialConditions pic = {
          charge, mass, temperature, drift, pitch, number_density, weight};

      this->particle_initial_conditions.emplace_back(pic);

      totalChargeDensity += charge * number_density;
      totalDensity += number_density;

      auto drift_speed = std::sqrt(2 * drift / mass);
      totalParallelCurrent += charge * number_density * drift_speed * pitch;
    }
    NESOASSERT(std::abs(totalChargeDensity) < 1e-14 * totalDensity,
               "The plasma must be neutral.");

    int stpcos = -1; //
    this->session->LoadParameter("subtract_total_parallel_current_off_species", stpcos, -1);
    if ((stpcos  > 0) && (stpcos <= this->num_species)) {
      auto icToChange = this->particle_initial_conditions[stpcos];
      NESOASSERT(icToChange.drift == 0.0, "The drift have started as 0");
      auto velocity = - totalParallelCurrent / icToChange.charge / icToChange.number_density;
      icToChange.pitch = sgn(velocity);
      icToChange.drift = 0.5 * icToChange.mass * std::pow(velocity, 2);
      if (rank == 0) {
        std::cout << "To offset current, the drift energy of species " << stpcos <<
          " is " << icToChange.drift << " with pitch " << icToChange.pitch <<
          " , whereas the temperature is " << icToChange.temperature << std::endl;
      }
    }

    // Add particle to the particle group
    this->add_particles();

    for (std::size_t s = 0; s < this->num_species; ++s) {
      auto pg = this->particle_groups[s];
      this->boris_integrators.emplace_back(std::make_shared<IntegratorBoris>(
          pg, this->dt));

      std::string filename = "MaxwellWave2D3V_particle_trajectory_" +
        std::to_string(s) + ".h5part";
      auto h5part = std::make_shared<H5Part>(
                 filename, this->particle_groups[s], Sym<REAL>("X"),
                 Sym<INT>("CELL_ID"), Sym<REAL>("V"), Sym<REAL>("E"), Sym<INT>("Q"),
                 Sym<INT>("M"), Sym<REAL>("B"), Sym<INT>("NESO_MPI_RANK"),
                 Sym<INT>("PARTICLE_ID"), Sym<REAL>("NESO_REFERENCE_POSITIONS"));
      this->h5parts.push_back(h5part);
    }
  };

  /**
   *  Write current particle state to trajectory.
   */
  inline void write() {
    for (auto h5part : this->h5parts) {
      h5part->write();
    }
  }

  /**
   *  Apply boundary conditions and transfer particles between MPI ranks.
   */
  inline void transfer_particles() {
    auto t0 = profile_timestamp();
    for (auto pg : this->particle_groups) {
      this->boundary_condition->execute(pg->position_dat);
      pg->hybrid_move();
      this->cell_id_translation->execute(pg->cell_id_dat);
      pg->cell_move();
    }
    this->sycl_target->profile_map.inc(
        "ChargedParticles", "transfer_particles", 1,
        profile_elapsed(t0, profile_timestamp()));
  }

  /**
   *  Free the object before MPI_Finalize is called.
   */
  inline void free() {
    for (auto h5part : this->h5parts) {
      h5part->close();
    }
    for (auto pg : this->particle_groups) {
      pg->free();
    }
    this->particle_mesh_interface->free();
    this->sycl_target->free();
  };

  /**
   * Boris
   */
  inline void accelerate(const double dt_fraction) {
    for (auto integrator : this->boris_integrators) {
      integrator->accelerate(dt_fraction);
    }
  }

  inline void advect(const double dt_fraction) {
    for (auto integrator : this->boris_integrators) {
      integrator->advect(dt_fraction);
    }
  }

  inline void communicate() {
    // positions were written so we apply boundary conditions and move
    // particles between ranks
    this->transfer_particles();
  }
  /**
   *  Get the Sym object for the ParticleDat holding particle charge multiplied
   * by its weight required for projection to charge density rho
   */
  inline Sym<REAL> get_rho_sym() { return Sym<REAL>("WQ"); }

  /**
   *  Get the Sym object for the ParticleDat holding w * q * v = j
   */
  inline Sym<REAL> get_current_sym() { return Sym<REAL>("WQV"); }

  //  /**
  //   *  Get the charge density of the system.
  //   */
  //  inline double get_charge_density() { return this->charge_density; }
};

#endif
