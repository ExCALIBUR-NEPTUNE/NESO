#ifndef __PARTSYS_BASE_H_
#define __PARTSYS_BASE_H_

#include <SolverUtils/EquationSystem.h>
#include <mpi.h>
#include <nektar_interface/geometry_transport/halo_extension.hpp>
#include <nektar_interface/particle_interface.hpp>
#include <neso_particles.hpp>
#include <type_traits>

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;

namespace NESO::Particles {

/// Struct used to set common options for particle systems
struct PartSysOptions {
  int extend_halos_offset = 0;
};

class PartSysBase {

public:
  // Some parameter names used in solver config files
  inline static const std::string NUM_PARTS_TOT_STR = "num_particles_total";
  inline static const std::string NUM_PARTS_PER_CELL_STR =
      "num_particles_per_cell";
  inline static const std::string PART_OUTPUT_FREQ_STR = "particle_output_freq";

  /// Total number of particles in simulation
  int64_t num_parts_tot;

  /// NESO-Particles ParticleGroup
  ParticleGroupSharedPtr particle_group;
  /// Compute target
  SYCLTargetSharedPtr sycl_target;

  /**
   * @brief Write particle parameter values to stdout for any parameter.
   *  @see also report_param()
   */
  inline void add_params_report() {
    std::cout << "Particle settings:" << std::endl;
    for (auto const &[param_lbl, param_str_val] : this->param_vals_to_report) {
      std::cout << "  " << param_lbl << ": " << param_str_val << std::endl;
    }
    std::cout << "============================================================="
                 "=========="
              << std::endl
              << std::endl;
  }

  /**
   * @brief Clear up memory related to the particle system
   */
  inline void free() {
    if (this->h5part_exists) {
      this->h5part->close();
    }
    this->particle_group->free();
    this->sycl_target->free();
    this->particle_mesh_interface->free();
  };

  /**
   *  @brief Write particle properties to an output file.
   *
   *  @param step Time step number.
   */
  inline void write(const int step) {
    if (this->h5part_exists) {
      if (this->sycl_target->comm_pair.rank_parent == 0) {
        nprint("Writing particle properties at step", step);
      }
      this->h5part->write();
    } else {
      if (this->sycl_target->comm_pair.rank_parent == 0) {
        nprint("Ignoring call to write particle data because an output file "
               "wasn't set up. init_output() not called?");
      }
    }
  }

protected:
  /**
   * Protected constructor to prohibit direct instantiation.
   *
   *  @param session Nektar++ session to use for parameters and simulation
   * specification.
   *  @param graph Nektar++ MeshGraph on which particles exist.
   *  @param comm (optional) MPI communicator to use - default MPI_COMM_WORLD.
   *
   */
  PartSysBase(const LU::SessionReaderSharedPtr session,
              const SD::MeshGraphSharedPtr graph, ParticleSpec particle_spec,
              MPI_Comm comm = MPI_COMM_WORLD,
              PartSysOptions options = PartSysOptions())
      : session(session), graph(graph), comm(comm), h5part_exists(false),
        ndim(graph->GetSpaceDimension()) {

    read_params();

    // Store options
    this->options = options;

    // Create interface between particles and nektar++
    this->particle_mesh_interface =
        std::make_shared<ParticleMeshInterface>(graph, 0, this->comm);
    extend_halos_fixed_offset(this->options.extend_halos_offset,
                              this->particle_mesh_interface);
    this->sycl_target =
        std::make_shared<SYCLTarget>(0, particle_mesh_interface->get_comm());
    this->nektar_graph_local_mapper = std::make_shared<NektarGraphLocalMapper>(
        this->sycl_target, this->particle_mesh_interface);
    this->domain = std::make_shared<Domain>(this->particle_mesh_interface,
                                            this->nektar_graph_local_mapper);

    // Create ParticleGroup
    this->particle_group =
        std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

    // Set up map between cell indices
    this->cell_id_translation = std::make_shared<CellIDTranslation>(
        this->sycl_target, this->particle_group->cell_id_dat,
        this->particle_mesh_interface);
  }

  /// Object used to map to/from nektar geometry ids to 0,N-1
  std::shared_ptr<CellIDTranslation> cell_id_translation;
  /// MPI communicator
  MPI_Comm comm;
  /// NESO-Particles domain.
  DomainSharedPtr domain;
  /// Pointer to Nektar Meshgraph object
  SD::MeshGraphSharedPtr graph;
  /// HDF5 output file
  std::shared_ptr<H5Part> h5part;
  /// HDF5 output file flag
  bool h5part_exists;
  /// Number of spatial dimensions being used
  const int ndim;
  /// Mapping instance to map particles into nektar++ elements.
  std::shared_ptr<NektarGraphLocalMapper> nektar_graph_local_mapper;
  /// Options struct
  PartSysOptions options;
  /// HMesh instance that allows particles to move over nektar++ meshes.
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  /// Pointer to Session object
  LU::SessionReaderSharedPtr session;

  /**
   * @brief Set up per-step particle output
   *
   *  @param fname Output filename. Default is 'particle_trajectory.h5part'.
   *  @param args Remaining arguments (variable length) should be sym instances
   *  indicating which ParticleDats are to be written.
   */
  template <typename... T>
  inline void init_output(std::string fname, T... args) {
    if (this->h5part_exists) {
      if (this->sycl_target->comm_pair.rank_parent == 0) {
        nprint("Ignoring (duplicate?) call to init_output().");
      }
    } else {
      // Create H5Part instance
      this->h5part =
          std::make_shared<H5Part>(fname, this->particle_group, args...);
      this->h5part_exists = true;
    }
  }

  /**
   * @brief Store particle param values in a map.Values are reported later via
   * add_params_report()
   *
   * @param label Label to attach to the parameter
   * @param value Value of the parameter that was set
   */
  template <typename T> void report_param(std::string label, T val) {
    // form stringstream and store string value in private map
    std::stringstream ss;
    ss << val;
    param_vals_to_report[label] = ss.str();
  }

  /**
   * @brief Read some parameters associated with all particle systems.
   */
  inline void read_params() {

    // Read total number of particles / number per cell from config
    int num_parts_per_cell, num_parts_tot;
    this->session->LoadParameter(NUM_PARTS_TOT_STR, num_parts_tot, -1);
    this->session->LoadParameter(NUM_PARTS_PER_CELL_STR, num_parts_per_cell,
                                 -1);

    if (num_parts_tot > 0) {
      this->num_parts_tot = num_parts_tot;
      if (num_parts_per_cell > 0) {
        nprint("Ignoring value of '" + NUM_PARTS_PER_CELL_STR +
               "' because  "
               "'" +
               NUM_PARTS_TOT_STR + "' was specified.");
      }
    } else {
      if (num_parts_per_cell > 0) {
        // Determine the global number of elements
        const int num_elements_local = this->graph->GetNumElements();
        int num_elements_global;
        MPICHK(MPI_Allreduce(&num_elements_local, &num_elements_global, 1,
                             MPI_INT, MPI_SUM, this->comm));

        // compute the global number of particles
        this->num_parts_tot =
            ((int64_t)num_elements_global) * num_parts_per_cell;

        report_param("Number of particles per cell/element",
                     num_parts_per_cell);
      } else {
        nprint("Particles disabled (Neither '" + NUM_PARTS_TOT_STR +
               "' or "
               "'" +
               NUM_PARTS_PER_CELL_STR + "' are set)");
      }
    }
    report_param("Total number of particles", this->num_parts_tot);

    // Output frequency
    int particle_output_freq;
    this->session->LoadParameter(PART_OUTPUT_FREQ_STR, particle_output_freq, 0);
    report_param("Output frequency (steps)", particle_output_freq);
  }

private:
  /// Map containing parameter name,value pairs to be written to stdout when the
  /// nektar equation system is initialised. Populated with report_param().
  std::map<std::string, std::string> param_vals_to_report;
};

} // namespace NESO::Particles
#endif