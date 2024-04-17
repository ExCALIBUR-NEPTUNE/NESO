#ifndef __PARTSYS_BASE_H_
#define __PARTSYS_BASE_H_

#include <SolverUtils/EquationSystem.h>
#include <mpi.h>
#include <nektar_interface/particle_interface.hpp>
#include <neso_particles.hpp>
#include <type_traits>

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;

namespace NESO::Particles {

class PartSysBase {

public:
  inline static const std::string NUM_PARTS_TOT_STR = "num_particles_total";
  inline static const std::string NUM_PARTS_PER_CELL_STR =
      "num_particles_per_cell";
  /// Total number of particles in simulation
  int64_t num_parts_tot;

  /// NESO-Particles ParticleGroup
  ParticleGroupSharedPtr particle_group;
  /// Compute target
  SYCLTargetSharedPtr sycl_target;

  /// Clear up memory
  inline void free() {
    if (this->h5part_exists) {
      this->h5part->close();
    }
    this->particle_group->free();
    this->sycl_target->free();
    this->particle_mesh_interface->free();
  };

protected:
  /**
   *  Create a new instance.
   *
   *  @param session Nektar++ session to use for parameters and simulation
   * specification.
   *  @param graph Nektar++ MeshGraph on which particles exist.
   *  @param comm (optional) MPI communicator to use - default MPI_COMM_WORLD.
   *
   */
  PartSysBase(const LU::SessionReaderSharedPtr session,
              const SD::MeshGraphSharedPtr graph, ParticleSpec particle_spec,
              MPI_Comm comm = MPI_COMM_WORLD)
      : session(session), graph(graph), comm(comm), h5part_exists(false),
        ndim(graph->GetSpaceDimension()) {

    set_num_parts_tot();

    // Create interface between particles and nektar++
    this->particle_mesh_interface =
        std::make_shared<ParticleMeshInterface>(graph, 0, this->comm);
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
  /// HMesh instance that allows particles to move over nektar++ meshes.
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  /// Pointer to Session object
  LU::SessionReaderSharedPtr session;

  inline void set_num_parts_tot() {

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
      } else {
        nprint("Particles disabled (Neither '" + NUM_PARTS_TOT_STR +
               "' or "
               "'" +
               NUM_PARTS_PER_CELL_STR + "' are set)");
      }
    }
  }
};

} // namespace NESO::Particles
#endif