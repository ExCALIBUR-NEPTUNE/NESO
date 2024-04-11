#ifndef __PARTSYS_BASE_H_
#define __PARTSYS_BASE_H_

#include <SolverUtils/EquationSystem.h>
#include <mpi.h>
#include <neso_particles.hpp>
#include <type_traits>

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;

namespace NESO::Particles {

class PartSysBase {

public:
  virtual ~PartSysBase() { free(); };

  /// NESO-Particles ParticleGroup
  ParticleGroupSharedPtr particle_group;
  /// Compute target
  SYCLTargetSharedPtr sycl_target;

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
  }

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

  /// Clear up memory
  inline void free() {
    if (this->h5part_exists) {
      this->h5part->close();
    }
    this->particle_group->free();
    this->particle_mesh_interface->free();
    this->sycl_target->free();
  };
};

} // namespace NESO::Particles
#endif