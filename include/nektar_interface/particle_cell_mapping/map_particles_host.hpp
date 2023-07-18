#ifndef __MAP_PARTICLES_HOST_H__
#define __MAP_PARTICLES_HOST_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

#include "../particle_mesh_interface.hpp"
#include "nektar_interface/parameter_store.hpp"
#include "particle_cell_mapping_common.hpp"
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar;
using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

/**
 *  Class to map particles into Nektar++ cells on the CPU host. Should work for
 *  all 2D and 3D elements.
 *
 *  Configurable with the following options in the passed ParameterStore:
 *  * MapParticlesHost/tol: Tolerance to apply when determining if a
 *    particle is within a geometry object (default 0.0).
 */
class MapParticlesHost {
protected:
  /// Disable (implicit) copies.
  MapParticlesHost(const MapParticlesHost &st) = delete;
  /// Disable (implicit) copies.
  MapParticlesHost &operator=(MapParticlesHost const &a) = delete;

  /// Tolerance to pass to Nektar++ for mapping.
  REAL tol;

  SYCLTargetSharedPtr sycl_target;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;

public:
  /**
   *  Constructor for mapping class.
   *
   *  @param sycl_target SYCLTarget on which to perform mapping.
   *  @param particle_mesh_interface ParticleMeshInterface containing 2D
   * Nektar++ cells.
   * @param config ParameterStore instance to pass tolerance to Nektar++
   * ContainsPoint.
   */
  MapParticlesHost(
      SYCLTargetSharedPtr sycl_target,
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      ParameterStoreSharedPtr config = std::make_shared<ParameterStore>());

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  triangles and quads.
   */
  void map(ParticleGroup &particle_group, const int map_cell = -1);
};

} // namespace NESO

#endif
