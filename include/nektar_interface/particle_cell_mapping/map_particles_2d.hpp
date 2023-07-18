#ifndef __MAP_PARTICLES_2D_H__
#define __MAP_PARTICLES_2D_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

#include "coarse_lookup_map.hpp"
#include "map_particles_2d_regular.hpp"
#include "nektar_interface/coordinate_mapping.hpp"
#include "nektar_interface/geometry_transport/shape_mapping.hpp"
#include "nektar_interface/parameter_store.hpp"
#include "nektar_interface/particle_mesh_interface.hpp"
#include "newton_geom_interfaces.hpp"
#include "particle_cell_mapping_common.hpp"

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

/**
 *  Class to map particles into Nektar++ cells.
 */
class MapParticles2D {
protected:
  /// Disable (implicit) copies.
  MapParticles2D(const MapParticles2D &st) = delete;
  /// Disable (implicit) copies.
  MapParticles2D &operator=(MapParticles2D const &a) = delete;

  SYCLTargetSharedPtr sycl_target;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;

  std::unique_ptr<MapParticlesCommon> map_particles_common;
  std::unique_ptr<MapParticles2DRegular> map_particles_2d_regular;
  std::unique_ptr<Newton::MapParticlesNewton<Newton::MappingQuadLinear2D>>
      map_particles_newton_linear_quad;

  int count_regular = 0;
  int count_deformed = 0;

public:
  /**
   *  Constructor for mapping class.
   *
   *  @param sycl_target SYCLTarget on which to perform mapping.
   *  @param particle_mesh_interface ParticleMeshInterface containing 2D
   * Nektar++ cells.
   * @param config ParameterStore instance to pass configuration options to
   * lower-level mappers.
   */
  MapParticles2D(
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
