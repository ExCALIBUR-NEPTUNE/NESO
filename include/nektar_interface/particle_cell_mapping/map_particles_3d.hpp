#ifndef __MAPPING_PARTICLES_3D_H__
#define __MAPPING_PARTICLES_3D_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <tuple>
#include <vector>

#include "coarse_lookup_map.hpp"
#include "map_particles_3d_regular.hpp"
#include "map_particles_host.hpp"
#include "nektar_interface/coordinate_mapping.hpp"
#include "nektar_interface/geometry_transport/shape_mapping.hpp"
#include "nektar_interface/parameter_store.hpp"
#include "nektar_interface/particle_mesh_interface.hpp"
#include "particle_cell_mapping_common.hpp"

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

#include "newton_geom_interfaces.hpp"

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

/**
 *  Class to map particles into Nektar++ cells.
 */
class MapParticles3D {
protected:
  /// Disable (implicit) copies.
  MapParticles3D(const MapParticles3D &st) = delete;
  /// Disable (implicit) copies.
  MapParticles3D &operator=(MapParticles3D const &a) = delete;

  SYCLTargetSharedPtr sycl_target;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;

  std::unique_ptr<MapParticlesCommon> map_particles_common;
  std::unique_ptr<MapParticlesHost> map_particles_host;
  std::unique_ptr<MapParticles3DRegular> map_particles_3d_regular;

  std::tuple<
      std::unique_ptr<Newton::MapParticlesNewton<Newton::MappingTetLinear3D>>,
      std::unique_ptr<Newton::MapParticlesNewton<Newton::MappingPrismLinear3D>>,
      std::unique_ptr<Newton::MapParticlesNewton<Newton::MappingHexLinear3D>>,
      std::unique_ptr<Newton::MapParticlesNewton<Newton::MappingPyrLinear3D>>>
      map_particles_3d_deformed_linear;

  template <typename T>
  inline void map_newton_internal(std::unique_ptr<T> &ptr,
                                  ParticleGroup &particle_group,
                                  const int map_cell) {
    if (ptr) {
      ptr->map(particle_group, map_cell);
    }
  }

public:
  /**
   *  Constructor for mapping class.
   *
   *  @param sycl_target SYCLTarget on which to perform mapping.
   *  @param particle_mesh_interface ParticleMeshInterface containing 3D
   * Nektar++ cells.
   * @param config ParameterStore configuration for lower level mapping classes.
   */
  MapParticles3D(
      SYCLTargetSharedPtr sycl_target,
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      ParameterStoreSharedPtr config = std::make_shared<ParameterStore>());

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  3D geometry objects
   */
  void map(ParticleGroup &particle_group, const int map_cell = -1);
};

} // namespace NESO

#endif
