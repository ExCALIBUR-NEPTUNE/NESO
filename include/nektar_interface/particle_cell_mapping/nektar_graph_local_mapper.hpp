#ifndef __NEKTAR_GRAPH_LOCAL_MAPPER_H__
#define __NEKTAR_GRAPH_LOCAL_MAPPER_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

#include "../coordinate_mapping.hpp"
#include "../particle_mesh_interface.hpp"
#include "coarse_lookup_map.hpp"
#include "particle_cell_mapping_2d.hpp"
#include "particle_cell_mapping_3d.hpp"

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

/**
 * Class to map particle positions to Nektar++ geometry objects.
 */
class NektarGraphLocalMapper : public LocalMapper {
private:
  SYCLTargetSharedPtr sycl_target;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  std::unique_ptr<MapParticles2D> map_particles_2d;
  std::unique_ptr<MapParticles3D> map_particles_3d;

public:
  ~NektarGraphLocalMapper(){};

  /**
   * Callback for ParticleGroup to execute for additional setup of the
   * LocalMapper that may involve the ParticleGroup.
   *
   * @param particle_group ParticleGroup instance.
   */
  void particle_group_callback(ParticleGroup &particle_group);

  /**
   *  Construct a new mapper object.
   *
   *  @param sycl_target SYCLTarget to use.
   *  @param particle_mesh_interface Interface between NESO-Particles and
   * Nektar++ mesh.
   *  @param tol Tolerance to pass to Nektar++ to bin particles into cells.
   *  @param config ParameterStore instance to configure lower level particle to
   * cell mappers.
   */
  NektarGraphLocalMapper(
      SYCLTargetSharedPtr sycl_target,
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      ParameterStoreSharedPtr config = std::make_shared<ParameterStore>());

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  geometry objects.
   */
  void map(ParticleGroup &particle_group, const int map_cell = -1);
};

} // namespace NESO

#endif
