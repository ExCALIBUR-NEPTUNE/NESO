#ifndef __MAP_PARTICLES_2D_REGULAR_H__
#define __MAP_PARTICLES_2D_REGULAR_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

#include "coarse_mappers_base.hpp"
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
 *  Class to map particles into regular (eRegular) triangles and quads.
 *
 *  Configurable with the following options in the passed ParameterStore:
 *  * MapParticles2DRegular/tol: Tolerance to apply when determining if a
 * particle is within a geometry object (default 0.0).
 *
 */
class MapParticles2DRegular : public CoarseMappersBase {
protected:
  /// Disable (implicit) copies.
  MapParticles2DRegular(const MapParticles2DRegular &st) = delete;
  /// Disable (implicit) copies.
  MapParticles2DRegular &operator=(MapParticles2DRegular const &a) = delete;

  ParticleMeshInterfaceSharedPtr particle_mesh_interface;

  int num_regular_geoms;
  /// The 3 vertices required by mapping from physical space to reference space.
  std::unique_ptr<BufferDeviceHost<double>> dh_vertices;

  /// Tolerance on the distance used to check if a particle is within a geometry
  /// object.
  REAL tol;

  template <typename U>
  inline void write_vertices_2d(U &geom, const int index, double *output) {
    int last_point_index = -1;
    if (geom->GetShapeType() == LibUtilities::eTriangle) {
      last_point_index = 2;
    } else if (geom->GetShapeType() == LibUtilities::eQuadrilateral) {
      last_point_index = 3;
    } else {
      NESOASSERT(false, "get_local_coords_2d Unknown shape type.");
    }
    const auto v0 = geom->GetVertex(0);
    const auto v1 = geom->GetVertex(1);
    const auto v2 = geom->GetVertex(last_point_index);

    NESOASSERT(v0->GetCoordim() == 2, "Expected v0->Coordim to be 2.");
    NESOASSERT(v1->GetCoordim() == 2, "Expected v1->Coordim to be 2.");
    NESOASSERT(v2->GetCoordim() == 2, "Expected v2->Coordim to be 2.");

    output[index * 6 + 0] = (*v0)[0];
    output[index * 6 + 1] = (*v0)[1];
    output[index * 6 + 2] = (*v1)[0];
    output[index * 6 + 3] = (*v1)[1];
    output[index * 6 + 4] = (*v2)[0];
    output[index * 6 + 5] = (*v2)[1];
  }

public:
  /**
   *  Create new instance for all 2D geometry objects in ParticleMeshInterface.
   *
   *  @param sycl_target SYCLTarget to use for computation.
   *  @param particle_mesh_interface ParticleMeshInterface containing graph.
   *  @param config ParameterStore instance to set allowable distance to mesh
   * cell tolerance.
   */
  MapParticles2DRegular(
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
