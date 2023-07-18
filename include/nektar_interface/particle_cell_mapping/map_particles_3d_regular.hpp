#ifndef __MAP_PARTICLES_3D_REGULAR_H__
#define __MAP_PARTICLES_3D_REGULAR_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <tuple>
#include <vector>

#include "coarse_mappers_base.hpp"
#include "nektar_interface/coordinate_mapping.hpp"
#include "nektar_interface/geometry_transport/shape_mapping.hpp"
#include "nektar_interface/particle_mesh_interface.hpp"
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
 *  * MapParticles3DRegular/tol: Tolerance to apply when determining if a
 * particle is within a geometry object (default 0.0).
 *
 */
class MapParticles3DRegular : public CoarseMappersBase {
protected:
  /// Disable (implicit) copies.
  MapParticles3DRegular(const MapParticles3DRegular &st) = delete;
  /// Disable (implicit) copies.
  MapParticles3DRegular &operator=(MapParticles3DRegular const &a) = delete;

  ParticleMeshInterfaceSharedPtr particle_mesh_interface;

  /// Tolerance to use to determine if a particle is within a geometry object.
  REAL tol;

  int num_regular_geoms;
  /// The 3 vertices required by mapping from physical space to reference space.
  std::unique_ptr<BufferDeviceHost<double>> dh_vertices;

  template <typename U>
  inline void write_vertices_3d(U &geom, const int index, double *output) {
    const auto shape_type = geom->GetShapeType();
    int index_v[4];
    index_v[0] = 0; // v0 is actually 0
    if (shape_type == LibUtilities::eHexahedron ||
        shape_type == LibUtilities::ePrism ||
        shape_type == LibUtilities::ePyramid) {
      index_v[1] = 1;
      index_v[2] = 3;
      index_v[3] = 4;
    } else if (shape_type == LibUtilities::eTetrahedron) {
      index_v[1] = 1;
      index_v[2] = 2;
      index_v[3] = 3;
    } else {
      NESOASSERT(false, "get_local_coords_3d Unknown shape type.");
    }

    for (int vx = 0; vx < 4; vx++) {
      auto vertex = geom->GetVertex(index_v[vx]);
      NESOASSERT(vertex->GetCoordim() == 3, "Expected Coordim to be 3.");
      NekDouble x, y, z;
      vertex->GetCoords(x, y, z);
      output[index * 12 + vx * 3 + 0] = x;
      output[index * 12 + vx * 3 + 1] = y;
      output[index * 12 + vx * 3 + 2] = z;
    }
  }

public:
  /**
   *  Create new mapper object for all 3D regular geometry objects in a
   *  ParticleMeshInterface.
   *
   *  @param sycl_target SYCLTarget to use for computation.
   *  @param particle_mesh_interface ParticleMeshInterface containing Nektar++
   *  MeshGraph.
   *  @param config ParameterStore instance to set allowable distance to mesh
   * cell tolerance.
   */
  MapParticles3DRegular(
      SYCLTargetSharedPtr sycl_target,
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      ParameterStoreSharedPtr config = std::make_shared<ParameterStore>());

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  3D Geometry objects.
   */
  void map(ParticleGroup &particle_group, const int map_cell = -1);
};

} // namespace NESO

#endif
