#ifndef __PARTICLE_CELL_MAPPING_3D_H__
#define __PARTICLE_CELL_MAPPING_3D_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <tuple>
#include <vector>

#include "../coordinate_mapping.hpp"
#include "../geometry_transport_2d.hpp"
#include "../geometry_transport_3d.hpp"
#include "../particle_mesh_interface.hpp"
#include "coarse_lookup_map.hpp"
#include "particle_cell_mapping_common.hpp"

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

#include "newton_geom_interfaces.hpp"

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

class MapParticles3DRegular {
protected:
  /// Disable (implicit) copies.
  MapParticles3DRegular(const MapParticles3DRegular &st) = delete;
  /// Disable (implicit) copies.
  MapParticles3DRegular &operator=(MapParticles3DRegular const &a) = delete;

  SYCLTargetSharedPtr sycl_target;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  std::unique_ptr<CoarseLookupMap> coarse_lookup_map;

  int num_regular_geoms;

  /// The nektar++ cell id for the cells indices pointed to from the map.
  std::unique_ptr<BufferDeviceHost<int>> dh_cell_ids;
  /// The MPI rank that owns the cell.
  std::unique_ptr<BufferDeviceHost<int>> dh_mpi_ranks;
  /// The type of the cell, i.e. a quad or a triangle.
  std::unique_ptr<BufferDeviceHost<int>> dh_type;
  /// The 3 vertices required by mapping from physical space to reference space.
  std::unique_ptr<BufferDeviceHost<double>> dh_vertices;

  std::unique_ptr<ErrorPropagate> ep;

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
   */
  MapParticles3DRegular(SYCLTargetSharedPtr sycl_target,
                        ParticleMeshInterfaceSharedPtr particle_mesh_interface);

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  3D Geometry objects.
   */
  void map(ParticleGroup &particle_group, const int map_cell = -1,
           const double tol = 0.0);
};

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
                                  const int map_cell, const double tol) {
    if (ptr) {
      ptr->map(particle_group, map_cell, tol);
    }
  }

public:
  /**
   *  Constructor for mapping class.
   *
   *  @param sycl_target SYCLTarget on which to perform mapping.
   *  @param particle_mesh_interface ParticleMeshInterface containing 2D
   * Nektar++ cells.
   */
  MapParticles3D(SYCLTargetSharedPtr sycl_target,
                 ParticleMeshInterfaceSharedPtr particle_mesh_interface);

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  triangles and quads.
   */
  void map(ParticleGroup &particle_group, const int map_cell = -1,
           const double tol = 1.0e-8);
};

} // namespace NESO

#endif
