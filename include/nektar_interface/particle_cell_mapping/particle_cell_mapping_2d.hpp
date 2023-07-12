#ifndef __PARTICLE_CELL_MAPPING_2D_H__
#define __PARTICLE_CELL_MAPPING_2D_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

#include "../coordinate_mapping.hpp"
#include "../geometry_transport_2d.hpp"
#include "../particle_mesh_interface.hpp"
#include "coarse_lookup_map.hpp"
#include "newton_geom_interfaces.hpp"
#include "particle_cell_mapping_common.hpp"

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

class MapParticles2DRegular {
protected:
  /// Disable (implicit) copies.
  MapParticles2DRegular(const MapParticles2DRegular &st) = delete;
  /// Disable (implicit) copies.
  MapParticles2DRegular &operator=(MapParticles2DRegular const &a) = delete;

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
   */
  MapParticles2DRegular(SYCLTargetSharedPtr sycl_target,
                        ParticleMeshInterfaceSharedPtr particle_mesh_interface);

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  triangles and quads.
   */
  void map(ParticleGroup &particle_group, const int map_cell = -1,
           const double tol = 0.0);
};

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
  std::unique_ptr<MapParticlesHost> map_particles_host;
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
   */
  MapParticles2D(SYCLTargetSharedPtr sycl_target,
                 ParticleMeshInterfaceSharedPtr particle_mesh_interface);

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  triangles and quads.
   */
  void map(ParticleGroup &particle_group, const int map_cell = -1,
           const double tol = 1.0e-10);
};

} // namespace NESO

#endif
