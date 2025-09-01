#ifndef __COMPOSITE_TRANSPORT_H_
#define __COMPOSITE_TRANSPORT_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <nektar_interface/geometry_transport/remote_geom.hpp>
#include <nektar_interface/particle_mesh_interface.hpp>
#include <nektar_interface/typedefs.hpp>

#include <cstdint>
#include <map>
#include <mpi.h>
#include <set>
#include <utility>
#include <vector>

namespace NESO::CompositeInteraction {

/**
 * High level class that collects geometry information for geometry objects
 * which are members of certain composites of interest.
 */
class CompositeTransport {
protected:
  const int ndim;
  MPI_Comm comm;
  bool allocated;
  int rank;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;

  std::shared_ptr<Particles::MeshHierarchyData::MeshHierarchyContainer<
      GeometryTransport::RemoteGeom<SpatialDomains::Geometry>>>
      mh_container;

  std::map<int, int> map_geom_id_to_composite_id;
  std::map<int, int> map_geom_id_to_owning_rank;

public:
  /// Disable (implicit) copies.
  CompositeTransport(const CompositeTransport &st) = delete;
  /// Disable (implicit) copies.
  CompositeTransport &operator=(CompositeTransport const &a) = delete;

  /// The composite indices for which the class detects intersections with.
  const std::vector<int> composite_indices;

  // Are the geoms in the mesh hierarchy cell owned or already requested
  std::set<INT> held_cells;

  // Cells that this rank contributed geoms to.
  std::vector<INT> contrib_cells;

  ~CompositeTransport() { this->free(); }

  /**
   * Free the data structure. Must be called collectively on the communicator.
   */
  void free();

  /**
   * Unpack the geometry for a given cell.
   *
   * @param[in] cell MeshHierarchy cell to unpack geometry for.
   * @param[in, out] remote_quads On return contains the unpacked remote quads
   * for the cell.
   * @param[in, out] remote_tris On return contains the unpacked remote
   * @param[in, out] remote_segments On return contains the unpacked remote
   * segments for the cell.
   */
  void get_geometry(
      const INT cell,
      std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::QuadGeom>>>
          &remote_quads,
      std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::TriGeom>>>
          &remote_tris,
      std::vector<std::shared_ptr<
          GeometryTransport::RemoteGeom<SpatialDomains::SegGeom>>>
          &remote_segments);

  /**
   * Collect on this MPI rank geometry information for requested MeshHierarchy
   * cells. Must be called collectively on the communicator.
   *
   * @param[in, out] cells_in MeshHierarchy cells which are required. On exit
   * hold the cells which are new to this MPI rank.
   * @returns Number of cells collected.
   */
  int collect_geometry(std::set<INT> &cells_in);

  /**
   *  Construct new transport instance for a given mesh and set of composite
   * indices.
   *
   *  @param particle_mesh_interface Mesh to collect geometry information on.
   *  @param composite_indices Composite indices to collect geometry objects
   * for.
   */
  CompositeTransport(ParticleMeshInterfaceSharedPtr particle_mesh_interface,
                     std::vector<int> &composite_indices);

  /**
   * Get the composite for a given geometry object.
   *
   * @param geom_id Global ID of geometry object.
   * @returns Composite ID containting given object.
   */
  int get_composite_id(const int geom_id);

  /**
   * Get the owning rank for a given geometry object.
   *
   * @param geom_id Global ID of geometry object.
   * @returns Owning rank for the given object.
   */
  int get_owning_rank(const int geom_id);
};

} // namespace NESO::CompositeInteraction

#endif
