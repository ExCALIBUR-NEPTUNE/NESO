#ifndef __NEKTAR_INTERFACE_COMPOSITE_INTERACTION_COMPOSITE_COLLECTIONS_H_
#define __NEKTAR_INTERFACE_COMPOSITE_INTERACTION_COMPOSITE_COLLECTIONS_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include "composite_collection.hpp"
#include "composite_transport.hpp"
#include "composite_utility.hpp"
#include <nektar_interface/geometry_transport/packed_geom_2d.hpp>
#include <nektar_interface/particle_cell_mapping/newton_geom_interfaces.hpp>
#include <nektar_interface/particle_mesh_interface.hpp>
#include <nektar_interface/typedefs.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace NESO::CompositeInteraction {

/**
 * Type to point to the normal vector for a boundary element.
 */
struct NormalData {
  REAL *d_normal;
};

/**
 * Device type to help finding the normal vector for a particle-boundary
 * interaction.
 */
struct NormalMapper {
  // Root of the tree containing normal data.
  BlockedBinaryNode<INT, NormalData, 8> *root;

  /**
   * Get a pointer to the normal data for an edge element.
   *
   * @param[in] global_id Global index of boundary element to retrive normal
   * for.
   * @param[in, out] normal Pointer to populate with address of normal vector.
   * @returns True if the boundary element is found otherwise false.
   */
  inline bool get(const INT global_id, REAL **normal) const {
    NormalData *node;
    bool *exists;
    const bool e = root->get_location(global_id, &exists, &node);
    *normal = node->d_normal;
    return e && (*exists);
  }
};

/**
 * Distributed data structure to hold the geometry information that intersects
 * each Mesh Hierarchy cell.
 */
class CompositeCollections {
public:
  using QuadDataDevice =
      typename Newton::MappingQuadLinear2DEmbed3D::DataDevice;
  using TriangleDataDevice =
      typename Newton::MappingTriangleLinear2DEmbed3D::DataDevice;

protected:
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;

  // Stack for device buffers (Newton X mapping data)
  std::stack<std::shared_ptr<BufferDevice<QuadDataDevice>>> geometry_data_quads;
  std::stack<std::shared_ptr<BufferDevice<TriangleDataDevice>>>
      geometry_data_tris;

  // Stack for device buffers (LinePlaneIntersections)
  std::stack<std::shared_ptr<BufferDevice<LinePlaneIntersection>>>
      stack_lpi_data;
  // Stack for device buffers (LineLineIntersections)
  std::stack<std::shared_ptr<BufferDevice<LineLineIntersection>>>
      stack_lli_data;
  // Stack for the device buffers (CompositeCollection)
  std::stack<std::shared_ptr<BufferDevice<CompositeCollection>>>
      stack_collection_data;
  // Stack for the device buffers (composite ids)
  std::stack<std::shared_ptr<BufferDevice<int>>> stack_composite_ids;
  // stack for device buffers (REAL data)
  std::stack<std::shared_ptr<BufferDevice<REAL>>> stack_real;

  // Cells already collected.
  std::set<INT> collected_cells;

  /**
   *  This method takes the packed geoms and creates the node the in blocked
   *  binary tree such that the kernel can access the geometry info.
   */
  void collect_cell(const INT cell);

  /// The normal data for each geometry object collected.
  std::shared_ptr<BlockedBinaryTree<INT, NormalData, 8>> map_normals;

  /// Inverse map from composite to boundary group.
  std::map<int, int> map_composite_to_group;

public:
  /// Disable (implicit) copies.
  CompositeCollections(const CompositeCollections &st) = delete;
  /// Disable (implicit) copies.
  CompositeCollections &operator=(CompositeCollections const &a) = delete;

  /// SYCLTarget to use for computation.
  SYCLTargetSharedPtr sycl_target;

  /// Map from composites to geometry objects held.
  std::map<int, std::map<int, std::shared_ptr<Geometry>>>
      map_composites_to_geoms;

  // Map from geometry ids to geometry objects.
  std::map<int, std::shared_ptr<Geometry>> map_geom_id_to_geoms;

  // Map from geometry ids to composite ids.
  std::map<int, int> map_geom_id_to_composite_id;

  /// The composite transport instance.
  std::shared_ptr<CompositeTransport> composite_transport;

  /// The container that holds the map from MeshHierarchy cells to the geometry
  /// objects for the composites in those cells (faces).
  std::shared_ptr<BlockedBinaryTree<INT, CompositeCollection *, 4>>
      map_cells_collections;

  /// Map from boundary group id to composites in the group.
  std::map<int, std::vector<int>> boundary_groups;

  /// Maximum number of modes for X maps.
  int max_num_modes;

  /**
   * Free the data structure. Must be called collectively on the communicator.
   */
  void free();

  /**
   *  Create new distributed container for geometry objects which are members
   *  of the passed composite indices.
   *
   *  @param sycl_target Compute device on which intersection computation will
   *  take place.
   *  @param particle_mesh_interface ParticleMeshInterface for the domain.
   *  @param boundary_groups Map from boundary group id to composite ids which
   *  form the group.
   */
  CompositeCollections(SYCLTargetSharedPtr sycl_target,
                       ParticleMeshInterfaceSharedPtr particle_mesh_interface,
                       std::map<int, std::vector<int>> boundary_groups);

  /**
   * Must be called collectively on the communicator. Collect on this MPI rank
   * all the geometry objects which intersect certain MeshHierarchy cells.
   *
   * @param cells MeshHierarchy cells to collect all geometry objects for.
   */
  void collect_geometry(std::set<INT> &cells);

  /**
   * Get a device callable mapper to map from global boundary indices to normal
   * vectors. This method should be called after @ref collect_geometry as the
   * map is populated with potentially relvant normal data on the fly.
   *
   * @returns Device copyable and callable mapper.
   */
  NormalMapper get_device_normal_mapper();
};

} // namespace NESO::CompositeInteraction

#endif
