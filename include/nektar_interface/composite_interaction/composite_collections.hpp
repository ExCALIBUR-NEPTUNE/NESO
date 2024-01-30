#ifndef __COMPOSITE_COLLECTIONS_H_
#define __COMPOSITE_COLLECTIONS_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <neso_particles.hpp>
using namespace NESO::Particles;

#include "composite_collection.hpp"
#include "composite_transport.hpp"
#include <nektar_interface/geometry_transport/packed_geom_2d.hpp>
#include <nektar_interface/particle_cell_mapping/newton_geom_interfaces.hpp>
#include <nektar_interface/particle_mesh_interface.hpp>

#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace NESO::CompositeInteraction {

/**
 * TODO
 */
class CompositeCollections {
protected:
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;

  // Stack for device buffers (Newton X mapping data)
  std::stack<std::shared_ptr<BufferDevice<unsigned char>>> stack_geometry_data;
  // Stack for device buffers (LinePlaneIntersections)
  std::stack<std::shared_ptr<BufferDevice<LinePlaneIntersection>>>
      stack_lpi_data;
  // Stack for the device buffers (CompositeCollection)
  std::stack<std::shared_ptr<BufferDevice<CompositeCollection>>>
      stack_collection_data;
  // Stack for the device buffers (composite ids)
  std::stack<std::shared_ptr<BufferDevice<int>>> stack_composite_ids;

  // Cells already collected.
  std::set<INT> collected_cells;

  /**
   *  This method takes the packed geoms and creates the node the in blocked
   *  binary tree such that the kernel can access the geometry info.
   */
  void collect_cell(const INT cell);

public:
  /// Disable (implicit) copies.
  CompositeCollections(const CompositeCollections &st) = delete;
  /// Disable (implicit) copies.
  CompositeCollections &operator=(CompositeCollections const &a) = delete;

  /// SYCLTarget to use for computation.
  SYCLTargetSharedPtr sycl_target;

  /// Map from composites to 2D geometry objects held.
  std::map<int, std::map<int, std::shared_ptr<Geometry2D>>>
      map_composites_to_geoms;

  /// The composite transport instance.
  std::unique_ptr<CompositeTransport> composite_transport;

  /// The container that holds the map from MeshHierarchy cells to the geometry
  /// objects for the composites in those cells.
  std::shared_ptr<BlockedBinaryTree<INT, CompositeCollection *, 4>>
      map_cells_collections;

  /**
   * TODO
   */
  void free();

  /**
   *  TODO
   */
  CompositeCollections(SYCLTargetSharedPtr sycl_target,
                       ParticleMeshInterfaceSharedPtr particle_mesh_interface,
                       std::vector<int> &composite_indices);

  /**
   * TODO
   */
  void collect_geometry(std::set<INT> &cells);
};

} // namespace NESO::CompositeInteraction

#endif
