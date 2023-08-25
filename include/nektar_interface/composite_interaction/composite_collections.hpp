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

  /**
   *  This method takes the packed geoms and creates the node the in blocked
   *  binary tree such that the kernel can access the geometry info.
   */
  inline void collect_cell(const INT cell) {

    std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::QuadGeom>>>
        remote_quads;
    std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::TriGeom>>>
        remote_tris;

    this->composite_transport->get_geometry(cell, remote_quads, remote_tris);

    const int num_quads = remote_quads.size();
    const int num_tris = remote_tris.size();

    Newton::MappingQuadLinear2DEmbed3D mapper_quads{};
    Newton::MappingTriangleLinear2DEmbed3D mapper_tris{};

    const int stride_quads = mapper_quads.data_size_device();
    const int stride_tris = mapper_tris.data_size_device();
    NESOASSERT(mapper_quads.data_size_host() == 0,
               "Expected 0 host bytes required for this mapper.");
    NESOASSERT(mapper_tris.data_size_host() == 0,
               "Expected 0 host bytes required for this mapper.");

    // we pack all the device data for the MeshHierachy cell into a single
    // device buffer
    const int num_bytes = stride_quads * num_quads + stride_tris * num_tris;
    const int offset_tris = stride_quads * num_quads;

    std::vector<unsigned char> buf(num_bytes);
    std::vector<LinePlaneIntersection> buf_lpi{};
    buf_lpi.reserve(num_quads + num_tris);

    unsigned char *map_data_quads = buf.data();
    unsigned char *map_data_tris = buf.data() + offset_tris;
    std::vector<int> composite_ids(num_quads + num_tris);

    for (int gx = 0; gx < num_quads; gx++) {
      auto remote_geom = remote_quads[gx];
      auto geom = remote_geom->geom;
      mapper_quads.write_data(geom, nullptr,
                              map_data_quads + gx * stride_quads);
      LinePlaneIntersection lpi(geom);
      buf_lpi.push_back(lpi);
      composite_ids[gx] = remote_geom->id;
    }

    for (int gx = 0; gx < num_tris; gx++) {
      auto remote_geom = remote_tris[gx];
      auto geom = remote_geom->geom;
      mapper_tris.write_data(geom, nullptr, map_data_tris + gx * stride_tris);
      LinePlaneIntersection lpi(geom);
      buf_lpi.push_back(lpi);
      composite_ids[num_quads + gx] = remote_geom->id;
    }

    // create a device buffer from the vector
    auto d_buf =
        std::make_shared<BufferDevice<unsigned char>>(this->sycl_target, buf);
    this->stack_geometry_data.push(d_buf);
    unsigned char *d_ptr = d_buf->ptr;

    // create the device buffer for the line plane intersection
    auto d_lpi_buf = std::make_shared<BufferDevice<LinePlaneIntersection>>(
        this->sycl_target, buf_lpi);
    this->stack_lpi_data.push(d_lpi_buf);
    LinePlaneIntersection *d_lpi_quads = d_lpi_buf->ptr;
    LinePlaneIntersection *d_lpi_tris = d_lpi_quads + num_quads;

    // device buffer for the composite ids
    auto d_ci_buf =
        std::make_shared<BufferDevice<int>>(this->sycl_target, composite_ids);
    this->stack_composite_ids.push(d_ci_buf);

    // create the CompositeCollection collection object that points to the
    // geometry data we just placed on the device
    std::vector<CompositeCollection> cc(1);
    cc[0].num_quads = num_quads;
    cc[0].num_tris = num_tris;
    cc[0].lpi_quads = d_lpi_quads;
    cc[0].lpi_tris = d_lpi_tris;
    cc[0].stride_quads = stride_quads;
    cc[0].stride_tris = stride_tris;
    cc[0].buf_quads = d_ptr;
    cc[0].buf_tris = d_ptr + offset_tris;
    cc[0].composite_ids_quads = d_ci_buf->ptr;
    cc[0].composite_ids_tris = d_ci_buf->ptr + num_quads;

    // create the device buffer that holds this CompositeCollection
    auto d_cc_buf = std::make_shared<BufferDevice<CompositeCollection>>(
        this->sycl_target, cc);
    this->stack_collection_data.push(d_cc_buf);

    // add the device pointer to the CompositeCollection we just created into
    // the BlockedBinaryTree
    this->map_cells_collections->add(cell, d_cc_buf->ptr);
  }

public:
  /// Disable (implicit) copies.
  CompositeCollections(const CompositeCollections &st) = delete;
  /// Disable (implicit) copies.
  CompositeCollections &operator=(CompositeCollections const &a) = delete;

  /// SYCLTarget to use for computation.
  SYCLTargetSharedPtr sycl_target;

  /// The composite transport instance.
  std::unique_ptr<CompositeTransport> composite_transport;

  /// The container that holds the map from MeshHierarchy cells to the geometry
  /// objects for the composites in those cells.
  std::shared_ptr<BlockedBinaryTree<INT, CompositeCollection *, 4>>
      map_cells_collections;

  /**
   * TODO
   */
  inline void free() { this->composite_transport->free(); }

  /**
   *  TODO
   */
  CompositeCollections(SYCLTargetSharedPtr sycl_target,
                       ParticleMeshInterfaceSharedPtr particle_mesh_interface,
                       std::vector<int> &composite_indices)
      : sycl_target(sycl_target),
        particle_mesh_interface(particle_mesh_interface) {

    this->composite_transport = std::make_unique<CompositeTransport>(
        particle_mesh_interface, composite_indices);

    this->map_cells_collections =
        std::make_shared<BlockedBinaryTree<INT, CompositeCollection *, 4>>(
            this->sycl_target);

    for (auto cx : this->composite_transport->held_cells) {
      this->collect_cell(cx);
    }
  }

  /**
   * TODO
   */
  inline void collect_geometry(std::set<INT> &cells) {
    this->composite_transport->collect_geometry(cells);
    for (auto cx : cells) {
      this->collect_cell(cx);
    }
  }
};

} // namespace NESO::CompositeInteraction

#endif
