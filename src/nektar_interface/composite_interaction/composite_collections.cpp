#include <nektar_interface/composite_interaction/composite_collections.hpp>

namespace NESO::CompositeInteraction {

void CompositeCollections::collect_cell(const INT cell) {
  // Don't reprocess a cell that has already been processed.
  if (this->collected_cells.count(cell)) {
    return;
  }
  this->collected_cells.insert(cell);

  std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::QuadGeom>>>
      remote_quads;
  std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::TriGeom>>>
      remote_tris;

  this->composite_transport->get_geometry(cell, remote_quads, remote_tris);

  const int num_quads = remote_quads.size();
  const int num_tris = remote_tris.size();

  if ((num_quads == 0) && (num_tris == 0)) {
    return;
  }

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
  std::vector<int> geom_ids(num_quads + num_tris);

  for (int gx = 0; gx < num_quads; gx++) {
    auto remote_geom = remote_quads[gx];
    auto geom = remote_geom->geom;
    mapper_quads.write_data(geom, nullptr, map_data_quads + gx * stride_quads);
    LinePlaneIntersection lpi(geom);
    buf_lpi.push_back(lpi);
    const auto composite_id = remote_geom->rank;
    composite_ids[gx] = composite_id;
    geom_ids[gx] = remote_geom->id;
    this->map_composites_to_geoms[composite_id][remote_geom->id] =
        std::dynamic_pointer_cast<Geometry2D>(geom);
  }

  for (int gx = 0; gx < num_tris; gx++) {
    auto remote_geom = remote_tris[gx];
    auto geom = remote_geom->geom;
    mapper_tris.write_data(geom, nullptr, map_data_tris + gx * stride_tris);
    LinePlaneIntersection lpi(geom);
    buf_lpi.push_back(lpi);
    const auto composite_id = remote_geom->rank;
    composite_ids[num_quads + gx] = composite_id;
    geom_ids[num_quads + gx] = remote_geom->id;
    this->map_composites_to_geoms[composite_id][remote_geom->id] =
        std::dynamic_pointer_cast<Geometry2D>(geom);
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
  // device buffer for the geom ids
  auto d_gi_buf =
      std::make_shared<BufferDevice<int>>(this->sycl_target, geom_ids);
  this->stack_composite_ids.push(d_gi_buf);

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
  cc[0].geom_ids_quads = d_gi_buf->ptr;
  cc[0].geom_ids_tris = d_gi_buf->ptr + num_quads;

  // create the device buffer that holds this CompositeCollection
  auto d_cc_buf = std::make_shared<BufferDevice<CompositeCollection>>(
      this->sycl_target, cc);
  this->stack_collection_data.push(d_cc_buf);

  // add the device pointer to the CompositeCollection we just created into
  // the BlockedBinaryTree
  this->map_cells_collections->add(cell, d_cc_buf->ptr);
}

void CompositeCollections::free() { this->composite_transport->free(); }

CompositeCollections::CompositeCollections(
    SYCLTargetSharedPtr sycl_target,
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

void CompositeCollections::collect_geometry(std::set<INT> &cells) {
  this->composite_transport->collect_geometry(cells);
  for (auto cx : cells) {
    this->collect_cell(cx);
  }
}

} // namespace NESO::CompositeInteraction
