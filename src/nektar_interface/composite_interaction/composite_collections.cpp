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
  std::vector<
      std::shared_ptr<GeometryTransport::RemoteGeom<SpatialDomains::SegGeom>>>
      remote_segments;
  this->composite_transport->get_geometry(cell, remote_quads, remote_tris,
                                          remote_segments);

  const int num_quads = remote_quads.size();
  const int num_tris = remote_tris.size();
  const int num_segments = remote_segments.size();
  // create the CompositeCollection collection object that points to the
  // geometry data we just placed on the device
  std::vector<CompositeCollection> cc(1);
  cc[0].num_quads = 0;
  cc[0].num_tris = 0;
  cc[0].num_segments = 0;

  // host space for the normals
  std::vector<REAL> h_normal_data;
  std::vector<INT> h_normal_ids;
  std::vector<REAL> normal_tmp;
  int normal_stride = -1;

  auto lambda_push_normal = [&](auto geom_id, auto geom) {
    get_normal_vector(geom, normal_tmp);
    h_normal_data.insert(h_normal_data.end(), normal_tmp.begin(),
                         normal_tmp.end());
    h_normal_ids.push_back(geom_id);
  };

  // host space for group ids
  std::vector<int> h_group_ids_quads;
  std::vector<int> h_group_ids_tris;
  std::vector<int> h_group_ids_segments;

  if ((num_quads > 0) || (num_tris > 0)) {

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

    normal_stride = 3;
    h_normal_data.reserve(3 * (num_quads + num_tris));
    h_normal_ids.reserve(num_quads + num_tris);
    h_group_ids_quads.reserve(num_quads);
    h_group_ids_tris.reserve(num_tris);

    for (int gx = 0; gx < num_quads; gx++) {
      auto remote_geom = remote_quads[gx];
      auto geom = remote_geom->geom;
      mapper_quads.write_data(geom, nullptr,
                              map_data_quads + gx * stride_quads);
      LinePlaneIntersection lpi(geom);
      buf_lpi.push_back(lpi);
      const auto composite_id = remote_geom->rank;
      composite_ids[gx] = composite_id;
      geom_ids[gx] = remote_geom->id;
      this->map_composites_to_geoms[composite_id][remote_geom->id] =
          std::dynamic_pointer_cast<Geometry>(geom);
      lambda_push_normal(remote_geom->id,
                         std::dynamic_pointer_cast<Geometry2D>(geom));
      h_group_ids_quads.push_back(
          this->map_composite_to_group.at(composite_id));
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
          std::dynamic_pointer_cast<Geometry>(geom);
      lambda_push_normal(remote_geom->id,
                         std::dynamic_pointer_cast<Geometry2D>(geom));
      h_group_ids_tris.push_back(this->map_composite_to_group.at(composite_id));
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

    // device buffers for group ids
    auto d_gr_quads_buf = std::make_shared<BufferDevice<int>>(
        this->sycl_target, h_group_ids_quads);
    this->stack_composite_ids.push(d_gr_quads_buf);
    auto d_gr_tris_buf = std::make_shared<BufferDevice<int>>(this->sycl_target,
                                                             h_group_ids_tris);
    this->stack_composite_ids.push(d_gr_tris_buf);

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
    cc[0].group_ids_quads = d_gr_quads_buf->ptr;
    cc[0].group_ids_tris = d_gr_tris_buf->ptr;
  }

  if (num_segments > 0) {
    std::vector<int> geom_ids_segments;
    geom_ids_segments.reserve(num_segments);
    std::vector<int> composite_ids_segments;
    composite_ids_segments.reserve(num_segments);
    std::vector<LineLineIntersection> lli_segments;
    lli_segments.reserve(num_segments);

    normal_stride = 2;
    h_normal_data.reserve(2 * num_segments);
    h_normal_ids.reserve(num_segments);
    h_group_ids_segments.reserve(num_segments);

    for (auto &rs : remote_segments) {
      composite_ids_segments.push_back(rs->rank);
      geom_ids_segments.push_back(rs->id);
      lli_segments.emplace_back(rs->geom);
      this->map_composites_to_geoms[rs->rank][rs->id] =
          std::dynamic_pointer_cast<Geometry>(rs->geom);
      lambda_push_normal(rs->id,
                         std::dynamic_pointer_cast<Geometry1D>(rs->geom));
      h_group_ids_segments.push_back(this->map_composite_to_group.at(rs->rank));
    }

    auto d_gi_buf = std::make_shared<BufferDevice<int>>(this->sycl_target,
                                                        geom_ids_segments);
    this->stack_composite_ids.push(d_gi_buf);
    auto d_ci_buf = std::make_shared<BufferDevice<int>>(this->sycl_target,
                                                        composite_ids_segments);
    this->stack_composite_ids.push(d_ci_buf);
    auto d_lli_buf = std::make_shared<BufferDevice<LineLineIntersection>>(
        this->sycl_target, lli_segments);
    this->stack_lli_data.push(d_lli_buf);

    // device buffers for group ids
    auto d_gr_segs_buf = std::make_shared<BufferDevice<int>>(
        this->sycl_target, h_group_ids_segments);
    this->stack_composite_ids.push(d_gr_segs_buf);

    cc[0].num_segments = num_segments;
    cc[0].lli_segments = d_lli_buf->ptr;
    cc[0].composite_ids_segments = d_ci_buf->ptr;
    cc[0].geom_ids_segments = d_gi_buf->ptr;
    cc[0].group_ids_segments = d_gr_segs_buf->ptr;
  }

  // Having no entry in the tree is faster than having an entry with zero quads,
  // triangles and segments defined. Hence only construct the node if it is
  // non-empty.
  if ((num_quads > 0) || (num_tris > 0) || (num_segments > 0)) {
    // create the device buffer that holds this CompositeCollection
    auto d_cc_buf = std::make_shared<BufferDevice<CompositeCollection>>(
        this->sycl_target, cc);
    this->stack_collection_data.push(d_cc_buf);

    // add the device pointer to the CompositeCollection we just created into
    // the BlockedBinaryTree
    this->map_cells_collections->add(cell, d_cc_buf->ptr);

    // write the normals
    NESOASSERT(normal_stride == 2 || normal_stride == 3, "Bad normal stride.");
    auto d_normal_data =
        std::make_shared<BufferDevice<REAL>>(this->sycl_target, h_normal_data);
    this->stack_real.push(d_normal_data);

    REAL *base_ptr = d_normal_data->ptr;
    for (int ix = 0; ix < h_normal_ids.size(); ix++) {
      this->map_normals->add(h_normal_ids.at(ix),
                             {base_ptr + normal_stride * ix});
    }
  }
}

void CompositeCollections::free() { this->composite_transport->free(); }

CompositeCollections::CompositeCollections(
    SYCLTargetSharedPtr sycl_target,
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    std::map<int, std::vector<int>> boundary_groups)
    : sycl_target(sycl_target),
      particle_mesh_interface(particle_mesh_interface),
      boundary_groups(boundary_groups) {

  std::size_t s = 0;
  for (auto &ix : boundary_groups) {
    s += ix.second.size();
  }
  std::vector<int> composite_indices;
  composite_indices.reserve(s);
  {
    std::set<int> composite_indices_set;
    for (auto &pairx : boundary_groups) {
      for (auto cx : pairx.second) {
        NESOASSERT(composite_indices_set.count(cx) == 0,
                   "Composite is listed more than once or is in multiple "
                   "boundary groups.");
        composite_indices_set.insert(cx);
        composite_indices.push_back(cx);
        this->map_composite_to_group[cx] = pairx.first;
      }
    }
  }

  this->composite_transport = std::make_shared<CompositeTransport>(
      particle_mesh_interface, composite_indices);

  this->map_cells_collections =
      std::make_shared<BlockedBinaryTree<INT, CompositeCollection *, 4>>(
          this->sycl_target);

  this->map_normals = std::make_shared<BlockedBinaryTree<INT, NormalData, 8>>(
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

NormalMapper CompositeCollections::get_device_normal_mapper() {
  NormalMapper mapper;
  mapper.root = this->map_normals->root;
  return mapper;
}

} // namespace NESO::CompositeInteraction
