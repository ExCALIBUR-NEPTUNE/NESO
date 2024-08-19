#include <nektar_interface/composite_interaction/composite_transport.hpp>

namespace NESO::CompositeInteraction {

void CompositeTransport::free() {
  this->composite_communication->free();
  if (this->allocated) {
    MPICHK(MPI_Comm_free(&this->comm));
    this->allocated = false;
  }
}

void CompositeTransport::get_geometry(
    const INT cell,
    std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::QuadGeom>>>
        &remote_quads,
    std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::TriGeom>>>
        &remote_tris) {

  remote_quads.clear();
  remote_tris.clear();
  const int size = this->packed_geoms_count[cell].value;
  if (size > 0) {
    unsigned char *buffer = this->packed_geoms.at(cell).data();
    std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::Geometry2D>>>
        remote_geoms;
    PackedGeoms2D packed_geoms_2d(buffer, size);
    packed_geoms_2d.unpack(remote_geoms);

    for (auto rgeom : remote_geoms) {
      auto shape_type = rgeom->geom->GetShapeType();
      NESOASSERT(shape_type == eQuadrilateral || shape_type == eTriangle,
                 "Expected Triangle or Quadrilateral");
      if (shape_type == eTriangle) {
        auto cgeom = std::make_shared<RemoteGeom2D<TriGeom>>(
            rgeom->rank, rgeom->id,
            std::dynamic_pointer_cast<TriGeom>(rgeom->geom));
        remote_tris.push_back(cgeom);
      } else {
        auto cgeom = std::make_shared<RemoteGeom2D<QuadGeom>>(
            rgeom->rank, rgeom->id,
            std::dynamic_pointer_cast<QuadGeom>(rgeom->geom));
        remote_quads.push_back(cgeom);
      }
    }
  }
}

int CompositeTransport::collect_geometry(std::set<INT> &cells_in) {

  // remove cells we already hold the geoms for
  std::set<INT> cells;
  for (auto cx : cells_in) {

    if (!this->held_cells.count(cx)) {
      cells.insert(cx);
      NESOASSERT(this->packed_geoms.count(cx) == 0,
                 "This cell is not collected but somehow this rank holds "
                 "geoms for the cell.");
    }
  }

  const int num_cells_to_collect_local = cells.size();
  int num_cells_to_collect_global_max = -1;
  MPICHK(MPI_Allreduce(&num_cells_to_collect_local,
                       &num_cells_to_collect_global_max, 1, MPI_INT, MPI_MAX,
                       this->comm));
  NESOASSERT(num_cells_to_collect_global_max > -1,
             "global max computation failed");

  // if no ranks require geoms we skip all this communication
  if (num_cells_to_collect_global_max) {

    auto mesh_hierarchy = this->particle_mesh_interface->mesh_hierarchy;

    // ranks sending geoms
    std::set<int> send_ranks_set;
    std::map<int, std::vector<std::int64_t>> rank_send_cells_map;
    const int mask = std::numeric_limits<int>::min();
    for (auto cx : cells) {
      const int owning_rank = mesh_hierarchy->get_owner(cx);
      // The MH may have cells that were never claimed and hence do not have
      // an owning rank. Not claimed implies no geoms intersect the MH cell.
      if (owning_rank != mask) {
        rank_send_cells_map[owning_rank].push_back(cx);
        send_ranks_set.insert(owning_rank);
      }
    }
    std::vector<int> send_ranks;
    send_ranks.reserve(send_ranks_set.size());
    std::vector<int> send_counts;
    send_counts.reserve(send_ranks_set.size());
    for (auto sx : send_ranks_set) {
      send_ranks.push_back(sx);
      send_counts.push_back(rank_send_cells_map.at(sx).size());
    }

    std::vector<int> recv_ranks;
    this->composite_communication->get_in_edges(send_ranks, recv_ranks);
    const int num_recv_ranks = recv_ranks.size();
    std::vector<int> recv_counts(num_recv_ranks);
    this->composite_communication->exchange_send_counts(
        send_ranks, recv_ranks, send_counts, recv_counts);

    std::map<int, std::vector<std::int64_t>> rank_recv_cells_map;
    for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
      const int remote_rank = recv_ranks[rankx];
      rank_recv_cells_map[remote_rank] =
          std::vector<std::int64_t>(recv_counts[rankx]);
    }

    this->composite_communication->exchange_requested_cells(
        send_ranks, recv_ranks, send_counts, recv_counts, rank_send_cells_map,
        rank_recv_cells_map);

    this->composite_communication->exchange_requested_cells_counts(
        send_ranks, recv_ranks, rank_send_cells_map, rank_recv_cells_map,
        this->packed_geoms_count);

    this->composite_communication->exchange_packed_cells(
        this->max_buf_size, send_ranks, recv_ranks, rank_send_cells_map,
        rank_recv_cells_map, this->packed_geoms_count, this->packed_geoms);
  }
  for (auto cx : cells) {
    this->held_cells.insert(cx);
  }
  return cells.size();
}

CompositeTransport::CompositeTransport(
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    std::vector<int> &composite_indices)
    : ndim(particle_mesh_interface->graph->GetMeshDimension()),
      composite_indices(composite_indices),
      particle_mesh_interface(particle_mesh_interface) {

  MPICHK(MPI_Comm_dup(particle_mesh_interface->get_comm(), &this->comm));
  this->allocated = true;
  MPICHK(MPI_Comm_rank(this->comm, &this->rank));

  this->composite_communication =
      std::make_unique<CompositeCommunication>(this->comm);

  auto graph = particle_mesh_interface->graph;
  // map from composite indices to CompositeSharedPtr
  auto graph_composites = graph->GetComposites();

  // Vector of geometry objects and the composite they correspond to
  std::vector<std::pair<SpatialDomains::GeometrySharedPtr, int>>
      geometry_composites;

  for (auto ix : composite_indices) {
    // check the composite of interest exists in the MeshGraph on this rank
    if (graph_composites.count(ix)) {
      // each composite has a vector of GeometrySharedPtrs in the m_geomVec
      // attribute
      auto geoms = graph_composites.at(ix)->m_geomVec;
      geometry_composites.reserve(geoms.size() + geometry_composites.size());
      for (auto &geom : geoms) {
        auto shape_type = geom->GetShapeType();
        NESOASSERT(
            ((ndim == 3) && (shape_type == LibUtilities::eTriangle ||
                             shape_type == LibUtilities::eQuadrilateral)) ||
                ((ndim == 2) && (shape_type == LibUtilities::eSegment)),
            "unknown composite shape type");
        geometry_composites.push_back({geom, ix});
      }
    }
  }

  // pack each geometry object and reuse the local_id which originally stored
  // local neso cell index to store composite index
  std::set<int> send_ranks_set;
  std::map<int, std::vector<
                    std::shared_ptr<RemoteGeom2D<SpatialDomains::Geometry2D>>>>
      rank_element_map;

  const double bounding_box_padding = 0.02;
  const int mask = std::numeric_limits<int>::min();

  for (auto geom_pair : geometry_composites) {
    // pack the geom and composite into a unsigned char buffer
    SpatialDomains::GeometrySharedPtr geom = geom_pair.first;
    const int composite = geom_pair.second;
    auto geom_2d = std::dynamic_pointer_cast<SpatialDomains::Geometry2D>(geom);

    // find all mesh hierarchy cells the geom intersects with
    std::deque<std::pair<INT, double>> cells;
    bounding_box_map(geom_2d, particle_mesh_interface->mesh_hierarchy, cells,
                     bounding_box_padding);
    auto rgeom_2d = std::make_shared<RemoteGeom2D<SpatialDomains::Geometry2D>>(
        composite, geom_2d->GetGlobalID(), geom_2d);
    for (auto cell_overlap : cells) {
      const INT cell = cell_overlap.first;
      const int owning_rank =
          particle_mesh_interface->mesh_hierarchy->get_owner(cell);
      // Some MH cells do not have owners as there was zero overlap between
      // elements and the MH cell
      if (owning_rank != mask) {
        send_ranks_set.insert(owning_rank);
        rank_element_map[owning_rank].push_back(rgeom_2d);
      }
    }
  }

  std::vector<int> send_ranks{};
  send_ranks.reserve(send_ranks_set.size());
  for (int rankx : send_ranks_set) {
    send_ranks.push_back(rankx);
  }

  // ranks this rank will recieve geoms it owns from
  std::vector<int> recv_ranks;
  this->composite_communication->get_in_edges(send_ranks, recv_ranks);

  // collect on each rank composites that intersect with the mesh hierarchy
  // cells the rank owns
  std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::Geometry2D>>>
      output_container;
  halo_exchange_geoms_2d(this->comm, send_ranks.size(), send_ranks,
                         recv_ranks.size(), recv_ranks, rank_element_map,
                         output_container);

  std::map<INT, std::vector<
                    std::shared_ptr<RemoteGeom2D<SpatialDomains::Geometry2D>>>>
      map_cell_rgeom;

  for (auto remote_geom : output_container) {
    auto geom = remote_geom->geom;

    // find all mesh hierarchy cells the geom intersects with
    std::deque<std::pair<INT, double>> cells;
    bounding_box_map(geom, particle_mesh_interface->mesh_hierarchy, cells,
                     bounding_box_padding);
    for (auto cell_overlap : cells) {
      const INT cell = cell_overlap.first;
      const int owning_rank =
          particle_mesh_interface->mesh_hierarchy->get_owner(cell);
      if (owning_rank == this->rank) {
        map_cell_rgeom[cell].push_back(remote_geom);
        this->held_cells.insert(cell);
      }
    }
  }

  size_t max_buf_size_tmps = 0;
  for (INT cell : this->held_cells) {
    PackedGeoms2D packed_geoms_2d(map_cell_rgeom.at(cell));
    packed_geoms[cell].insert(std::end(packed_geoms[cell]),
                              std::begin(packed_geoms_2d.buf),
                              std::end(packed_geoms_2d.buf));
    max_buf_size_tmps =
        std::max(max_buf_size_tmps, packed_geoms.at(cell).size());
    this->packed_geoms_count[cell].value = packed_geoms.at(cell).size();
  }

  // This object holds 2D geoms for all mh cells this rank owns
  // Add the MH cells which don't intersect a 2D geom.
  for (auto &cellx : particle_mesh_interface->owned_mh_cells) {
    this->held_cells.insert(cellx);
  }

  std::uint64_t max_buf_size_tmpi =
      static_cast<std::uint64_t>(max_buf_size_tmps);
  MPICHK(MPI_Allreduce(&max_buf_size_tmpi, &this->max_buf_size, 1, MPI_UINT64_T,
                       MPI_MAX, this->comm));

  for (auto &cell : packed_geoms) {
    cell.second.resize(this->max_buf_size);
  }
}

} // namespace NESO::CompositeInteraction
