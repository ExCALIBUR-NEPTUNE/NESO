#include <nektar_interface/composite_interaction/composite_transport.hpp>

namespace NESO::CompositeInteraction {

void CompositeTransport::free() {
  if (this->allocated) {
    this->mh_container->free();
    this->mh_container = nullptr;
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

  std::vector<GeometryTransport::RemoteGeom<SpatialDomains::Geometry>> geoms;
  this->mh_container->get(cell, geoms);

  if (geoms.size() > 0) {
    if (this->ndim == 3) {
      for (auto gx : geoms) {
        auto shape_type = gx.geom->GetShapeType();
        if (shape_type == LibUtilities::eTriangle) {
          auto ptr = std::dynamic_pointer_cast<TriGeom>(gx.geom);
          NESOASSERT(ptr.get() != nullptr, "bad cast of ptr to TriGeom");
          remote_tris.push_back(
              std::make_shared<RemoteGeom2D<TriGeom>>(gx.rank, gx.id, ptr));
        } else {
          auto ptr = std::dynamic_pointer_cast<QuadGeom>(gx.geom);
          NESOASSERT(ptr.get() != nullptr, "bad cast of ptr to QuadGeom");
          remote_quads.push_back(
              std::make_shared<RemoteGeom2D<QuadGeom>>(gx.rank, gx.id, ptr));
        }
      }
    } else if (this->ndim == 2) {

    } else {
      NESOASSERT(false, "not implemented in 1D");
    }
  }
}

int CompositeTransport::collect_geometry(std::set<INT> &cells_in) {

  // remove cells we already hold the geoms for
  std::set<INT> cells;
  for (auto cx : cells_in) {
    if (!this->held_cells.count(cx)) {
      cells.insert(cx);
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
    std::vector<INT> cells_vector;
    cells_vector.reserve(cells.size());
    for (auto cx : cells) {
      cells_vector.push_back(cx);
    }

    this->mh_container->gather(cells_vector);
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
  std::map<INT,
           std::vector<GeometryTransport::RemoteGeom<SpatialDomains::Geometry>>>
      cell_geom_map;
  std::set<INT> contrib_cells_set;

  const double bounding_box_padding = 0.02;
  const int mask = std::numeric_limits<int>::min();

  for (auto geom_pair : geometry_composites) {
    SpatialDomains::GeometrySharedPtr geom = geom_pair.first;
    const int composite = geom_pair.second;

    // find all mesh hierarchy cells the geom intersects with
    std::deque<std::pair<INT, double>> cells;
    bounding_box_map(geom, particle_mesh_interface->mesh_hierarchy, cells,
                     bounding_box_padding);

    GeometryTransport::RemoteGeom<SpatialDomains::Geometry> rgeom(
        composite, geom->GetGlobalID(), geom);

    for (auto cell_overlap : cells) {
      const INT cell = cell_overlap.first;
      const int owning_rank =
          particle_mesh_interface->mesh_hierarchy->get_owner(cell);
      // Some MH cells do not have owners as there was zero overlap between
      // elements and the MH cell
      if (owning_rank != mask) {
        cell_geom_map[cell].push_back(rgeom);
        contrib_cells_set.insert(cell);
      }
    }
  }

  this->contrib_cells.reserve(contrib_cells_set.size());
  for (auto cx : contrib_cells_set) {
    this->contrib_cells.push_back(cx);
  }

  this->mh_container =
      std::make_shared<Particles::MeshHierarchyData::MeshHierarchyContainer<
          GeometryTransport::RemoteGeom<SpatialDomains::Geometry>>>(
          particle_mesh_interface->get_mesh_hierarchy(), cell_geom_map);
}

} // namespace NESO::CompositeInteraction
