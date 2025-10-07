#include <nektar_interface/composite_interaction/composite_interaction_boundary_conditions.hpp>

namespace NESO::CompositeInteraction {

CompositeIntersectionBoundaryConditions::
    CompositeIntersectionBoundaryConditions(
        const LibUtilities::SessionReaderSharedPtr &session,
        const SpatialDomains::MeshGraphSharedPtr &graph)
    :

      SpatialDomains::BoundaryConditions(session, graph) {

  int index = 0;
  for (auto ix : m_boundaryRegions) {
    std::vector<int> composite_indices;
    composite_indices.reserve(ix.second->size());
    for (auto jx : *ix.second) {
      const int composite_index = jx.first;
      NESOASSERT(!this->map_composite_to_exp_index.count(composite_index),
                 "Composite index already in map.");
      this->map_composite_to_exp_index[composite_index] = index++;
      composite_indices.push_back(composite_index);
    }
    this->boundary_groups[ix.first] = composite_indices;
  }

  const int comm_size_session = session->GetComm()->GetSize();
  int size;
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  NESOASSERT(size == comm_size_session,
             "MPI Comm missmatch between what was assumed and what was found "
             "at runtime.");

  std::set<int> bids_local;
  for (auto ix : this->boundary_groups) {
    bids_local.insert(ix.first);
  }

  std::set<int> bids_global =
      Particles::set_all_reduce_union(bids_local, MPI_COMM_WORLD);

  for (auto bx : bids_global) {
    std::set<int> cids_local;
    for (auto cx : this->boundary_groups[bx]) {
      cids_local.insert(cx);
    }
    std::set<int> cids_global =
        Particles::set_all_reduce_union(cids_local, MPI_COMM_WORLD);
    this->boundary_groups[bx].clear();
    this->boundary_groups[bx].reserve(cids_global.size());
    for (int cx : cids_global) {
      this->boundary_groups[bx].push_back(cx);
    }
  }
}

std::map<int, std::vector<int>>
CompositeIntersectionBoundaryConditions::get_boundary_groups() {
  return this->boundary_groups;
}

int CompositeIntersectionBoundaryConditions::get_expansion_index(
    const int composite_index) {
  if (this->map_composite_to_exp_index.count(composite_index)) {
    return this->map_composite_to_exp_index[composite_index];
  } else {
    return -1;
  }
}

SpatialDomains::MeshGraphSharedPtr
CompositeIntersectionBoundaryConditions::get_mesh_graph() {
  return this->m_meshGraph;
}
} // namespace NESO::CompositeInteraction
