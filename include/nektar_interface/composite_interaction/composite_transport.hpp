#ifndef __COMPOSITE_TRANSPORT_H_
#define __COMPOSITE_TRANSPORT_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <neso_particles.hpp>
using namespace NESO::Particles;

#include <nektar_interface/geometry_transport/halo_extension.hpp>
#include <nektar_interface/geometry_transport/packed_geom_2d.hpp>
#include <nektar_interface/particle_mesh_interface.hpp>

#include <cstdint>
#include <map>
#include <mpi.h>
#include <set>
#include <utility>
#include <vector>

#include "composite_communication.hpp"

namespace NESO::CompositeInteraction {

/**
 * TODO
 */
class CompositeTransport {
protected:
  const int ndim;
  std::unique_ptr<CompositeCommunication> composite_communication;
  MPI_Comm comm;
  bool allocated;
  // map from mesh hierarchy cells to the packed geoms for that cell
  std::map<INT, std::vector<unsigned char>> packed_geoms;
  int rank;
  // the maximum size of the packed geoms accross all ranks
  std::uint64_t max_buf_size;

  ParticleMeshInterfaceSharedPtr particle_mesh_interface;

public:
  /// Disable (implicit) copies.
  CompositeTransport(const CompositeTransport &st) = delete;
  /// Disable (implicit) copies.
  CompositeTransport &operator=(CompositeTransport const &a) = delete;

  /// The composite indices for which the class detects intersections with.
  const std::vector<int> composite_indices;

  ~CompositeTransport() { this->free(); }

  /**
   *  TODO
   */
  inline void free() {
    this->composite_communication->free();
    if (this->allocated) {
      MPICHK(MPI_Comm_free(&this->comm));
      this->allocated = false;
    }
  }

  /**
   * TODO
   */
  inline void collect_geometry(std::set<INT> &cells) {

    // remove cells we already hold the geoms for
    std::set<INT> cells_tmp;
    for (auto cx : cells) {
      cells_tmp.insert(cx);
    }
    cells.clear();
    for (auto cx : cells_tmp) {
      if (!packed_geoms.count(cx)) {
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

      auto mesh_hierarchy = this->particle_mesh_interface->mesh_hierarchy;

      // ranks sending geoms
      std::set<int> send_ranks_set;
      std::map<int, std::vector<std::int64_t>> rank_send_cells_map;
      for (auto cx : cells) {
        const int owning_rank = mesh_hierarchy->get_owner(cx);
        rank_send_cells_map[owning_rank].push_back(cx);
        send_ranks_set.insert(owning_rank);
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

      this->composite_communication->exchange_packed_cells(
          this->max_buf_size, send_ranks, recv_ranks, rank_send_cells_map,
          rank_recv_cells_map, this->packed_geoms);
    }
  }

  /**
   *  TODO
   */
  CompositeTransport(ParticleMeshInterfaceSharedPtr particle_mesh_interface,
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
          NESOASSERT(shape_type == LibUtilities::eTriangle ||
                         shape_type == LibUtilities::eQuadrilateral,
                     "unknown composite shape type");
          geometry_composites.push_back({geom, ix});
        }
      }
    }

    // pack each geometry object and reuse the local_id which originally stored
    // local neso cell index to store composite index
    set<int> send_ranks_set;
    std::map<
        int,
        std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::Geometry2D>>>>
        rank_element_map;

    const double bounding_box_padding = 0.02;

    for (auto geom_pair : geometry_composites) {
      // pack the geom and composite into a unsigned char buffer
      SpatialDomains::GeometrySharedPtr geom = geom_pair.first;
      const int composite = geom_pair.second;
      auto geom_2d =
          std::dynamic_pointer_cast<SpatialDomains::Geometry2D>(geom);

      // find all mesh hierarchy cells the geom intersects with
      std::deque<std::pair<INT, double>> cells;
      bounding_box_map(geom_2d, particle_mesh_interface->mesh_hierarchy, cells,
                       bounding_box_padding);
      auto rgeom_2d =
          std::make_shared<RemoteGeom2D<SpatialDomains::Geometry2D>>(
              0, composite, geom_2d);
      for (auto cell_overlap : cells) {
        const INT cell = cell_overlap.first;
        const int owning_rank =
            particle_mesh_interface->mesh_hierarchy->get_owner(cell);
        send_ranks_set.insert(owning_rank);
        rank_element_map[owning_rank].push_back(rgeom_2d);
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

    size_t max_buf_size_tmps = 0;
    for (auto remote_geom : output_container) {
      auto geom = remote_geom->geom;
      const int composite = remote_geom->id;
      // find all mesh hierarchy cells the geom intersects with
      std::deque<std::pair<INT, double>> cells;
      bounding_box_map(geom, particle_mesh_interface->mesh_hierarchy, cells,
                       bounding_box_padding);

      GeometryTransport::PackedGeom2D packed_geom_2d(0, composite, geom);
      for (auto cell_overlap : cells) {
        const INT cell = cell_overlap.first;
        const int owning_rank =
            particle_mesh_interface->mesh_hierarchy->get_owner(cell);
        if (owning_rank == this->rank) {
          packed_geoms[cell].insert(std::end(packed_geoms[cell]),
                                    std::begin(packed_geom_2d.buf),
                                    std::end(packed_geom_2d.buf));
          max_buf_size_tmps =
              std::max(max_buf_size_tmps, packed_geoms[cell].size());
        }
      }
    }
    std::uint64_t max_buf_size_tmpi =
        static_cast<std::uint64_t>(max_buf_size_tmps);
    MPICHK(MPI_Allreduce(&max_buf_size_tmpi, &this->max_buf_size, 1,
                         MPI_UINT64_T, MPI_MAX, this->comm));

    for (auto &cell : packed_geoms) {
      cell.second.resize(this->max_buf_size);
    }
  }
};

} // namespace NESO::CompositeInteraction

#endif
