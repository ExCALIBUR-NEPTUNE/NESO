#ifndef __HALO_EXTENSION_H_
#define __HALO_EXTENSION_H_

#include "particle_mesh_interface.hpp"
#include <map>
#include <memory>
#include <mpi.h>
#include <set>
#include <vector>

using namespace NESO;

namespace NESO {

/**
 *  TODO
 */
inline void halo_extend_exchange_send_sizes(
    MPI_Comm comm, const int num_send_ranks, std::vector<int> &send_ranks,
    std::vector<int> &send_sizes, const int num_recv_ranks,
    std::vector<int> &recv_ranks, std::vector<int> &recv_sizes) {
  // send the packed sizes to the remote ranks
  std::vector<MPI_Request> recv_requests(num_recv_ranks);
  // non-blocking recv packed geom sizes
  for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
    const int remote_rank = recv_ranks[rankx];
    MPICHK(MPI_Irecv(recv_sizes.data() + rankx, 1, MPI_INT, remote_rank, 452,
                     comm, recv_requests.data() + rankx));
  }
  // send sizes to remote ranks
  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    const int remote_rank = send_ranks[rankx];
    MPICHK(MPI_Send(send_sizes.data() + rankx, 1, MPI_INT, remote_rank, 452,
                    comm));
  }

  // wait for recv sizes to be recvd
  MPICHK(MPI_Waitall(num_recv_ranks, recv_requests.data(), MPI_STATUS_IGNORE));
}

/**
 *  TODO
 */
template <typename T>
inline void halo_extend_exchange_geoms_2d(
    MPI_Comm comm, const int num_send_ranks, std::vector<int> &send_ranks,
    const int num_recv_ranks, std::vector<int> &recv_ranks,
    std::map<int, std::map<int, std::shared_ptr<T>>> &rank_element_map,
    std::map<int, int> &original_owners,
    std::vector<std::shared_ptr<RemoteGeom2D<T>>> &output_container) {

  // map from remote MPI ranks to packed geoms
  std::map<int, std::shared_ptr<PackedGeoms2D>> rank_pack_geom_map;
  // pack the local geoms for each remote rank
  std::vector<int> send_sizes(num_send_ranks);
  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    const int remote_rank = send_ranks[rankx];
    rank_pack_geom_map[remote_rank] = std::make_shared<PackedGeoms2D>(
        original_owners, rank_element_map[remote_rank]);
    send_sizes[rankx] =
        static_cast<int>(rank_pack_geom_map[remote_rank]->buf.size());
  }

  // send the packed sizes to the remote ranks
  std::vector<MPI_Request> recv_requests(num_recv_ranks);
  std::vector<int> recv_sizes(num_recv_ranks);
  // non-blocking recv packed geom sizes
  for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
    const int remote_rank = recv_ranks[rankx];
    MPICHK(MPI_Irecv(recv_sizes.data() + rankx, 1, MPI_INT, remote_rank, 451,
                     comm, recv_requests.data() + rankx));
  }
  // send sizes to remote ranks
  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    const int remote_rank = send_ranks[rankx];
    MPICHK(MPI_Send(send_sizes.data() + rankx, 1, MPI_INT, remote_rank, 451,
                    comm));
  }

  // wait for recv sizes to be recvd
  MPICHK(MPI_Waitall(num_recv_ranks, recv_requests.data(), MPI_STATUS_IGNORE));

  // allocate space for the recv'd geometry objects
  const int max_recv_size =
      (num_recv_ranks > 0)
          ? *std::max_element(std::begin(recv_sizes), std::end(recv_sizes))
          : 0;
  std::vector<unsigned char> recv_buffer(max_recv_size * num_recv_ranks);

  // recv packed geoms
  for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
    const int remote_rank = recv_ranks[rankx];
    const int num_recv_bytes = recv_sizes[rankx];
    MPICHK(MPI_Irecv(recv_buffer.data() + rankx * max_recv_size, num_recv_bytes,
                     MPI_UNSIGNED_CHAR, remote_rank, 461, comm,
                     recv_requests.data() + rankx));
  }

  // send packed geoms
  std::vector<MPI_Request> send_requests(num_send_ranks);
  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    const int remote_rank = send_ranks[rankx];
    const int num_send_bytes = send_sizes[rankx];
    NESOASSERT(num_send_bytes == rank_pack_geom_map[remote_rank]->buf.size(),
               "buffer size missmatch");
    MPICHK(MPI_Isend(rank_pack_geom_map[remote_rank]->buf.data(),
                     num_send_bytes, MPI_UNSIGNED_CHAR, remote_rank, 461, comm,
                     send_requests.data() + rankx));
  }
  MPICHK(MPI_Waitall(num_recv_ranks, recv_requests.data(), MPI_STATUS_IGNORE));

  // unpack the recv'd geoms
  for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
    PackedGeoms2D remote_packed_geoms(
        recv_buffer.data() + rankx * max_recv_size, recv_sizes[rankx]);
    remote_packed_geoms.unpack(output_container);
  }

  // wait for the sends to complete
  std::vector<MPI_Status> send_status(num_send_ranks);
  MPICHK(MPI_Waitall(num_send_ranks, send_requests.data(), send_status.data()));
}

/**
 * TODO
 */
inline void extend_halos_fixed_offset(
    const int width, ParticleMeshInterfaceSharedPtr particle_mesh_interface) {

  MPI_Comm comm = particle_mesh_interface->comm;
  MPI_Win recv_win;
  int *recv_win_data;
  MPICHK(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, comm,
                          &recv_win_data, &recv_win));
  recv_win_data[0] = 0;
  MPI_Request request_barrier;
  MPICHK(MPI_Ibarrier(comm, &request_barrier));

  const int ndim = particle_mesh_interface->ndim;
  auto &owned_mh_cells = particle_mesh_interface->owned_mh_cells;
  auto &unowned_mh_cells = particle_mesh_interface->unowned_mh_cells;
  auto &mesh_hierarchy = particle_mesh_interface->mesh_hierarchy;
  const INT ncells_fine = mesh_hierarchy->ncells_fine;

  INT offset_starts[3] = {0, 0, 0};
  INT offset_ends[3] = {1, 1, 1};
  for (int dimx = 0; dimx < ndim; dimx++) {
    offset_starts[dimx] = -width;
    offset_ends[dimx] = width + 1;
  }

  INT cell_counts[3];
  for (int dimx = 0; dimx < ndim; dimx++) {
    cell_counts[dimx] = mesh_hierarchy->dims[dimx] * ncells_fine;
  }

  std::vector<INT> base_cells;
  base_cells.reserve(owned_mh_cells.size() + unowned_mh_cells.size());
  std::set<INT> owned_cells;
  for (auto cellx : owned_mh_cells) {
    owned_cells.insert(cellx);
    base_cells.push_back(cellx);
  }
  for (auto cellx : unowned_mh_cells) {
    base_cells.push_back(cellx);
  }

  std::set<INT> remote_cells;
  for (auto cellx : base_cells) {
    INT global_tuple_mh[6];
    INT global_tuple[3];
    mesh_hierarchy->linear_to_tuple_global(cellx, global_tuple_mh);
    // convert the mesh hierary tuple format into a more standard tuple format
    for (int dimx = 0; dimx < ndim; dimx++) {
      const INT cart_index_dim =
          global_tuple_mh[dimx] * ncells_fine + global_tuple_mh[dimx + ndim];
      global_tuple[dimx] = cart_index_dim;
    }

    // loop over the offsets
    INT ox[3];
    for (ox[2] = offset_starts[2]; ox[2] < offset_ends[2]; ox[2]++) {
      for (ox[1] = offset_starts[1]; ox[1] < offset_ends[1]; ox[1]++) {
        for (ox[0] = offset_starts[0]; ox[0] < offset_ends[0]; ox[0]++) {
          // compute the cell from the offset
          for (int dimx = 0; dimx < ndim; dimx++) {
            const INT offset_dim_linear =
                (global_tuple[dimx] + ox[dimx] + cell_counts[dimx]) %
                cell_counts[dimx];
            // convert back to a mesh hierarchy tuple index
            auto pq =
                std::div((long long)offset_dim_linear, (long long)ncells_fine);
            global_tuple_mh[dimx] = pq.quot;
            global_tuple_mh[dimx + ndim] = pq.rem;
          }
          const INT offset_linear =
              mesh_hierarchy->tuple_to_linear_global(global_tuple_mh);
          // if this rank owns this cell then there is nothing to do
          if (!owned_cells.count(offset_linear)) {
            remote_cells.insert(offset_linear);
          }
        }
      }
    }
  }

  /* N.B. From here onwards "send" ranks are remote MPI ranks this rank will
   * send geometry objects to and "recv" ranks are those this rank will recv
   * geometry objects from.
   */

  // collect the owners of the remote ranks
  std::map<int, std::vector<int64_t>> rank_cells_map;
  std::vector<int> recv_ranks;
  for (auto cellx : remote_cells) {
    const int remote_rank = mesh_hierarchy->get_owner(cellx);
    if ((remote_rank >= 0) &&
        (remote_rank < particle_mesh_interface->comm_size)) {
      rank_cells_map[remote_rank].push_back(cellx);
      recv_ranks.push_back(remote_rank);
    }
  }
  const int num_recv_ranks = recv_ranks.size();

  MPICHK(MPI_Wait(&request_barrier, MPI_STATUS_IGNORE));
  // start to setup the communication pattern
  const int one[1] = {1};
  int recv[1];
  for (int rank : recv_ranks) {
    MPICHK(MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, recv_win));
    MPICHK(MPI_Get_accumulate(one, 1, MPI_INT, recv, 1, MPI_INT, rank, 0, 1,
                              MPI_INT, MPI_SUM, recv_win));
    MPICHK(MPI_Win_unlock(rank, recv_win));
  }

  MPICHK(MPI_Barrier(comm));
  const int num_send_ranks = recv_win_data[0];
  MPICHK(MPI_Win_free(&recv_win));

  // each remote rank will send the number of cells it requires geoms for
  std::vector<int> send_metadata(num_send_ranks);
  std::vector<MPI_Request> recv_requests(num_send_ranks);
  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    MPICHK(MPI_Irecv(send_metadata.data() + rankx, 1, MPI_INT, MPI_ANY_SOURCE,
                     145, comm, recv_requests.data() + rankx));
  }

  for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
    const int remote_rank = recv_ranks[rankx];
    int metadata[1];
    metadata[0] = rank_cells_map[remote_rank].size();
    MPICHK(MPI_Send(metadata, 1, MPI_INT, remote_rank, 145, comm));
  }

  // wait for recv sizes to be recvd
  std::vector<MPI_Status> recv_status(num_send_ranks);
  MPICHK(MPI_Waitall(num_send_ranks, recv_requests.data(), recv_status.data()));
  std::vector<int> send_ranks(num_send_ranks);
  std::map<int, std::vector<int64_t>> send_cells;
  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    const int remote_rank = recv_status[rankx].MPI_SOURCE;
    send_ranks[rankx] = remote_rank;
    // allocate space to store the requested cell indices
    std::vector<int64_t> tmp_cells(send_metadata[rankx]);
    send_cells[remote_rank] = tmp_cells;
  }

  /* We now hold:
   * 1) the cells to collect (recv) from and the ranks that hold them
   * 2) the ranks that requested cells from this rank (send ranks)
   * 3) the number of cells each rank will request.
   */

  // send recv the requested mesh hierarch cells
  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    const int num_cells = send_metadata[rankx];
    const int remote_rank = send_ranks[rankx];
    NESOASSERT(send_cells[remote_rank].size() == num_cells,
               "recv allocation missmatch");
    MPICHK(MPI_Irecv(send_cells[remote_rank].data(), num_cells, MPI_INT64_T,
                     MPI_ANY_SOURCE, 146, comm, recv_requests.data() + rankx));
  }
  for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
    const int remote_rank = recv_ranks[rankx];
    const int num_cells = rank_cells_map[remote_rank].size();
    MPICHK(MPI_Send(rank_cells_map[remote_rank].data(), num_cells, MPI_INT64_T,
                    remote_rank, 146, comm));
  }
  MPICHK(MPI_Waitall(num_send_ranks, recv_requests.data(), MPI_STATUS_IGNORE));

  std::set<INT> send_cells_set;
  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    const int remote_rank = send_ranks[rankx];
    for (auto cellx : send_cells[remote_rank]) {
      send_cells_set.insert(cellx);
    }
  }

  // collect local geoms to send
  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>> geoms_2d;
  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms_3d;
  std::map<int, int> original_owners;
  const int comm_rank = particle_mesh_interface->comm_rank;

  // cells to geometry objects to pack
  std::map<INT, std::vector<int>> cells_to_geoms_2d;
  std::map<INT, std::vector<int>> cells_to_geoms_3d;

  /*  In 2D we only need to pack the 2D quads and triangles for the requested
   *  mesh hierarchy cells.
   *
   *  In 3D we need to collect the 3D objects that cover the requested cells.
   *  Pack the 2D objects needed to create those 3D objects and the description
   *  of how to rebuild the object.
   */

  get_all_elements_2d(particle_mesh_interface->graph, geoms_2d);
  for (auto geom : geoms_2d) {
    const int gid = geom.second->GetGlobalID();
    NESOASSERT(original_owners.count(gid) == 0, "Geometry id already in map");
    original_owners[gid] = comm_rank;
  }
  for (auto geom : particle_mesh_interface->remote_triangles) {
    geoms_2d[geom->id] = std::dynamic_pointer_cast<Geometry2D>(geom->geom);
    NESOASSERT(original_owners.count(geom->id) == 0,
               "Geometry id already in map");
    original_owners[geom->id] = geom->rank;
  }
  for (auto geom : particle_mesh_interface->remote_quads) {
    geoms_2d[geom->id] = std::dynamic_pointer_cast<Geometry2D>(geom->geom);
    NESOASSERT(original_owners.count(geom->id) == 0,
               "Geometry id already in map");
    original_owners[geom->id] = geom->rank;
  }

  if (ndim == 2) {
    std::deque<std::pair<INT, double>> cells;
    for (auto geom : geoms_2d) {
      const int gid = geom.second->GetGlobalID();
      bounding_box_map(geom.second, mesh_hierarchy, cells);
      for (auto cell : cells) {
        if (send_cells_set.count(cell.first)) {
          cells_to_geoms_2d[cell.first].push_back(gid);
        }
      }
    }

  } else if (ndim == 3) {
    get_all_elements_3d(particle_mesh_interface->graph, geoms_3d);
    for (auto geom : geoms_3d) {
      const int gid = geom.second->GetGlobalID();
      NESOASSERT(original_owners.count(gid) == 0, "Geometry id already in map");
      original_owners[gid] = comm_rank;
    }
    for (auto geom : particle_mesh_interface->remote_geoms_3d) {
      geoms_3d[geom->id] = std::dynamic_pointer_cast<Geometry3D>(geom->geom);
      NESOASSERT(original_owners.count(geom->id) == 0,
                 "Geometry id already in map");
      original_owners[geom->id] = geom->rank;
    }
    for (auto geom : geoms_3d) {
      const int gid = geom.second->GetGlobalID();
      std::deque<std::pair<INT, double>> cells;
      bounding_box_map(geom.second, mesh_hierarchy, cells);
      for (auto cell : cells) {
        if (send_cells_set.count(cell.first)) {
          cells_to_geoms_3d[cell.first].push_back(gid);
        }
      }
    }
  }

  // For each remote rank we create the element maps to pack
  std::map<int, std::map<int, std::shared_ptr<Nektar::SpatialDomains::TriGeom>>>
      rank_triangle_map;
  std::map<int,
           std::map<int, std::shared_ptr<Nektar::SpatialDomains::QuadGeom>>>
      rank_quad_map;
  std::map<int,
           std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>>
      rank_geoms_3d_map;

  auto lambda_push_ptr = [&](const int remote_rank, const int element_id) {
    ASSERTL0(geoms_2d.count(element_id), "geometry id not found in map");
    auto element_ptr = geoms_2d[element_id];
    if (element_ptr->GetShapeType() == LibUtilities::eTriangle) {
      rank_triangle_map[remote_rank][element_id] =
          std::dynamic_pointer_cast<TriGeom>(element_ptr);
    } else if (element_ptr->GetShapeType() == LibUtilities::eQuadrilateral) {
      rank_quad_map[remote_rank][element_id] =
          std::dynamic_pointer_cast<QuadGeom>(element_ptr);
    } else {
      ASSERTL0(false, "unknown element type.");
    }
  };

  for (const int remote_rank : send_ranks) {
    for (const INT cell : send_cells[remote_rank]) {
      if (ndim == 2) {
        for (const int element_id : cells_to_geoms_2d[cell]) {
          lambda_push_ptr(remote_rank, element_id);
        }
      } else if (ndim == 3) {
        for (const int element_3d_id : cells_to_geoms_3d[cell]) {
          auto element_ptr = geoms_3d[element_3d_id];
          rank_geoms_3d_map[remote_rank][element_3d_id] = element_ptr;
        }
      }
    }
  }

  /**
   *  We now finally have a map from remote mpi rank to 2D and 3D geoms to pack
   *  and send to that remote mpi rank.
   */

  std::vector<std::shared_ptr<RemoteGeom2D<TriGeom>>> tmp_remote_tris;
  std::vector<std::shared_ptr<RemoteGeom2D<QuadGeom>>> tmp_remote_quads;
  // In both 2D and 3D there are 2D faces/geoms to exchange
  halo_extend_exchange_geoms_2d(comm, num_send_ranks, send_ranks,
                                num_recv_ranks, recv_ranks, rank_triangle_map,
                                original_owners, tmp_remote_tris);
  halo_extend_exchange_geoms_2d(comm, num_send_ranks, send_ranks,
                                num_recv_ranks, recv_ranks, rank_quad_map,
                                original_owners, tmp_remote_quads);

  // We may have just recv'd a geom that this rank originally sent to a remote
  // as a halo object so we sort the geoms into remote and owned again. We also
  // will have re-recv'd remote geoms we already held.

  for (auto geom : tmp_remote_tris) {
    const int geom_id = geom->geom->GetGlobalID();
    // Is this geom new?
    if (!geoms_2d.count(geom_id)) {
      particle_mesh_interface->remote_triangles.push_back(geom);
      geoms_2d[geom_id] = std::dynamic_pointer_cast<Geometry2D>(geom->geom);
    }
  }
  for (auto geom : tmp_remote_quads) {
    const int geom_id = geom->geom->GetGlobalID();
    // Is this geom new?
    if (!geoms_2d.count(geom_id)) {
      particle_mesh_interface->remote_quads.push_back(geom);
      geoms_2d[geom_id] = std::dynamic_pointer_cast<Geometry2D>(geom->geom);
    }
  }
  // In the 2D case the halos are now extended. In the 3D case we need to
  // exchange and rebuild 3D geometry objects.
  if (ndim == 3) {
    std::vector<int> send_sizes(num_send_ranks);
    std::map<int, std::vector<int>> deconstructed_geoms;
    deconstuct_per_rank_geoms_3d(original_owners, geoms_2d, geoms_3d,
                                 rank_geoms_3d_map, num_send_ranks, send_ranks,
                                 send_sizes, deconstructed_geoms,
                                 rank_triangle_map, rank_quad_map);

    std::vector<int> recv_sizes(num_recv_ranks);
    halo_extend_exchange_send_sizes(comm, num_send_ranks, send_ranks,
                                    send_sizes, num_recv_ranks, recv_ranks,
                                    recv_sizes);
    std::vector<int> packed_geoms;
    sendrecv_geoms_3d(comm, deconstructed_geoms, num_send_ranks, send_ranks,
                      send_sizes, num_recv_ranks, recv_ranks, recv_sizes,
                      packed_geoms);

    // rebuild the 2D rank element maps required to build the 3D objects
    std::map<int, std::map<int, std::shared_ptr<Geometry2D>>>
        rank_element_map_2d;
    for (auto geom : geoms_2d) {
      const int geom_id = geom.first;
      NESOASSERT(original_owners.count(geom_id), "Geom id not found in map");
      const int rank = original_owners[geom_id];
      rank_element_map_2d[rank][geom_id] = geom.second;
    }
    // rebuild the recv'd 3D geoms
    std::vector<std::shared_ptr<RemoteGeom3D>> tmp_remote_geoms_3d;
    reconstruct_geoms_3d(rank_element_map_2d, packed_geoms,
                         tmp_remote_geoms_3d);

    for (auto geom : tmp_remote_geoms_3d) {
      const int geom_id = geom->geom->GetGlobalID();
      // Is this geom new?
      if (!geoms_3d.count(geom_id)) {
        particle_mesh_interface->remote_geoms_3d.push_back(geom);
      }
    }
  }
}

} // namespace NESO

#endif
