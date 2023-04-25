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
 *
 */
inline void
get_mesh_hierarchy_cells(const int width,
                         ParticleMeshInterfaceSharedPtr particle_mesh_interface,
                         std::set<INT> &remote_cells) {
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
}

/**
 *  TODO
 */
inline int
get_map_rank_to_cells(ParticleMeshInterfaceSharedPtr particle_mesh_interface,
                      std::set<INT> &remote_cells,
                      std::map<int, std::vector<int64_t>> &rank_cells_map,
                      std::vector<int> &recv_ranks) {

  const int comm_rank = particle_mesh_interface->comm_rank;
  auto &mesh_hierarchy = particle_mesh_interface->mesh_hierarchy;
  for (auto cellx : remote_cells) {
    const int remote_rank = mesh_hierarchy->get_owner(cellx);
    if ((remote_rank >= 0) &&
        (remote_rank < particle_mesh_interface->comm_size) &&
        (remote_rank != comm_rank)) {
      rank_cells_map[remote_rank].push_back(cellx);
    }
  }
  recv_ranks.reserve(rank_cells_map.size());
  for (const auto &remote_rank_pair : rank_cells_map) {
    recv_ranks.push_back(remote_rank_pair.first);
  }

  const int num_recv_ranks = recv_ranks.size();
  return num_recv_ranks;
}

/**
 * TODO
 */
inline int halo_get_num_send_ranks(MPI_Comm comm,
                                   std::vector<int> &recv_ranks) {

  MPI_Win recv_win;
  int *recv_win_data;
  MPICHK(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, comm,
                          &recv_win_data, &recv_win));
  recv_win_data[0] = 0;
  MPI_Request request_barrier;
  MPICHK(MPI_Ibarrier(comm, &request_barrier));
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

  return num_send_ranks;
}

/**
 *  TODO
 */
inline void
halo_get_send_cells(MPI_Comm comm, const int num_send_ranks,
                    std::vector<int> &send_ranks, const int num_recv_ranks,
                    std::vector<int> &recv_ranks,
                    std::map<int, std::vector<int64_t>> &send_cells,
                    std::map<int, std::vector<int64_t>> &rank_cells_map) {

  // each remote rank will send the number of cells it requires geoms for
  std::vector<int> send_metadata(num_send_ranks);

  std::vector<MPI_Request> recv_requests(num_send_ranks);
  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    MPICHK(MPI_Irecv(send_metadata.data() + rankx, 1, MPI_INT, MPI_ANY_SOURCE,
                     145, comm, recv_requests.data() + rankx));
  }

  for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
    const int remote_rank = recv_ranks.at(rankx);
    int metadata[1];
    metadata[0] = rank_cells_map.at(remote_rank).size();
    MPICHK(MPI_Send(metadata, 1, MPI_INT, remote_rank, 145, comm));
  }

  // wait for recv sizes to be recvd
  std::vector<MPI_Status> recv_status(num_send_ranks);
  MPICHK(MPI_Waitall(num_send_ranks, recv_requests.data(), recv_status.data()));

  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    const int remote_rank = recv_status.at(rankx).MPI_SOURCE;
    send_ranks[rankx] = remote_rank;
    // allocate space to store the requested cell indices
    std::vector<int64_t> tmp_cells(send_metadata.at(rankx));
    send_cells[remote_rank] = tmp_cells;
  }

  /* We now hold:
   * 1) the cells to collect (recv) from and the ranks that hold them
   * 2) the ranks that requested cells from this rank (send ranks)
   * 3) the number of cells each rank will request.
   */

  // send recv the requested mesh hierarch cells
  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    const int num_cells = send_metadata.at(rankx);
    const int remote_rank = send_ranks.at(rankx);
    NESOASSERT(send_cells.at(remote_rank).size() == num_cells,
               "recv allocation missmatch");
    MPICHK(MPI_Irecv(send_cells.at(remote_rank).data(), num_cells, MPI_INT64_T,
                     MPI_ANY_SOURCE, 146, comm, recv_requests.data() + rankx));
  }
  for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
    const int remote_rank = recv_ranks.at(rankx);
    const int num_cells = rank_cells_map.at(remote_rank).size();
    MPICHK(MPI_Send(rank_cells_map.at(remote_rank).data(), num_cells,
                    MPI_INT64_T, remote_rank, 146, comm));
  }
  MPICHK(
      MPI_Waitall(num_send_ranks, recv_requests.data(), MPI_STATUSES_IGNORE));
}

/**
 * TODO
 */
inline void
halo_get_send_cells_set(std::vector<int> &send_ranks,
                        std::map<int, std::vector<int64_t>> &send_cells,
                        std::set<INT> &send_cells_set) {
  const int num_send_ranks = send_ranks.size();
  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    const int remote_rank = send_ranks.at(rankx);
    for (auto cellx : send_cells.at(remote_rank)) {
      send_cells_set.insert(cellx);
    }
  }
}

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
  MPICHK(
      MPI_Waitall(num_recv_ranks, recv_requests.data(), MPI_STATUSES_IGNORE));
}

/**
 *  TODO
 */
template <typename T>
inline void halo_extend_exchange_geoms_2d(
    MPI_Comm comm, const int num_send_ranks, std::vector<int> &send_ranks,
    const int num_recv_ranks, std::vector<int> &recv_ranks,
    std::map<int, std::vector<std::shared_ptr<RemoteGeom2D<T>>>>
        &rank_element_map,
    std::vector<std::shared_ptr<RemoteGeom2D<T>>> &output_container) {

  // map from remote MPI ranks to packed geoms
  std::map<int, std::shared_ptr<PackedGeoms2D>> rank_pack_geom_map;
  // pack the local geoms for each remote rank
  std::vector<int> send_sizes(num_send_ranks);
  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    const int remote_rank = send_ranks[rankx];
    rank_pack_geom_map[remote_rank] =
        std::make_shared<PackedGeoms2D>(rank_element_map[remote_rank]);
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
  MPICHK(
      MPI_Waitall(num_recv_ranks, recv_requests.data(), MPI_STATUSES_IGNORE));

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
  MPICHK(
      MPI_Waitall(num_recv_ranks, recv_requests.data(), MPI_STATUSES_IGNORE));

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
 *  TODO
 */
template <typename T, typename U>
inline void halo_rebuild_rank_element_map(
    std::map<int, std::map<int, std::shared_ptr<T>>> &rank_element_map,
    std::vector<std::shared_ptr<U>> &geoms) {
  for (auto &geom : geoms) {
    const int rank = geom->rank;
    const int gid = geom->id;
    rank_element_map[rank][gid] = geom;
  }
}

/**
 *   TODO
 */
inline void halo_get_rank_to_geoms_2d(
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>>
        &rank_geoms_2d_map_local) {

  const int comm_rank = particle_mesh_interface->comm_rank;
  /// map from geom id to geom of locally owned 2D objects.
  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>
      geoms_2d_local;
  get_all_elements_2d(particle_mesh_interface->graph, geoms_2d_local);

  rank_geoms_2d_map_local[comm_rank] = geoms_2d_local;
  for (auto geom : particle_mesh_interface->remote_triangles) {
    const int rank = geom->rank;
    const int gid = geom->id;
    rank_geoms_2d_map_local[rank][gid] =
        std::dynamic_pointer_cast<Geometry2D>(geom->geom);
  }
  for (auto geom : particle_mesh_interface->remote_quads) {
    const int rank = geom->rank;
    const int gid = geom->id;
    rank_geoms_2d_map_local[rank][gid] =
        std::dynamic_pointer_cast<Geometry2D>(geom->geom);
  }
}

/**
 *   TODO
 */
inline void halo_get_rank_to_geoms_3d(
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>>
        &rank_geoms_3d_map_local) {

  const int comm_rank = particle_mesh_interface->comm_rank;
  /// map from geom id to geom of locally owned 3D objects.
  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>
      geoms_3d_local;
  get_all_elements_3d(particle_mesh_interface->graph, geoms_3d_local);

  rank_geoms_3d_map_local[comm_rank] = geoms_3d_local;
  for (auto geom : particle_mesh_interface->remote_geoms_3d) {
    const int rank = geom->rank;
    const int gid = geom->id;
    rank_geoms_3d_map_local[rank][geom->id] =
        std::dynamic_pointer_cast<Geometry3D>(geom->geom);
  }
}

/**
 *  TODO
 */
template <typename T>
inline void get_cells_to_geoms_map(
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    std::map<int, std::map<int, std::shared_ptr<T>>> &rank_geoms_map_local,
    std::set<INT> &send_cells_set,
    std::map<INT, std::vector<std::pair<int, int>>> &cells_to_geoms) {
  const auto mesh_hierarchy = particle_mesh_interface->mesh_hierarchy;
  for (const auto &rank_geoms : rank_geoms_map_local) {
    const int rank = rank_geoms.first;
    for (auto geom : rank_geoms.second) {
      const int gid = geom.second->GetGlobalID();
      std::deque<std::pair<INT, double>> cells;
      bounding_box_map(geom.second, mesh_hierarchy, cells);
      for (auto cell : cells) {
        if (send_cells_set.count(cell.first)) {
          cells_to_geoms[cell.first].push_back({rank, gid});
        }
      }
    }
  }
}

/**
 * TODO
 */
template <typename T>
inline void halo_unpack_2D_geoms(
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>>
        &rank_geoms_2d_map_local,
    std::vector<std::shared_ptr<RemoteGeom2D<T>>> &tmp_remote_geoms,
    std::vector<std::shared_ptr<RemoteGeom2D<T>>> &output_container) {
  for (auto remote_geom : tmp_remote_geoms) {
    const int rank = remote_geom->rank;
    const int gid = remote_geom->id;
    const auto geom = remote_geom->geom;
    const bool new_rank = !rank_geoms_2d_map_local.count(rank);
    const bool new_geom =
        new_rank ? true : !rank_geoms_2d_map_local[rank].count(gid);
    if (new_geom) {
      output_container.push_back(remote_geom);
      rank_geoms_2d_map_local[rank][gid] =
          std::dynamic_pointer_cast<Geometry2D>(geom);
    }
  }
}

/**
 * TODO
 */
inline void extend_halos_fixed_offset(
    const int width, ParticleMeshInterfaceSharedPtr particle_mesh_interface) {

  MPI_Comm comm = particle_mesh_interface->comm;
  const int comm_rank = particle_mesh_interface->comm_rank;
  const int ndim = particle_mesh_interface->ndim;
  std::set<INT> remote_cells;
  auto &mesh_hierarchy = particle_mesh_interface->mesh_hierarchy;

  get_mesh_hierarchy_cells(width, particle_mesh_interface, remote_cells);

  /* N.B. From here onwards "send" ranks are remote MPI ranks this rank will
   * send geometry objects to and "recv" ranks are those this rank will recv
   * geometry objects from.
   */

  // collect the owners of the remote ranks
  std::map<int, std::vector<int64_t>> rank_cells_map;
  std::vector<int> recv_ranks;
  const int num_recv_ranks = get_map_rank_to_cells(
      particle_mesh_interface, remote_cells, rank_cells_map, recv_ranks);
  const int num_send_ranks = halo_get_num_send_ranks(comm, recv_ranks);

  std::vector<int> send_ranks(num_send_ranks);
  std::map<int, std::vector<int64_t>> send_cells;

  halo_get_send_cells(comm, num_send_ranks, send_ranks, num_recv_ranks,
                      recv_ranks, send_cells, rank_cells_map);
  std::set<INT> send_cells_set;
  halo_get_send_cells_set(send_ranks, send_cells, send_cells_set);

  // collect local geoms to send

  /*  In 2D we only need to pack the 2D quads and triangles for the requested
   *  mesh hierarchy cells.
   *
   *  In 3D we need to collect the 3D objects that cover the requested cells.
   *  Pack the 2D objects needed to create those 3D objects and the description
   *  of how to rebuild the object.
   */

  /* In 2D we assume that a unique rank owns each 2D object.
   * In 3D we assume that a unique rank owns each 3D object - this means 2D
   * objects may be shared between multiple ranks.
   */

  std::map<int,
           std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>>
      rank_geoms_2d_map_local;
  halo_get_rank_to_geoms_2d(particle_mesh_interface, rank_geoms_2d_map_local);

  std::map<int,
           std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>>
      rank_geoms_3d_map_local;
  halo_get_rank_to_geoms_3d(particle_mesh_interface, rank_geoms_3d_map_local);

  // cells to geometry objects to pack
  std::map<INT, std::vector<std::pair<int, int>>> cells_to_geoms_2d;
  std::map<INT, std::vector<std::pair<int, int>>> cells_to_geoms_3d;

  if (ndim == 2) {
    get_cells_to_geoms_map(particle_mesh_interface, rank_geoms_2d_map_local,
                           send_cells_set, cells_to_geoms_2d);
  } else if (ndim == 3) {
    get_cells_to_geoms_map(particle_mesh_interface, rank_geoms_3d_map_local,
                           send_cells_set, cells_to_geoms_3d);
  }

  std::map<int, std::set<std::pair<int, int>>> rank_geoms_2d_set;
  std::map<int, std::set<std::pair<int, int>>> rank_geoms_3d_set;
  for (const int remote_rank : send_ranks) {
    for (const INT cell : send_cells[remote_rank]) {
      if (ndim == 2) {
        for (const auto rank_element : cells_to_geoms_2d[cell]) {
          rank_geoms_2d_set[remote_rank].insert(rank_element);
        }
      } else if (ndim == 3) {
        for (const auto rank_element : cells_to_geoms_3d[cell]) {
          rank_geoms_3d_set[remote_rank].insert(rank_element);
          // push the 2D faces onto the 2D send set
          const int orig_rank = rank_element.first;
          const int orig_gid = rank_element.second;
          const auto geom = rank_geoms_3d_map_local[orig_rank][orig_gid];
          const int num_faces = geom->GetNumFaces();
          for (int facex = 0; facex < num_faces; facex++) {
            auto face_geom = geom->GetFace(facex);
            const int face_gid = face_geom->GetGlobalID();
            rank_geoms_2d_set[remote_rank].insert({orig_rank, face_gid});
          }
        }
      }
    }
  }

  // For each remote rank we create the element maps to pack
  std::map<int, std::vector<std::shared_ptr<
                    RemoteGeom2D<Nektar::SpatialDomains::TriGeom>>>>
      rank_triangle_map;
  std::map<int, std::vector<std::shared_ptr<
                    RemoteGeom2D<Nektar::SpatialDomains::QuadGeom>>>>
      rank_quad_map;
  std::map<int, std::vector<std::shared_ptr<RemoteGeom3D>>> rank_geoms_3d_map;

  for (const auto rank_geoms : rank_geoms_2d_set) {
    const int remote_rank = rank_geoms.first;
    for (const auto rank_gid : rank_geoms.second) {
      const int original_rank = rank_gid.first;
      const int gid = rank_gid.second;
      const auto geom = rank_geoms_2d_map_local[original_rank][gid];
      if (geom->GetShapeType() == ShapeType::eTriangle) {
        rank_triangle_map[remote_rank].push_back(
            std::make_shared<RemoteGeom2D<TriGeom>>(
                original_rank, gid, std::dynamic_pointer_cast<TriGeom>(geom)));
      } else if (geom->GetShapeType() == ShapeType::eQuadrilateral) {
        rank_quad_map[remote_rank].push_back(
            std::make_shared<RemoteGeom2D<QuadGeom>>(
                original_rank, gid, std::dynamic_pointer_cast<QuadGeom>(geom)));
      }
    }
  }

  for (const auto rank_geoms : rank_geoms_3d_set) {
    const int remote_rank = rank_geoms.first;
    for (const auto rank_gid : rank_geoms.second) {
      const int original_rank = rank_gid.first;
      const int gid = rank_gid.second;
      const auto geom = rank_geoms_2d_map_local[original_rank][gid];
      rank_geoms_3d_map[remote_rank].push_back(
          std::make_shared<RemoteGeom3D>(original_rank, gid, geom));
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
                                tmp_remote_tris);
  halo_extend_exchange_geoms_2d(comm, num_send_ranks, send_ranks,
                                num_recv_ranks, recv_ranks, rank_quad_map,
                                tmp_remote_quads);

  // We may have just recv'd a geom that this rank originally sent to a remote
  // as a halo object so we sort the geoms into remote and owned again. We also
  // will have re-recv'd remote geoms we already held.
  halo_unpack_2D_geoms(rank_geoms_2d_map_local, tmp_remote_tris,
                       particle_mesh_interface->remote_triangles);
  halo_unpack_2D_geoms(rank_geoms_2d_map_local, tmp_remote_quads,
                       particle_mesh_interface->remote_quads);

  if (ndim == 3) {

    std::vector<int> send_sizes(num_send_ranks);
    std::map<int, std::vector<int>> deconstructed_geoms;

    for (int rankx = 0; rankx < num_send_ranks; rankx++) {
      const int remote_rank = send_ranks[rankx];
      for (auto remote_geom : rank_geoms_3d_map[remote_rank]) {
        const int original_rank = remote_geom->rank;
        const int gid = remote_geom->id;
        const auto geom = remote_geom->geom;
        deconstruct_geoms_base_3d(original_rank, gid, geom,
                                  deconstructed_geoms[remote_rank]);
      }
      send_sizes[rankx] = deconstructed_geoms[remote_rank].size();
    }

    std::vector<int> recv_sizes(num_recv_ranks);
    halo_extend_exchange_send_sizes(comm, num_send_ranks, send_ranks,
                                    send_sizes, num_recv_ranks, recv_ranks,
                                    recv_sizes);
    std::vector<int> packed_geoms;
    sendrecv_geoms_3d(comm, deconstructed_geoms, num_send_ranks, send_ranks,
                      send_sizes, num_recv_ranks, recv_ranks, recv_sizes,
                      packed_geoms);

    // rebuild the recv'd 3D geoms
    std::vector<std::shared_ptr<RemoteGeom3D>> tmp_remote_geoms_3d;
    reconstruct_geoms_3d(rank_geoms_2d_map_local, packed_geoms,
                         tmp_remote_geoms_3d);

    for (auto geom : tmp_remote_geoms_3d) {
      const int original_rank = geom->rank;
      const int gid = geom->geom->GetGlobalID();
      NESOASSERT(gid == geom->id, "ID missmatch");
      const bool new_rank = !rank_geoms_3d_map_local.count(original_rank);
      const bool new_geom =
          (new_rank) ? true
                     : !rank_geoms_3d_map_local.at(original_rank).count(gid);
      // Is this geom new?
      if (new_geom) {
        rank_geoms_3d_map_local[original_rank][gid] = geom->geom;
        particle_mesh_interface->remote_geoms_3d.push_back(geom);
      }
    }
  }
}

} // namespace NESO

#endif