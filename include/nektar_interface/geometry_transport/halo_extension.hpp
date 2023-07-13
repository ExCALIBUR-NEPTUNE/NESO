#ifndef __HALO_EXTENSION_H_
#define __HALO_EXTENSION_H_

#include "nektar_interface/particle_mesh_interface.hpp"
#include <map>
#include <memory>
#include <mpi.h>
#include <set>
#include <vector>

using namespace NESO;

namespace NESO {

/**
 *  For an input offset "width" find all the mesh hierarchy cells within width
 *  of this MPI rank. This MPI rank is defined as the mesh hierarchy cells this
 *  MPI owns and the mesh hierarchy cells this MPI overlaps with a geometry
 *  object.
 *
 *  @param[in] width Stencil width in each coordinate direction. Greater or
 *  equal to zero. A value of zero is useful to extend the halos over all mesh
 *  hierarchy cells touched by geometry objects this MPI rank owns.
 *  @param[in] particle_mesh_interface ParticleMeshInterface to extend the halos
 * of.
 *  @param[in, out] remote_cells Set of MeshHierarchy cells which are not owned
 *  by this MPI rank but are within "width" of mesh hierarchy cells which are
 *  owned by this MPI rank or cells which have non-zero overlap with the
 *  bounding box of geometry objects which are owned by this MPI rank.
 */
void halo_get_mesh_hierarchy_cells(
    const int width, ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    std::set<INT> &remote_cells);

/**
 *  For a set of MeshHierarchy cells find the corresponding remote MPI ranks
 * that own these cells. Ignore cells owned by this MPI rank. Build a map from
 * remote MPI rank to MeshHierarchy cells owned by that remote rank.
 *
 *  @param[in] particle_mesh_interface ParticleMeshInterface to use.
 *  @param[in] remote_cells Set of MeshHierarchy cells to find owing ranks for.
 *  @param[in, out] rank_cells_map Output map from MPI rank to owned
 * MeshHierarchy cells.
 *  @param[in, out] recv_ranks Vector of remote ranks (equal to the keys of
 * rank_cells_map).
 *  @returns Number of remote MPI ranks in recv_ranks_vector.
 */
int halo_get_map_rank_to_cells(
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    std::set<INT> &remote_cells,
    std::map<int, std::vector<int64_t>> &rank_cells_map,
    std::vector<int> &recv_ranks);

/**
 * Get on each MPI rank the number of remote MPI ranks which hold this rank in
 * a vector of remote ranks. For the halo extend use case this is used to
 * determine how many remote ranks require geometry objects from this rank.
 * This function must be called collectively on the MPI communicator.
 *
 * @param[in] comm MPI communicator to use.
 * @param[in] recv_ranks Vector of remote MPI ranks this rank should
 * receive geometry objects from.
 * @returns Number of remote MPI ranks to setup communication with.
 */
int halo_get_num_send_ranks(MPI_Comm comm, std::vector<int> &recv_ranks);

/**
 *  Exchange which cells are required to be communicated between MPI ranks. In
 * the halo exchange use case rank i requests a list of cells from rank j. Rank
 * i is the "receiving" MPI rank as in the context of the halo exchange "send"
 * and "recv" are relative to the direction of travel of the geometry objects.
 * Must be called collectively on the MPI communicator.
 *
 *  @param[in] comm MPI communicator to use.
 *  @param[in] num_send_ranks Number of remote MPI ranks this rank will send
 * geometry objects to.
 *  @param[in] send_ranks Vector of MPI ranks this MPI rank will send geometry
 * objects to.
 *  @param[in] num_recv_ranks Number of remote MPI ranks this MPI rank will
 * receive geometry objects from.
 *  @param[in] recv_ranks Vector of MPI ranks to receive geometry objects from.
 *  @param[in, out] send_cells Map from remote MPI rank to a vector of
 * MeshHierarchy cells requested by the each remote rank. These are the cells
 * this rank should pack and send to the send_ranks.
 *  @param[in] Map from each receive rank to the vector of MeshHierarchy cells
 * this rank will request from the remote MPI rank.
 */
void halo_get_send_cells(MPI_Comm comm, const int num_send_ranks,
                         std::vector<int> &send_ranks, const int num_recv_ranks,
                         std::vector<int> &recv_ranks,
                         std::map<int, std::vector<int64_t>> &send_cells,
                         std::map<int, std::vector<int64_t>> &rank_cells_map);
/**
 * Create the set of MeshHierarchy cells which have been requested by remote MPI
 * ranks.
 *
 * @param[in] send_ranks Vector of remote MPI ranks which have requested
 * MeshHierarchy cells.
 * @param[in] send_cells Map from MPI ranks to requested MeshHierarchy cells.
 * @param[in, out] send_cells_set Output set of MeshHierarchy cells which have
 * been requested.
 */
void halo_get_send_cells_set(std::vector<int> &send_ranks,
                             std::map<int, std::vector<int64_t>> &send_cells,
                             std::set<INT> &send_cells_set);

/**
 *  Exchange between MPI ranks the number of elements that are to be exchanged
 *  such that buffers can be allocated. Must be called collectively on the
 *  communicator.
 *
 *  @param[in] comm MPI communicator.
 *  @param[in] num_send_ranks Number of MPI ranks this rank will send geometry
 * objects to.
 *  @param[in] send_ranks Vector of MPI ranks to send geometry objects to.
 *  @param[in] recv_ranks Vector of MPI ranks to receive geometry objects from.
 *  @param[in, out] Number of elements to expect to receive from each remote MPI
 * rank.
 */
void halo_exchange_send_sizes(MPI_Comm comm, const int num_send_ranks,
                              std::vector<int> &send_ranks,
                              std::vector<int> &send_sizes,
                              const int num_recv_ranks,
                              std::vector<int> &recv_ranks,
                              std::vector<int> &recv_sizes);

/**
 *  Exchange 2D geometry objects (Quads and Triangles) between MPI ranks. Must
 *  be called collectively on the communicator.
 *
 *  @param[in] comm MPI communicator to use.
 *  @param[in] num_send_ranks Number of remote MPI ranks to send objects to.
 *  @param[in] send_ranks Vector of remote MPI objects to send objects to.
 *  @param[in] num_recv_ranks Number of remote MPI ranks to receive objects
 * from.
 *  @param[in] recv_ranks Vector of ranks to receive objects from.
 *  @param[in] rank_element_map Map from remote MPI rank to vector of geometry
 *  objects to send to the remote MPI rank.
 *  @param[in, out] output_container Container to push received geometry objects
 * onto.
 */
template <typename T>
inline void halo_exchange_geoms_2d(
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
 *  Extend a map from MPI rank to a map from geometry id to geometry shared
 * pointer.
 *
 *  @param[in, out] rank_element_map Map to extend.
 *  @param[in] geoms Vector of geometry objects (RemoteGeom2D or RemoteGeom3D)
 * to extend map with.
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
 *   Build a map from MPI rank to 2D geometry objects that originate from that
 * MPI rank using geometry objects in the graph on this rank and geometry
 * objects in the halos on this rank.
 *
 *   @param[in] particle_mesh_interface ParticleMeshInterface to use as a source
 * of geometry objects.
 *   @param[in,out] rank_geoms_2d_map_local Output map.
 */
void halo_get_rank_to_geoms_2d(
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>>
        &rank_geoms_2d_map_local);

/**
 *   Build a map from MPI rank to 3D geometry objects that originate from that
 * MPI rank using geometry objects in the graph on this rank and geometry
 * objects in the halos on this rank.
 *
 *   @param[in] particle_mesh_interface ParticleMeshInterface to use as a source
 * of geometry objects.
 *   @param[in,out] rank_geoms_3d_map_local Output map.
 */
void halo_get_rank_to_geoms_3d(
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>>
        &rank_geoms_3d_map_local);

/**
 *  Build a map from MeshHierarchy cells to a vector of pairs storing the
 *  original MPI rank of a geometry object and the geometry id of the object.
 *
 *  @param[in] particle_mesh_interface ParticleMeshInterface to use as a source
 * of geometry objects.
 *  @param[in] rank_geoms_map_local Map from MPI rank to geometry objects that
 * originate from that MPI rank.
 *  @param[in] send_cells_set Set of MeshHierarchy cells to create the map for.
 *  @param[in, out] cells_to_geoms Map from MeshHierarchy cells listed in
 *  send_cells_set to original MPI ranks and geometry ids.
 */
template <typename T>
inline void halo_get_cells_to_geoms_map(
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
 * Unpack 2D geometry objects into a container. Ignore geometry objects that
 * originate from this MPI rank.
 *
 * @param[in, out] rank_geoms_2d_map_local Map from MPI ranks to originating
 * MPI ranks and geometry ids. Will be extended with the new geometry objects.
 * @param[in] tmp_remote_geoms RemoteGeom2D instances which will be kept if new
 * or discarded if already held.
 * @param[in, out] output_container Container to push new RemoteGeom2D instances
 * onto.
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
 * Extend the halo regions of a ParticleMeshInterface. Consider all
 * MeshHierarchy (MH) cells which are either owned by this MPI rank or were
 * claimed by this MPI rank but ultimately are not owned by this rank. These two
 * sets of MH cells are all cells where there is a non-empty intersection with
 * the bounding box of a Nektar++ geometry object which is owned by this MPI
 * rank.
 *
 * For each MH cell in this set consider all MH cells which are within the
 * passed offset but are not owned by this MPI rank "new MH cells". For each
 * new MH cell collect on this MPI rank all Nektar++ geometry objects where the
 * bounding box of that object intersects a cell in the new MH cells list.
 *
 * For all non-negative offsets this function call grows the size of the halo
 * regions. With an offset of zero the halo regions are extended with the
 * geometry objects that intersect the claimed but not owned MH cells. A
 * negative offset does not error and does not modify the halo regions. This
 * function must be called collectively on the MPI communicator stored in the
 * ParticleMeshInterface.
 *
 * @param[in] offset Integer offset to apply to MeshHierarchy cells in all
 * coordinate directions.
 * @param[in,out] particle_mesh_interface ParticleMeshInterface to extend the
 * halos of.
 */
void extend_halos_fixed_offset(
    const int offset, ParticleMeshInterfaceSharedPtr particle_mesh_interface);

} // namespace NESO

#endif
