#ifndef __GEOMETRY_TRANSPORT_2D_H__
#define __GEOMETRY_TRANSPORT_2D_H__

// Nektar++ Includes
#include <SpatialDomains/MeshGraph.h>

// System includes
#include <iostream>
#include <map>
#include <mpi.h>
#include <vector>

#include "shape_mapping.hpp"

using namespace std;
using namespace Nektar;
using namespace Nektar::SpatialDomains;

#include "packed_geoms_2d.hpp"
#include "remote_geom_2d.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;

namespace NESO {

/*
 *  Collect all the 2D Geometry objects of type T from all the other MPI ranks.
 */
template <typename T>
std::vector<std::shared_ptr<RemoteGeom2D<T>>>
get_all_remote_geoms_2d(MPI_Comm comm,
                        std::map<int, std::shared_ptr<T>> &geom_map) {
  int rank, size;
  MPICHK(MPI_Comm_rank(comm, &rank));
  MPICHK(MPI_Comm_size(comm, &size));

  PackedGeoms2D local_packed_geoms(rank, geom_map);

  int max_buf_size, this_buf_size;
  this_buf_size = local_packed_geoms.buf.size();

  MPICHK(
      MPI_Allreduce(&this_buf_size, &max_buf_size, 1, MPI_INT, MPI_MAX, comm));

  // simplify send/recvs by choosing a buffer size of the maximum over all
  // ranks
  local_packed_geoms.buf.resize(max_buf_size);

  std::vector<unsigned char> buf_send(max_buf_size);
  std::vector<unsigned char> buf_recv(max_buf_size);

  memcpy(buf_send.data(), local_packed_geoms.buf.data(), max_buf_size);

  unsigned char *ptr_send = buf_send.data();
  unsigned char *ptr_recv = buf_recv.data();
  MPI_Status status;

  std::vector<std::shared_ptr<RemoteGeom2D<T>>> remote_geoms{};

  auto lambda_pos_mod = [](int i, int n) { return (i % n + n) % n; };

  // cyclic passing of remote geometry objects.
  for (int shiftx = 0; shiftx < (size - 1); shiftx++) {

    int rank_send = lambda_pos_mod(rank + 1, size);
    int rank_recv = lambda_pos_mod(rank - 1, size);

    MPICHK(MPI_Sendrecv(ptr_send, max_buf_size, MPI_BYTE, rank_send, rank,
                        ptr_recv, max_buf_size, MPI_BYTE, rank_recv, rank_recv,
                        comm, &status));

    PackedGeoms2D remote_packed_geoms(ptr_recv, max_buf_size);
    remote_packed_geoms.unpack(remote_geoms);

    unsigned char *ptr_tmp = ptr_send;
    ptr_send = ptr_recv;
    ptr_recv = ptr_tmp;
  }

  return remote_geoms;
}

/**
 * Get all 2D geometry objects from a Nektar++ MeshGraph
 *
 * @param[in] graph MeshGraph instance.
 * @param[in,out] std::map of Nektar++ Geometry2D pointers.
 */
void get_all_elements_2d(
    Nektar::SpatialDomains::MeshGraphSharedPtr &graph,
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>> &geoms);

/**
 * Get a local 2D geometry object from a Nektar++ MeshGraph
 *
 * @param graph Nektar++ MeshGraph to return geometry object from.
 * @returns Local 2D geometry object.
 */
Geometry2DSharedPtr
get_element_2d(Nektar::SpatialDomains::MeshGraphSharedPtr &graph);

/**
 *  Add remote 2D objects to a map from geometry ids to shared pointers.
 *
 *  @param[in] remote_geoms Vector of remote geometry objects.
 *  @param[in,out] new_map Output element map (appended to).
 */
template <typename T>
inline void combine_remote_geoms_2d(
    std::vector<std::shared_ptr<RemoteGeom2D<T>>> &remote_geoms,
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>
        &new_map) {
  for (const auto &ix : remote_geoms) {
    new_map[ix->id] = std::dynamic_pointer_cast<Geometry2D>(ix.geom);
  }
}

/**
 *  Add remote 2D objects (typed) to a vector of remote geometry objects
 * (generic 2D type).
 *
 *  @param[in] remote_geoms Vector of remote geometry objects.
 *  @param[in,out] remote_geoms_2d Output vector of 2D geometry types.
 */
template <typename T>
inline void combine_remote_geoms_2d(
    std::vector<std::shared_ptr<RemoteGeom2D<T>>> &remote_geoms,
    std::vector<std::shared_ptr<RemoteGeom2D<Geometry2D>>> &remote_geoms_2d) {
  remote_geoms_2d.reserve(remote_geoms_2d.size() + remote_geoms.size());
  for (const auto &ix : remote_geoms) {
    remote_geoms_2d.push_back(
        std::make_shared<RemoteGeom2D<Geometry2D>>(ix->rank, ix->id, ix->geom));
  }
}

} // namespace NESO

#endif
