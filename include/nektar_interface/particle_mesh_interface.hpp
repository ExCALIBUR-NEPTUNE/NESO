#ifndef __PARTICLE_MESH_INTERFACE_H__
#define __PARTICLE_MESH_INTERFACE_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <vector>

#include <mpi.h>

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

#include "bounding_box_intersection.hpp"
#include "geometry_transport/geometry_transport.hpp"
#include "particle_boundary_conditions.hpp"

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

typedef std::map<INT, std::deque<int>> MHGeomMap;

/**
 *  Simple wrapper around an int and float to use for assembling cell claim
 *  weights.
 */
class ClaimWeight {
private:
public:
  /// The integer weight that will be used to make the claim.
  int weight;
  /// A floating point weight for reference/testing.
  double weightf;
  ~ClaimWeight(){};
  ClaimWeight() : weight(0), weightf(0.0){};
};

/**
 *  Container to collect claim weights local to this rank before passing them
 *  to the mesh hierarchy. Local collection prevents excessive MPI RMA comms.
 */
class LocalClaim {
private:
public:
  /// Map from global cell indices of a MeshHierarchy to ClaimWeights.
  std::map<int64_t, ClaimWeight> claim_weights;
  /// Set of cells which claims were made for.
  std::set<int64_t> claim_cells;
  ~LocalClaim(){};
  LocalClaim(){};
  /**
   *  Claim a cell with passed weights.
   *
   *  @param index Global linear index of cell in MeshHierarchy.
   *  @param weight Integer claim weight, this will be passed to the
   * MeshHierarchy to claim the cell.
   *  @param weightf Floating point weight for reference/testing.
   */
  inline void claim(const int64_t index, const int weight,
                    const double weightf) {
    if (weight > 0.0) {
      this->claim_cells.insert(index);
      auto current_claim = this->claim_weights[index];
      if (weight > current_claim.weight) {
        this->claim_weights[index].weight = weight;
        this->claim_weights[index].weightf = weightf;
      }
    }
  }
};

/**
 * Convert a mesh index (index_x, index_y, ...) for this cartesian mesh to
 * the format for a MeshHierarchy: (coarse_x, coarse_y,.., fine_x,
 * fine_y,...).
 *
 * @param ndim Number of dimensions.
 * @param index_mesh Tuple index into cartesian grid of cells.
 * @param mesh_hierarchy MeshHierarchy instance.
 * @param index_mh Output index in the MeshHierarchy.
 */
inline void
mesh_tuple_to_mh_tuple(const int ndim, const int64_t *index_mesh,
                       std::shared_ptr<MeshHierarchy> mesh_hierarchy,
                       INT *index_mh) {
  for (int dimx = 0; dimx < ndim; dimx++) {
    auto pq = std::div((long long)index_mesh[dimx],
                       (long long)mesh_hierarchy->ncells_dim_fine);
    index_mh[dimx] = pq.quot;
    index_mh[dimx + ndim] = pq.rem;
  }
}

/**
 *  Use the bounds of an element in 1D to compute the overlap area with a given
 *  cell. Passed bounds should be shifted relative to an origin of 0.
 *
 *  @param lhs Lower bound of element.
 *  @param rhs Upepr bound of element.
 *  @param cell Cell index (base 0).
 *  @param cell_width_fine Width of each cell.
 */
inline double overlap_1d(const double lhs, const double rhs, const int cell,
                         const double cell_width_fine) {

  const double cell_start = cell * cell_width_fine;
  const double cell_end = cell_start + cell_width_fine;

  // if the overlap is empty then the area is 0.
  if (rhs <= cell_start) {
    return 0.0;
  } else if (lhs >= cell_end) {
    return 0.0;
  }

  const double interval_start = std::max(cell_start, lhs);
  const double interval_end = std::min(cell_end, rhs);
  const double area = interval_end - interval_start;

  return (area > 0.0) ? area : 0.0;
}

/**
 * Compute all claims to cells, and associated weights, for the passed element
 * using the element bounding box.
 *
 * @param[in] element Nektar++ mesh element to use.
 * @param[in] mesh_hierarchy MeshHierarchy instance which cell claims will be
 * made into.
 * @param[in,out] cells Mesh heirarchy cells covered by bounding box.
 */
template <typename T>
inline void bounding_box_map(T element,
                             std::shared_ptr<MeshHierarchy> mesh_hierarchy,
                             std::deque<std::pair<INT, double>> &cells) {
  cells.clear();

  auto element_bounding_box = element->GetBoundingBox();
  const int ndim = mesh_hierarchy->ndim;
  auto origin = mesh_hierarchy->origin;

  int cell_starts[3] = {0, 0, 0};
  int cell_ends[3] = {1, 1, 1};
  double shifted_bounding_box[6];

  // For each dimension compute the starting and ending cells overlapped by
  // this element by using the bounding box. This gives an iteration set of
  // cells touched by this element's bounding box.
  for (int dimx = 0; dimx < ndim; dimx++) {
    const double lhs_point = element_bounding_box[dimx] - origin[dimx];
    const double rhs_point = element_bounding_box[dimx + 3] - origin[dimx];
    shifted_bounding_box[dimx] = lhs_point;
    shifted_bounding_box[dimx + 3] = rhs_point;
    int lhs_cell = lhs_point * mesh_hierarchy->inverse_cell_width_fine;
    int rhs_cell = rhs_point * mesh_hierarchy->inverse_cell_width_fine + 1;

    const int64_t ncells_dim_fine =
        mesh_hierarchy->ncells_dim_fine * mesh_hierarchy->dims[dimx];

    lhs_cell = (lhs_cell < 0) ? 0 : lhs_cell;
    lhs_cell = (lhs_cell >= ncells_dim_fine) ? ncells_dim_fine : lhs_cell;
    rhs_cell = (rhs_cell < 0) ? 0 : rhs_cell;
    rhs_cell = (rhs_cell > ncells_dim_fine) ? ncells_dim_fine : rhs_cell;

    cell_starts[dimx] = lhs_cell;
    cell_ends[dimx] = rhs_cell;
  }

  const double cell_width_fine = mesh_hierarchy->cell_width_fine;
  const double inverse_cell_volume = 1.0 / std::pow(cell_width_fine, ndim);

  // mesh tuple index
  int64_t index_mesh[3];
  // mesh_hierarchy tuple index
  INT index_mh[6];

  // For each cell compute the overlap with the element and use the overlap
  // volume to compute a claim weight (as a ratio of the volume of the cell).
  for (int cz = cell_starts[2]; cz < cell_ends[2]; cz++) {
    index_mesh[2] = cz;
    double area_z = 1.0;
    if (ndim > 2) {
      area_z = overlap_1d(shifted_bounding_box[2], shifted_bounding_box[2 + 3],
                          cz, cell_width_fine);
    }

    for (int cy = cell_starts[1]; cy < cell_ends[1]; cy++) {
      index_mesh[1] = cy;
      double area_y = 1.0;
      if (ndim > 1) {
        area_y = overlap_1d(shifted_bounding_box[1],
                            shifted_bounding_box[1 + 3], cy, cell_width_fine);
      }

      for (int cx = cell_starts[0]; cx < cell_ends[0]; cx++) {
        index_mesh[0] = cx;
        const double area_x =
            overlap_1d(shifted_bounding_box[0], shifted_bounding_box[0 + 3], cx,
                       cell_width_fine);

        const double volume = area_x * area_y * area_z;

        if (volume > 0.0) {
          mesh_tuple_to_mh_tuple(ndim, index_mesh, mesh_hierarchy, index_mh);
          const INT index_global =
              mesh_hierarchy->tuple_to_linear_global(index_mh);
          cells.push_back({index_global, volume});
        }
      }
    }
  }
}

/**
 * Compute all claims to cells, and associated weights, for the passed element
 * using the element bounding box.
 *
 * @param element_id Integer element id for Nektar++ map.
 * @param element Nektar++ mesh element to use.
 * @param mesh_hierarchy MeshHierarchy instance which cell claims will be made
 * into.
 * @param local_claim LocalClaim instance in which cell claims are being
 * collected into.
 * @param mh_geom_map MHGeomMap from MeshHierarchy global cells ids to Nektar++
 * element ids.
 */
template <typename T>
inline void bounding_box_claim(int element_id, T element,
                               std::shared_ptr<MeshHierarchy> mesh_hierarchy,
                               LocalClaim &local_claim,
                               MHGeomMap &mh_geom_map) {

  std::deque<std::pair<INT, double>> cells;
  bounding_box_map(element, mesh_hierarchy, cells);

  const int ndim = mesh_hierarchy->ndim;
  const double cell_width_fine = mesh_hierarchy->cell_width_fine;
  const double inverse_cell_volume = 1.0 / std::pow(cell_width_fine, ndim);

  for (const auto &cell_volume : cells) {
    const INT index_global = cell_volume.first;
    const double volume = cell_volume.second;
    const double ratio = volume * inverse_cell_volume;
    const int weight = 1000000.0 * ratio;
    local_claim.claim(index_global, weight, ratio);
    mh_geom_map[index_global].push_back(element_id);
  }
}

/**
 *  Create a MeshHierarchy around a Nektar++ graph. Also handles exchange of
 *  Nektar++ geometry objects to cover the owned cells in the MeshHierarchy.
 */
class ParticleMeshInterface : public HMesh {

private:
  MPI_Win recv_win;
  int *recv_win_data;

  /**
   *  Get the MPI remote MPI ranks this rank expects to send geometry objects
   * to.
   */
  template <typename T>
  inline int exchange_get_send_ranks(
      std::map<int, std::shared_ptr<T>> &element_map, MHGeomMap &mh_geom_map,
      std::map<int, std::map<int, std::shared_ptr<T>>> &rank_element_map,
      std::vector<int> &send_ranks) {

    std::set<int> send_ranks_set;
    for (const INT &cell : this->unowned_mh_cells) {
      const int remote_rank = this->mesh_hierarchy->get_owner(cell);
      NESOASSERT(remote_rank >= 0, "Owning rank is negative.");
      NESOASSERT(remote_rank < this->comm_size, "Owning rank too large.");
      NESOASSERT(remote_rank != this->comm_rank,
                 "Trying to send geoms to self.");
      send_ranks_set.insert(remote_rank);
      for (const int &geom_id : mh_geom_map[cell]) {
        rank_element_map[remote_rank][geom_id] = element_map[geom_id];
      }
    }

    const int num_send_ranks = send_ranks_set.size();
    send_ranks.reserve(num_send_ranks);
    for (auto &rankx : send_ranks_set) {
      send_ranks.push_back(rankx);
    }

    return num_send_ranks;
  }

  /**
   * Reset the local data structures before call to
   * exchange_finalise_send_counts.
   */
  inline MPI_Request exchange_init_send_counts() {
    NESOASSERT(this->recv_win_data != nullptr, "recv win is not allocated.");
    this->recv_win_data[0] = 0;
    MPI_Request request_barrier;
    MPICHK(MPI_Ibarrier(this->comm, &request_barrier));
    return request_barrier;
  }

  /**
   * Exchange send counts of geometry objecys between MPI ranks.
   */
  inline int exchange_finalise_send_counts(MPI_Request &request_barrier,
                                           std::vector<int> &send_ranks) {

    NESOASSERT(this->recv_win_data != nullptr, "recv win is not allocated.");
    MPICHK(MPI_Wait(&request_barrier, MPI_STATUS_IGNORE));
    // start to setup the communication pattern
    const int one[1] = {1};
    int recv[1];
    for (int rank : send_ranks) {
      MPICHK(MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, this->recv_win));
      MPICHK(MPI_Get_accumulate(one, 1, MPI_INT, recv, 1, MPI_INT, rank, 0, 1,
                                MPI_INT, MPI_SUM, this->recv_win));
      MPICHK(MPI_Win_unlock(rank, this->recv_win));
    }

    MPICHK(MPI_Barrier(this->comm));
    const int num_recv_ranks = this->recv_win_data[0];
    return num_recv_ranks;
  }

  /**
   *  Get the ranks to recv from for exchange operations
   */
  inline void exchange_get_recv_ranks(const int num_send_ranks,
                                      std::vector<int> &send_ranks,
                                      std::vector<int> &send_sizes,
                                      const int num_recv_ranks,
                                      std::vector<int> &recv_ranks,
                                      std::vector<int> &recv_sizes) {
    // send the packed sizes to the remote ranks
    std::vector<MPI_Request> recv_requests(num_recv_ranks);
    // non-blocking recv packed geom sizes
    for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
      MPICHK(MPI_Irecv(recv_sizes.data() + rankx, 1, MPI_INT, MPI_ANY_SOURCE,
                       45, this->comm, recv_requests.data() + rankx));
    }
    // send sizes to remote ranks
    for (int rankx = 0; rankx < num_send_ranks; rankx++) {
      const int remote_rank = send_ranks[rankx];
      MPICHK(MPI_Send(send_sizes.data() + rankx, 1, MPI_INT, remote_rank, 45,
                      this->comm));
    }

    // wait for recv sizes to be recvd
    std::vector<MPI_Status> recv_status(num_recv_ranks);

    MPICHK(
        MPI_Waitall(num_recv_ranks, recv_requests.data(), recv_status.data()));

    for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
      const int remote_rank = recv_status[rankx].MPI_SOURCE;
      recv_ranks[rankx] = remote_rank;
    }
  }

  /**
   *  Pack and exchange 2D geometry objects between MPI ranks.
   */
  template <typename T>
  inline void exchange_packed_2d(
      const int num_send_ranks,
      std::map<int, std::map<int, std::shared_ptr<T>>> &rank_element_map,
      std::vector<int> &send_ranks,
      std::vector<std::shared_ptr<RemoteGeom2D<T>>> &output_container) {

    auto request_barrier = exchange_init_send_counts();

    // map from remote MPI ranks to packed geoms
    std::map<int, std::shared_ptr<PackedGeoms2D>> rank_pack_geom_map;
    // pack the local geoms for each remote rank
    std::vector<int> send_sizes(num_send_ranks);
    for (int rankx = 0; rankx < num_send_ranks; rankx++) {
      const int remote_rank = send_ranks[rankx];
      rank_pack_geom_map[remote_rank] = std::make_shared<PackedGeoms2D>(
          this->comm_rank, rank_element_map[remote_rank]);
      send_sizes[rankx] =
          static_cast<int>(rank_pack_geom_map[remote_rank]->buf.size());
    }

    // determine the number of remote ranks that will send geoms to this rank
    const int num_recv_ranks =
        exchange_finalise_send_counts(request_barrier, send_ranks);

    std::vector<int> recv_sizes(num_recv_ranks);
    std::vector<int> recv_ranks(num_recv_ranks);
    exchange_get_recv_ranks(num_send_ranks, send_ranks, send_sizes,
                            num_recv_ranks, recv_ranks, recv_sizes);

    // allocate space for the recv'd geometry objects
    const int max_recv_size =
        (num_recv_ranks > 0)
            ? *std::max_element(std::begin(recv_sizes), std::end(recv_sizes))
            : 0;
    std::vector<unsigned char> recv_buffer(max_recv_size * num_recv_ranks);

    // wait for recv sizes to be recvd
    std::vector<MPI_Status> recv_status(num_recv_ranks);
    std::vector<MPI_Request> recv_requests(num_recv_ranks);

    // recv packed geoms
    for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
      const int remote_rank = recv_ranks[rankx];
      const int num_recv_bytes = recv_sizes[rankx];
      MPICHK(MPI_Irecv(recv_buffer.data() + rankx * max_recv_size,
                       num_recv_bytes, MPI_UNSIGNED_CHAR, remote_rank, 46,
                       this->comm, recv_requests.data() + rankx));
    }

    // send packed geoms
    std::vector<MPI_Request> send_requests(num_send_ranks);
    for (int rankx = 0; rankx < num_send_ranks; rankx++) {
      const int remote_rank = send_ranks[rankx];
      const int num_send_bytes = send_sizes[rankx];
      NESOASSERT(num_send_bytes == rank_pack_geom_map[remote_rank]->buf.size(),
                 "buffer size missmatch");
      MPICHK(MPI_Isend(rank_pack_geom_map[remote_rank]->buf.data(),
                       num_send_bytes, MPI_UNSIGNED_CHAR, remote_rank, 46,
                       this->comm, send_requests.data() + rankx));
    }
    MPICHK(
        MPI_Waitall(num_recv_ranks, recv_requests.data(), recv_status.data()));

    // unpack the recv'd geoms
    for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
      PackedGeoms2D remote_packed_geoms(
          recv_buffer.data() + rankx * max_recv_size, recv_sizes[rankx]);
      remote_packed_geoms.unpack(output_container);
    }

    // wait for the sends to complete
    std::vector<MPI_Status> send_status(num_send_ranks);
    MPICHK(
        MPI_Waitall(num_send_ranks, send_requests.data(), send_status.data()));
  }

  /**
   *  Determine 2D elements to send to remote ranks and exchange them to build
   *  2D halos.
   */
  template <typename T>
  inline void exchange_geometry_2d(
      std::map<int, std::shared_ptr<T>> &element_map, MHGeomMap &mh_geom_map,
      std::vector<std::shared_ptr<RemoteGeom2D<T>>> &output_container) {

    // map from mpi rank to element ids
    std::map<int, std::map<int, std::shared_ptr<T>>> rank_element_map;
    // Set of remote ranks to send to
    std::vector<int> send_ranks;
    // Get the ranks to send to
    const int num_send_ranks = exchange_get_send_ranks(
        element_map, mh_geom_map, rank_element_map, send_ranks);
    exchange_packed_2d(num_send_ranks, rank_element_map, send_ranks,
                       output_container);
  }

  /**
   *  Find a global bounding box around the computational domain.
   */
  inline void compute_bounding_box(
      std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>
          &geoms_2d,
      std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>
          &geoms_3d) {

    // Get a local and global bounding box for the mesh
    reset_bounding_box(this->bounding_box);

    int64_t num_elements = 0;
    if (this->ndim == 2) {
      for (auto &e : geoms_2d) {
        NESOASSERT(e.first == e.second->GetGlobalID(), "GlobalID != key");
        expand_bounding_box(e.second, this->bounding_box);
        num_elements++;
      }
    } else if (this->ndim == 3) {
      for (auto &e : geoms_3d) {
        NESOASSERT(e.first == e.second->GetGlobalID(), "GlobalID != key");
        expand_bounding_box(e.second, this->bounding_box);
        num_elements++;
      }
    } else {
      NESOASSERT(false, "unknown spatial dimension");
    }
    this->cell_count = num_elements;

    MPICHK(MPI_Allreduce(this->bounding_box.data(),
                         this->global_bounding_box.data(), 3, MPI_DOUBLE,
                         MPI_MIN, this->comm));
    MPICHK(MPI_Allreduce(this->bounding_box.data() + 3,
                         this->global_bounding_box.data() + 3, 3, MPI_DOUBLE,
                         MPI_MAX, this->comm));
  }

  /**
   *  Create a mesh hierarchy based on the bounding box and number of elements.
   */
  inline void create_mesh_hierarchy() {

    // Compute a set of coarse mesh sizes and dimensions for the mesh hierarchy
    double min_extent = std::numeric_limits<double>::max();
    double max_extent = std::numeric_limits<double>::min();
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      const double tmp_global_extent =
          this->global_bounding_box[dimx + 3] - this->global_bounding_box[dimx];
      const double tmp_extent =
          this->bounding_box[dimx + 3] - this->bounding_box[dimx];
      this->extents[dimx] = tmp_extent;
      this->global_extents[dimx] = tmp_global_extent;

      min_extent = std::min(min_extent, tmp_global_extent);
      max_extent = std::max(max_extent, tmp_global_extent);
    }
    NESOASSERT(min_extent > 0.0, "Minimum extent is <= 0");
    double coarse_cell_size;
    const double extent_ratio = max_extent / min_extent;
    if (extent_ratio >= 2.0) {
      coarse_cell_size = min_extent;
    } else {
      coarse_cell_size = max_extent;
    }

    std::vector<int> dims(this->ndim);
    std::vector<double> origin(this->ndim);

    int64_t hm_cell_count = 1;
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      origin[dimx] = this->global_bounding_box[dimx];
      const int tmp_dim =
          std::ceil(this->global_extents[dimx] / coarse_cell_size);
      dims[dimx] = tmp_dim;
      hm_cell_count *= ((int64_t)tmp_dim);
    }

    this->ncells_coarse = hm_cell_count;

    int64_t global_num_elements;
    int64_t local_num_elements = this->cell_count;
    MPICHK(MPI_Allreduce(&local_num_elements, &global_num_elements, 1,
                         MPI_INT64_T, MPI_SUM, this->comm));

    // compute a subdivision order that would result in the same order of fine
    // cells in the mesh hierarchy as mesh elements in Nektar++
    const double inverse_ndim = 1.0 / ((double)this->ndim);
    const int matching_subdivision_order =
        std::ceil((((double)std::log(global_num_elements)) -
                   ((double)std::log(hm_cell_count))) *
                  inverse_ndim);

    // apply the offset to this order and compute the used subdivision order
    this->subdivision_order = std::max(0, matching_subdivision_order +
                                              this->subdivision_order_offset);

    // create the mesh hierarchy
    this->mesh_hierarchy = std::make_shared<MeshHierarchy>(
        this->comm, this->ndim, dims, origin, coarse_cell_size,
        this->subdivision_order);

    this->ncells_fine = static_cast<int>(this->mesh_hierarchy->ncells_fine);
  }

  /**
   *  Loop over cells that were claimed locally and claim them in the mesh
   * hierarchy.
   */
  inline void claim_mesh_hierarchy_cells(LocalClaim &local_claim) {
    // claim cells in the mesh hierarchy
    mesh_hierarchy->claim_initialise();

    for (auto &cellx : local_claim.claim_cells) {
      mesh_hierarchy->claim_cell(cellx,
                                 local_claim.claim_weights[cellx].weight);
    }
    mesh_hierarchy->claim_finalise();
  }

  /**
   *  For each Nektar++ element claim cells based on bounding box.
   */
  template <typename T>
  inline void claim_cells(std::map<int, std::shared_ptr<T>> &geoms,
                          LocalClaim &local_claim, MHGeomMap &mh_geom_map) {

    for (auto &e : geoms) {
      bounding_box_claim(e.first, e.second, mesh_hierarchy, local_claim,
                         mh_geom_map);
    }
  }

  /**
   *  Find the cells which were claimed by this rank but are acutally owned by
   *  a remote rank
   */
  inline void get_unowned_cells(LocalClaim &local_claim) {
    std::stack<INT> owned_cell_stack;
    std::stack<INT> unowned_cell_stack;
    for (auto &cellx : local_claim.claim_cells) {
      const int owning_rank = this->mesh_hierarchy->get_owner(cellx);
      if (owning_rank == this->comm_rank) {
        owned_cell_stack.push(cellx);
      } else {
        unowned_cell_stack.push(cellx);
      }
    }
    this->owned_mh_cells.reserve(owned_cell_stack.size());
    while (!owned_cell_stack.empty()) {
      this->owned_mh_cells.push_back(owned_cell_stack.top());
      owned_cell_stack.pop();
    }
    this->unowned_mh_cells.reserve(unowned_cell_stack.size());
    while (!unowned_cell_stack.empty()) {
      this->unowned_mh_cells.push_back(unowned_cell_stack.top());
      unowned_cell_stack.pop();
    }
  }

  /**
   * Create halos on a 2D mesh.
   */
  inline void create_halos_2d(TriGeomMap &triangles, QuadGeomMap &quads,
                              MHGeomMap &mh_geom_map_tri,
                              MHGeomMap &mh_geom_map_quad) {

    MPICHK(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm,
                            &this->recv_win_data, &this->recv_win));

    // exchange geometry objects between ranks
    this->exchange_geometry_2d(triangles, mh_geom_map_tri,
                               this->remote_triangles);
    this->exchange_geometry_2d(quads, mh_geom_map_quad, this->remote_quads);

    MPICHK(MPI_Win_free(&this->recv_win));
    this->recv_win_data = nullptr;
  }

  /**
   * Wrapper around exchange_packed_2d for building 3D halos.
   */
  template <typename T>
  inline void exchange_2d_send_wrapper(
      std::map<int, std::map<int, std::shared_ptr<T>>> &rank_element_map,
      std::vector<std::shared_ptr<RemoteGeom2D<T>>> &output_container) {
    std::vector<int> send_ranks;
    send_ranks.reserve(rank_element_map.size());
    int num_send_ranks = 0;
    for (auto rankx : rank_element_map) {
      send_ranks.push_back(rankx.first);
      num_send_ranks++;
    }
    exchange_packed_2d(num_send_ranks, rank_element_map, send_ranks,
                       output_container);
  }

  /**
   * Create halos on a 3D mesh
   */
  inline void create_halos_3d(
      std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>
          &geoms_2d,
      std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>
          &geoms_3d,
      MHGeomMap &mh_geom_map) {

    // exchange geometry objects between ranks
    // map from mpi rank to element ids
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>>
        rank_element_map;
    // Set of remote ranks to send to
    std::vector<int> send_ranks;
    // Get the ranks to send to
    const int num_send_ranks = exchange_get_send_ranks(
        geoms_3d, mh_geom_map, rank_element_map, send_ranks);

    // for each rank we will send to loop over the 3d geoms to send and extract
    // the triangles/quads that construct those geoms

    // the information required to reconstruct the 3D geoms on each remote rank
    std::map<int, std::vector<int>> deconstructed_geoms;
    std::vector<int> send_sizes(num_send_ranks);
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::TriGeom>>>
        rank_triangle_map;
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::QuadGeom>>>
        rank_quad_map;

    deconstuct_per_rank_geoms_3d(
        comm_rank, geoms_2d, rank_element_map, num_send_ranks, send_ranks,
        send_sizes, deconstructed_geoms, rank_triangle_map, rank_quad_map);

    // send to remote MPI ranks
    MPICHK(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm,
                            &this->recv_win_data, &this->recv_win));

    exchange_2d_send_wrapper(rank_triangle_map, this->remote_triangles);
    exchange_2d_send_wrapper(rank_quad_map, this->remote_quads);

    // empty local element maps to free memory
    rank_triangle_map.clear();
    rank_quad_map.clear();
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>>
        rank_element_map_2d;

    // rebuild the 2D maps to recreate the 3D geoms
    for (const auto &geom : this->remote_triangles) {
      const int rank = geom->rank;
      const int gid = geom->id;
      const auto geom_ptr = geom->geom;
      rank_element_map_2d[rank][gid] =
          std::dynamic_pointer_cast<SpatialDomains::Geometry2D>(geom_ptr);
    }
    for (const auto &geom : this->remote_quads) {
      const int rank = geom->rank;
      const int gid = geom->id;
      const auto geom_ptr = geom->geom;
      rank_element_map_2d[rank][gid] =
          std::dynamic_pointer_cast<SpatialDomains::Geometry2D>(geom_ptr);
    }

    // exchange the 3D geometry information
    // determine the number of remote ranks that will send geoms to this rank
    auto request_barrier = exchange_init_send_counts();
    const int num_recv_ranks =
        exchange_finalise_send_counts(request_barrier, send_ranks);
    std::vector<int> recv_sizes(num_recv_ranks);
    std::vector<int> recv_ranks(num_recv_ranks);
    exchange_get_recv_ranks(num_send_ranks, send_ranks, send_sizes,
                            num_recv_ranks, recv_ranks, recv_sizes);
    MPICHK(MPI_Win_free(&this->recv_win));
    this->recv_win_data = nullptr;

    // recv_ranks now contains remote ranks that will send 3D deconstructed
    // objects. recv_sizes now contains how many ints each of these ranks will
    // send
    std::vector<int> packed_geoms;
    sendrecv_geoms_3d(this->comm, deconstructed_geoms, num_send_ranks,
                      send_ranks, send_sizes, num_recv_ranks, recv_ranks,
                      recv_sizes, packed_geoms);

    // rebuild element maps using objects recieved from remote ranks
    reconstruct_geoms_3d(rank_element_map_2d, packed_geoms,
                         this->remote_geoms_3d);
  }

  /**
   *  Construct the list of neighbour MPI ranks from the remote geometry
   *  objects.
   */
  inline void collect_neighbour_ranks() {
    this->neighbour_ranks.clear();
    std::set<int> remote_rank_set;
    // The ranks that own the copied geometry objects are ranks which local
    // communication patterns should be setup with.
    if (this->ndim == 2) {
      for (auto &geom : this->remote_triangles) {
        remote_rank_set.insert(geom->rank);
      }
      for (auto &geom : this->remote_quads) {
        remote_rank_set.insert(geom->rank);
      }
    } else if (this->ndim == 3) {
      for (auto &geom : this->remote_geoms_3d) {
        remote_rank_set.insert(geom->rank);
      }
    } else {
      NESOASSERT(false, "Unexpected number of dimensions.");
    }
    this->neighbour_ranks.reserve(remote_rank_set.size());
    for (auto rankx : remote_rank_set) {
      this->neighbour_ranks.push_back(rankx);
    }
  }

public:
  /// Disable (implicit) copies.
  ParticleMeshInterface(const ParticleMeshInterface &st) = delete;
  /// Disable (implicit) copies.
  ParticleMeshInterface &operator=(ParticleMeshInterface const &a) = delete;

  /// The Nektar++ graph on which the instance is based.
  Nektar::SpatialDomains::MeshGraphSharedPtr graph;
  /// Number of dimensions (physical).
  int ndim;
  /// Subdivision order of MeshHierarchy.
  int subdivision_order;
  /// Subdivision order offset used to create MeshHierarchy.
  int subdivision_order_offset;
  /// MPI Communicator used.
  MPI_Comm comm;
  /// MPI rank on communicator.
  int comm_rank;
  /// Size of MPI communicator.
  int comm_size;
  /// Underlying MeshHierarchy instance.
  std::shared_ptr<MeshHierarchy> mesh_hierarchy;
  /// Number of cells, i.e. Number of Nektar++ elements on this rank.
  int cell_count;
  /// Number of coarse cells in MeshHierarchy.
  int ncells_coarse;
  /// Number of fine cells per coarse cell in MeshHierarchy.
  int ncells_fine;
  /// Bounding box for elements on this rank.
  std::array<double, 6> bounding_box;
  /// Global bounding box for all elements in graph.
  std::array<double, 6> global_bounding_box;
  /// Local extents of local bounding box.
  std::array<double, 3> extents;
  /// Global extents of global bounding box.
  std::array<double, 3> global_extents;
  /// Vector of nearby ranks which local exchange patterns can be setup with.
  std::vector<int> neighbour_ranks;
  /// Vector of MeshHierarchy cells which are owned by this rank.
  std::vector<INT> owned_mh_cells;
  /// Vector of MeshHierarchy cells which were claimed but are not owned by this
  /// rank.
  std::vector<INT> unowned_mh_cells;
  /// Vector of remote TriGeom objects which have been copied to this rank.
  std::vector<std::shared_ptr<RemoteGeom2D<TriGeom>>> remote_triangles;
  /// Vector of remote QuadGeom objects which have been copied to this rank.
  std::vector<std::shared_ptr<RemoteGeom2D<QuadGeom>>> remote_quads;
  /// Vector of remote 3D geometry objects which have been copied to this rank.
  std::vector<std::shared_ptr<RemoteGeom3D>> remote_geoms_3d;

  ~ParticleMeshInterface() {}

  /**
   *  Create new ParticleMeshInterface.
   *
   *  @param graph Nektar++ MeshGraph to use.
   *  @param subdivision_order_offset Offset to the computed subdivision order.
   *  An offset of 0 will match the order of elements in the MeshGraph.
   *  @param comm MPI Communicator to use.
   */
  ParticleMeshInterface(Nektar::SpatialDomains::MeshGraphSharedPtr graph,
                        const int subdivision_order_offset = 0,
                        MPI_Comm comm = MPI_COMM_WORLD)
      : graph(graph), subdivision_order_offset(subdivision_order_offset),
        comm(comm) {

    this->ndim = graph->GetMeshDimension();
    MPICHK(MPI_Comm_rank(this->comm, &this->comm_rank));
    MPICHK(MPI_Comm_size(this->comm, &this->comm_size));

    NESOASSERT(graph->GetCurvedEdges().size() == 0,
               "Curved edge found in graph.");
    NESOASSERT(graph->GetCurvedFaces().size() == 0,
               "Curved face found in graph.");

    auto triangles = graph->GetAllTriGeoms();
    auto quads = graph->GetAllQuadGeoms();
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>> geoms_2d;
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms_3d;
    get_all_elements_2d(graph, geoms_2d);
    get_all_elements_3d(graph, geoms_3d);

    // Get a local and global bounding box for the mesh
    this->compute_bounding_box(geoms_2d, geoms_3d);
    // create a mesh hierarchy
    this->create_mesh_hierarchy();

    // assemble the cell claim weights locally for cells of interest to this
    // rank
    LocalClaim local_claim;
    MHGeomMap mh_geom_map_tri;
    MHGeomMap mh_geom_map_quad;
    MHGeomMap mh_geom_map_3d;
    if (this->ndim == 2) {
      this->claim_cells(triangles, local_claim, mh_geom_map_tri);
      this->claim_cells(quads, local_claim, mh_geom_map_quad);
    } else if (this->ndim == 3) {
      this->claim_cells(geoms_3d, local_claim, mh_geom_map_3d);
    } else {
      NESOASSERT(false, "unsupported spatial dimension");
    }
    claim_mesh_hierarchy_cells(local_claim);

    // get the MeshHierarchy global cells owned by this rank and those that
    // where successfully claimed by another rank.
    this->get_unowned_cells(local_claim);

    if (this->ndim == 2) {
      this->create_halos_2d(triangles, quads, mh_geom_map_tri,
                            mh_geom_map_quad);
    } else if (this->ndim == 3) {
      this->create_halos_3d(geoms_2d, geoms_3d, mh_geom_map_3d);
    } else {
      NESOASSERT(false, "unsupported spatial dimension");
    }
  }

  /**
   * Get the MPI communicator of the mesh.
   *
   * @returns MPI communicator.
   */
  inline MPI_Comm get_comm() { return this->comm; };
  /**
   *  Get the number of dimensions of the mesh.
   *
   *  @returns Number of mesh dimensions.
   */
  inline int get_ndim() { return this->ndim; };
  /**
   *  Get the Mesh dimensions.
   *
   *  @returns Mesh dimensions.
   */
  inline std::vector<int> &get_dims() { return this->mesh_hierarchy->dims; };
  /**
   * Get the subdivision order of the mesh.
   *
   * @returns Subdivision order.
   */
  inline int get_subdivision_order() { return this->subdivision_order; };
  /**
   * Get the total number of cells in the mesh on this MPI rank, i.e. the
   * number of Nektar++ elements on this MPI rank.
   *
   * @returns Total number of mesh cells on this MPI rank.
   */
  inline int get_cell_count() { return this->cell_count; };
  /**
   * Get the mesh width of the coarse cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy coarse cell width.
   */
  inline double get_cell_width_coarse() {
    return this->mesh_hierarchy->cell_width_coarse;
  };
  /**
   * Get the mesh width of the fine cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy fine cell width.
   */
  inline double get_cell_width_fine() {
    return this->mesh_hierarchy->cell_width_fine;
  };
  /**
   * Get the inverse mesh width of the coarse cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy inverse coarse cell width.
   */
  inline double get_inverse_cell_width_coarse() {
    return this->mesh_hierarchy->inverse_cell_width_coarse;
  };
  /**
   * Get the inverse mesh width of the fine cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy inverse fine cell width.
   */
  inline double get_inverse_cell_width_fine() {
    return this->mesh_hierarchy->inverse_cell_width_fine;
  };
  /**
   *  Get the global number of coarse cells.
   *
   *  @returns Global number of coarse cells.
   */
  inline int get_ncells_coarse() { return this->ncells_coarse; };
  /**
   *  Get the number of fine cells per coarse cell.
   *
   *  @returns Number of fine cells per coarse cell.
   */
  inline int get_ncells_fine() { return this->ncells_fine; };
  /**
   * Get the MeshHierarchy instance placed over the mesh.
   *
   * @returns MeshHierarchy placed over the mesh.
   */
  inline std::shared_ptr<MeshHierarchy> get_mesh_hierarchy() {
    return this->mesh_hierarchy;
  };
  /**
   *  Free the mesh and associated communicators.
   */
  inline void free() { this->mesh_hierarchy->free(); }
  /**
   *  Get a std::vector of MPI ranks which should be used to setup local
   *  communication patterns.
   *
   *  @returns std::vector of MPI ranks.
   */
  inline std::vector<int> &get_local_communication_neighbours() {
    this->collect_neighbour_ranks();
    return this->neighbour_ranks;
  };
  /**
   *  Get a point in the domain that should be in, or at least close to, the
   *  sub-domain on this MPI process. Useful for parallel initialisation.
   *
   *  @param point Pointer to array of size equal to at least the number of mesh
   * dimensions.
   */
  inline void get_point_in_subdomain(double *point) {

    auto graph = this->graph;
    NESOASSERT(this->ndim == 2 || this->ndim == 3,
               "Expected 2 or 3 position components");

    // Find a local geometry object
    GeometrySharedPtr geom;
    if (this->ndim == 2) {
      geom = std::dynamic_pointer_cast<Geometry>(get_element_2d(graph));
    } else {
      geom = std::dynamic_pointer_cast<Geometry>(get_element_3d(graph));
    }
    NESOASSERT(geom != nullptr, "Geom pointer is null.");

    // Get the average of the geoms vertices as a point in the domain
    const int num_verts = geom->GetNumVerts();
    auto v0 = geom->GetVertex(0);
    Array<OneD, NekDouble> coords(3);
    v0->GetCoords(coords);
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      point[dimx] = coords[dimx];
    }
    for (int vx = 1; vx < num_verts; vx++) {
      auto v = geom->GetVertex(vx);
      v->GetCoords(coords);
      for (int dimx = 0; dimx < this->ndim; dimx++) {
        point[dimx] += coords[dimx];
      }
    }
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      point[dimx] /= ((double)num_verts);
    }

    Array<OneD, NekDouble> mid(3);
    mid[2] = 0;
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      mid[dimx] = point[dimx];
    }

    // If somehow the average is not in the domain use the first vertex
    if (!geom->ContainsPoint(mid)) {
      v0->GetCoords(coords);
      for (int dimx = 0; dimx < this->ndim; dimx++) {
        const auto p = coords[dimx];
        point[dimx] = p;
        mid[dimx] = p;
      }
    }

    NESOASSERT(geom->ContainsPoint(mid), "Geom should contain this point");
  };
};

typedef std::shared_ptr<ParticleMeshInterface> ParticleMeshInterfaceSharedPtr;

} // namespace NESO
#endif
