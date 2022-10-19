#ifndef __PARTICLE_INTERFACE_H__
#define __PARTICLE_INTERFACE_H__

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
#include "geometry_transport_2d.hpp"
#include "particle_boundary_conditions.hpp"

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

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
          const double ratio = volume * inverse_cell_volume;
          const int weight = 1000000.0 * ratio;
          mesh_tuple_to_mh_tuple(ndim, index_mesh, mesh_hierarchy, index_mh);
          const INT index_global =
              mesh_hierarchy->tuple_to_linear_global(index_mh);

          local_claim.claim(index_global, weight, ratio);
          mh_geom_map[index_global].push_back(element_id);
        }
      }
    }
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

  template <typename T>
  inline void exchange_geometry_2d(
      std::map<int, std::shared_ptr<T>> &element_map, MHGeomMap &mh_geom_map,
      std::vector<INT> &unowned_mh_cells,
      std::vector<std::shared_ptr<RemoteGeom2D<T>>> &output_container) {

    // map from mpi rank to element ids
    std::map<int, std::map<int, std::shared_ptr<T>>> rank_element_map;
    // Set of remote ranks to send to
    std::set<int> send_ranks_set;
    for (const INT &cell : unowned_mh_cells) {
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

    std::vector<int> send_ranks;
    const int num_send_ranks = send_ranks_set.size();
    send_ranks.reserve(num_send_ranks);
    for (auto &rankx : send_ranks_set) {
      send_ranks.push_back(rankx);
    }

    this->recv_win_data[0] = 0;
    MPI_Request request_barrier;
    MPICHK(MPI_Ibarrier(this->comm, &request_barrier));

    // map from remote MPI ranks to packed geoms
    std::map<int, std::shared_ptr<PackedGeoms2D>> rank_pack_geom_map;
    // pack the local geoms for each remote rank
    std::vector<int> send_packed_sizes(num_send_ranks);
    for (int rankx = 0; rankx < num_send_ranks; rankx++) {
      const int remote_rank = send_ranks[rankx];
      rank_pack_geom_map[remote_rank] = std::make_shared<PackedGeoms2D>(
          this->comm_rank, rank_element_map[remote_rank]);
      send_packed_sizes[rankx] =
          static_cast<int>(rank_pack_geom_map[remote_rank]->buf.size());
    }

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

    // send the packed sizes to the remote ranks
    std::vector<MPI_Request> recv_requests(num_recv_ranks);
    std::vector<int> recv_sizes(num_recv_ranks);
    // non-blocking recv packed geom sizes
    for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
      MPICHK(MPI_Irecv(recv_sizes.data() + rankx, 1, MPI_INT, MPI_ANY_SOURCE,
                       45, this->comm, recv_requests.data() + rankx));
    }
    // send sizes to remote ranks
    for (int rankx = 0; rankx < num_send_ranks; rankx++) {
      const int remote_rank = send_ranks[rankx];
      MPICHK(MPI_Send(send_packed_sizes.data() + rankx, 1, MPI_INT, remote_rank,
                      45, this->comm));
    }

    // wait for recv sizes to be recvd
    std::vector<MPI_Status> recv_status(num_recv_ranks);
    std::vector<int> recv_ranks(num_recv_ranks);

    MPICHK(
        MPI_Waitall(num_recv_ranks, recv_requests.data(), recv_status.data()));

    for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
      const int remote_rank = recv_status[rankx].MPI_SOURCE;
      recv_ranks[rankx] = remote_rank;
    }

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
      MPICHK(MPI_Irecv(recv_buffer.data() + rankx * max_recv_size,
                       num_recv_bytes, MPI_UNSIGNED_CHAR, remote_rank, 46,
                       this->comm, recv_requests.data() + rankx));
    }

    // send packed geoms
    std::vector<MPI_Request> send_requests(num_send_ranks);
    for (int rankx = 0; rankx < num_send_ranks; rankx++) {
      const int remote_rank = send_ranks[rankx];
      const int num_send_bytes = send_packed_sizes[rankx];
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
  /// Vector of remote TriGeom objects which have been copied to this rank.
  std::vector<std::shared_ptr<RemoteGeom2D<TriGeom>>> remote_triangles;
  /// Vector of remote QuadGeom objects which have been copied to this rank.
  std::vector<std::shared_ptr<RemoteGeom2D<QuadGeom>>> remote_quads;

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
      : graph(graph), comm(comm) {

    this->ndim = graph->GetMeshDimension();

    NESOASSERT(graph->GetCurvedEdges().size() == 0,
               "Curved edge found in graph.");
    NESOASSERT(graph->GetCurvedFaces().size() == 0,
               "Curved face found in graph.");
    NESOASSERT(graph->GetAllTetGeoms().size() == 0,
               "Tet element found in graph.");
    NESOASSERT(graph->GetAllPyrGeoms().size() == 0,
               "Pyr element found in graph.");
    NESOASSERT(graph->GetAllPrismGeoms().size() == 0,
               "Prism element found in graph.");
    NESOASSERT(graph->GetAllHexGeoms().size() == 0,
               "Hex element found in graph.");

    auto triangles = graph->GetAllTriGeoms();
    auto quads = graph->GetAllQuadGeoms();

    // Get a local and global bounding box for the mesh
    for (int dimx = 0; dimx < 3; dimx++) {
      this->bounding_box[dimx] = std::numeric_limits<double>::max();
      this->bounding_box[dimx + 3] = std::numeric_limits<double>::min();
    }

    int64_t num_elements = 0;
    for (auto &e : triangles) {
      NESOASSERT(e.first == e.second->GetGlobalID(), "GlobalID != key");
      expand_bounding_box(e.second, this->bounding_box);
      num_elements++;
    }
    for (auto &e : quads) {
      NESOASSERT(e.first == e.second->GetGlobalID(), "GlobalID != key");
      expand_bounding_box(e.second, this->bounding_box);
      num_elements++;
    }

    this->cell_count = num_elements;

    MPICHK(MPI_Allreduce(this->bounding_box.data(),
                         this->global_bounding_box.data(), 3, MPI_DOUBLE,
                         MPI_MIN, this->comm));
    MPICHK(MPI_Allreduce(this->bounding_box.data() + 3,
                         this->global_bounding_box.data() + 3, 3, MPI_DOUBLE,
                         MPI_MAX, this->comm));

    // Compute a set of coarse mesh sizes and dimensions for the mesh hierarchy
    double min_extent = std::numeric_limits<double>::max();
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      const double tmp_global_extent =
          this->global_bounding_box[dimx + 3] - this->global_bounding_box[dimx];
      const double tmp_extent =
          this->bounding_box[dimx + 3] - this->bounding_box[dimx];
      this->extents[dimx] = tmp_extent;
      this->global_extents[dimx] = tmp_global_extent;

      min_extent = std::min(min_extent, tmp_global_extent);
    }
    NESOASSERT(min_extent > 0.0, "Minimum extent is <= 0");

    std::vector<int> dims(this->ndim);
    std::vector<double> origin(this->ndim);

    int64_t hm_cell_count = 1;
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      origin[dimx] = this->global_bounding_box[dimx];
      const int tmp_dim = std::ceil(this->global_extents[dimx] / min_extent);
      dims[dimx] = tmp_dim;
      hm_cell_count *= ((int64_t)tmp_dim);
    }

    this->ncells_coarse = hm_cell_count;

    int64_t global_num_elements;
    MPICHK(MPI_Allreduce(&num_elements, &global_num_elements, 1, MPI_INT64_T,
                         MPI_SUM, this->comm));

    // compute a subdivision order that would result in the same order of fine
    // cells in the mesh hierarchy as mesh elements in Nektar++
    const double inverse_ndim = 1.0 / ((double)this->ndim);
    const int matching_subdivision_order =
        std::ceil((((double)std::log(global_num_elements)) -
                   ((double)std::log(hm_cell_count))) *
                  inverse_ndim);

    // apply the offset to this order and compute the used subdivision order
    this->subdivision_order =
        std::max(0, matching_subdivision_order + subdivision_order_offset);

    // create the mesh hierarchy
    this->mesh_hierarchy =
        std::make_shared<MeshHierarchy>(this->comm, this->ndim, dims, origin,
                                        min_extent, this->subdivision_order);

    this->ncells_fine = static_cast<int>(this->mesh_hierarchy->ncells_fine);

    // assemble the cell claim weights locally for cells of interest to this
    // rank
    LocalClaim local_claim;
    MHGeomMap mh_geom_map_tri;
    MHGeomMap mh_geom_map_quad;
    for (auto &e : triangles) {
      bounding_box_claim(e.first, e.second, mesh_hierarchy, local_claim,
                         mh_geom_map_tri);
    }
    for (auto &e : quads) {
      bounding_box_claim(e.first, e.second, mesh_hierarchy, local_claim,
                         mh_geom_map_quad);
    }

    // claim cells in the mesh hierarchy
    mesh_hierarchy->claim_initialise();
    for (auto &cellx : local_claim.claim_cells) {
      mesh_hierarchy->claim_cell(cellx,
                                 local_claim.claim_weights[cellx].weight);
    }
    mesh_hierarchy->claim_finalise();

    MPICHK(MPI_Comm_rank(this->comm, &this->comm_rank));
    MPICHK(MPI_Comm_size(this->comm, &this->comm_size));

    // get the MeshHierarchy global cells owned by this rank and those that
    // where successfully claimed by another rank.
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
    std::vector<INT> unowned_mh_cells;
    unowned_mh_cells.reserve(unowned_cell_stack.size());
    while (!unowned_cell_stack.empty()) {
      unowned_mh_cells.push_back(unowned_cell_stack.top());
      unowned_cell_stack.pop();
    }

    MPICHK(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm,
                            &this->recv_win_data, &this->recv_win));

    // exchange geometry objects between ranks
    this->exchange_geometry_2d(triangles, mh_geom_map_tri, unowned_mh_cells,
                               this->remote_triangles);
    this->exchange_geometry_2d(quads, mh_geom_map_quad, unowned_mh_cells,
                               this->remote_quads);

    MPICHK(MPI_Win_free(&this->recv_win));
    this->recv_win_data = nullptr;

    // The ranks that own the copied geometry objects are ranks which local
    // communication patterns should be setup with.
    std::set<int> remote_rank_set;
    for (auto &geom : this->remote_triangles) {
      remote_rank_set.insert(geom->rank);
    }
    for (auto &geom : this->remote_quads) {
      remote_rank_set.insert(geom->rank);
    }
    this->neighbour_ranks.reserve(remote_rank_set.size());
    for (auto rankx : remote_rank_set) {
      this->neighbour_ranks.push_back(rankx);
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
  virtual inline std::shared_ptr<MeshHierarchy> get_mesh_hierarchy() {
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
  virtual inline std::vector<int> &get_local_communication_neighbours() {
    return this->neighbour_ranks;
  };
};

typedef std::shared_ptr<ParticleMeshInterface> ParticleMeshInterfaceSharedPtr;

/**
 * Class to map particle positions to Nektar++ cells. Implemented for triangles
 * and quads.
 */
class NektarGraphLocalMapperT : public LocalMapper {
private:
  SYCLTargetSharedPtr sycl_target;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  const double tol;

public:
  ~NektarGraphLocalMapperT(){};

  /**
   * Callback for ParticleGroup to execute for additional setup of the
   * LocalMapper that may involve the ParticleGroup.
   *
   * @param particle_group ParticleGroup instance.
   */
  inline void particle_group_callback(ParticleGroup &particle_group) {

    particle_group.add_particle_dat(
        ParticleDat(particle_group.sycl_target,
                    ParticleProp(Sym<REAL>("NESO_REFERENCE_POSITIONS"),
                                 particle_group.domain->mesh->get_ndim()),
                    particle_group.domain->mesh->get_cell_count()));
  };

  /**
   *  Construct a new mapper object.
   *
   *  @param sycl_target SYCLTarget to use.
   *  @param particle_mesh_interface Interface between NESO-Particles and
   * Nektar++ mesh.
   *  @param tol Tolerance to pass to Nektar++ to bin particles into cells.
   */
  NektarGraphLocalMapperT(
      SYCLTargetSharedPtr sycl_target,
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      const double tol = 1.0e-10)
      : sycl_target(sycl_target),
        particle_mesh_interface(particle_mesh_interface), tol(tol){};

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  triangles and quads.
   */
  inline void map(ParticleGroup &particle_group, const int map_cell = -1) {

    ParticleDatSharedPtr<REAL> &position_dat = particle_group.position_dat;
    ParticleDatSharedPtr<REAL> &ref_position_dat =
        particle_group[Sym<REAL>("NESO_REFERENCE_POSITIONS")];
    ParticleDatSharedPtr<INT> &cell_id_dat = particle_group.cell_id_dat;
    ParticleDatSharedPtr<INT> &mpi_rank_dat = particle_group.mpi_rank_dat;

    auto t0 = profile_timestamp();
    const int rank = this->sycl_target->comm_pair.rank_parent;
    const int ndim = this->particle_mesh_interface->ndim;
    const int ncell = this->particle_mesh_interface->get_cell_count();
    const int nrow_max = mpi_rank_dat->cell_dat.get_nrow_max();
    auto graph = this->particle_mesh_interface->graph;

    CellDataT<REAL> particle_positions(sycl_target, nrow_max,
                                       position_dat->ncomp);
    CellDataT<REAL> ref_particle_positions(sycl_target, nrow_max,
                                           ref_position_dat->ncomp);
    CellDataT<INT> mpi_ranks(sycl_target, nrow_max, mpi_rank_dat->ncomp);
    CellDataT<INT> cell_ids(sycl_target, nrow_max, cell_id_dat->ncomp);

    EventStack event_stack;

    int cell_start, cell_end;
    if (map_cell < 0) {
      cell_start = 0;
      cell_end = ncell;
    } else {
      cell_start = map_cell;
      cell_end = map_cell + 1;
    }

    for (int cellx = cell_start; cellx < cell_end; cellx++) {
      // for (int cellx = 0; cellx < ncell; cellx++) {

      auto t0_copy_from = profile_timestamp();
      position_dat->cell_dat.get_cell_async(cellx, particle_positions,
                                            event_stack);
      ref_position_dat->cell_dat.get_cell_async(cellx, ref_particle_positions,
                                                event_stack);
      mpi_rank_dat->cell_dat.get_cell_async(cellx, mpi_ranks, event_stack);
      cell_id_dat->cell_dat.get_cell_async(cellx, cell_ids, event_stack);

      event_stack.wait();
      sycl_target->profile_map.inc(
          "NektarGraphLocalMapperT", "copy_from", 0,
          profile_elapsed(t0_copy_from, profile_timestamp()));

      const int nrow = mpi_rank_dat->cell_dat.nrow[cellx];
      Array<OneD, NekDouble> global_coord(3);
      Array<OneD, NekDouble> local_coord(3);
      auto point = std::make_shared<PointGeom>(ndim, -1, 0.0, 0.0, 0.0);

      for (int rowx = 0; rowx < nrow; rowx++) {

        if ((mpi_ranks)[1][rowx] < 0) {

          // copy the particle position into a nektar++ point format
          for (int dimx = 0; dimx < ndim; dimx++) {
            global_coord[dimx] = particle_positions[dimx][rowx];
            local_coord[dimx] = ref_particle_positions[dimx][rowx];
          }

          // update the PointGeom
          point->UpdatePosition(global_coord[0], global_coord[1],
                                global_coord[2]);

          auto t0_nektar_lookup = profile_timestamp();
          // get the elements that could contain the point
          auto element_ids = graph->GetElementsContainingPoint(point);
          // test the possible local geometry elements
          NekDouble dist;

          bool geom_found = false;
          // check the original nektar++ geoms
          for (auto &ex : element_ids) {
            Geometry2DSharedPtr geom_2d = graph->GetGeometry2D(ex);
            geom_found = geom_2d->ContainsPoint(global_coord, local_coord,
                                                this->tol, dist);
            if (geom_found) {
              (mpi_ranks)[1][rowx] = rank;
              (cell_ids)[0][rowx] = ex;
              for (int dimx = 0; dimx < ndim; dimx++) {
                ref_particle_positions[dimx][rowx] = local_coord[dimx];
              }
              break;
            }
          }
          sycl_target->profile_map.inc(
              "NektarGraphLocalMapperT", "map_nektar", 0,
              profile_elapsed(t0_nektar_lookup, profile_timestamp()));

          auto t0_halo_lookup = profile_timestamp();
          // containing geom not found in the set of owned geoms, now check the
          // remote geoms
          if (!geom_found) {
            for (auto &remote_geom :
                 this->particle_mesh_interface->remote_triangles) {
              geom_found = remote_geom->geom->ContainsPoint(
                  global_coord, local_coord, this->tol, dist);
              if (geom_found) {
                (mpi_ranks)[1][rowx] = remote_geom->rank;
                (cell_ids)[0][rowx] = remote_geom->id;
                for (int dimx = 0; dimx < ndim; dimx++) {
                  ref_particle_positions[dimx][rowx] = local_coord[dimx];
                }
                break;
              }
            }
          }
          if (!geom_found) {
            for (auto &remote_geom :
                 this->particle_mesh_interface->remote_quads) {
              geom_found = remote_geom->geom->ContainsPoint(
                  global_coord, local_coord, this->tol, dist);
              if (geom_found) {
                (mpi_ranks)[1][rowx] = remote_geom->rank;
                (cell_ids)[0][rowx] = remote_geom->id;
                for (int dimx = 0; dimx < ndim; dimx++) {
                  ref_particle_positions[dimx][rowx] = local_coord[dimx];
                }
                break;
              }
            }
          }
          sycl_target->profile_map.inc(
              "NektarGraphLocalMapperT", "map_halo", 0,
              profile_elapsed(t0_halo_lookup, profile_timestamp()));
          // if a geom is not found and there is a non-null global MPI rank then
          // this function was called after the global move and the lack of a
          // local cell / mpi rank is a fatal error.
          if (((mpi_ranks)[0][rowx] > -1) && !geom_found) {
            NESOASSERT(false, "No local geometry found for particle");
          }
        }
      }

      auto t0_copy_to = profile_timestamp();
      ref_position_dat->cell_dat.set_cell_async(cellx, ref_particle_positions,
                                                event_stack);
      mpi_rank_dat->cell_dat.set_cell_async(cellx, mpi_ranks, event_stack);
      cell_id_dat->cell_dat.set_cell_async(cellx, cell_ids, event_stack);
      event_stack.wait();

      sycl_target->profile_map.inc(
          "NektarGraphLocalMapperT", "copy_to", 0,
          profile_elapsed(t0_copy_to, profile_timestamp()));
    }
    sycl_target->profile_map.inc("NektarGraphLocalMapperT", "map", 1,
                                 profile_elapsed(t0, profile_timestamp()));
  };
};

/**
 *  Class to convert Nektar++ global ids of geometry objects to ids that can be
 *  used by NESO-Particles.
 */
class CellIDTranslation {
private:
  SYCLTargetSharedPtr sycl_target;
  ParticleDatSharedPtr<INT> cell_id_dat;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  BufferDeviceHost<int> id_map;
  int shift;

public:
  ~CellIDTranslation(){};

  /// Map from NESO-Particles ids to nektar++ global ids.
  std::vector<int> map_to_nektar;

  /**
   * Create a new geometry id mapper.
   *
   * @param sycl_target Compute device to use.
   * @param cell_id_dat ParticleDat of cell ids.
   * @param particle_mesh_interface Interface object between Nektar++ graph and
   * NESO-Particles.
   */
  CellIDTranslation(SYCLTargetSharedPtr sycl_target,
                    ParticleDatSharedPtr<INT> cell_id_dat,
                    ParticleMeshInterfaceSharedPtr particle_mesh_interface)
      : sycl_target(sycl_target), cell_id_dat(cell_id_dat),
        particle_mesh_interface(particle_mesh_interface),
        id_map(sycl_target, 1) {

    auto graph = this->particle_mesh_interface->graph;
    auto triangles = graph->GetAllTriGeoms();
    auto quads = graph->GetAllQuadGeoms();

    int id_min = std::numeric_limits<int>::max();
    int id_max = std::numeric_limits<int>::min();

    const int nelements = triangles.size() + quads.size();
    this->map_to_nektar.resize(nelements);
    int index = 0;
    for (auto &geom : triangles) {
      const int id = geom.second->GetGlobalID();
      NESOASSERT(geom.first == id, "Expected these ids to match");
      id_min = std::min(id_min, id);
      id_max = std::max(id_max, id);
      this->map_to_nektar[index++] = id;
    }
    for (auto &geom : quads) {
      const int id = geom.second->GetGlobalID();
      NESOASSERT(geom.first == id, "Expected these ids to match");
      id_min = std::min(id_min, id);
      id_max = std::max(id_max, id);
      this->map_to_nektar[index++] = id;
    }
    NESOASSERT(index == nelements, "element count missmatch");
    this->shift = id_min;
    const int shifted_max = id_max - id_min;
    id_map.realloc_no_copy(shifted_max + 1);

    for (int ex = 0; ex < nelements; ex++) {
      const int lookup_index = this->map_to_nektar[ex] - this->shift;
      this->id_map.h_buffer.ptr[lookup_index] = ex;
    }
    this->id_map.host_to_device();
  };

  /**
   *  Loop over all particles and map cell ids from Nektar++ cell ids to
   *  NESO-Particle cells ids.
   */
  inline void execute() {
    auto t0 = profile_timestamp();

    auto pl_iter_range = this->cell_id_dat->get_particle_loop_iter_range();
    auto pl_stride = this->cell_id_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = this->cell_id_dat->get_particle_loop_npart_cell();

    auto k_cell_id_dat = this->cell_id_dat->cell_dat.device_ptr();
    const auto k_lookup_map = this->id_map.d_buffer.ptr;
    const INT k_shift = this->shift;

    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const INT nektar_cell = k_cell_id_dat[cellx][0][layerx];
                const INT shifted_nektar_cell = nektar_cell - k_shift;
                const INT neso_cell = k_lookup_map[shifted_nektar_cell];
                k_cell_id_dat[cellx][0][layerx] = neso_cell;

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target->profile_map.inc("CellIDTranslation", "execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));
  };
};

#endif
