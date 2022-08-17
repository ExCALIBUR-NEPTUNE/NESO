#ifndef __PARTICLE_INTERFACE_H__
#define __PARTICLE_INTERFACE_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stack>
#include <vector>

#include <mpi.h>

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

#include "bounding_box_intersection.hpp"

using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

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
 *  to the mesh heirarchy. Local collection prevents excessive MPI RMA comms.
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
inline void mesh_tuple_to_mh_tuple(const int ndim, const int64_t *index_mesh,
                                   MeshHierarchy &mesh_hierarchy,
                                   INT *index_mh) {
  for (int dimx = 0; dimx < ndim; dimx++) {
    auto pq = std::div((long long)index_mesh[dimx],
                       (long long)mesh_hierarchy.ncells_dim_fine);
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
 * @param element Nektar++ mesh element to use.
 * @param mesh_hierarchy MeshHierarchy instance which cell claims will be made
 * into.
 * @param local_claim LocalClaim instance in which cell claims are being
 * collected into.
 */
template <typename T>
inline void bounding_box_claim(T element, MeshHierarchy &mesh_hierarchy,
                               LocalClaim &local_claim) {

  auto element_bounding_box = element->GetBoundingBox();
  const int ndim = mesh_hierarchy.ndim;
  auto origin = mesh_hierarchy.origin;

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
    int lhs_cell = lhs_point * mesh_hierarchy.inverse_cell_width_fine;
    int rhs_cell = rhs_point * mesh_hierarchy.inverse_cell_width_fine + 1;

    const int64_t ncells_dim_fine =
        mesh_hierarchy.ncells_dim_fine * mesh_hierarchy.dims[dimx];

    lhs_cell = (lhs_cell < 0) ? 0 : lhs_cell;
    lhs_cell = (lhs_cell >= ncells_dim_fine) ? ncells_dim_fine : lhs_cell;
    rhs_cell = (rhs_cell < 0) ? 0 : rhs_cell;
    rhs_cell = (rhs_cell > ncells_dim_fine) ? ncells_dim_fine : rhs_cell;

    cell_starts[dimx] = lhs_cell;
    cell_ends[dimx] = rhs_cell;
  }

  const double cell_width_fine = mesh_hierarchy.cell_width_fine;
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
        const double ratio = volume * inverse_cell_volume;
        const int weight = 1000000.0 * ratio;
        mesh_tuple_to_mh_tuple(ndim, index_mesh, mesh_hierarchy, index_mh);
        const INT index_global =
            mesh_hierarchy.tuple_to_linear_global(index_mh);

        local_claim.claim(index_global, weight, ratio);
      }
    }
  }
}

class ParticleMeshInterface : public HMesh {

private:
public:
  Nektar::SpatialDomains::MeshGraphSharedPtr graph;
  int ndim;
  int subdivision_order;
  MPI_Comm comm;
  MeshHierarchy mesh_hierarchy;
  int cell_count;
  int ncells_coarse;
  int ncells_fine;
  std::array<double, 6> bounding_box;
  std::array<double, 6> global_bounding_box;
  std::array<double, 3> extents;
  std::array<double, 3> global_extents;
  std::vector<int> neighbour_ranks;
  std::vector<INT> owned_mh_cells;

  ~ParticleMeshInterface() {}
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
      expand_bounding_box(e.second, this->bounding_box);
      num_elements++;
    }
    for (auto &e : quads) {
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
    // cells in the mesh heirarchy as mesh elements in Nektar++
    const double inverse_ndim = 1.0 / ((double)this->ndim);
    const int matching_subdivision_order =
        std::ceil((((double)std::log(global_num_elements)) -
                   ((double)std::log(hm_cell_count))) *
                  inverse_ndim);

    // apply the offset to this order and compute the used subdivision order
    this->subdivision_order =
        std::max(0, matching_subdivision_order + subdivision_order_offset);

    // create the mesh hierarchy
    this->mesh_hierarchy = MeshHierarchy(this->comm, this->ndim, dims, origin,
                                         min_extent, this->subdivision_order);

    this->ncells_fine = static_cast<int>(this->mesh_hierarchy.ncells_fine);

    // assemble the cell claim weights locally for cells of interest to this
    // rank
    LocalClaim local_claim;
    for (auto &e : triangles) {
      bounding_box_claim(e.second, mesh_hierarchy, local_claim);
    }
    for (auto &e : quads) {
      bounding_box_claim(e.second, mesh_hierarchy, local_claim);
    }

    // claim cells in the mesh hierarchy
    mesh_hierarchy.claim_initialise();
    for (auto &cellx : local_claim.claim_cells) {
      mesh_hierarchy.claim_cell(cellx, local_claim.claim_weights[cellx].weight);
    }
    mesh_hierarchy.claim_finalise();

    int rank;
    MPICHK(MPI_Comm_rank(this->comm, &rank));

    // get the MeshHierarchy global cells owned by this rank
    std::stack<INT> cell_stack;
    for (auto &cellx : local_claim.claim_cells) {
      const int owning_rank = this->mesh_hierarchy.get_owner(cellx);
      if (owning_rank == rank) {
        cell_stack.push(cellx);
      }
    }
    this->owned_mh_cells.reserve(cell_stack.size());
    while (!cell_stack.empty()) {
      this->owned_mh_cells.push_back(cell_stack.top());
      cell_stack.pop();
    }

    // get copies of remote geometry objects required to cover the owned cells
    MeshHierarchyBoundingBoxIntersection mhbbi(this->mesh_hierarchy,
                                               this->owned_mh_cells);
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
  inline std::vector<int> &get_dims() { return this->mesh_hierarchy.dims; };
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
    return this->mesh_hierarchy.cell_width_coarse;
  };
  /**
   * Get the mesh width of the fine cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy fine cell width.
   */
  inline double get_cell_width_fine() {
    return this->mesh_hierarchy.cell_width_fine;
  };
  /**
   * Get the inverse mesh width of the coarse cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy inverse coarse cell width.
   */
  inline double get_inverse_cell_width_coarse() {
    return this->mesh_hierarchy.inverse_cell_width_coarse;
  };
  /**
   * Get the inverse mesh width of the fine cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy inverse fine cell width.
   */
  inline double get_inverse_cell_width_fine() {
    return this->mesh_hierarchy.inverse_cell_width_fine;
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
  virtual inline MeshHierarchy *get_mesh_hierarchy() {
    return &this->mesh_hierarchy;
  };
  /**
   *  Free the mesh and associated communicators.
   */
  inline void free() { this->mesh_hierarchy.free(); }
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

#endif
