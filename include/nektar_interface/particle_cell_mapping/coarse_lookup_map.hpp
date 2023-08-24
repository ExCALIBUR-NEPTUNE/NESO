#ifndef __COARSE_LOOKUP_MAP_H_
#define __COARSE_LOOKUP_MAP_H_

#include "../bounding_box_intersection.hpp"
#include "../utility_mesh_cartesian.hpp"

#include <array>
#include <list>
#include <map>
#include <neso_particles.hpp>
#include <utility>
#include <vector>

using namespace NESO::Particles;

namespace NESO {

/**
 * A coarse lookup map is a Cartesian mesh overlay that is imposed over a
 * collection of local and remote geometry objects. A map is built from each
 * Cartesian cell to the geometry objects which have a non-zero intersection
 * between their bounding box and the cell.
 *
 * For each Cartesian cell, the set of geometry objects are candidate cells. If
 * a lookup point is contained within the Nektar++ mesh then at least one of
 * these candidate cells should contain the lookup point. Candidate cells are
 * ordered by overlap volume with the Cartesian mesh cell.
 */
class CoarseLookupMap {
protected:
  /// Disable (implicit) copies.
  CoarseLookupMap(const CoarseLookupMap &st) = delete;
  /// Disable (implicit) copies.
  CoarseLookupMap &operator=(CoarseLookupMap const &a) = delete;

  const int ndim;
  SYCLTargetSharedPtr sycl_target;

  template <typename U>
  inline void
  add_geom_to_map(std::map<int, std::list<std::pair<double, int>>> &geom_map,
                  U &geom, const int gid) {
    // get the bounding box of the geom
    auto geom_bounding_box = geom->GetBoundingBox();

    std::array<double, 3> cell_starts = {0, 0, 0};
    std::array<double, 3> cell_ends = {0, 0, 0};
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      const int start = this->cartesian_mesh->get_cell_in_dimension(
          dimx, geom_bounding_box[dimx]);
      const int end = this->cartesian_mesh->get_cell_in_dimension(
          dimx, geom_bounding_box[dimx + 3]);
      cell_starts[dimx] = start;
      cell_ends[dimx] = end;
    }

    std::array<double, 3> cell_tuple;
    std::array<double, 6> cell_bounding_box;

    for (int cz = cell_starts[2]; cz <= cell_ends[2]; cz++) {
      cell_tuple[2] = cz;
      for (int cy = cell_starts[1]; cy <= cell_ends[1]; cy++) {
        cell_tuple[1] = cy;
        for (int cx = cell_starts[0]; cx <= cell_ends[0]; cx++) {
          cell_tuple[0] = cx;
          const int cell_index =
              this->cartesian_mesh->get_linear_cell_index(cell_tuple);
          this->cartesian_mesh->get_bounding_box(cell_tuple, cell_bounding_box);
          const double overlap_estimate = bounding_box_intersection(
              this->ndim, geom_bounding_box, cell_bounding_box);
          if (overlap_estimate > 0.0) {
            geom_map[cell_index].push_back({overlap_estimate, gid});
          }
        }
      }
    }
  }

public:
  /// The stride between lists of candidate cells for each Cartesian mesh cell.
  int map_stride;
  /// The Cartesian mesh over the owned and halo cells.
  std::unique_ptr<DeviceCartesianMesh> cartesian_mesh;
  /// The map from cells in Cartesian mesh to candidate cells in the lookup
  /// arrays.
  std::unique_ptr<BufferDeviceHost<int>> dh_map;
  /// The number of candidate cells for each Cartesian mesh cell.
  std::unique_ptr<BufferDeviceHost<int>> dh_map_sizes;
  /// Map from Nektar++ global id to local id in the cartesian map.
  std::map<int, int> gid_to_lookup_id;

  /**
   * Create new lookup map from Cartesian cells to Nektar++ geometry elements.
   *
   * @param ndim Number of dimensions (e.g. 2 or 3).
   * @param sycl_target SYCLTarget to build lookup map on.
   * @param geoms_local Map of local geometry objects, i.e. geometry objects
   * that exist in the MeshGraph on this MPI rank.
   * @param geoms_remote Vector of remote geometry objects which have been
   * copied onto this MPI rank as halo objects.
   */
  template <typename T, typename U>
  CoarseLookupMap(const int ndim, SYCLTargetSharedPtr sycl_target,
                  std::map<int, std::shared_ptr<T>> geoms_local,
                  std::vector<std::shared_ptr<U>> geoms_remote)
      : ndim(ndim), sycl_target(sycl_target) {

    std::array<double, 6> bounding_box_tmp = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::array<double, 3> extents_tmp = {0.0, 0.0, 0.0};

    // Create a bounding box around passed cells.
    reset_bounding_box(bounding_box_tmp);
    int cell_count = 0;
    for (auto &geom : geoms_local) {
      expand_bounding_box(geom.second, bounding_box_tmp);
      cell_count++;
    }
    for (auto &geom : geoms_remote) {
      expand_bounding_box(geom->geom, bounding_box_tmp);
      cell_count++;
    }

    // create a Cartesian mesh over the passed geoms
    double min_extent = std::numeric_limits<double>::max();
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      const double dim_extent =
          bounding_box_tmp[dimx + 3] - bounding_box_tmp[dimx];
      extents_tmp[dimx] = dim_extent;
      min_extent = std::min(min_extent, dim_extent);
    }

    const int target_cell_count = std::pow(2, ndim) * cell_count;
    double base_extent = min_extent / 4;
    std::array<int, 3> grid_cell_counts;
    int grid_linear_cell_count = 1;
    for (int dimx = 0; dimx < ndim; dimx++) {
      const int tmp_cell_count = extents_tmp[dimx] / base_extent;
      grid_cell_counts[dimx] = tmp_cell_count;
      grid_linear_cell_count *= tmp_cell_count;
    }

    // expand the overlayed mesh to have the requested cell count
    if (grid_linear_cell_count < target_cell_count) {

      const double diff_factor =
          ((double)target_cell_count) / ((double)grid_linear_cell_count);
      const int diff_scaling =
          std::ceil(std::exp(std::log(diff_factor) / ((double)ndim)));

      grid_linear_cell_count = 1;
      for (int dimx = 0; dimx < ndim; dimx++) {
        int tmp_cell_count = grid_cell_counts[dimx];
        tmp_cell_count *= diff_scaling;
        grid_cell_counts[dimx] = tmp_cell_count;
        grid_linear_cell_count *= tmp_cell_count;
      }

      NESOASSERT(grid_linear_cell_count >= target_cell_count,
                 "Target cell count not reached.");
    }

    // create a mesh object
    std::vector<double> origin(ndim);
    std::vector<double> extents(ndim);
    std::vector<int> cell_counts(ndim);
    for (int dimx = 0; dimx < ndim; dimx++) {
      origin[dimx] = bounding_box_tmp[dimx];
      extents[dimx] = extents_tmp[dimx];
      cell_counts[dimx] = grid_cell_counts[dimx];
    }

    this->cartesian_mesh = std::make_unique<DeviceCartesianMesh>(
        this->sycl_target, this->ndim, origin, extents, cell_counts);

    // map from cartesian cells to nektar mesh cells
    std::map<int, std::list<std::pair<double, int>>> geom_map;
    int cell_index = 0;
    for (auto &geom : geoms_local) {
      const int geom_id = geom.second->GetGlobalID();
      this->add_geom_to_map(geom_map, geom.second, cell_index);
      gid_to_lookup_id[geom_id] = cell_index;
      cell_index++;
    }
    for (auto &geom : geoms_remote) {
      const int geom_id = geom->id;
      this->add_geom_to_map(geom_map, geom->geom, cell_index);
      gid_to_lookup_id[geom_id] = cell_index;
      cell_index++;
    }

    int max_map_size = 0;
    // sort the maps in order of largest covering area.
    for (auto &cell : geom_map) {
      cell.second.sort();
      cell.second.reverse();
      max_map_size =
          std::max(max_map_size, static_cast<int>(cell.second.size()));
    }

    this->map_stride = max_map_size;
    const int mesh_cell_count = this->cartesian_mesh->get_cell_count();
    this->dh_map = std::make_unique<BufferDeviceHost<int>>(
        this->sycl_target, this->map_stride * mesh_cell_count);
    this->dh_map_sizes = std::make_unique<BufferDeviceHost<int>>(
        this->sycl_target, mesh_cell_count);

    for (int cellx = 0; cellx < mesh_cell_count; cellx++) {
      this->dh_map_sizes->h_buffer.ptr[cellx] = 0;
    }

    for (auto &cell : geom_map) {
      const int mesh_cell_id = cell.first;
      NESOASSERT(mesh_cell_id >= 0, "Bad cell index (low).");
      NESOASSERT(mesh_cell_id < mesh_cell_count, "Bad cell index (high).");
      int inner_index = 0;
      for (auto &gid : cell.second) {
        this->dh_map->h_buffer
            .ptr[mesh_cell_id * this->map_stride + inner_index] = gid.second;
        inner_index++;
      }
      this->dh_map_sizes->h_buffer.ptr[mesh_cell_id] = inner_index;
      NESOASSERT(inner_index <= this->map_stride, "Map size exceeds stride.");
    }

    this->dh_map->host_to_device();
    this->dh_map_sizes->host_to_device();
  }
};

}; // namespace NESO

#endif
