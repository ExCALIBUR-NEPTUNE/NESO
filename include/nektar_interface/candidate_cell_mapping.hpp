#ifndef __CANDIDATE_CELL_MAPPING
#define __CANDIDATE_CELL_MAPPING

#include "bounding_box_intersection.hpp"
#include "particle_mesh_interface.hpp"
#include "utility_mesh_cartesian.hpp"

#include <array>
#include <list>
#include <map>
#include <neso_particles.hpp>
#include <utility>
#include <vector>

using namespace NESO::Particles;

namespace NESO {

/**
 *  Class to facilitate mapping points to Nektar++ geometry objects.
 */
class CandidateCellMapper {

protected:
  /// Disable (implicit) copies.
  CandidateCellMapper(const CandidateCellMapper &st) = delete;
  /// Disable (implicit) copies.
  CandidateCellMapper &operator=(CandidateCellMapper const &a) = delete;

  SYCLTargetSharedPtr sycl_target;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  std::array<double, 6> bounding_box;
  std::array<double, 3> extents;
  int ndim;

  template <typename U>
  inline void
  process_geom(std::map<int, std::list<std::pair<double, int>>> &geom_map,
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

  template <typename U>
  inline void write_vertices(U &geom, const int index, double *output) {
    int last_point_index;
    if (geom->GetShapeType() == LibUtilities::eTriangle) {
      last_point_index = 2;
    } else if (geom->GetShapeType() == LibUtilities::eQuadrilateral) {
      last_point_index = 3;
    } else {
      NESOASSERT(false, "get_local_coords_2d Unknown shape type.");
    }
    const auto v0 = geom->GetVertex(0);
    const auto v1 = geom->GetVertex(1);
    const auto v2 = geom->GetVertex(last_point_index);

    NESOASSERT(v0->GetCoordim() == 2, "Expected v0->Coordim to be 2.");
    NESOASSERT(v1->GetCoordim() == 2, "Expected v1->Coordim to be 2.");
    NESOASSERT(v2->GetCoordim() == 2, "Expected v2->Coordim to be 2.");

    output[index * 6 + 0] = (*v0)[0];
    output[index * 6 + 1] = (*v0)[1];
    output[index * 6 + 2] = (*v1)[0];
    output[index * 6 + 3] = (*v1)[1];
    output[index * 6 + 4] = (*v2)[0];
    output[index * 6 + 5] = (*v2)[1];
  }

public:
  /// TriGeom index
  static constexpr int index_tri_geom = 0;
  /// QuadGeom index
  static constexpr int index_quad_geom = 1;
  /// The stride between lists of candidate cells for each Cartesian mesh cell.
  int map_stride;
  /// The Cartesian mesh over the owned an halo cells.
  std::unique_ptr<DeviceCartesianMesh> cartesian_mesh;
  /// The nektar++ cell id for the cells indices pointed to from the map.
  std::unique_ptr<BufferDeviceHost<int>> dh_cell_ids;
  /// The MPI rank that owns the cell.
  std::unique_ptr<BufferDeviceHost<int>> dh_mpi_ranks;
  /// The type of the cell, i.e. a quad or a triangle.
  std::unique_ptr<BufferDeviceHost<int>> dh_type;
  /// The 3 vertices required by mapping from physical space to reference space.
  std::unique_ptr<BufferDeviceHost<double>> dh_vertices;
  /// The map from cells in Cartesian mesh to candidate cells in the lookup
  /// arrays.
  std::unique_ptr<BufferDeviceHost<int>> dh_map;
  /// The number of candidate cells for each Cartesian mesh cell.
  std::unique_ptr<BufferDeviceHost<int>> dh_map_sizes;

  /**
   *  Create instance for particular mesh interface.
   *
   *  @param sycl_target SYCLTarget instance to use.
   *  @param particle_mesh_interface Interface to build maps for.
   */
  CandidateCellMapper(SYCLTargetSharedPtr sycl_target,
                      ParticleMeshInterfaceSharedPtr particle_mesh_interface)
      : sycl_target(sycl_target),
        particle_mesh_interface(particle_mesh_interface) {

    // Create a bounding box around owned cells and halo cells.
    reset_bounding_box(this->bounding_box);
    // expand around the locally owned mesh cells
    expand_bounding_box_array(particle_mesh_interface->bounding_box,
                              this->bounding_box);
    int cell_count = particle_mesh_interface->cell_count;

    // expand around the remotely owned cells that form the halo
    this->ndim = particle_mesh_interface->ndim;
    NESOASSERT(ndim == 2, "Only defined for 2D");
    for (auto &geom : particle_mesh_interface->remote_triangles) {
      expand_bounding_box(geom->geom, this->bounding_box);
      cell_count++;
    }
    for (auto &geom : particle_mesh_interface->remote_quads) {
      expand_bounding_box(geom->geom, this->bounding_box);
      cell_count++;
    }

    // create a Cartesian mesh over the owned domain.
    double min_extent = std::numeric_limits<double>::max();
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      const double dim_extent =
          this->bounding_box[dimx + 3] - this->bounding_box[dimx];
      this->extents[dimx] = dim_extent;
      min_extent = std::min(min_extent, dim_extent);
    }

    const int target_cell_count = std::pow(2, ndim) * cell_count;
    double base_extent = min_extent / 4;
    std::array<int, 3> grid_cell_counts;
    int grid_linear_cell_count = 1;
    for (int dimx = 0; dimx < ndim; dimx++) {
      const int tmp_cell_count = this->extents[dimx] / base_extent;
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
      origin[dimx] = this->bounding_box[dimx];
      extents[dimx] = this->extents[dimx];
      cell_counts[dimx] = grid_cell_counts[dimx];
    }
    this->cartesian_mesh = std::make_unique<DeviceCartesianMesh>(
        this->sycl_target, this->ndim, origin, extents, cell_counts);

    // create map from cartesian cells to mesh cells by constructing the map in
    // reverse
    auto quads = particle_mesh_interface->graph->GetAllQuadGeoms();
    auto triangles = particle_mesh_interface->graph->GetAllTriGeoms();

    // map from cartesian cells to nektar mesh cells
    std::map<int, std::list<std::pair<double, int>>> geom_map;
    this->dh_cell_ids =
        std::make_unique<BufferDeviceHost<int>>(this->sycl_target, cell_count);
    this->dh_mpi_ranks =
        std::make_unique<BufferDeviceHost<int>>(this->sycl_target, cell_count);
    this->dh_type =
        std::make_unique<BufferDeviceHost<int>>(this->sycl_target, cell_count);
    this->dh_vertices = std::make_unique<BufferDeviceHost<double>>(
        this->sycl_target, cell_count * 6);

    const int rank = this->sycl_target->comm_pair.rank_parent;

    int cell_index = 0;
    for (auto &geom : quads) {
      this->process_geom(geom_map, geom.second, cell_index);
      const int id = geom.second->GetGlobalID();
      NESOASSERT(id == geom.first, "ID missmatch");
      this->dh_cell_ids->h_buffer.ptr[cell_index] = id;
      this->dh_mpi_ranks->h_buffer.ptr[cell_index] = rank;
      this->dh_type->h_buffer.ptr[cell_index] = index_quad_geom;
      this->write_vertices(geom.second, cell_index,
                           this->dh_vertices->h_buffer.ptr);
      cell_index++;
    }
    for (auto &geom : triangles) {
      this->process_geom(geom_map, geom.second, cell_index);
      const int id = geom.second->GetGlobalID();
      NESOASSERT(id == geom.first, "ID missmatch");
      this->dh_cell_ids->h_buffer.ptr[cell_index] = id;
      this->dh_mpi_ranks->h_buffer.ptr[cell_index] = rank;
      this->dh_type->h_buffer.ptr[cell_index] = index_tri_geom;
      this->write_vertices(geom.second, cell_index,
                           this->dh_vertices->h_buffer.ptr);
      cell_index++;
    }
    for (auto &geom : particle_mesh_interface->remote_quads) {
      this->process_geom(geom_map, geom->geom, cell_index);
      this->dh_cell_ids->h_buffer.ptr[cell_index] = geom->id;
      this->dh_mpi_ranks->h_buffer.ptr[cell_index] = geom->rank;
      this->dh_type->h_buffer.ptr[cell_index] = index_quad_geom;
      this->write_vertices(geom->geom, cell_index,
                           this->dh_vertices->h_buffer.ptr);
      cell_index++;
    }
    for (auto &geom : particle_mesh_interface->remote_triangles) {
      this->process_geom(geom_map, geom->geom, cell_index);
      this->dh_cell_ids->h_buffer.ptr[cell_index] = geom->id;
      this->dh_mpi_ranks->h_buffer.ptr[cell_index] = geom->rank;
      this->dh_type->h_buffer.ptr[cell_index] = index_tri_geom;
      this->write_vertices(geom->geom, cell_index,
                           this->dh_vertices->h_buffer.ptr);
      cell_index++;
    }

    NESOASSERT(cell_index == cell_count, "missmatch in cell counts");

    this->dh_cell_ids->host_to_device();
    this->dh_mpi_ranks->host_to_device();
    this->dh_type->host_to_device();
    this->dh_vertices->host_to_device();

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
