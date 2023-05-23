#ifndef __CANDIDATE_CELL_MAPPING
#define __CANDIDATE_CELL_MAPPING

#include "bounding_box_intersection.hpp"
#include "geometry_transport_3d.hpp"
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
 *  Class to facilitate mapping points to Nektar++ geometry objects. This class
 *  places a Cartesian grid over the locally owned mesh cells and remotely
 *  owned halo cells. A map is built from each Cartesian mesh cell to Nektar++
 *  cells (local and halo) which have a non-empty bounding box intersection.
 *  These are the "candidate" cells. For each candidate cell the owning rank
 *  and geometry id is recorded. Along with rank and geometry id the vertices
 *  required by the Nektar++ mapping algorithm are recorded.
 *
 *  To use these maps:
 *  1) Find the Cartesian mesh cell that contains the input coordinate.
 *  2) Loop over the Nektar++ mesh elements that overlap that Cartesian mesh
 *     cell.
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
  /*
   *  In 2D -> 3 points each 2 doubles.
   *  In 3D -> 4 points each 3 doubles.
   */
  int geom_vertex_stride;

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
  inline void write_vertices_2d(U &geom, const int index, double *output) {
    int last_point_index = -1;
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

    NESOASSERT(this->geom_vertex_stride == 6, "Unexpected geom stride.");
    output[index * 6 + 0] = (*v0)[0];
    output[index * 6 + 1] = (*v0)[1];
    output[index * 6 + 2] = (*v1)[0];
    output[index * 6 + 3] = (*v1)[1];
    output[index * 6 + 4] = (*v2)[0];
    output[index * 6 + 5] = (*v2)[1];
  }

  template <typename U>
  inline void write_vertices_3d(U &geom, const int index, double *output) {
    const auto shape_type = geom->GetShapeType();
    int index_v[4];
    index_v[0] = 0; // v0 is actually 0
    if (shape_type == LibUtilities::eHexahedron ||
        shape_type == LibUtilities::ePrism ||
        shape_type == LibUtilities::ePyramid) {
      index_v[1] = 1;
      index_v[2] = 3;
      index_v[3] = 4;
    } else if (shape_type == LibUtilities::eTetrahedron) {
      index_v[1] = 1;
      index_v[2] = 2;
      index_v[3] = 3;
    } else {
      NESOASSERT(false, "get_local_coords_3d Unknown shape type.");
    }

    NESOASSERT(this->geom_vertex_stride == 12, "Unexpected geom stride.");
    for (int vx = 0; vx < 4; vx++) {
      auto vertex = geom->GetVertex(index_v[vx]);
      NESOASSERT(vertex->GetCoordim() == 3, "Expected Coordim to be 3.");
      NekDouble x, y, z;
      vertex->GetCoords(x, y, z);
      output[index * this->geom_vertex_stride + vx * 3 + 0] = x;
      output[index * this->geom_vertex_stride + vx * 3 + 1] = y;
      output[index * this->geom_vertex_stride + vx * 3 + 2] = z;
    }
  }

public:
  /// The stride between lists of candidate cells for each Cartesian mesh cell.
  int map_stride;
  /// The Cartesian mesh over the owned and halo cells.
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
    NESOASSERT(ndim == 2 || ndim == 3, "Only defined for 2D and 3D");
    if (this->ndim == 2) {
      for (auto &geom : particle_mesh_interface->remote_triangles) {
        expand_bounding_box(geom->geom, this->bounding_box);
        cell_count++;
      }
      for (auto &geom : particle_mesh_interface->remote_quads) {
        expand_bounding_box(geom->geom, this->bounding_box);
        cell_count++;
      }
    } else {
      NESOASSERT(ndim == 3, "Expected ndim == 3");
      for (auto &geom : particle_mesh_interface->remote_geoms_3d) {
        expand_bounding_box(geom->geom, this->bounding_box);
        cell_count++;
      }
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

    this->geom_vertex_stride = (this->ndim == 2) ? 6 : 12;

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
        this->sycl_target, cell_count * geom_vertex_stride);

    const int rank = this->sycl_target->comm_pair.rank_parent;
    if (this->ndim == 2) {
      const int index_tri_geom =
          shape_type_to_int(LibUtilities::ShapeType::eTriangle);
      const int index_quad_geom =
          shape_type_to_int(LibUtilities::ShapeType::eQuadrilateral);

      int cell_index = 0;
      for (auto &geom : quads) {
        this->process_geom(geom_map, geom.second, cell_index);
        const int id = geom.second->GetGlobalID();
        NESOASSERT(id == geom.first, "ID mismatch");
        this->dh_cell_ids->h_buffer.ptr[cell_index] = id;
        this->dh_mpi_ranks->h_buffer.ptr[cell_index] = rank;
        this->dh_type->h_buffer.ptr[cell_index] = index_quad_geom;
        this->write_vertices_2d(geom.second, cell_index,
                                this->dh_vertices->h_buffer.ptr);
        cell_index++;
      }
      for (auto &geom : triangles) {
        this->process_geom(geom_map, geom.second, cell_index);
        const int id = geom.second->GetGlobalID();
        NESOASSERT(id == geom.first, "ID mismatch");
        this->dh_cell_ids->h_buffer.ptr[cell_index] = id;
        this->dh_mpi_ranks->h_buffer.ptr[cell_index] = rank;
        this->dh_type->h_buffer.ptr[cell_index] = index_tri_geom;
        this->write_vertices_2d(geom.second, cell_index,
                                this->dh_vertices->h_buffer.ptr);
        cell_index++;
      }
      for (auto &geom : particle_mesh_interface->remote_quads) {
        this->process_geom(geom_map, geom->geom, cell_index);
        this->dh_cell_ids->h_buffer.ptr[cell_index] = geom->id;
        this->dh_mpi_ranks->h_buffer.ptr[cell_index] = geom->rank;
        this->dh_type->h_buffer.ptr[cell_index] = index_quad_geom;
        this->write_vertices_2d(geom->geom, cell_index,
                                this->dh_vertices->h_buffer.ptr);
        cell_index++;
      }
      for (auto &geom : particle_mesh_interface->remote_triangles) {
        this->process_geom(geom_map, geom->geom, cell_index);
        this->dh_cell_ids->h_buffer.ptr[cell_index] = geom->id;
        this->dh_mpi_ranks->h_buffer.ptr[cell_index] = geom->rank;
        this->dh_type->h_buffer.ptr[cell_index] = index_tri_geom;
        this->write_vertices_2d(geom->geom, cell_index,
                                this->dh_vertices->h_buffer.ptr);
        cell_index++;
      }
      NESOASSERT(cell_index == cell_count, "mismatch in cell counts");
    } else if (this->ndim == 3) {
      // loop over owned geoms
      // map from geom id to geom of locally owned 3D objects.
      std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>
          geoms_3d_local;
      get_all_elements_3d(particle_mesh_interface->graph, geoms_3d_local);
      int cell_index = 0;
      for (auto &geom : geoms_3d_local) {
        this->process_geom(geom_map, geom.second, cell_index);
        const int id = geom.second->GetGlobalID();
        NESOASSERT(id == geom.first, "ID mismatch");
        this->dh_cell_ids->h_buffer.ptr[cell_index] = id;
        this->dh_mpi_ranks->h_buffer.ptr[cell_index] = rank;
        this->dh_type->h_buffer.ptr[cell_index] =
            shape_type_to_int(geom.second->GetShapeType());
        this->write_vertices_3d(geom.second, cell_index,
                                this->dh_vertices->h_buffer.ptr);
        cell_index++;
      }
      for (auto &geom : particle_mesh_interface->remote_geoms_3d) {
        this->process_geom(geom_map, geom, cell_index);
        this->dh_cell_ids->h_buffer.ptr[cell_index] = geom->id;
        this->dh_mpi_ranks->h_buffer.ptr[cell_index] = geom->rank;
        this->dh_type->h_buffer.ptr[cell_index] =
            shape_type_to_int(geom->geom->GetShapeType());
        this->write_vertices_3d(geom->geom, cell_index,
                                this->dh_vertices->h_buffer.ptr);
        cell_index++;
      }
      NESOASSERT(cell_index == cell_count, "mismatch in cell counts");
    } else {
      NESOASSERT(false, "unsupported number of dimensions");
    }

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

/**
 * TODO
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
   * TODO
   */
  template <typename T, typename U>
  CoarseLookupMap(const int ndim, SYCLTargetSharedPtr sycl_target,
                  std::map<int, std::shared_ptr<T>> geoms_local,
                  std::vector<std::shared_ptr<U>> geoms_remote)
      : ndim(ndim), sycl_target(sycl_target) {

    std::array<double, 6> bounding_box_tmp;
    std::array<double, 3> extents_tmp;

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
