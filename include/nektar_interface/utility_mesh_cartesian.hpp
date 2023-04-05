#ifndef __UTILITY_MESH_CARTESIAN_H_
#define __UTILITY_MESH_CARTESIAN_H_

#include <neso_particles.hpp>

#include <memory>
#include <vector>

using namespace NESO::Particles;

namespace NESO {

/**
 *  Holds the metadata that describes a structured Cartesian mesh.
 */
class DeviceCartesianMesh {
protected:
  /// Disable (implicit) copies.
  DeviceCartesianMesh(const DeviceCartesianMesh &st) = delete;
  /// Disable (implicit) copies.
  DeviceCartesianMesh &operator=(DeviceCartesianMesh const &a) = delete;

public:
  /// SYCL target this mesh exists on.
  SYCLTargetSharedPtr sycl_target;
  /// Number of mesh dimensions.
  const int ndim;
  /// The origin of the mesh.
  std::shared_ptr<BufferDeviceHost<double>> dh_origin;
  /// The extents of the mesh.
  std::shared_ptr<BufferDeviceHost<double>> dh_extents;
  /// The number of cells in each dimension.
  std::shared_ptr<BufferDeviceHost<int>> dh_cell_counts;
  /// The width of the cells in each dimension.
  std::shared_ptr<BufferDeviceHost<double>> dh_cell_widths;
  /// The inverse of the cells in each dimension.
  std::shared_ptr<BufferDeviceHost<double>> dh_inverse_cell_widths;

  /**
   *  Create new mesh on the passed SYCL device.
   *
   *  @param[in] sycl_target SYCLTarget to use.
   *  @param[in] ndim Number of mesh dimensions.
   *  @param[in] origin Origin of the mesh.
   *  @param[in] extents Extent of the mesh in each dimension.
   *  @param[in] cell_counts Number of mesh cells in each dimension.
   */
  DeviceCartesianMesh(SYCLTargetSharedPtr sycl_target, const int ndim,
                      const std::vector<double> origin,
                      const std::vector<double> extents,
                      const std::vector<int> cell_counts)
      : sycl_target(sycl_target), ndim(ndim) {

    NESOASSERT(origin.size() >= ndim, "Origin vector has size less than ndim.");
    NESOASSERT(extents.size() >= ndim,
               "Extents vector has size less than ndim.");
    NESOASSERT(cell_counts.size() >= ndim,
               "Cell counts vector has size less than ndim.");

    this->dh_origin =
        std::make_shared<BufferDeviceHost<double>>(this->sycl_target, ndim);
    this->dh_extents =
        std::make_shared<BufferDeviceHost<double>>(this->sycl_target, ndim);
    this->dh_cell_counts =
        std::make_shared<BufferDeviceHost<int>>(this->sycl_target, ndim);
    this->dh_cell_widths =
        std::make_shared<BufferDeviceHost<double>>(this->sycl_target, ndim);
    this->dh_inverse_cell_widths =
        std::make_shared<BufferDeviceHost<double>>(this->sycl_target, ndim);

    for (int dimx = 0; dimx < ndim; dimx++) {
      this->dh_origin->h_buffer.ptr[dimx] = origin[dimx];
      const double extent = extents[dimx];
      NESOASSERT(extent > 0, "An extent is not strictly positive.");
      this->dh_extents->h_buffer.ptr[dimx] = extent;
      const int cell_count = cell_counts[dimx];
      NESOASSERT(cell_count > 0, "A cell count is not strictly positive.");
      this->dh_cell_counts->h_buffer.ptr[dimx] = cell_count;
      const double cell_width = extent / ((double)cell_counts[dimx]);
      this->dh_cell_widths->h_buffer.ptr[dimx] = cell_width;
      this->dh_inverse_cell_widths->h_buffer.ptr[dimx] = 1.0 / cell_width;
    }

    this->dh_origin->host_to_device();
    this->dh_extents->host_to_device();
    this->dh_cell_counts->host_to_device();
    this->dh_cell_widths->host_to_device();
    this->dh_inverse_cell_widths->host_to_device();
  }

  /**
   *  Return the cell which contains a point in the specified dimension.
   *
   *  @param[in] dim Dimension to find cell in.
   *  @param[in] point Coordinate in the requested dimension.
   *  @returns Containing cell in dimension.
   */
  inline int get_cell_in_dimension(const int dim, const double point) {
    NESOASSERT(point >= (this->dh_origin->h_buffer.ptr[dim] - 1.0e-8),
               "Point is below lower bound.");
    NESOASSERT(point <= (this->dh_origin->h_buffer.ptr[dim] +
                         this->dh_extents->h_buffer.ptr[dim] + 1.0e-8),
               "Point is above upper bound.");
    NESOASSERT((dim > -1) && (dim < this->ndim), "Bad dimension passed.");

    const double shifted_point = point - this->dh_origin->h_buffer.ptr[dim];
    double cell_float =
        shifted_point * this->dh_inverse_cell_widths->h_buffer.ptr[dim];
    int cell = cell_float;

    cell = (cell < 0) ? 0 : cell;
    cell = (cell >= this->dh_cell_counts->h_buffer.ptr[dim])
               ? this->dh_cell_counts->h_buffer.ptr[dim] - 1
               : cell;
    return cell;
  }

  /**
   *  Convert an index as a tuple to a linear index.
   *
   *  @param[in] cell_tuple Subscriptable host object with cell indices in each
   *  dimension.
   *  @returns Linear cell index.
   */
  template <typename T> inline int get_linear_cell_index(const T &cell_tuple) {
    auto cell_counts = this->dh_cell_counts->h_buffer.ptr;

    int idx = cell_tuple[this->ndim - 1];
    for (int dimx = (this->ndim - 2); dimx >= 0; dimx--) {
      idx *= cell_counts[dimx];
      idx += cell_tuple[dimx];
    }

    return idx;
  }

  /**
   * Get a Nektar++ format bounding box for a cell.
   *
   * @param[in] cell_tuple Subscriptable cell index tuple.
   * @param[out] bounding_box Bounding box.
   */
  template <typename T>
  inline void get_bounding_box(const T &cell_tuple,
                               std::array<double, 6> &bounding_box) {
    auto cell_counts = this->dh_cell_counts->h_buffer.ptr;
    auto cell_widths = this->dh_cell_widths->h_buffer.ptr;
    auto origin = this->dh_origin->h_buffer.ptr;

    for (int dimx = 0; dimx < this->ndim; dimx++) {
      const int cell = cell_tuple[dimx];
      NESOASSERT(cell >= 0, "Bad cell index (below 0).");
      NESOASSERT(cell < cell_counts[dimx],
                 "Bad cell index (greater than cell count).");
      const double lb = origin[dimx] + cell * cell_widths[dimx];
      const double ub = origin[dimx] + (cell + 1) * cell_widths[dimx];
      bounding_box[dimx] = lb;
      bounding_box[dimx + 3] = ub;
    }
  }

  /**
   *  Get the number of cells in the mesh.
   */
  inline int get_cell_count() {
    int count = 1;
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      count *= this->dh_cell_counts->h_buffer.ptr[dimx];
    }
    return count;
  }
};

} // namespace NESO

#endif
