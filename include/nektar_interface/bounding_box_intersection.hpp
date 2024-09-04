#ifndef __BOUNDING_BOX_INTERSECTION_H__
#define __BOUNDING_BOX_INTERSECTION_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stack>
#include <vector>

#include <mpi.h>

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

namespace NESO {

/**
 * Compute the overlap between two Nektar++ format bounding boxes.
 *
 * @param[in] ndim Actual number of dimensions (Nektar++ uses 6 element bounding
 * boxes always).
 * @param[in] a First bounding box.
 * @param[in] b Second bounding box.
 */
inline double bounding_box_intersection(const int ndim,
                                        const std::array<double, 6> &a,
                                        const std::array<double, 6> &b) {
  auto lambda_interval_overlap = [](const double la, const double ua,
                                    const double lb, const double ub) {
    if (ua <= lb) {
      return 0.0;
    } else if (la >= ub) {
      return 0.0;
    } else {
      const double interval_start = std::max(lb, la);
      const double interval_end = std::min(ub, ua);
      const double area = interval_end - interval_start;
      return area;
    }
  };

  double area = 1.0;
  for (int dimx = 0; dimx < ndim; dimx++) {
    area *= lambda_interval_overlap(a[dimx], a[dimx + 3], b[dimx], b[dimx + 3]);
  }
  return area;
}

/**
 * Resets a Nektar++ compatible bounding box (Array length 6 typically) to a
 * default that safely works with expand_bounding_box_array. Lower bounds are
 * set to MAX_DOUBLE and Upper bounds are set to MIN_DOUBLE.
 *
 * @param[in] bounding_box Bounding box to be initialised/reset.
 */
template <std::size_t ARRAY_LENGTH>
inline void reset_bounding_box(std::array<double, ARRAY_LENGTH> &bounding_box) {
  static_assert(ARRAY_LENGTH % 2 == 0, "Array size should be even.");
  constexpr int ndim = ARRAY_LENGTH / 2;

  for (int dimx = 0; dimx < ndim; dimx++) {
    bounding_box[dimx] = std::numeric_limits<double>::max();
    bounding_box[dimx + ndim] = std::numeric_limits<double>::lowest();
  }
}

/**
 *  Extend the bounds of a bounding box to include the passed bounding box. It
 *  is assumed that the bounds are stored in an array of size 2x(ndim) and the
 *  first ndim elements are the lower bounds and the remaining ndim elements
 *  are the upper bounds. Number of dimensions, ndim, is deduced from the array
 * size.
 *
 *  @param bounding_box_in Bounding box to use to extend the accumulation
 *  bounding box.
 *  @param bounding_box Bounding box to extend using element.
 */
template <std::size_t ARRAY_LENGTH>
inline void
expand_bounding_box_array(std::array<double, ARRAY_LENGTH> &bounding_box_in,
                          std::array<double, ARRAY_LENGTH> &bounding_box) {

  static_assert(ARRAY_LENGTH % 2 == 0, "Array size should be even.");
  constexpr int ndim = ARRAY_LENGTH / 2;

  for (int dimx = 0; dimx < ndim; dimx++) {
    bounding_box[dimx] = std::min(bounding_box[dimx], bounding_box_in[dimx]);
    bounding_box[dimx + ndim] =
        std::max(bounding_box[dimx + ndim], bounding_box_in[dimx + ndim]);
  }
}

/**
 *  Extend the bounds of a bounding box to include the given element.
 *
 *  @param element Nektar++ element that includes a GetBoundingBox method.
 *  @param bounding_box Bounding box to extend using element.
 */
template <typename T, std::size_t ARRAY_LENGTH>
inline void
expand_bounding_box(T element, std::array<double, ARRAY_LENGTH> &bounding_box) {

  auto element_bounding_box = element->GetBoundingBox();
  expand_bounding_box_array(element_bounding_box, bounding_box);
}

/**
 *  Get the bounding box for a fine cell in the MeshHierarchy in the same
 *  format as Nektar++.
 *
 *  @param mesh_hierarchy MeshHierarchy instance.
 *  @param global_cell Linear global cell index of cell.
 *  @param bounding_box Output array for bounding box.
 */
inline void get_bounding_box(std::shared_ptr<MeshHierarchy> mesh_hierarchy,
                             const INT global_cell,
                             std::array<double, 6> &bounding_box) {

  INT index[6];
  mesh_hierarchy->linear_to_tuple_global(global_cell, index);
  const int ndim = mesh_hierarchy->ndim;
  const double cell_width_coarse = mesh_hierarchy->cell_width_coarse;
  const double cell_width_fine = mesh_hierarchy->cell_width_fine;
  auto origin = mesh_hierarchy->origin;

  for (int dimx = 0; dimx < ndim; dimx++) {
    const double lhs = origin[dimx] + index[dimx] * cell_width_coarse +
                       index[dimx + ndim] * cell_width_fine;
    const double rhs = lhs + cell_width_fine;
    bounding_box[dimx] = lhs;
    bounding_box[dimx + 3] = rhs;
  }
}

/**
 *  Determine if [lhs_a, rhs_a] and [lhs_b, rhs_b] intersect.
 *
 *  @param lhs_a LHS of first interval.
 *  @param rhs_a RHS of first interval.
 *  @param lhs_b LHS of second interval.
 *  @param rhs_b RHS of second interval.
 *  @returns true if intervals intersect otherwise false.
 */
inline bool interval_intersect(const double lhs_a, const double rhs_a,
                               const double lhs_b, const double rhs_b) {
  return ((rhs_a >= lhs_b) && (lhs_a <= rhs_b));
}

/**
 *  Determine if two bounding boxes intersect.
 *
 *  @param ndim Number of dimensions.
 *  @param bounding_box_a First bounding box.
 *  @param bounding_box_b Second bounding box.
 *  @returns true if intervals intersect otherwise false.
 */
template <std::size_t ARRAY_LENGTH>
inline bool
bounding_box_intersect(const int ndim,
                       std::array<double, ARRAY_LENGTH> &bounding_box_a,
                       std::array<double, ARRAY_LENGTH> &bounding_box_b) {

  static_assert(ARRAY_LENGTH % 2 == 0, "Array size should be even.");
  constexpr int stride = ARRAY_LENGTH / 2;
  bool flag = true;
  for (int dimx = 0; dimx < ndim; dimx++) {
    flag = flag && interval_intersect(
                       bounding_box_a[dimx], bounding_box_a[dimx + stride],
                       bounding_box_b[dimx], bounding_box_b[dimx + stride]);
  }
  return flag;
}

/**
 *  Class to determine if a bounding box, e.g. from a Nektar++ element,
 *  intersects with a set of MeshHierarchy cells.
 */
class MeshHierarchyBoundingBoxIntersection {
private:
public:
  const int ndim;
  std::vector<std::array<double, 6>> bounding_boxes;
  std::array<double, 6> bounding_box;

  ~MeshHierarchyBoundingBoxIntersection() {};
  /**
   *  Create container of bounding boxes on which intersection tests can be
   *  performed.
   *
   *  @param mesh_hierarchy MeshHierarchy instance holding cells.
   *  @param owned_cells Vector of linear global cell indices in MeshHierarchy.
   */
  MeshHierarchyBoundingBoxIntersection(
      std::shared_ptr<MeshHierarchy> mesh_hierarchy,
      std::vector<INT> &owned_cells)
      : ndim(mesh_hierarchy->ndim) {

    // Get the bounding boxes for the owned cell and a bounding box for all
    // owned cells.
    for (int dimx = 0; dimx < 3; dimx++) {
      this->bounding_box[dimx] = std::numeric_limits<double>::max();
      this->bounding_box[dimx + 3] = std::numeric_limits<double>::lowest();
    }
    const int num_cells = owned_cells.size();
    this->bounding_boxes.resize(num_cells);
    for (int cellx = 0; cellx < num_cells; cellx++) {
      const INT cell = owned_cells[cellx];
      get_bounding_box(mesh_hierarchy, cell, this->bounding_boxes[cellx]);
      expand_bounding_box_array(this->bounding_boxes[cellx],
                                this->bounding_box);
    }
  };

  /**
   *  Test if a bounding box intersects with any of the held bounding boxes.
   *
   *  @param query_bounding_box Bounding box in Nektar++ format.
   */
  template <std::size_t ARRAY_LENGTH>
  inline bool intersects(std::array<double, ARRAY_LENGTH> &query_bounding_box) {
    // first test the bounding box around all held bounding boxes.
    if (!bounding_box_intersect(this->ndim, this->bounding_box,
                                query_bounding_box)) {
      return false;
    } else {
      for (auto &box : this->bounding_boxes) {
        if (bounding_box_intersect(this->ndim, box, query_bounding_box)) {
          return true;
        }
      }
      return false;
    }
  }
};

} // namespace NESO

#endif
