#ifndef __COORDINATE_MAPPING_H
#define __COORDINATE_MAPPING_H
#include "nektar_interface/geometry_transport_3d.hpp"
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar;

namespace NESO {
namespace GeometryInterface {
struct Tetrahedron {

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi Local coordinate to map to collapsed coordinate. Must be a
   * subscriptable type for indices 0,1,2.
   *  @param[in, out] eta Local collapsed coordinate. Must be a subscriptable
   * type for indices 0,1,2.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed(const T *xi, T *eta) {
    NekDouble d2 = 1.0 - xi[2];
    NekDouble d12 = -xi[1] - xi[2];
    if (fabs(d2) < NekConstants::kNekZeroTol) {
      if (d2 >= 0.) {
        d2 = NekConstants::kNekZeroTol;
      } else {
        d2 = -NekConstants::kNekZeroTol;
      }
    }
    if (fabs(d12) < NekConstants::kNekZeroTol) {
      if (d12 >= 0.) {
        d12 = NekConstants::kNekZeroTol;
      } else {
        d12 = -NekConstants::kNekZeroTol;
      }
    }
    eta[0] = 2.0 * (1.0 + xi[0]) / d12 - 1.0;
    eta[1] = 2.0 * (1.0 + xi[1]) / d2 - 1.0;
    eta[2] = xi[2];
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta Local collapsed coordinate to map to local coordinate. Must
   * be a subscriptable type for indices 0,1,2.
   *  @param[in, out] xi Local coordinate. Must be a
   * subscriptable type for indices 0,1,2.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord(const T *eta, T *xi) {
    xi[1] = (1.0 + eta[0]) * (1.0 - eta[2]) * 0.5 - 1.0;
    xi[0] = (1.0 + eta[0]) * (-xi[1] - eta[2]) * 0.5 - 1.0;
    xi[2] = eta[2];
  }

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi Local coordinate to map to collapsed coordinate.
   *  @param[in, out] eta Local collapsed coordinate.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed(const sycl::vec<T, 3> &xi,
                                         sycl::vec<T, 3> &eta) {
    NekDouble d2 = 1.0 - xi[2];
    NekDouble d12 = -xi[1] - xi[2];
    if (fabs(d2) < NekConstants::kNekZeroTol) {
      if (d2 >= 0.) {
        d2 = NekConstants::kNekZeroTol;
      } else {
        d2 = -NekConstants::kNekZeroTol;
      }
    }
    if (fabs(d12) < NekConstants::kNekZeroTol) {
      if (d12 >= 0.) {
        d12 = NekConstants::kNekZeroTol;
      } else {
        d12 = -NekConstants::kNekZeroTol;
      }
    }
    eta[0] = 2.0 * (1.0 + xi[0]) / d12 - 1.0;
    eta[1] = 2.0 * (1.0 + xi[1]) / d2 - 1.0;
    eta[2] = xi[2];
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta Local collapsed coordinate to map to local coordinate.
   *  @param[in, out] xi Local coordinate.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord(const sycl::vec<T, 3> &eta,
                                         sycl::vec<T, 3> &xi) {
    xi[1] = (1.0 + eta[0]) * (1.0 - eta[2]) * 0.5 - 1.0;
    xi[0] = (1.0 + eta[0]) * (-xi[1] - eta[2]) * 0.5 - 1.0;
    xi[2] = eta[2];
  }
};

struct Pyramid {

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi Local coordinate to map to collapsed coordinate. Must be a
   * subscriptable type for indices 0,1,2.
   *  @param[in, out] eta Local collapsed coordinate. Must be a subscriptable
   * type for indices 0,1,2.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed(const T *xi, T *eta) {
    NekDouble d2 = 1.0 - xi[2];
    if (fabs(d2) < NekConstants::kNekZeroTol) {
      if (d2 >= 0.) {
        d2 = NekConstants::kNekZeroTol;
      } else {
        d2 = -NekConstants::kNekZeroTol;
      }
    }
    eta[2] = xi[2]; // eta_z = xi_z
    eta[1] = 2.0 * (1.0 + xi[1]) / d2 - 1.0;
    eta[0] = 2.0 * (1.0 + xi[0]) / d2 - 1.0;
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta Local collapsed coordinate to map to local coordinate. Must
   * be a subscriptable type for indices 0,1,2.
   *  @param[in, out] xi Local coordinate. Must be a
   * subscriptable type for indices 0,1,2.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord(const T *eta, T *xi) {
    xi[0] = (1.0 + eta[0]) * (1.0 - eta[2]) * 0.5 - 1.0;
    xi[1] = (1.0 + eta[1]) * (1.0 - eta[2]) * 0.5 - 1.0;
    xi[2] = eta[2];
  }

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi Local coordinate to map to collapsed coordinate.
   *  @param[in, out] eta Local collapsed coordinate.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed(const sycl::vec<T, 3> &xi,
                                         sycl::vec<T, 3> &eta) {
    NekDouble d2 = 1.0 - xi[2];
    if (fabs(d2) < NekConstants::kNekZeroTol) {
      if (d2 >= 0.) {
        d2 = NekConstants::kNekZeroTol;
      } else {
        d2 = -NekConstants::kNekZeroTol;
      }
    }
    eta[2] = xi[2]; // eta_z = xi_z
    eta[1] = 2.0 * (1.0 + xi[1]) / d2 - 1.0;
    eta[0] = 2.0 * (1.0 + xi[0]) / d2 - 1.0;
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta Local collapsed coordinate to map to local coordinate.
   *  @param[in, out] xi Local coordinate.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord(const sycl::vec<T, 3> &eta,
                                         sycl::vec<T, 3> &xi) {
    xi[0] = (1.0 + eta[0]) * (1.0 - eta[2]) * 0.5 - 1.0;
    xi[1] = (1.0 + eta[1]) * (1.0 - eta[2]) * 0.5 - 1.0;
    xi[2] = eta[2];
  }
};

struct Prism {

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi Local coordinate to map to collapsed coordinate. Must be a
   * subscriptable type for indices 0,1,2.
   *  @param[in, out] eta Local collapsed coordinate. Must be a subscriptable
   * type for indices 0,1,2.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed(const T *xi, T *eta) {
    NekDouble d2 = 1.0 - xi[2];
    if (fabs(d2) < NekConstants::kNekZeroTol) {
      if (d2 >= 0.) {
        d2 = NekConstants::kNekZeroTol;
      } else {
        d2 = -NekConstants::kNekZeroTol;
      }
    }
    eta[2] = xi[2]; // eta_z = xi_z
    eta[1] = xi[1]; // eta_y = xi_y
    eta[0] = 2.0 * (1.0 + xi[0]) / d2 - 1.0;
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta Local collapsed coordinate to map to local coordinate. Must
   * be a subscriptable type for indices 0,1,2.
   *  @param[in, out] xi Local coordinate. Must be a
   * subscriptable type for indices 0,1,2.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord(const T *eta, T *xi) {
    xi[0] = (1.0 + eta[0]) * (1.0 - eta[2]) * 0.5 - 1.0;
    xi[1] = eta[1];
    xi[2] = eta[2];
  }

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi Local coordinate to map to collapsed coordinate.
   *  @param[in, out] eta Local collapsed coordinate.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed(const sycl::vec<T, 3> &xi,
                                         sycl::vec<T, 3> &eta) {
    NekDouble d2 = 1.0 - xi[2];
    if (fabs(d2) < NekConstants::kNekZeroTol) {
      if (d2 >= 0.) {
        d2 = NekConstants::kNekZeroTol;
      } else {
        d2 = -NekConstants::kNekZeroTol;
      }
    }
    eta[2] = xi[2]; // eta_z = xi_z
    eta[1] = xi[1]; // eta_y = xi_y
    eta[0] = 2.0 * (1.0 + xi[0]) / d2 - 1.0;
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta Local collapsed coordinate to map to local coordinate.
   *  @param[in, out] xi Local coordinate.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord(const sycl::vec<T, 3> &eta,
                                         sycl::vec<T, 3> &xi) {
    xi[0] = (1.0 + eta[0]) * (1.0 - eta[2]) * 0.5 - 1.0;
    xi[1] = eta[1];
    xi[2] = eta[2];
  }
};

struct Hexahedron {

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi Local coordinate to map to collapsed coordinate. Must be a
   * subscriptable type for indices 0,1,2.
   *  @param[in, out] eta Local collapsed coordinate. Must be a subscriptable
   * type for indices 0,1,2.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed(const T *xi, T *eta) {
    eta[0] = xi[0];
    eta[1] = xi[1];
    eta[2] = xi[2];
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta Local collapsed coordinate to map to local coordinate. Must
   * be a subscriptable type for indices 0,1,2.
   *  @param[in, out] xi Local coordinate. Must be a
   * subscriptable type for indices 0,1,2.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord(const T *eta, T *xi) {
    xi[0] = eta[0];
    xi[1] = eta[1];
    xi[2] = eta[2];
  }
  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi Local coordinate to map to collapsed coordinate.
   *  @param[in, out] eta Local collapsed coordinate.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed(const sycl::vec<T, 3> &xi,
                                         sycl::vec<T, 3> &eta) {
    eta = xi;
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta Local collapsed coordinate to map to local coordinate.
   *  @param[in, out] xi Local coordinate.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord(const sycl::vec<T, 3> &eta,
                                         sycl::vec<T, 3> &xi) {
    xi = eta;
  }
};

/**
 * TODO
 */
template <typename T>
inline void loc_collapsed_to_loc_coord(const int shape_type,
                                       const sycl::vec<T, 3> &eta,
                                       sycl::vec<T, 3> &xi) {

  /*
  Tet

    xi[1] = (1.0 + eta[0]) * (1.0 - eta[2]) * 0.5 - 1.0;
    xi[0] = (1.0 + eta[0]) * (-xi[1] - eta[2]) * 0.5 - 1.0;
    xi[2] = eta[2];

  Pyrimid
    xi[0] = (1.0 + eta[0]) * (1.0 - eta[2]) * 0.5 - 1.0;
    xi[1] = (1.0 + eta[1]) * (1.0 - eta[2]) * 0.5 - 1.0;
    xi[2] = eta[2];

  Prism
    xi[0] = (1.0 + eta[0]) * (1.0 - eta[2]) * 0.5 - 1.0;
    xi[1] = eta[1];
    xi[2] = eta[2];

  hex:
    xi = eta;


  a = 1.0 + eta[0];
  b = 1.0 - eta[2];
  c = (a) * (b) * 0.5 - 1.0;


  Tet

    xi[1] = c;
    xi[0] = (a) * (-xi[1] - eta[2]) * 0.5 - 1.0;
    xi[2] = eta[2];

  Pyrimid

    xi[1] = (1.0 + eta[1]) * (b) * 0.5 - 1.0;
    xi[0] = c;
    xi[2] = eta[2];

  Prism

    xi[1] = eta[1];
    xi[0] = c;
    xi[2] = eta[2];

  hex:
    xi = eta;
    */

  constexpr int shape_type_tet = shape_type_to_int(LibUtilities::eTetrahedron);
  constexpr int shape_type_pyr = shape_type_to_int(LibUtilities::ePyramid);
  constexpr int shape_type_hex = shape_type_to_int(LibUtilities::eHexahedron);

  const T eta0 = eta[0];
  const T eta1 = eta[1];
  const T eta2 = eta[2];

  const T a = 1.0 + eta0;
  const T b = 1.0 - eta2;
  const T c = (a) * (b)*0.5 - 1.0;
  const T xi1 =
      (shape_type == shape_type_tet)
          ? c
          : ((shape_type == shape_type_pyr) ? (1.0 + eta1) * (b)*0.5 - 1.0
                                            : eta1);
  const T tet_x = (1.0 + eta0) * (-xi1 - eta2) * 0.5 - 1.0;
  xi[0] = (shape_type == shape_type_tet)
              ? tet_x
              : ((shape_type == shape_type_hex) ? eta0 : c);
  xi[1] = xi1;
  xi[2] = eta2;
}

/**
 * TODO
 */
template <typename T>
inline void loc_coord_to_loc_collapsed(const int shape_type,
                                       const sycl::vec<T, 3> &xi,
                                       sycl::vec<T, 3> &eta) {

  constexpr int shape_type_tet = shape_type_to_int(LibUtilities::eTetrahedron);
  constexpr int shape_type_pyr = shape_type_to_int(LibUtilities::ePyramid);
  constexpr int shape_type_hex = shape_type_to_int(LibUtilities::eHexahedron);

  const T xi0 = xi[0];
  const T xi1 = xi[1];
  const T xi2 = xi[2];

  NekDouble d2 = 1.0 - xi2;
  if (fabs(d2) < NekConstants::kNekZeroTol) {
    if (d2 >= 0.) {
      d2 = NekConstants::kNekZeroTol;
    } else {
      d2 = -NekConstants::kNekZeroTol;
    }
  }
  NekDouble d12 = -xi1 - xi2;
  if (fabs(d12) < NekConstants::kNekZeroTol) {
    if (d12 >= 0.) {
      d12 = NekConstants::kNekZeroTol;
    } else {
      d12 = -NekConstants::kNekZeroTol;
    }
  }

  const T id2x2 = 2.0 / d2;
  const T a = 1.0 + xi0;
  const T b = (1.0 + xi1) * id2x2 - 1.0;
  const T c = a * id2x2 - 1.0;
  const T d = 2.0 * a / d12 - 1.0;

  eta[0] = (shape_type == shape_type_tet)   ? d
           : (shape_type == shape_type_hex) ? xi0
                                            : c;

  eta[1] = ((shape_type == shape_type_tet) || (shape_type == shape_type_pyr))
               ? b
               : xi1;
  eta[2] = xi2;
}

/**
 * TODO
 */
template <typename T>
inline bool
clamp_loc_coord(T *coord, const T tol = std::numeric_limits<T>::epsilon()) {

  bool clamp = false;
  if (!std::isfinite(*coord)) {
    *coord = 0.0;
    clamp = true;
  } else if ((*coord) < -(1.0 + tol)) {
    *coord = -(1.0 + tol);
    clamp = true;
  } else if ((*coord) > (1.0 + tol)) {
    *coord = 1.0 + tol;
    clamp = true;
  }
  return clamp;
}


/**
 * TODO
 */
template <typename T>
inline bool
clamp_loc_coords(T *coord0, T *coord1, const T tol = std::numeric_limits<T>::epsilon()) {
  return clamp_loc_coord(coord0, tol) || clamp_loc_coord(coord1, tol);
}


/**
 * TODO
 */
template <typename T>
inline bool
clamp_loc_coords(T *coord0, T *coord1, T * coord2, const T tol = std::numeric_limits<T>::epsilon()) {
  return clamp_loc_coord(coord0, tol) || clamp_loc_coords(coord1, coord2, tol);
}


} // namespace GeometryInterface
} // namespace NESO

#endif
