#ifndef __COORDINATE_MAPPING_H
#define __COORDINATE_MAPPING_H
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp> // this pulls in the sycl headers namespace

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
template <typename T>
inline void
loc_collapsed_to_loc_coord(const int shape_type, const int shape_type_tet,
                           const int shape_type_pyr, const int shape_type_prism,
                           const int shape_type_hex, const sycl::vec<T, 3> &eta,
                           sycl::vec<T, 3> &xi) {

  /*


Tet

  xi[0] = (1.0 + eta[0]) * (-xi[1] - eta[2]) * 0.5 - 1.0;
  xi[1] = (1.0 + eta[0]) * (1.0 - eta[2]) * 0.5 - 1.0;
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

  xi[0] = (a) * (-xi[1] - eta[2]) * 0.5 - 1.0;
  xi[1] = c;
  xi[2] = eta[2];

Pyrimid
  xi[0] = c;
  xi[1] = (1.0 + eta[1]) * (b) * 0.5 - 1.0;
  xi[2] = eta[2];

Prism
  xi[0] = c;
  xi[1] = eta[1];
  xi[2] = eta[2];

hex:
  xi = eta;

  */

  const T a = 1.0 + eta[0];
  const T b = 1.0 - eta[2];
  const T c = (a) * (b)*0.5 - 1.0;
  const T xi1 =
      (shape_type == shape_type_tet)
          ? c
          : ((shape_type == shape_type_pyr) ? (1.0 + eta[1]) * (b)*0.5 - 1.0
                                            : eta[1]);
  const T tet_x = (1.0 + eta[0]) * (-xi1 - eta[2]) * 0.5 - 1.0;
  xi[0] = (shape_type == shape_type_tet)
              ? tet_x
              : ((shape_type == shape_type_hex) ? eta[0] : c);
  xi[1] = xi1;
  xi[2] = eta[2];
}
} // namespace GeometryInterface
} // namespace NESO

#endif