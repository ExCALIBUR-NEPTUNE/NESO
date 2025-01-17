#ifndef __COORDINATE_MAPPING_H
#define __COORDINATE_MAPPING_H
#include "nektar_interface/geometry_transport/shape_mapping.hpp"
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar;
using namespace NESO::Particles;

namespace NESO {
namespace GeometryInterface {

/**
 *  Abstract base class for converting betwen local coordinates and local
 *  collapsed coordinates in 3D.
 */
template <typename SPECIALISATION> struct BaseCoordinateMapping3D {

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi0 Local coordinate to map to collapsed coordinate. Component
   * x.
   *  @param[in] xi1 Local coordinate to map to collapsed coordinate. Component
   * y.
   *  @param[in] xi2 Local coordinate to map to collapsed coordinate. Component
   * z.
   *  @param[in, out] eta0 Local collapsed coordinate. Component x.
   *  @param[in, out] eta1 Local collapsed coordinate. Component y.
   *  @param[in, out] eta2 Local collapsed coordinate. Component z.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed(const T xi0, const T xi1, const T xi2,
                                         T *eta0, T *eta1, T *eta2) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.loc_coord_to_loc_collapsed_v(xi0, xi1, xi2, eta0, eta1, eta2);
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta0 Local collapsed coordinate. Component x.
   *  @param[in] eta1 Local collapsed coordinate. Component y.
   *  @param[in] eta2 Local collapsed coordinate. Component z.
   *  @param[in, out] xi0 Local coordinate to map to collapsed coordinate.
   * Component x.
   *  @param[in, out] xi1 Local coordinate to map to collapsed coordinate.
   * Component y.
   *  @param[in, out] xi2 Local coordinate to map to collapsed coordinate.
   * Component z.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord(const T eta0, const T eta1,
                                         const T eta2, T *xi0, T *xi1, T *xi2) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.loc_collapsed_to_loc_coord_v(eta0, eta1, eta2, xi0, xi1, xi2);
  }

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi Local coordinate to map to collapsed coordinate. Must be a
   * subscriptable type for indices 0,1,2.
   *  @param[in, out] eta Local collapsed coordinate. Must be a subscriptable
   * type for indices 0,1,2.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed(const T &xi, T &eta) {
    loc_coord_to_loc_collapsed(xi[0], xi[1], xi[2], &eta[0], &eta[1], &eta[2]);
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
  inline void loc_collapsed_to_loc_coord(const T &eta, T &xi) {
    loc_collapsed_to_loc_coord(eta[0], eta[1], eta[2], &xi[0], &xi[1], &xi[2]);
  }
};

/**
 *  Abstract base class for converting betwen local coordinates and local
 *  collapsed coordinates in 2D.
 */
template <typename SPECIALISATION> struct BaseCoordinateMapping2D {

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi0 Local coordinate to map to collapsed coordinate. Component
   * x.
   *  @param[in] xi1 Local coordinate to map to collapsed coordinate. Component
   * y.
   *  @param[in, out] eta0 Local collapsed coordinate. Component x.
   *  @param[in, out] eta1 Local collapsed coordinate. Component y.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed(const T xi0, const T xi1, T *eta0,
                                         T *eta1) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.loc_coord_to_loc_collapsed_v(xi0, xi1, eta0, eta1);
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta0 Local collapsed coordinate. Component x.
   *  @param[in] eta1 Local collapsed coordinate. Component y.
   *  @param[in, out] xi0 Local coordinate to map to collapsed coordinate.
   * Component x.
   *  @param[in, out] xi1 Local coordinate to map to collapsed coordinate.
   * Component y.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord(const T eta0, const T eta1, T *xi0,
                                         T *xi1) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.loc_collapsed_to_loc_coord_v(eta0, eta1, xi0, xi1);
  }

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi Local coordinate to map to collapsed coordinate. Must be a
   * subscriptable type for indices 0,1.
   *  @param[in, out] eta Local collapsed coordinate. Must be a subscriptable
   * type for indices 0,1.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed(const T &xi, T &eta) {
    loc_coord_to_loc_collapsed(xi[0], xi[1], &eta[0], &eta[1]);
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta Local collapsed coordinate to map to local coordinate. Must
   * be a subscriptable type for indices 0,1.
   *  @param[in, out] xi Local coordinate. Must be a
   * subscriptable type for indices 0,1.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord(const T &eta, T &xi) {
    loc_collapsed_to_loc_coord(eta[0], eta[1], &xi[0], &xi[1]);
  }
};

struct Tetrahedron : BaseCoordinateMapping3D<Tetrahedron> {

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi0 Local coordinate to map to collapsed coordinate. Component
   * x.
   *  @param[in] xi1 Local coordinate to map to collapsed coordinate. Component
   * y.
   *  @param[in] xi2 Local coordinate to map to collapsed coordinate. Component
   * z.
   *  @param[in, out] eta0 Local collapsed coordinate. Component x.
   *  @param[in, out] eta1 Local collapsed coordinate. Component y.
   *  @param[in, out] eta2 Local collapsed coordinate. Component z.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed_v(const T xi0, const T xi1,
                                           const T xi2, T *eta0, T *eta1,
                                           T *eta2) {
    NekDouble d2 = 1.0 - xi2;
    NekDouble d12 = -xi1 - xi2;
    if (sycl::fabs(d2) < NekConstants::kNekZeroTol) {
      if (d2 >= 0.) {
        d2 = NekConstants::kNekZeroTol;
      } else {
        d2 = -NekConstants::kNekZeroTol;
      }
    }
    if (sycl::fabs(d12) < NekConstants::kNekZeroTol) {
      if (d12 >= 0.) {
        d12 = NekConstants::kNekZeroTol;
      } else {
        d12 = -NekConstants::kNekZeroTol;
      }
    }
    *eta0 = 2.0 * (1.0 + xi0) / d12 - 1.0;
    *eta1 = 2.0 * (1.0 + xi1) / d2 - 1.0;
    *eta2 = xi2;
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta0 Local collapsed coordinate. Component x.
   *  @param[in] eta1 Local collapsed coordinate. Component y.
   *  @param[in] eta2 Local collapsed coordinate. Component z.
   *  @param[in, out] xi0 Local coordinate to map to collapsed coordinate.
   * Component x.
   *  @param[in, out] xi1 Local coordinate to map to collapsed coordinate.
   * Component y.
   *  @param[in, out] xi2 Local coordinate to map to collapsed coordinate.
   * Component z.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord_v(const T eta0, const T eta1,
                                           const T eta2, T *xi0, T *xi1,
                                           T *xi2) {
    *xi1 = (1.0 + eta1) * (1.0 - eta2) * 0.5 - 1.0;
    *xi0 = (1.0 + eta0) * (-(*xi1) - eta2) * 0.5 - 1.0;
    *xi2 = eta2;
  }
};

struct Pyramid : BaseCoordinateMapping3D<Pyramid> {

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi0 Local coordinate to map to collapsed coordinate. Component
   * x.
   *  @param[in] xi1 Local coordinate to map to collapsed coordinate. Component
   * y.
   *  @param[in] xi2 Local coordinate to map to collapsed coordinate. Component
   * z.
   *  @param[in, out] eta0 Local collapsed coordinate. Component x.
   *  @param[in, out] eta1 Local collapsed coordinate. Component y.
   *  @param[in, out] eta2 Local collapsed coordinate. Component z.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed_v(const T xi0, const T xi1,
                                           const T xi2, T *eta0, T *eta1,
                                           T *eta2) {
    NekDouble d2 = 1.0 - xi2;
    if (sycl::fabs(d2) < NekConstants::kNekZeroTol) {
      if (d2 >= 0.) {
        d2 = NekConstants::kNekZeroTol;
      } else {
        d2 = -NekConstants::kNekZeroTol;
      }
    }
    *eta2 = xi2; // eta_z = xi_z
    *eta1 = 2.0 * (1.0 + xi1) / d2 - 1.0;
    *eta0 = 2.0 * (1.0 + xi0) / d2 - 1.0;
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta0 Local collapsed coordinate. Component x.
   *  @param[in] eta1 Local collapsed coordinate. Component y.
   *  @param[in] eta2 Local collapsed coordinate. Component z.
   *  @param[in, out] xi0 Local coordinate to map to collapsed coordinate.
   * Component x.
   *  @param[in, out] xi1 Local coordinate to map to collapsed coordinate.
   * Component y.
   *  @param[in, out] xi2 Local coordinate to map to collapsed coordinate.
   * Component z.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord_v(const T eta0, const T eta1,
                                           const T eta2, T *xi0, T *xi1,
                                           T *xi2) {
    *xi0 = (1.0 + eta0) * (1.0 - eta2) * 0.5 - 1.0;
    *xi1 = (1.0 + eta1) * (1.0 - eta2) * 0.5 - 1.0;
    *xi2 = eta2;
  }
};

struct Prism : BaseCoordinateMapping3D<Prism> {

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi0 Local coordinate to map to collapsed coordinate. Component
   * x.
   *  @param[in] xi1 Local coordinate to map to collapsed coordinate. Component
   * y.
   *  @param[in] xi2 Local coordinate to map to collapsed coordinate. Component
   * z.
   *  @param[in, out] eta0 Local collapsed coordinate. Component x.
   *  @param[in, out] eta1 Local collapsed coordinate. Component y.
   *  @param[in, out] eta2 Local collapsed coordinate. Component z.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed_v(const T xi0, const T xi1,
                                           const T xi2, T *eta0, T *eta1,
                                           T *eta2) {
    NekDouble d2 = 1.0 - xi2;
    if (sycl::fabs(d2) < NekConstants::kNekZeroTol) {
      if (d2 >= 0.) {
        d2 = NekConstants::kNekZeroTol;
      } else {
        d2 = -NekConstants::kNekZeroTol;
      }
    }
    *eta2 = xi2; // eta_z = xi_z
    *eta1 = xi1; // eta_y = xi_y
    *eta0 = 2.0 * (1.0 + xi0) / d2 - 1.0;
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta0 Local collapsed coordinate. Component x.
   *  @param[in] eta1 Local collapsed coordinate. Component y.
   *  @param[in] eta2 Local collapsed coordinate. Component z.
   *  @param[in, out] xi0 Local coordinate to map to collapsed coordinate.
   * Component x.
   *  @param[in, out] xi1 Local coordinate to map to collapsed coordinate.
   * Component y.
   *  @param[in, out] xi2 Local coordinate to map to collapsed coordinate.
   * Component z.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord_v(const T eta0, const T eta1,
                                           const T eta2, T *xi0, T *xi1,
                                           T *xi2) {
    *xi0 = (1.0 + eta0) * (1.0 - eta2) * 0.5 - 1.0;
    *xi1 = eta1;
    *xi2 = eta2;
  }
};

struct Hexahedron : BaseCoordinateMapping3D<Hexahedron> {

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi0 Local coordinate to map to collapsed coordinate. Component
   * x.
   *  @param[in] xi1 Local coordinate to map to collapsed coordinate. Component
   * y.
   *  @param[in] xi2 Local coordinate to map to collapsed coordinate. Component
   * z.
   *  @param[in, out] eta0 Local collapsed coordinate. Component x.
   *  @param[in, out] eta1 Local collapsed coordinate. Component y.
   *  @param[in, out] eta2 Local collapsed coordinate. Component z.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed_v(const T xi0, const T xi1,
                                           const T xi2, T *eta0, T *eta1,
                                           T *eta2) {
    *eta0 = xi0;
    *eta1 = xi1;
    *eta2 = xi2;
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta0 Local collapsed coordinate. Component x.
   *  @param[in] eta1 Local collapsed coordinate. Component y.
   *  @param[in] eta2 Local collapsed coordinate. Component z.
   *  @param[in, out] xi0 Local coordinate to map to collapsed coordinate.
   * Component x.
   *  @param[in, out] xi1 Local coordinate to map to collapsed coordinate.
   * Component y.
   *  @param[in, out] xi2 Local coordinate to map to collapsed coordinate.
   * Component z.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord_v(const T eta0, const T eta1,
                                           const T eta2, T *xi0, T *xi1,
                                           T *xi2) {
    *xi0 = eta0;
    *xi1 = eta1;
    *xi2 = eta2;
  }
};

struct Quadrilateral : BaseCoordinateMapping2D<Quadrilateral> {

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi0 Local coordinate to map to collapsed coordinate. Component
   * x.
   *  @param[in] xi1 Local coordinate to map to collapsed coordinate. Component
   * y.
   *  @param[in, out] eta0 Local collapsed coordinate. Component x.
   *  @param[in, out] eta1 Local collapsed coordinate. Component y.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed_v(const T xi0, const T xi1, T *eta0,
                                           T *eta1) {
    *eta0 = xi0;
    *eta1 = xi1;
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta0 Local collapsed coordinate. Component x.
   *  @param[in] eta1 Local collapsed coordinate. Component y.
   *  @param[in, out] xi0 Local coordinate to map to collapsed coordinate.
   * Component x.
   *  @param[in, out] xi1 Local coordinate to map to collapsed coordinate.
   * Component y.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord_v(const T eta0, const T eta1, T *xi0,
                                           T *xi1) {
    *xi0 = eta0;
    *xi1 = eta1;
  }
};

struct Triangle : BaseCoordinateMapping2D<Triangle> {

  /**
   *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
   *
   *  @param[in] xi0 Local coordinate to map to collapsed coordinate. Component
   * x.
   *  @param[in] xi1 Local coordinate to map to collapsed coordinate. Component
   * y.
   *  @param[in, out] eta0 Local collapsed coordinate. Component x.
   *  @param[in, out] eta1 Local collapsed coordinate. Component y.
   */
  template <typename T>
  inline void loc_coord_to_loc_collapsed_v(const T xi0, const T xi1, T *eta0,
                                           T *eta1) {
    const NekDouble d1_original = 1.0 - xi1;
    const bool mask_small_cond =
        (sycl::fabs(d1_original) < NekConstants::kNekZeroTol);
    NekDouble d1 = d1_original;
    d1 = (mask_small_cond && (d1 >= 0.0))
             ? NekConstants::kNekZeroTol
             : ((mask_small_cond && (d1 < 0.0)) ? -NekConstants::kNekZeroTol
                                                : d1);
    *eta0 = 2. * (1. + xi0) / d1 - 1.0;
    *eta1 = xi1;
  }

  /**
   *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
   *
   *  @param[in] eta0 Local collapsed coordinate. Component x.
   *  @param[in] eta1 Local collapsed coordinate. Component y.
   *  @param[in, out] xi0 Local coordinate to map to collapsed coordinate.
   * Component x.
   *  @param[in, out] xi1 Local coordinate to map to collapsed coordinate.
   * Component y.
   */
  template <typename T>
  inline void loc_collapsed_to_loc_coord_v(const T eta0, const T eta1, T *xi0,
                                           T *xi1) {
    *xi0 = (1.0 + eta0) * (1.0 - eta1) * 0.5 - 1.0;
    *xi1 = eta1;
  }
};

/**
 *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
 *
 *  @param[in] shape_type Integer denoting shape type found by cast of Nektar++
 *  shape type enum to int.
 *  @param[in] eta0 Local collapsed coordinate to map to local coordinate.
 *  @param[in] eta1 Local collapsed coordinate to map to local coordinate.
 *  @param[in] eta2 Local collapsed coordinate to map to local coordinate.
 *  @param[in, out] xi0 Local coordinate.
 *  @param[in, out] xi1 Local coordinate.
 *  @param[in, out] xi2 Local coordinate.
 */
template <typename T>
inline void loc_collapsed_to_loc_coord(const int shape_type, const T eta0,
                                       const T eta1, const T eta2, T *xi0,
                                       T *xi1, T *xi2) {

  /*
  Tet

    xi[1] = (1.0 + eta[1]) * (1.0 - eta[2]) * 0.5 - 1.0;
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

  const REAL a = 1.0 + eta0;
  const REAL b = 1.0 - eta2;
  const REAL c = (1.0 + eta1) * (b)*0.5 - 1.0;
  const REAL d = (a) * (b)*0.5 - 1.0;
  *xi1 = (shape_type == shape_type_tet)
             ? c
             : ((shape_type == shape_type_pyr) ? (1.0 + eta1) * (b)*0.5 - 1.0
                                               : eta1);
  const REAL tet_x = (1.0 + eta0) * (-(*xi1) - eta2) * 0.5 - 1.0;
  *xi0 = (shape_type == shape_type_tet)
             ? tet_x
             : ((shape_type == shape_type_hex) ? eta0 : d);
  *xi2 = eta2;
}

/**
 *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
 *
 *  @param[in] shape_type Integer denoting shape type found by cast of Nektar++
 *  shape type enum to int.
 *  @param[in] eta Local collapsed coordinate to map to local coordinate. Must
 * be a subscriptable type for indices 0,1,2.
 *  @param[in, out] xi Local coordinate. Must be a
 * subscriptable type for indices 0,1,2.
 */
template <typename T>
inline void loc_collapsed_to_loc_coord(const int shape_type, const T &eta,
                                       T &xi) {
  loc_collapsed_to_loc_coord(shape_type, eta[0], eta[1], eta[2], &xi[0], &xi[1],
                             &xi[2]);
}

/**
 *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
 *
 *  @param[in] shape_type Integer denoting shape type found by cast of Nektar++
 *  @param[in] xi0 Local coordinate to map to collapsed coordinate, x component.
 *  @param[in] xi1 Local coordinate to map to collapsed coordinate, y component.
 *  @param[in] xi2 Local coordinate to map to collapsed coordinate, z component.
 *  @param[in, out] eta0 Local collapsed coordinate, x component.
 *  @param[in, out] eta1 Local collapsed coordinate, y component.
 *  @param[in, out] eta2 Local collapsed coordinate, z component.
 */
template <typename T>
inline void loc_coord_to_loc_collapsed_3d(const int shape_type, const T xi0,
                                          const T xi1, const T xi2, T *eta0,
                                          T *eta1, T *eta2) {

  constexpr int shape_type_tet = shape_type_to_int(LibUtilities::eTetrahedron);
  constexpr int shape_type_pyr = shape_type_to_int(LibUtilities::ePyramid);
  constexpr int shape_type_hex = shape_type_to_int(LibUtilities::eHexahedron);

  NekDouble d2 = 1.0 - xi2;
  if (sycl::fabs(d2) < NekConstants::kNekZeroTol) {
    if (d2 >= 0.) {
      d2 = NekConstants::kNekZeroTol;
    } else {
      d2 = -NekConstants::kNekZeroTol;
    }
  }
  NekDouble d12 = -xi1 - xi2;
  if (sycl::fabs(d12) < NekConstants::kNekZeroTol) {
    if (d12 >= 0.) {
      d12 = NekConstants::kNekZeroTol;
    } else {
      d12 = -NekConstants::kNekZeroTol;
    }
  }

  const REAL id2x2 = 2.0 / d2;
  const REAL a = 1.0 + xi0;
  const REAL b = (1.0 + xi1) * id2x2 - 1.0;
  const REAL c = a * id2x2 - 1.0;
  const REAL d = 2.0 * a / d12 - 1.0;

  *eta0 = (shape_type == shape_type_tet)   ? d
          : (shape_type == shape_type_hex) ? xi0
                                           : c;

  *eta1 = ((shape_type == shape_type_tet) || (shape_type == shape_type_pyr))
              ? b
              : xi1;
  *eta2 = xi2;
}

/**
 *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
 *
 *  @param[in] shape_type Integer denoting shape type found by cast of Nektar++
 *  @param[in] xi Local coordinate to map to collapsed coordinate. Must be a
 * subscriptable type for indices 0,1,2.
 *  @param[in, out] eta Local collapsed coordinate. Must be a subscriptable
 * type for indices 0,1,2.
 */
template <typename T>
inline void loc_coord_to_loc_collapsed_3d(const int shape_type, const T &xi,
                                          T &eta) {
  loc_coord_to_loc_collapsed_3d(shape_type, xi[0], xi[1], xi[2], &eta[0],
                                &eta[1], &eta[2]);
}

/**
 *  Map the local coordinate (xi) to the local collapsed coordinate (eta).
 *
 *  @param[in] shape_type Integer denoting shape type found by cast of Nektar++
 *  @param[in] xi0 Local coordinate to map to collapsed coordinate. Coordinate
 * x.
 *  @param[in] xi1 Local coordinate to map to collapsed coordinate. Coordinate
 * y.
 *  @param[in, out] eta0 Local collapsed coordinate. Coordinate x.
 *  @param[in, out] eta1 Local collapsed coordinate. Coordinate y.
 */
template <typename T>
inline void loc_coord_to_loc_collapsed_2d(const int shape_type, const T xi0,
                                          const T xi1, T *eta0, T *eta1) {

  constexpr int shape_type_tri = shape_type_to_int(LibUtilities::eTriangle);
  const NekDouble d1_original = 1.0 - xi1;
  const bool mask_small_cond =
      (sycl::fabs(d1_original) < NekConstants::kNekZeroTol);
  NekDouble d1 = d1_original;
  d1 =
      (mask_small_cond && (d1 >= 0.0))
          ? NekConstants::kNekZeroTol
          : ((mask_small_cond && (d1 < 0.0)) ? -NekConstants::kNekZeroTol : d1);
  *eta0 = (shape_type_tri == shape_type) ? 2. * (1. + xi0) / d1 - 1.0 : xi0;
  *eta1 = xi1;
}

/**
 *  Map the local collapsed coordinate (eta) to the local coordinate (xi).
 *
 *  @param[in] shape_type Integer denoting shape type found by cast of Nektar++
 *  shape type enum to int.
 *  @param[in] eta0 Local collapsed coordinate to map to local coordinate.
 *  @param[in] eta1 Local collapsed coordinate to map to local coordinate.
 *  @param[in, out] xi0 Local coordinate.
 *  @param[in, out] xi1 Local coordinate.
 */
template <typename T>
inline void loc_collapsed_to_loc_coord_2d(const int shape_type, const T eta0,
                                          const T eta1, T *xi0, T *xi1) {
  constexpr int shape_type_tri = shape_type_to_int(LibUtilities::eTriangle);
  *xi0 = (shape_type == shape_type_tri)
             ? (1.0 + eta0) * (1.0 - eta1) * 0.5 - 1.0
             : eta0;
  *xi1 = eta1;
}

/**
 * Clamp a single coordinate to the interval [-1-tol, 1+tol] for a given
 * tolerance.
 *
 * @param[in, out] coord Value to clamp into interval.
 * @param[in] tol Optional input tolerance, default machine precision for
 * templated type.
 * @returns True if value was clamped otherwise false.
 */
template <typename T>
inline bool clamp_loc_coord(T *coord,
                            const T tol = std::numeric_limits<T>::epsilon()) {

  bool clamp = false;
  if (!sycl::isfinite(*coord)) {
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
 * Clamp a two coordinates to the interval [-1-tol, 1+tol] for a given
 * tolerance.
 *
 * @param[in, out] coord0 Value to clamp into interval.
 * @param[in, out] coord1 Value to clamp into interval.
 * @param[in] tol Optional input tolerance, default machine precision for
 * templated type.
 * @returns True if any value was clamped otherwise false.
 */
template <typename T>
inline bool clamp_loc_coords(T *coord0, T *coord1,
                             const T tol = std::numeric_limits<T>::epsilon()) {
  return clamp_loc_coord(coord0, tol) || clamp_loc_coord(coord1, tol);
}

/**
 * Clamp a three coordinates to the interval [-1-tol, 1+tol] for a given
 * tolerance.
 *
 * @param[in, out] coord0 Value to clamp into interval.
 * @param[in, out] coord1 Value to clamp into interval.
 * @param[in, out] coord2 Value to clamp into interval.
 * @param[in] tol Optional input tolerance, default machine precision for
 * templated type.
 * @returns True if any value was clamped otherwise false.
 */
template <typename T>
inline bool clamp_loc_coords(T *coord0, T *coord1, T *coord2,
                             const T tol = std::numeric_limits<T>::epsilon()) {
  return clamp_loc_coord(coord0, tol) || clamp_loc_coords(coord1, coord2, tol);
}

} // namespace GeometryInterface
} // namespace NESO

#endif
