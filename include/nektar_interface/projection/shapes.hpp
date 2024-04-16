#pragma once
#include "quad.hpp"
#include "tri.hpp"
#if 0
#include <CL/sycl.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <type_traits>

namespace NESO::Project {
namespace Private::Shape {
// Ugly but can't static cast a sycl vector so covering
// the cases I need so the collapse code works on vectors and doubles
template <typename T, typename U, typename Q> inline auto convert(U &in) {
  if constexpr (std::is_same<U, cl::sycl::vec<Q, 1>>::value ||
                std::is_same<U, cl::sycl::vec<Q, 2>>::value ||
                std::is_same<U, cl::sycl::vec<Q, 4>>::value ||
                std::is_same<U, cl::sycl::vec<Q, 8>>::value) {
    return in.template convert<T>();
  } else {
    return static_cast<T>(in);
  }
}

template <typename T> inline auto abs(T a) { return cl::sycl::abs(a); }

template <> inline auto abs<bool>(bool a) { return static_cast<long>(a); }

constexpr double Tol = 1.0E-12;

template <typename T, typename U>
inline void loc_coord_to_loc_collapsed(T const xi0, T const xi1, T &eta0,
                                       T &eta1) {
  auto d1_origional = T(1.0) - xi1;
  // This is prob not the "right" way to do this
  auto zeroTol = T(Tol);
  // Have to abs, for some reason sycl vec leaves -1 not 1 in the mask
  // ... still want it to work for T==double
  auto mask_small = Private::Shape::abs(cl::sycl::fabs(d1_origional) < zeroTol);
  zeroTol = cl::sycl::copysign(zeroTol, d1_origional);
  auto fmask =
      Private::Shape::convert<U, decltype(mask_small), long>(mask_small);
  auto d1 = (T(1.0) - fmask) * d1_origional + fmask * zeroTol;
  eta0 = T(2.0) * (T(1.0) + xi0) / d1 - T(1.0);
  eta1 = xi1;
}
} // namespace Private::Shape

struct eTri {
  static constexpr Nektar::LibUtilities::ShapeType shape_type 
      = Nektar::LibUtilities::eTriangle;
  static constexpr int dim = 2;
  template <typename T>
  inline static void loc_coord_to_loc_collapsed(T const xi0, T const xi1,
                                                T &eta0, T &eta1) {
    Private::Shape::loc_coord_to_loc_collapsed<T, double>(xi0, xi1, eta0, eta1);
  }
};

struct eQuad {
  static constexpr Nektar::LibUtilities::ShapeType 
      shape_type = Nektar::LibUtilities::eQuadrilateral;
  static constexpr int dim = 2;
  template <typename T> 
  static void loc_coord_to_loc_collapsed(T &xi0, T &xi1){};
};
} // namespace NESO::Project
#endif
