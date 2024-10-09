#pragma once
#include "constants.hpp"
#include "unroll.hpp"
#include <CL/sycl.hpp>
namespace NESO::Project::Util::Private {

template <typename T> inline auto NESO_ALWAYS_INLINE to_mask_vec(T a) {
  return cl::sycl::abs(a);
}

template <> inline auto NESO_ALWAYS_INLINE to_mask_vec<bool>(bool a) {
  return static_cast<long>(a);
}
// cast from type U to type T
// special case if U is a sycl vector to call convert function
// ugly but can't think of anything better
template <typename T, typename U, typename Q>
inline auto NESO_ALWAYS_INLINE convert(U &in) {
  if constexpr (std::is_same<U, cl::sycl::vec<Q, 1>>::value ||
                std::is_same<U, cl::sycl::vec<Q, 2>>::value ||
                std::is_same<U, cl::sycl::vec<Q, 4>>::value ||
                std::is_same<U, cl::sycl::vec<Q, 8>>::value) {
    return in.template convert<T>();
  } else {
    return static_cast<T>(in);
  }
}
// Collapse one coord to point (I think)
// A bit complicated to support support sycl vectors
// also probably pre-optimised without checking if
// branches where especially bad
template <typename T>
inline auto NESO_ALWAYS_INLINE collapse_coords(T const x, T const d) {
  auto dprime = T(1.0) - d;
  auto zeroTol = T(Constants::Tolerance);
  auto mask_small =
      Util::Private::to_mask_vec(cl::sycl::fabs(dprime) < zeroTol);
  zeroTol = cl::sycl::copysign(zeroTol, dprime);
  auto fmask =
      Util::Private::convert<T, decltype(mask_small), long>(mask_small);
  dprime = (T(1.0) - fmask) * dprime + fmask * zeroTol;
  return T(2.0) * (T(1.0) + x) / dprime - T(1.0);
}
} // namespace NESO::Project::Util::Private
