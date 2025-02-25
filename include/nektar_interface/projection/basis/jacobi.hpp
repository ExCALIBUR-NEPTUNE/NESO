#ifndef _NESO_NEKTAR_INTERFACE_PROJECTION_BASIS_JACOBI_HPP
#define _NESO_NEKTAR_INTERFACE_PROJECTION_BASIS_JACOBI_HPP
#include <cmath>
#include <cstdint>
#include <utilities/unroll.hpp>

namespace NESO::Basis::Private {
template <int64_t n> inline int64_t NESO_ALWAYS_INLINE  pochhammer(const int64_t m) {
  return (m + (n - 1)) * pochhammer<n - 1>(m);
}
template <> inline int64_t NESO_ALWAYS_INLINE pochhammer<0>([[maybe_unused]] const int64_t m) {
  return 1;
}

template <typename T, int64_t p, int64_t alpha, int64_t beta>
inline auto NESO_ALWAYS_INLINE jacobi([[maybe_unused]] T const z) {
  static_assert(p >= 0, "p must be >= 0");
  if constexpr (p == 0)
    return T(1.0);
  else if constexpr (p == 1)
    return T(T(0.5) *
             (T(2.0) * T(alpha + 1) + T(alpha + beta + 2) * (z - T(1.0))));
  else if constexpr (p == 2)
    return T(T(0.125)) *
           (T(T(4.0) * (alpha + T(1.0)) * (alpha + T(2.0))) +
            T(T(4.0) * (alpha + beta + T(3.0)) * (alpha + T(2.0))) *
                (z - (T(1.0))) +
            T((alpha + beta + T(3.0))) * T((alpha + beta + T(4.0))) *
                (z - (T(1.0))) * (z - (T(1.0))));
  else {
    auto n = p - 1;
    auto pn = jacobi<T, p - 1, alpha, beta>(z);
    auto pnm1 = jacobi<T, p - 2, alpha, beta>(z);
    auto coeff_pnp1 =
        T(T(2.0) * (n + T(1.0))
          * (n + alpha + beta + T(1.0)) * (T(2.0) * n + alpha + beta));
    auto coeff_pn = T((T(2.0) * n + alpha + beta + T(1.0)) *
                      (alpha * alpha - beta * beta)) +
                    T(pochhammer<3>(2 * (p - 1) + alpha + beta)) * z;
    auto coeff_pnm1 = T(-T(2.0) * (n + alpha) * (n + beta) *
                        (T(2.0) * n + alpha + beta + T(2.0)));
    return T((T(1.0) / coeff_pnp1) * (coeff_pn * pn + coeff_pnm1 * pnm1));
  }
}
} // namespace NESO::Basis::Private
#endif
