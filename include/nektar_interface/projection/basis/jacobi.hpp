#ifndef _NESO_NEKTAR_INTERFACE_PROJECTION_BASIS_JACOBI_HPP
#define _NESO_NEKTAR_INTERFACE_PROJECTION_BASIS_JACOBI_HPP
#include <cmath>
#include <cstdint>

namespace NESO::Basis::Private {
template <int64_t n> inline int64_t pochhammer(const int64_t m) {
  return (m + (n - 1)) * pochhammer<n - 1>(m);
}
template <> inline int64_t pochhammer<0>([[maybe_unused]] const int64_t m) {
  return 1;
}

template <typename T, int64_t p, int64_t alpha, int64_t beta>
inline auto __attribute__((always_inline)) jacobi([[maybe_unused]] T const z) {
  static_assert(p >= 0, "p must be >= 0");
  if constexpr (p == 0)
    return 1.0;
  else if constexpr (p == 1)
    return T(0.5 * (2.0 * T(alpha + 1) + T(alpha + beta + 2) * (z - 1.0)));
  else if constexpr (p == 2)
    return T(0.125) *
           (T(4.0 * (alpha + 1.0) * (alpha + 2.0)) +
            T(4.0 * (alpha + beta + 3.0) * (alpha + 2.0)) * (z - (1.0)) +
            T((alpha + beta + 3.0)) * T((alpha + beta + 4.0)) * (z - (1.0)) *
                (z - (1.0)));
  else {
    auto n = p - 1;
    auto pn = jacobi<T, p - 1, alpha, beta>(z);
    auto pnm1 = jacobi<T, p - 2, alpha, beta>(z);
    auto coeff_pnp1 = T(2.0 * (n + 1.0) // 6
                        * (n + alpha + beta + 1.0) * (2.0 * n + alpha + beta));
    auto coeff_pn =
        T((2.0 * n + alpha + beta + 1.0) * (alpha * alpha - beta * beta)) +
        T(pochhammer<3>(2 * (p - 1) + alpha + beta)) * z;
    auto coeff_pnm1 =
        T(-2.0 * (n + alpha) * (n + beta) * (2.0 * n + alpha + beta + 2.0));
    return T((1.0 / coeff_pnp1) * (coeff_pn * pn + coeff_pnm1 * pnm1));
  }
}
} // namespace NESO::Basis::Private
#endif
