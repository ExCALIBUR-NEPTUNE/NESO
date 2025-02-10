#ifndef _NESO_NEKTAR_INTERFACE_PROJECTION_BASIS_BASIS_HPP
#define _NESO_NEKTAR_INTERFACE_PROJECTION_BASIS_BASIS_HPP
#include <cmath>
#include <cstdint>

#include "../unroll.hpp"
#include "jacobi.hpp"
#include "power.hpp"
#include <utilities/static_for.hpp>
namespace NESO::Basis {
namespace Private {
template <int64_t P, int64_t Q> constexpr int64_t max() {
  return P > Q ? P : Q;
}
} // namespace Private
template <typename T, int32_t N, int32_t alpha, int32_t beta>
inline auto NESO_ALWAYS_INLINE eModA(T z) {
  if constexpr (N == 0)
    return T(0.5) * (T(1.0) - z);
  else if constexpr (N == 1)
    return T(0.5) * (T(1.0) + z);
  else
    return T(0.25) * (T(1.0) - z) * (T(1.0) + z) *
           Private::jacobi<T, N - 2, alpha, beta>(z);
}

template <typename T, int32_t N, int32_t alpha, int32_t beta>
inline auto NESO_ALWAYS_INLINE eModA(T z, T *output, int32_t stride) {
  const T b0 = T(0.5) * (T(1.0) - z);
  const T b1 = T(0.5) * (T(1.0) + z);
  output[0] = b0;
  output[stride] = b1;
  Utilities::static_for<N - 2>([&](auto idx) {
    assert((2 + idx.value) < N);
    output[(2 + idx.value) * stride] =
        b0 * b1 * Private::jacobi<T, idx.value, alpha, beta>(z);
  });
}

template <int32_t N> constexpr inline auto NESO_ALWAYS_INLINE eModA_len() {
  return N;
}

template <typename T, int32_t N, int32_t alpha, int32_t beta>
inline auto NESO_ALWAYS_INLINE eModB(T z, T *output, int32_t stride) {
  T b0 = T(1.0);
  const T b1 = T(0.5) * (T(1.0) + z);
  Utilities::static_for<N>([&](auto p) {
    Utilities::static_for<N - p.value>([&](auto q) {
      if constexpr (p.value == 0) {
        *output = eModA<T, q.value, alpha, beta>(z);
        output += stride;
      } else if constexpr (q.value == 0) {
        *output = b0;
        output += stride;
      } else {
        *output =
            b0 * b1 * Private::jacobi<T, q.value - 1, 2 * p.value - 1, 1>(z);
        output += stride;
      }
    });
    b0 *= T(0.5) * (T(1.0) - z);
  });
}

template <typename T, int32_t p, int32_t q, int32_t alpha, int32_t beta>
inline auto NESO_ALWAYS_INLINE eModB(T z) {
  if constexpr (p == 0) {
    return eModA<T, q, alpha, beta>(z);
  } else if constexpr (q == 0) {
    T b0 = Private::power<T, p>::_(T(0.5) * (T(1.0) - z));
    return b0;
  } else {
    T b0 = Private::power<T, p>::_(T(0.5) * (T(1.0) - z));
    T const b1 = T(0.5) * (T(1.0) + z);
    return b1 * b0 * Private::jacobi<T, q - 1, 2 * p - 1, 1>(z);
  }
}

template <int32_t N> constexpr inline auto NESO_ALWAYS_INLINE eModB_len() {
  return N * (N + 1) >> 1;
}

template <typename T, int32_t N, int32_t alpha, int32_t beta>
inline auto NESO_ALWAYS_INLINE eModC(T z, T *output, int32_t stride) {
  Utilities::static_for<N>([&](auto p) {
    Utilities::static_for<N - p.value>([&](auto q) {
      Utilities::static_for<N - p.value - q.value>([&](auto r) {
        *output = eModB<T, p.value + q.value, r.value, alpha, beta>(z);
        output += stride;
      });
    });
  });
}

// TODO double check this is correct sum of N triangle numbers
template <int32_t N> inline constexpr auto NESO_ALWAYS_INLINE eModC_len() {
  return N * (N + 1) * (N + 2) / 6;
}

template <typename T, int32_t N, int32_t alpha, int32_t beta>
inline auto NESO_ALWAYS_INLINE eModPyrC(T z, T *output, int32_t stride) {
  const T b0 = T(0.5) * (T(1.0) - z);
  const T b1 = T(0.5) * (T(1.0) + z);
  Utilities::static_for<N>([&](auto p) {
    Utilities::static_for<N>([&](auto q) {
      constexpr auto max = Private::max<p.value, q.value>();
      Utilities::static_for<N - max>([&](auto r) {
        if constexpr (p.value == 0)
          *output = eModB<T, q.value, r.value, alpha, beta>(z);
        else if constexpr (p.value == 1) {
          auto constexpr m = q.value == 0 ? 1 : q.value;
          *output = eModB<T, m, r.value, alpha, beta>(z);
        } else if constexpr (q.value < 2) {
          *output = eModB<T, p.value, r.value, alpha, beta>(z);
        } else if constexpr (r.value == 0) {
          *output = Private::power<T, p.value + q.value - 2>::_(b0);
        } else {
          *output =
              b1 * Private::power<T, p.value + q.value - 2>::_(b0) *
              Private::jacobi<T, r.value - 1, 2 * p.value + 2 * q.value - 3, 1>(
                  z);
        }
        output += stride;
      });
    });
  });
}

// Sum of squares
template <int32_t N> inline constexpr auto NESO_ALWAYS_INLINE eModPyrC_len() {
  return N * (N + 1) * (2 * N + 1) / 6;
}

} // namespace NESO::Basis
#endif
