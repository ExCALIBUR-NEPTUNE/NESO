#pragma once
#include <cmath>
#include <cstdint>

#include "power.hpp"
#include "static_for.hpp"
#include "jacobi.hpp"

namespace NESO::Basis {
template <typename T, int64_t N, int64_t alpha, int64_t beta>
inline double __attribute__((always_inline)) eModA(T z) {
  if constexpr (N == 0)
    return 0.5 * (1.0 - z);
  else if constexpr (N == 1)
    return 0.5 * (1.0 + z);
  else
    return 0.25 * (1.0 - z) * (1.0 + z) *
           Private::jacobi<T, N - 2, alpha, beta>(z);
}

template <typename T, int64_t N, int64_t stride, int64_t alpha, int64_t beta>
inline auto __attribute__((always_inline)) eModA(T z, T *output) {
  const T b0 = 0.5 * (1.0 - z);
  const T b1 = 0.5 * (1.0 + z);
  output[0] = b0;
  output[stride] = b1;
  Private::static_for<N - 2>([&](auto idx) { 
    assert((2 + idx.value) < N);
    output[(2 + idx.value) * stride] =
        b0 * b1 * Private::jacobi<T, idx.value, alpha, beta>(z);
  });
}

template <typename T, int64_t N, int64_t stride, int64_t alpha, int64_t beta>
inline auto __attribute__((always_inline)) eModB(T z, T *output) {
  T b0 = 1.0;
  const T b1 = 0.5 * (1.0 + z);
  Private::static_for<N>([&](auto p) {
    Private::static_for<N - p.value>([&](auto q) {
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
    b0 *= 0.5 * (1.0 - z);
  });
}

template <typename T, int64_t p, int64_t q, int64_t alpha, int64_t beta>
inline auto __attribute__((always_inline)) eModB(T z) {
  if constexpr (p == 0)
    return eModA<T, q, alpha, beta>(z);
  T b0 = Private::power<T, p>::_(0.5 * (1.0 - z));
  if constexpr (q == 0)
    return b0;
  T const b1 = 0.5 * (1.0 + z);
  return b1 * b0 * Private::jacobi<T, q - 1, 2 * p - 1, 1>(z);
}

template <typename T, int64_t N, int64_t stride, int64_t alpha, int64_t beta>
inline auto __attribute__((always_inline)) eModC(T z, T *output) {
  Private::static_for<N>([&](auto p) {
    Private::static_for<N - p.value>([&](auto q) {
      Private::static_for<N - p.value - q.value>([&](auto r) {
        *output = eModB<T, p.value + q.value, r.value, alpha, beta>(z);
        output += stride;
      });
    });
  });
}
#if 0
template <typename T, int64_t N, int64_t stride, int64_t alpha, int64_t beta>
inline auto __attribute__((always_inline)) eModPyrC(T z, T *output)
{
    const T b0 = 0.5 * (1.0 - z);
    const T b1 = 0.5 * (1.0 + z);
   Private::static_for<N>([&](auto p) {
       Private::static_for<N>([&](auto q) {
           Private::static_for<N - Private::max<p.value, q.value>()>([&](auto r) {
                if constexpr (p.value == 0)
                    *output = eModB<T, q.value, r.value, alpha, beta>(z);
                else if constexpr (p.value == 1) {
                    auto m = q.value == 0 ? 1 : q.value;
                    *output = eModB<T, m, r.value, alpha, beta>(z);
                } else {
                    T b0pow =Private::power<T, p.value + q.value - 2>::_(b0);
                    if constexpr (q.value < 2)
                        *output = b0pow;
                    else
                        *output = b1 * b0pow *
                                  Private::jacobi<T, r.value - 1,
                                         2 * p.value + 2 * q.value - 3, 1>(z);
                }
                output += stride;
            });
        });
    });
}
#endif

} // namespace NESO::Basis
