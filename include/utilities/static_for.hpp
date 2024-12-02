#ifndef _NESO_UTILITIES_BASIS_STATIC_FOR_HPP
#define _NESO_UTILITIES_BASIS_STATIC_FOR_HPP
#include <cstdint>
#include <utility>
namespace NESO::Utilities {
// Static for loop found on stackoverflow
// https://stackoverflow.com/questions/37602057/why-isnt-a-for-loop-a-compile-time-expression
// Usage eg:
// static_for<N>([&] (auto idx) { /* do some code with idx.value */;});
template <class F, int32_t... I>
inline void __attribute__((always_inline))
static_for(F func, std::integer_sequence<int32_t, I...>) {
  (func(std::integral_constant<int32_t, I>{}), ...);
}

template <int32_t N, typename F>
inline void __attribute__((always_inline)) static_for(F func) {
  if constexpr (N >= 0) {
    static_for(func, std::make_integer_sequence<int32_t, N>());
  }
}
} // namespace NESO::Utilities
#endif
