#pragma once
#include <cstdint>
#include <utility>
namespace NESO::Basis::Private {
// Static for loop found on stackoverflow
// https://stackoverflow.com/questions/37602057/why-isnt-a-for-loop-a-compile-time-expression
// Usage eg:
// static_for<N>([&] (auto idx) { printf("%ld\n",idx.value);});
template <int64_t N>
struct Number {
    static const constexpr auto value = N;
};
#if !defined(EXCLUDE_SYCL_HEADERS) && defined(__INTEL_LLVM_COMPILER)
#include <CL/sycl.hpp>
#define NESO_SYCL_EXTERN extern SYCL_EXTERNAL
#else
#define NESO_SYCL_EXTERN
#endif
template <class F, int64_t... Is>
NESO_SYCL_EXTERN inline void __attribute__((always_inline))
static_for(F func, std::integer_sequence<int64_t, Is...>)
{
    (func(Number<Is>{}), ...);
}

template <int64_t N, typename F>
NESO_SYCL_EXTERN inline void __attribute__((always_inline)) static_for(F func)
{
    if constexpr (N >= 0) {
        static_for(func, std::make_integer_sequence<int64_t, N>());
    }
}
}
