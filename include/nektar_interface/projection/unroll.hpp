#pragma once
#if defined(__clang__)
#define NESO_UNROLL_LOOP _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define NESO_UNROLL_LOOP _Pragma("GCC unroll 20")
#endif

#if defined(__clang__) || defined(__GNUC__)
#define NESO_ALWAYS_INLINE __attribute__((always_inline))
#endif
