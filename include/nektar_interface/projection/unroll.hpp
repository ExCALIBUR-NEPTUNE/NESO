#ifndef _NESO_NEKTAR_INTERFACE_PROJECTION_UNROLL_HPP
#define _NESO_NEKTAR_INTERFACE_PROJECTION_UNROLL_HPP
#if defined(__clang__)
#define NESO_UNROLL_LOOP _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define NESO_UNROLL_LOOP _Pragma("GCC unroll 20")
#elif
#define NESO_UNROLL_LOOP
#endif

#if defined(__clang__) || defined(__GNUC__)
#define NESO_ALWAYS_INLINE __attribute__((always_inline))
#else
#define NESO_ALWAYS_INLINE
#endif
#endif
