#ifndef _NESO_CONSTANTS_HPP
#define _NESO_CONSTANTS_HPP

namespace NESO::Constants {
// Needs to be at least 9 for the unit tests
#ifndef NESO_MAX_NUMMODES
#define NESO_MAX_NUMMODES 9
#endif
#ifndef NESO_MIN_NUMMODES
#define NESO_MIN_NUMMODES 1
#endif
#ifndef NESO_PREF_LOCAL_SIZE
#define NESO_PREF_LOCAL_SIZE 128
#endif
#ifndef NESO_VECTOR_WIDTH
#define NESO_VECTOR_WIDTH 4
#endif
#ifndef NESO_GPU_WARP_SIZE
#define NESO_GPU_WARP_SIZE 32
#endif

constexpr int max_nummodes = NESO_MAX_NUMMODES;
constexpr int min_nummodes = NESO_MIN_NUMMODES;
// sycl terminology - block size in cuda
constexpr int preferred_local_size = NESO_PREF_LOCAL_SIZE;
constexpr int vector_width = NESO_VECTOR_WIDTH;
constexpr int gpu_warp_size = NESO_GPU_WARP_SIZE;
// Always the same no point being configurable
constexpr int beta = 1;
constexpr int alpha = 1;
constexpr int cpu_stride = 1;
} // namespace NESO::Constants
#endif
