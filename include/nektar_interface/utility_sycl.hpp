#ifndef __UTILITY_SYCL_H_
#define __UTILITY_SYCL_H_

#include <map>
#include <memory>
#include <neso_particles.hpp>
#include <string>
#include <type_traits>
#include <vector>

using namespace NESO::Particles;

namespace NESO {

/**
 * For an iteration set size N and local workgroup size L determine M such
 * that M>=N and M % L == 0.
 *
 * @param N Actual iteration set size.
 * @param L Local workgroup size.
 * @returns M such that M >= N and M % L == 0.
 */
inline std::size_t get_global_size(const std::size_t N, const std::size_t L) {
  // TODO Upstream to NESO-Particles
  const auto div_mod =
      std::div(static_cast<long long>(N), static_cast<long long>(L));
  const std::size_t outer_size =
      static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1));
  return outer_size * L;
}

/**
 * For a given local workgroup size determine a global size sufficient for all
 * cells in a ParticleDat.
 *
 * @param particle_dat ParticleDat to use as iteration set.
 * @param local_size Local workgroup size.
 */
template <typename T>
inline std::size_t
get_particle_loop_global_size(ParticleDatSharedPtr<T> particle_dat,
                              const std::size_t local_size) {
  // TODO Upstream to NESO-Particles
  const std::size_t N =
      static_cast<std::size_t>(particle_dat->cell_dat.get_nrow_max());
  return get_global_size(N, local_size);
}

#ifndef NESO_VECTOR_LENGTH
#ifdef NESO_PARTICLES_VECTOR_LENGTH
#define NESO_VECTOR_LENGTH NESO_PARTICLES_VECTOR_LENGTH
#else
#define NESO_VECTOR_LENGTH 4 // TODO MAKE THIS CONFIGURATION CMAKE TIME
#endif
#endif

#ifndef NESO_VECTOR_BLOCK_FACTOR
#define NESO_VECTOR_BLOCK_FACTOR 8 // TODO MAKE THIS CONFIGURATION CMAKE TIME
#endif

#define NESO_VECTOR_BLOCK_SIZE (NESO_VECTOR_LENGTH * NESO_VECTOR_BLOCK_FACTOR)

/**
 *  For an input integer L >= 0 return smallest M such that M >= L and M %
 * NESO_VECTOR_LENGTH == 0.
 *
 *  @param L Input length.
 *  @returns Output length M.
 */
template <typename T> inline T pad_to_vector_length(const T L) {
  const T rem_L = L % NESO_VECTOR_LENGTH;
  if ((rem_L) == 0) {
    return L;
  } else {
    return L + (NESO_VECTOR_LENGTH - rem_L);
  }
}

} // namespace NESO

#endif
