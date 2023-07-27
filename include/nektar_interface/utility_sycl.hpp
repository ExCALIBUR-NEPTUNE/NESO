#ifndef __UTILITY_SYCL_H_
#define __UTILITY_SYCL_H_

#include <map>
#include <memory>
#include <neso_particles.hpp>
#include <string>
#include <vector>

using namespace NESO::Particles;

namespace NESO {

/**
 *  Get a number of local work items that should not exceed the maximum
 *  available local memory on the device.
 *
 *  @param sycl_target SYCLTarget instance local memory will be allocated on.
 *  @param num_bytes Number of bytes requested per work item.
 *  @param default_num Default number of work items.
 *  @returns Number of work items.
 */
inline std::size_t get_num_local_work_items(SYCLTargetSharedPtr sycl_target,
                                            const std::size_t num_bytes,
                                            const std::size_t default_num) {
  sycl::device device = sycl_target->device;
  auto local_mem_exists =
      device.is_host() ||
      (device.get_info<sycl::info::device::local_mem_type>() !=
       sycl::info::local_mem_type::none);
  auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();

  const std::size_t max_num_workitems = local_mem_size / num_bytes;
  // find the max power of two that does not exceed the number of work items.
  const std::size_t two_power = log2(max_num_workitems);
  const std::size_t max_base_two_num_workitems = std::pow(2, two_power);

  const std::size_t deduced_num_work_items =
      std::min(default_num, max_base_two_num_workitems);
  NESOASSERT((deduced_num_work_items > 0),
             "Deduced number of work items is not strictly positive.");

  const std::size_t local_mem_bytes = deduced_num_work_items * num_bytes;
  if ((!local_mem_exists) || (local_mem_size < local_mem_bytes)) {
    NESOASSERT(false, "Not enough local memory");
  }
  return deduced_num_work_items;
}

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
#define NESO_VECTOR_LENGTH 1
#endif
#endif

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
