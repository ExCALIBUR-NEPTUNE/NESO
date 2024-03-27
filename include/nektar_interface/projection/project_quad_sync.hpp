#pragma once
#include "basis/basis.hpp"
#include "constants.hpp"
#include "device_data.hpp"
#include "restrict.hpp"
#include "shapes.hpp"
#include "unroll.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include <cstdio>
namespace NESO::Project {

/* Testing suggests this is the best kernel (I can think of) for GPUs (or A100
 * at least)
 *  - The calculated basis is stored so that all the 0-modes for all the
 *  particles in the local group are first, then 1-modes etc.
 *  i.e.
 *    local_mem = |mode_0_par_0, mode_0_par_1, mode_0_par2,.....|
 * this means each thread in the local group writes to adjacent addresses and
 * avoids bank conflicts
 *
 * After the sync each thread is now responsible for a dof (not a particle)
 * In order to avoid conflicts when accumulating each particle's contribution
 * we have to pad the array by one or each thread in the warp would be accessing
 * an address on the same bank
 * i.e.
 * thread 0 would access
 *   mode1[0];
 * and thread 1 would access
 *   mode1[128 = 0 mod16] or mode1[129 = 1 mod16] with the padding
 */
namespace Private::GPU {
template <int nmode, typename T, int alpha, int beta>
void NESO_ALWAYS_INLINE fill_local_mem_quad(T eta0, T eta1, T qoi,
                                            T *NESO_RESTRICT local0,
                                            T *NESO_RESTRICT local1) {
  Basis::eModA<T, nmode, Constants::gpu_stride, alpha, beta>(eta0, local0);
  Basis::eModA<T, nmode, Constants::gpu_stride, alpha, beta>(eta1, local1);
  for (int qx = 0; qx < nmode; ++qx) {
    local1[qx * Constants::gpu_stride] *= qoi;
  }
}
template <int nmode, typename T>
T NESO_ALWAYS_INLINE reduce_dof_quad(int idx_local, int count,
                                     T *NESO_RESTRICT mode0,
                                     T *NESO_RESTRICT mode1) {
  int i = idx_local / nmode;
  int j = idx_local % nmode;
  double dof = 0.0;
  for (int k = 0; k < count; ++k) {
    dof += mode1[i * Constants::gpu_stride + k] *
           mode0[j * Constants::gpu_stride + k];
  }
  return dof;
}
} // namespace Private::GPU

// round-up N to nearest multiple of SIZE
#define ROUND_UP_TO(SIZE, N) (SIZE) * ((((N)-1) / (SIZE)) + 1)

template <int nmode, typename T, int alpha, int beta>
sycl::event project_gpu(DeviceData<T, eQuad> &data, int componant,
                        sycl::queue &queue) {
  std::size_t outer_size = ROUND_UP_TO(Constants::local_size, data.nrow_max);

  sycl::nd_range<2> range(sycl::range<2>(data.ncells, outer_size),
                          sycl::range<2>(1, Constants::local_size));

  auto ev = queue.submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        local_mem(sycl::range<1>((Constants::gpu_stride) * (2 * nmode)), cgh);
    cgh.parallel_for<>(range, [=](sycl::nd_item<2> idx) {
      const int cellx = data.cell_ids[idx.get_global_id(0)];
      long npart = data.par_per_cell[cellx];
      if (npart == 0)
        return;
      int idx_local = idx.get_local_id(1);
      const int layerx = idx.get_global_id(1);
      if (layerx < npart) {
        Private::GPU::fill_local_mem_quad<nmode,T,alpha,beta>(
            data.positions[cellx][0][layerx], data.positions[cellx][1][layerx],
            data.input[cellx][componant][layerx], &local_mem[idx_local],
            (&local_mem[idx_local]) + Constants::gpu_stride * nmode);
      }
      idx.barrier(sycl::access::fence_space::local_space);
      auto ndof = nmode * nmode;
      long nthd = idx.get_local_range(1);
      const auto cell_dof = &data.dofs[data.dof_offsets[cellx]];
      auto count =
          std::min(nthd, std::max(long{0}, npart - layerx + idx_local));
      auto mode0 = &local_mem[0];
      auto mode1 = mode0 + nmode * Constants::gpu_stride;
      while (idx_local < ndof) {
        auto temp = Private::GPU::reduce_dof_quad<nmode, T>(idx_local, count,
                                                            mode0, mode1);
        sycl::atomic_ref<double, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
            coeff_atomic_ref(cell_dof[idx_local]);
        coeff_atomic_ref.fetch_add(temp);
        idx_local += nthd;
      }    
  });
});
return ev;
}

} // namespace NESO::Project
