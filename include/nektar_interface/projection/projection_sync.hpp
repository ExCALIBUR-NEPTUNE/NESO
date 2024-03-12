#pragma once
#include "constants.hpp"
#include "..basis/basis.hpp"
#include <CL/sycl.hpp>

#include <cmath>
#include <limits>
namespace NESO {
namespace MetaJacobi {


/*
 * Testing suggests this is the best kernel (I can think of) for GPUs (or A100 at least)
 *  - The calculated basis is stored so that all the 0-modes for all the
 *  particles in the local group are first, then 1-modes etc.
 *  i.e.
 *    local_mem = |mode_0_par_0, mode_0_par_1, mode_0_par2,.....|
 * this means each thread in the local group writes to adjacent addresses and avoids bank conflicts
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

template <int nmode, typename T, int alpha, int beta>
sycl::event 
project_synchronised_quad(T *dofs, int *dof_offsets, int ncell,
        int max_par_in_cell, int *cell_ids,
        int *par_per_cell, T ***positions,
        T ***input, int componant,
        sycl::queue &queue) {
  // Will just round up max_par_in_cell to nearest multiple of local_size
  std::size_t outer_size = Constants::local_size *
                           (((max_par_in_cell - 1) / Constants::local_size) + 1);
  // auto res = std::div(max_par_in_cell, Constants::local_size);
  // auto outer_size = res.quot + (res.rem != 0);
  sycl::range out_range = sycl::range<2>(ncell, outer_size);
  sycl::range in_range = sycl::range<2>(1, Constants::local_size);

  return queue.submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        local_mem(sycl::range<1>((Constants::gpu_stride) * (2 * nmode)), cgh);
        //local_mem(sycl::range<1>((15) * (2 * nmode)), cgh);
    cgh.parallel_for<>(
        sycl::nd_range<2>(out_range, in_range), [=](sycl::nd_item<2> idx) {
          // Helper function to compute eModified_A
          const int cellx = cell_ids[idx.get_global_id(0)];
          long npart = par_per_cell[cellx];
          if (npart == 0)
            return;
          int idx_local = idx.get_local_id(1);
          const int layerx = idx.get_global_id(1);
#ifndef NDEBUG
          if (idx_local == 0) {
          for (int i = 0; i < 129*8; ++i)
          local_mem[i] = std::numeric_limits<double>::signaling_NaN();
          }
#endif
          idx.barrier();
          if (layerx < npart) {
            //printf("sync -- CELL %d, PART %ld, LX %d, LOCAL %d\n",cellx,npart, layerx, idx_local);
          //printf("%ld (%ld, %ld) - (%ld, %ld) \n", 
          //        npart,
          //        idx.get_global_id(0),
          //        idx.get_global_id(1),
          //        idx.get_local_id(0),
          //        idx.get_local_id(1));
            const double eta0 = positions[cellx][0][layerx];
            const double eta1 = positions[cellx][1][layerx];
            const double value = input[cellx][componant][layerx];

            auto local0 = &local_mem[idx_local];
            auto local1 = local0 + Constants::gpu_stride * nmode;

            eModA<T, nmode, Constants::gpu_stride, alpha, beta>(eta0, local0);
            eModA<T, nmode, Constants::gpu_stride, alpha, beta>(eta1, local1);
            for (int qx = 0; qx < nmode; ++qx) {
              local1[qx * Constants::gpu_stride] *= value;
            }
          }
                 
          idx.barrier(cl::sycl::access::fence_space::local_space);
          //idx.barrier();
          auto ndof = nmode * nmode;
          long nthd = idx.get_local_range(1);
          const auto cell_dof = &dofs[dof_offsets[cellx]];
          auto count =
              std::min(nthd, std::max(long{0}, npart - layerx + idx_local));
          auto mode0 = &local_mem[0];
          auto mode1 = mode0 + nmode * Constants::gpu_stride;

          while (idx_local < ndof) {
            int i = idx_local / nmode;
            int j = idx_local % nmode;
            double temp = 0.0;
            for (int k = 0; k < count; ++k) {
              temp += mode1[i * Constants::gpu_stride + k] *
                      mode0[j * Constants::gpu_stride + k];
            }
            //printf("sync %d (%d,%d), %f %d %lx\n",cellx, i,j, temp, idx_local, cell_dof);
            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>
                coeff_atomic_ref(cell_dof[idx_local]);
            coeff_atomic_ref.fetch_add(temp);
            idx_local += nthd;
          }
        });
  });
}
} // namespace MetaJacobi
} // namespace NESO
