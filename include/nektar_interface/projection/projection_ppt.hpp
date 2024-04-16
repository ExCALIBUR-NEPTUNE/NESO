#pragma once
#include "..basis/basis.hpp"
#include "constants.hpp"
#include <CL/sycl.hpp>

#if defined(__clang__)
#define NESO_UNROLL_PRAGMA _Pragma("clang unroll loop(full)")
#elif defined(__GNUC__)
#define NESO_UNROLL_PRAGMA _Pragma("GCC unroll 20")
#endif

namespace NESO {
namespace MetaJacobi {

template <int nmode, typename T, int alpha, int beta>
sycl::event project_thread_per_cell_quad(T *dofs, int *dof_offsets, int ncell,
                                         [[maybe_unused]] int max_par_in_cell,
                                         int *cell_ids, int *par_per_cell,
                                         T ***positions, T ***input,
                                         int componant, sycl::queue &queue) {

  sycl::range<1> range{static_cast<std::size_t>(ncell)};
  return queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<>(range, [=](sycl::id<1> idx) {
      auto cellx = cell_ids[idx];
      auto npart = par_per_cell[cellx];
      if (npart == 0)
        return;
      auto cell_dof = &dofs[dof_offsets[cellx]];
      // printf("PPT  -- CELL %d, PART %d\n",cellx,npart);
      for (int part = 0; part < npart; ++part) {
        auto eta0 = positions[cellx][0][part];
        auto eta1 = positions[cellx][1][part];
        auto qoi = input[cellx][componant][part];
        T local0[nmode];
        T local1[nmode];
        Basis::eModA<T, nmode, Constants::cpu_stride, alpha, beta>(eta0,
                                                                   local0);
        Basis::eModA<T, nmode, Constants::cpu_stride, alpha, beta>(eta1,
                                                                   local1);
        NESO_UNROLL_PRAGMA
        for (int i = 0; i < nmode; ++i) {
          double temp = local1[i] * qoi;
          NESO_UNROLL_PRAGMA
          for (int j = 0; j < nmode; ++j) {
            cell_dof[j + nmode * i] += temp * local0[j];
          }
        }
      }
      // for (int i = 0; i < nmode; ++i) {
      //     for (int j = 0; j < nmode; ++j) {
      //       printf("ppt %d (%d,%d), %f %d %lx\n",cellx, i,j, cell_dof[j+nmode
      //       *i], j + nmode * i, cell_dof);
      //     }
      //   }
    });
  });
}

} // namespace MetaJacobi
} // namespace NESO
