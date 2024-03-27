#pragma once
#include "../basis/basis.hpp"
#include "constants.hpp"
#include "device_data.hpp"
#include "shapes_old.hpp"
#include "unroll.hpp"
#include <CL/sycl.hpp>

namespace sycl=cl::sycl;

namespace NESO::Project {

namespace Private::CPU {
template <int nmode, typename T, int alpha, int beta>
inline NESO_ALWAYS_INLINE void quad_ppt(const double eta0, const double eta1,
                                        const double qoi, double *dofs) {
  T local0[nmode];
  T local1[nmode];
  Basis::eModA<T, nmode, Constants::cpu_stride, alpha, beta>(eta0, local0);
  Basis::eModA<T, nmode, Constants::cpu_stride, alpha, beta>(eta1, local1);
  NESO_UNROLL_LOOP
  for (int i = 0; i < nmode; ++i) {
    double temp = local1[i] * qoi;
    NESO_UNROLL_LOOP
    for (int j = 0; j < nmode; ++j) {
      dofs[j + nmode * i] += temp * local0[j];
    }
  }
} 
}

template <int nmode, typename T, int alpha, int beta,typename P>
sycl::event NESO_ALWAYS_INLINE project_cpu(DeviceData<T,eQuad> &data, 
                         int componant,
                        sycl::queue &queue) 
{
  sycl::range<1> range{static_cast<std::size_t>(data.ncells)};
  return queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<>(range, [=](sycl::id<1> idx) {
      auto cellx = data.cell_ids[idx];
      auto npart = data.par_per_cell[cellx];
      if (npart == 0)
        return;
      auto cell_dof = &data.dofs[data.dof_offsets[cellx]];
      for (int part = 0; part < npart; ++part) {
        auto eta0 = data.positions[cellx][0][part];
        auto eta1 = data.positions[cellx][1][part];
        auto qoi = data.input[cellx][componant][part];
        Private::CPU::quad_ppt<nmode, T, alpha, beta>(eta0, eta1, qoi, cell_dof);
      }
    });
  });
}
} // namespace NESO::Project
