#pragma once
#include "../basis/basis.hpp"
#include "constants.hpp"
#include "device_data.hpp"
//#include "shapes_old.hpp"
#include "shapes.hpp"
#include "unroll.hpp"

#include <CL/sycl.hpp>

namespace NESO::Project {
namespace Private::TPP {
template <int nmode, typename T, int alpha, int beta>
inline void NESO_ALWAYS_INLINE project_tri(const double eta0, const double eta1, const double qoi,
             double *dofs) {
  T local0[nmode];
  T local1[(nmode * (nmode + 1))/2];
  Basis::eModA<T, nmode, Constants::cpu_stride, alpha, beta>(eta0, local0);
  Basis::eModB<T, nmode, Constants::cpu_stride, alpha, beta>(eta1, local1);
  int mode = 0;
  NESO_UNROLL_LOOP
  for (int i = 0; i < nmode; ++i) {
    NESO_UNROLL_LOOP
    for (int j = 0; j < nmode - i; ++j) {
      double temp = (mode == 1) ? 1.0 : local0[i];
      dofs[mode] += temp * local1[mode] * qoi;
      mode++;
    }
  }
}
} // namespace Private::CPU
template <int nmode, typename T, int alpha, int beta,typename P>
cl::sycl::event NESO_ALWAYS_INLINE project_cpu(DeviceData<T, eTriangle> &data, int componant,
                            cl::sycl::queue &queue) {
  cl::sycl::range<1> range{static_cast<std::size_t>(data.ncells)};
  return queue.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<>(range, [=](cl::sycl::id<1> idx) {
      auto cellx = data.cell_ids[idx];
      auto npart = data.par_per_cell[cellx];
      if (npart == 0)
        return;
      auto cell_dof = &data.dofs[data.dof_offsets[cellx]];
      for (int part = 0; part < npart; ++part) {
        auto xi0 = data.positions[cellx][0][part];
        auto xi1 = data.positions[cellx][1][part];
        T eta0, eta1;
        eTriangle::template loc_coord_to_loc_collapsed<T>(xi0, xi1, eta0, eta1);
        auto qoi = data.input[cellx][componant][part];
        eTriangle::template project_tpp<nmode,T,alpha,beta>(eta0,eta1,qoi,cell_dof);
        //Private::TPP::project_tri<nmode, T, alpha, beta>(eta0, eta1, qoi, cell_dof);
      }
    });
  });
}
} // namespace NESO::Project
