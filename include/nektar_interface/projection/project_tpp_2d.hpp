#pragma once
#include "device_data.hpp"
#include "shapes.hpp"
#include <CL/sycl.hpp>

namespace NESO::Project {
template <int nmode, typename T, int alpha, int beta, typename Shape>
cl::sycl::event project_tpp(DeviceData<T, Shape> &data, int componant,
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
        Shape::template loc_coord_to_loc_collapsed<T>(xi0, xi1, eta0, eta1);
        auto qoi = data.input[cellx][componant][part];
        Shape::template project_tpp<nmode, T, alpha, beta>(eta0, eta1, qoi,
                                                           cell_dof);
      }
    });
  });
}
} // namespace NESO::Project
