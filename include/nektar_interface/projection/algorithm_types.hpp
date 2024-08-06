#pragma once
#include <CL/sycl.hpp>
#include "basis/basis.hpp"
#include "constants.hpp"
#include "device_data.hpp"
#include "restrict.hpp"
#include "shapes.hpp"
#include "unroll.hpp"

namespace NESO::Project {

struct ThreadPerCell3D {
  template <int nmode, typename T, int alpha, int beta, typename Shape>
  cl::sycl::event static inline project(DeviceData<T> &data, int componant,
                                        cl::sycl::queue &queue) {
    cl::sycl::range<1> range{static_cast<size_t>(data.ncells)};
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
          auto xi2 = data.positions[cellx][2][part];
          T eta0, eta1, eta2;
          Shape::template loc_coord_to_loc_collapsed<T>(xi0, xi1, xi2, eta0,
                                                        eta1, eta2);
          auto qoi = data.input[cellx][componant][part];
          Shape::template project_one_particle<nmode, T, alpha, beta>(
              eta0, eta1, eta2, qoi, cell_dof);
        }
      });
    });
  }
};

struct ThreadPerCell2D {
  template <int nmode, typename T, int alpha, int beta, typename Shape>
  cl::sycl::event static inline project(DeviceData<T> &data, int componant,
                                        cl::sycl::queue &queue) {
    cl::sycl::range<1> range{static_cast<size_t>(data.ncells)};
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
          Shape::template project_one_particle<nmode, T, alpha, beta>(
              eta0, eta1, qoi, cell_dof);
        }
      });
    });
  }
};

#define ROUND_UP_TO(SIZE, N) (SIZE) * ((((N)-1) / (SIZE)) + 1)
struct ThreadPerDof3D {

  template <int nmode, typename T, int alpha, int beta, typename Shape>
  cl::sycl::event static inline project(DeviceData<T> &data, int componant,
                                        cl::sycl::queue &queue) {

    std::size_t outer_size = ROUND_UP_TO(Constants::local_size, data.nrow_max);

    cl::sycl::nd_range<2> range(cl::sycl::range<2>(data.ncells, outer_size),
                                cl::sycl::range<2>(1, Constants::local_size));

    auto ev = queue.submit([&](cl::sycl::handler &cgh) {
      cl::sycl::local_accessor<double> local_mem0{
          Shape::template local_mem_size<nmode, 0>(), cgh};
      cl::sycl::local_accessor<double> local_mem1{
          Shape::template local_mem_size<nmode, 1>(), cgh};
      cl::sycl::local_accessor<double> local_mem2{
          Shape::template local_mem_size<nmode, 2>(), cgh};

      cgh.parallel_for<>(range, [=](cl::sycl::nd_item<2> idx) {
        const int cellx = data.cell_ids[idx.get_global_id(0)];
        long npart = data.par_per_cell[cellx];

        if (npart == 0)
          return;

        int idx_local = idx.get_local_id(1);
        const int layerx = idx.get_global_id(1);

        if (layerx < npart) {
          auto xi0 = data.positions[cellx][0][layerx];
          auto xi1 = data.positions[cellx][1][layerx];
          auto xi2 = data.positions[cellx][2][layerx];
          T eta0, eta1, eta2;
          Shape::template loc_coord_to_loc_collapsed<T>(xi0, xi1, xi2, eta0,
                                                        eta1, eta2);
          Shape::template fill_local_mem<nmode, T, alpha, beta>(
              eta0, eta1, eta2, data.input[cellx][componant][layerx],
              &local_mem0[idx_local], &local_mem1[idx_local],
              &local_mem2[idx_local]);
        }

        idx.barrier(cl::sycl::access::fence_space::local_space);

        auto ndof = Shape::template get_ndof<nmode>();
        long nthd = idx.get_local_range(1);
        const auto cell_dof = &data.dofs[data.dof_offsets[cellx]];
        auto count =
            std::min(nthd, std::max(long{0}, npart - layerx + idx_local));
        auto mode0 = &local_mem0[0];
        auto mode1 = &local_mem1[0];
        auto mode2 = &local_mem2[0];

        while (idx_local < ndof) {
          auto temp = Shape::template reduce_dof<nmode, T>(idx_local, count,
                                                           mode0, mode1, mode2);
          cl::sycl::atomic_ref<double, cl::sycl::memory_order::relaxed,
                               cl::sycl::memory_scope::device>
              coeff_atomic_ref(cell_dof[idx_local]);
          coeff_atomic_ref.fetch_add(temp);
          idx_local += nthd;
        }
      });
    });
    return ev;
  }
};
struct ThreadPerDof2D {
  // Project (in 2D) using one thread per particle, then one thread per dof
  template <int nmode, typename T, int alpha, int beta, typename Shape>
  cl::sycl::event static inline project(DeviceData<T> &data, int componant,
                                        cl::sycl::queue &queue) {

    std::size_t outer_size = ROUND_UP_TO(Constants::local_size, data.nrow_max);

    cl::sycl::nd_range<2> range(cl::sycl::range<2>(data.ncells, outer_size),
                                cl::sycl::range<2>(1, Constants::local_size));

    auto ev = queue.submit([&](cl::sycl::handler &cgh) {
      cl::sycl::local_accessor<double> local_mem0{
          Shape::template local_mem_size<nmode, 0>(), cgh};
      cl::sycl::local_accessor<double> local_mem1{
          Shape::template local_mem_size<nmode, 1>(), cgh};

      cgh.parallel_for<>(range, [=](cl::sycl::nd_item<2> idx) {
        const int cellx = data.cell_ids[idx.get_global_id(0)];
        long npart = data.par_per_cell[cellx];

        if (npart == 0)
          return;

        int idx_local = idx.get_local_id(1);
        const int layerx = idx.get_global_id(1);

        if (layerx < npart) {
          auto xi0 = data.positions[cellx][0][layerx];
          auto xi1 = data.positions[cellx][1][layerx];
          T eta0, eta1;
          Shape::template loc_coord_to_loc_collapsed<T>(xi0, xi1, eta0, eta1);
          Shape::template fill_local_mem<nmode, T, alpha, beta>(
              eta0, eta1, data.input[cellx][componant][layerx],
              &local_mem0[idx_local], &local_mem1[idx_local]);
        }

        idx.barrier(cl::sycl::access::fence_space::local_space);

        auto ndof = Shape::template get_ndof<nmode>();
        long nthd = idx.get_local_range(1);
        const auto cell_dof = &data.dofs[data.dof_offsets[cellx]];
        auto count =
            std::min(nthd, std::max(long{0}, npart - layerx + idx_local));
        auto mode0 = &local_mem0[0];
        auto mode1 = &local_mem1[0];

        while (idx_local < ndof) {
          auto temp = Shape::template reduce_dof<nmode, T>(idx_local, count,
                                                           mode0, mode1);
          cl::sycl::atomic_ref<double, cl::sycl::memory_order::relaxed,
                               cl::sycl::memory_scope::device>
              coeff_atomic_ref(cell_dof[idx_local]);
          coeff_atomic_ref.fetch_add(temp);
          idx_local += nthd;
        }
      });
    });
    return ev;
  }
};
#undef ROUND_UP_TO
} // namespace NESO::Project
