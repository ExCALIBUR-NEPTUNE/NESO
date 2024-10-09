#pragma once

#include "constants.hpp"
#include "device_data.hpp"
#include "shapes.hpp"
#include <CL/sycl.hpp>
#include <limits>
#include <optional>

#define ROUND_UP_TO_MULTIPLE(MULTIPLE, X)                                      \
  (MULTIPLE) * ((((X) - 1) / (MULTIPLE)) + 1)
#define ROUND_DOWN_TO_MULTIPLE(MULTIPLE, X) (MULTIPLE) * ((X) / (MULTIPLE))

namespace NESO::Project {
namespace Private {
template <int nmode, typename Shape>
static inline uint64_t total_local_slots() {
  if constexpr (Shape::dim == 2) {
    return Shape::template local_mem_size<nmode, 0>(1) +
           Shape::template local_mem_size<nmode, 1>(1);
  } else if constexpr (Shape::dim == 3) {
    return Shape::template local_mem_size<nmode, 0>(1) +
           Shape::template local_mem_size<nmode, 1>(1) +
           Shape::template local_mem_size<nmode, 2>(1);
  } else {
    static_assert(false, "Only support 2 and 3 dimensional cell types");
    // unreachable
    return -1;
  }
}

template <int nmode, typename T, typename Shape>
static inline uint64_t calc_max_local_size(cl::sycl::queue &queue,
                                           int preferred_block_size) {
  auto dev = queue.get_device();
  // We subtract 1024 because cuda reserves that much per SM for itself
  // Don't know if there is some "sycl" way to get this info
  // TODO: Find the corresponding number for AMD and Intel
  uint64_t local_mem_max =
      (dev.get_info<cl::sycl::info::device::local_mem_size>() - 1024) /
      sizeof(T);
  uint64_t local_required = total_local_slots<nmode, Shape>();
  if (local_required * (preferred_block_size + 1) < local_mem_max) {
    return preferred_block_size;
  }
  // check it's possible to find anything
  // Make sure a != 0 in next step
  if (local_required > local_mem_max) {
    return 0;
  }
  // Need (a + 1) * local_required <= local_mem_max
  //      a <= local_mem_max/local_requred - 1
  uint64_t a = (local_mem_max / local_required) - 1;
  // not sure about the sub_group_sizes thing but is the closest I can find
  size_t min_sub_group_size =
      dev.is_gpu()
          ? (dev.get_info<cl::sycl::info::device::sub_group_sizes>()[0])
          : 1;
  // Will return 0 if can't find something that is a multiple of
  //(min_sub_group_size + 1) that fits
  return ROUND_DOWN_TO_MULTIPLE(min_sub_group_size, a);
}
} // namespace Private

struct ThreadPerCell {
  template <int nmode, typename T, int alpha, int beta, typename Shape>
  std::optional<cl::sycl::event> static inline project(DeviceData<T> &data,
                                                       int componant,
                                                       cl::sycl::queue &queue) {
    cl::sycl::range<1> range{static_cast<size_t>(data.ncells)};
    if constexpr (Shape::dim == 2) {
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
    } else {
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
  }
};

// Untested on Microsoft
// c++20 has a solution for this std::source_location::function_name
#ifdef _MSC_VER
#define PRETTY_FUNCTION __FUNCSIG__
#else
#define PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif

struct ThreadPerDof {

  template <int nmode, typename T, int alpha, int beta, typename Shape>
  std::optional<cl::sycl::event> static inline project(DeviceData<T> &data,
                                                       int componant,
                                                       cl::sycl::queue &queue) {

    // TODO: What to do when local_size comes back as zero
    // 1. Theoretically can use fewer than the warp size for the block size
    //    just be slow
    // 2. Crash always
    // 3. Crash in debug/warn in release <--- doing this for now, crashing will
    //    compilcate the benchmarking in release
    auto local_size = Private::calc_max_local_size<nmode, T, Shape>(
        queue, Constants::preferred_block_size);
    if (local_size == 0) {
      fprintf(stderr,
              "%s: This kernel uses too much local(shared) memory for your "
              "device\n",
              PRETTY_FUNCTION);
      // Crash in debug version
      assert(false);
      // Just return empty optional in release mode
      // Doing this so the benchmarks can run without crashing
      // TODO: This needs a better solution I could raise an exception (blurg)
      // or make it return std::option (more rusty)
      return {};
    }

    std::size_t outer_size = ROUND_UP_TO_MULTIPLE(local_size, data.nrow_max);

    cl::sycl::nd_range<2> range(cl::sycl::range<2>(data.ncells, outer_size),
                                cl::sycl::range<2>(1, local_size));
    // in practice won't overflow a int
    assert((local_size + 1) < std::numeric_limits<int>::max());
    int stride = (int)local_size + 1;
    if constexpr (Shape::dim == 2) {
      return queue.submit([&](cl::sycl::handler &cgh) {
        cl::sycl::local_accessor<T> local_mem0{
            Shape::template local_mem_size<nmode, 0>(stride), cgh};
        cl::sycl::local_accessor<T> local_mem1{
            Shape::template local_mem_size<nmode, 1>(stride), cgh};

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
                &local_mem0[idx_local], &local_mem1[idx_local], stride);
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
            auto temp = Shape::template reduce_dof<nmode, T>(
                idx_local, count, mode0, mode1, stride);
            cl::sycl::atomic_ref<T, cl::sycl::memory_order::relaxed,
                                 cl::sycl::memory_scope::device>
                coeff_atomic_ref(cell_dof[idx_local]);
            coeff_atomic_ref.fetch_add(temp);
            idx_local += nthd;
          }
        });
      });
    } else {
      return queue.submit([&](cl::sycl::handler &cgh) {
        cl::sycl::local_accessor<T> local_mem0{
            Shape::template local_mem_size<nmode, 0>(stride), cgh};
        cl::sycl::local_accessor<T> local_mem1{
            Shape::template local_mem_size<nmode, 1>(stride), cgh};
        cl::sycl::local_accessor<T> local_mem2{
            Shape::template local_mem_size<nmode, 2>(stride), cgh};

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
                &local_mem2[idx_local], stride);
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
            auto temp = Shape::template reduce_dof<nmode, T>(
                idx_local, count, mode0, mode1, mode2, stride);
            cl::sycl::atomic_ref<T, cl::sycl::memory_order::relaxed,
                                 cl::sycl::memory_scope::device>
                coeff_atomic_ref(cell_dof[idx_local]);
            coeff_atomic_ref.fetch_add(temp);
            idx_local += nthd;
          }
        });
      });
    }
  }
};
#undef ROUND_UP_TO_MULTIPLE
#undef ROUND_DOWN_TO_MULTIPLE
#undef PRETTY_FUNCTION
} // namespace NESO::Project
