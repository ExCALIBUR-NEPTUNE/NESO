#ifndef _NESO_NEKTAR_INTERFACE_PROJECTION_ALGORITHM_TYPES_HPP
#define _NESO_NEKTAR_INTERFACE_PROJECTION_ALGORITHM_TYPES_HPP

#include "device_data.hpp"
#include "shapes.hpp"
#include <limits>
#include <neso_constants.hpp>
#include <neso_particles/sycl_typedefs.hpp>
#include <optional>
#include <type_traits>

#define ROUND_UP_TO_MULTIPLE(MULTIPLE, X)                                      \
  (MULTIPLE) * ((((X) - 1) / (MULTIPLE)) + 1)
#define ROUND_DOWN_TO_MULTIPLE(MULTIPLE, X) (MULTIPLE) * ((X) / (MULTIPLE))

namespace NESO::Project {
namespace Private {

template <typename T>
auto inline NESO_ALWAYS_INLINE
get_par_idx(DeviceData<T, NESO::Project::NoFilter> const &data, int, int n,
            int) {
  return n;
}

template <typename T>
auto inline NESO_ALWAYS_INLINE
get_par_idx(DeviceData<T, NESO::Project::ApplyFilter> const &data, int cell,
            int n, int npart) {
  assert(n < npart);
  return data.filter[cell][0][n];
}

} // namespace Private

struct ThreadPerCell {
  template <int nmode, typename T, int alpha, int beta, typename Shape,
            typename Filter>
  std::optional<sycl::event> static inline project(DeviceData<T, Filter> &data,
                                                   int componant,
                                                   sycl::queue &queue) {
    sycl::range<1> range{static_cast<size_t>(data.ncells)};
    if constexpr (Shape::dim == 2) {
      return queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(range, [=](sycl::id<1> idx) {
          auto cellx = data.cell_ids[idx];
          auto npart = data.par_per_cell[cellx];

          if (npart == 0)
            return;

          auto cell_dof = &data.dofs[data.dof_offsets[cellx]];
          for (int part = 0; part < npart; ++part) {
            int par_id =
                Private::get_par_idx(data, cellx, part, npart);
            auto xi0 = data.positions[cellx][0][par_id];
            auto xi1 = data.positions[cellx][1][par_id];
            T eta0, eta1;
            Shape::template loc_coord_to_loc_collapsed<T>(xi0, xi1, eta0, eta1);
            auto qoi = data.input[cellx][componant][par_id];
            Shape::template project_one_particle<nmode, T, alpha, beta>(
                eta0, eta1, qoi, cell_dof);
          }
        });
      });
    } else {
      return queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(range, [=](sycl::id<1> idx) {
          auto cellx = data.cell_ids[idx];
          auto npart = data.par_per_cell[cellx];

          if (npart == 0)
            return;

          auto cell_dof = &data.dofs[data.dof_offsets[cellx]];
          for (int part = 0; part < npart; ++part) {
            int par_id =
                Private::get_par_idx(data, cellx, part, npart);
            auto xi0 = data.positions[cellx][0][par_id];
            auto xi1 = data.positions[cellx][1][par_id];
            auto xi2 = data.positions[cellx][2][par_id];
            T eta0, eta1, eta2;
            Shape::template loc_coord_to_loc_collapsed<T>(xi0, xi1, xi2, eta0,
                                                          eta1, eta2);
            auto qoi = data.input[cellx][componant][par_id];
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
#elif defined(__clang__) || defined(__GNUC__)
#define PRETTY_FUNCTION __PRETTY_FUNCTION__
#else
#define PRETTY_FUNCTION __func__
#endif

struct ThreadPerDof {
private:
  template <int nmode, typename Shape>
  static inline uint64_t total_local_slots() {

    static_assert(Shape::dim == 2 || Shape::dim == 3,
                  "Only support 2 and 3 dimensional cell types");
    if constexpr (Shape::dim == 2) {
      return Shape::template local_mem_size<nmode, 0>(1) +
             Shape::template local_mem_size<nmode, 1>(1);
    } else {
      return Shape::template local_mem_size<nmode, 0>(1) +
             Shape::template local_mem_size<nmode, 1>(1) +
             Shape::template local_mem_size<nmode, 2>(1);
    }
  }

  template <int nmode, typename T, typename Shape>
  static inline uint64_t calc_max_local_size(sycl::queue &queue,
                                             int preferred_local_size,
                                             bool is_gpu) {
    auto dev = queue.get_device();
    auto max_size = static_cast<int>(
        dev.template get_info<sycl::info::device::max_work_group_size>());
    if (preferred_local_size > max_size) {
      preferred_local_size = max_size;
    }
    uint64_t local_mem_max =
        (dev.template get_info<sycl::info::device::local_mem_size>()) /
        sizeof(T);
    uint64_t local_required = total_local_slots<nmode, Shape>();
    if (local_required * (preferred_local_size + 1) < local_mem_max) {
      return preferred_local_size;
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
        is_gpu ? (dev.get_info<sycl::info::device::sub_group_sizes>()[0]) : 1;
    // Will return 0 if can't find something that is a multiple of
    //(min_sub_group_size + 1) that fits
    return ROUND_DOWN_TO_MULTIPLE(min_sub_group_size, a);
  }

public:
  template <int nmode, typename T, int alpha, int beta, typename Shape,
            typename Filter>
  std::optional<sycl::event> static inline project(DeviceData<T, Filter> &data,
                                                   int componant,
                                                   sycl::queue &queue) {
    bool is_gpu = queue.get_device().is_gpu();
    int local_size = calc_max_local_size<nmode, T, Shape>(
        queue, Constants::preferred_local_size, is_gpu);
    if (!local_size) {
      fprintf(stderr,
              "%s: This function uses too much local (shared) memory for your "
              "device\n",
              PRETTY_FUNCTION);
      // Don't fail here would be inconvenient in benchmark and tests
      // but need to check the return value is not empty in use
      return std::nullopt;
    }

    std::size_t outer_size = ROUND_UP_TO_MULTIPLE(local_size, data.nrow_max);
    // NOTE: If on NVIDIA then AFAICT outer_size < INT_MAX but nrow_max is an int
    // all we can do is check that the ROUND_UP won't overflow
    // but can probably get a tighter bound than this
    // These are hardcoded nvidia numbers - can't find them from sycl
    constexpr int max_y = std::numeric_limits<int>::max();
    constexpr int max_x = (1 << 16) - 1;
    if (is_gpu &&
        ((data.nrow_max > (max_y - local_size)) || (data.ncells > max_x))) {
      fprintf(stderr,
              "%s: requested number of work items exceeds max grid size (+/- "
              "rounding) on gpu device\n",
              PRETTY_FUNCTION);
      return std::nullopt;
    }
    // TODO: IMO not plausable to get row_max near INT_MAX becasue of memory
    // limits. But is possible to have > 2^16 cells so need to chunk
    // the kernel up if greater than this
    sycl::nd_range<2> range(sycl::range<2>(data.ncells, outer_size),
                            sycl::range<2>(1, local_size));
    int stride = local_size + 1;
    if constexpr (Shape::dim == 2) {
      return queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<T> local_mem0{
            Shape::template local_mem_size<nmode, 0>(stride), cgh};
        sycl::local_accessor<T> local_mem1{
            Shape::template local_mem_size<nmode, 1>(stride), cgh};

        cgh.parallel_for<>(range, [=](sycl::nd_item<2> idx) {
          const int cellx = data.cell_ids[idx.get_global_id(0)];
          long npart = data.par_per_cell[cellx];
          if (npart == 0)
            return;
          int idx_local = idx.get_local_id(1);
          int idx_global = idx.get_global_id(1);

          const int layerx =
              Private::get_par_idx(data, cellx, idx_global, npart);
          if (idx_global < npart) {
            auto xi0 = data.positions[cellx][0][layerx];
            auto xi1 = data.positions[cellx][1][layerx];
            T eta0, eta1;
            Shape::template loc_coord_to_loc_collapsed<T>(xi0, xi1, eta0, eta1);
            Shape::template fill_local_mem<nmode, T, alpha, beta>(
                eta0, eta1, data.input[cellx][componant][layerx],
                &local_mem0[idx_local], &local_mem1[idx_local], stride);
          }

          idx.barrier(sycl::access::fence_space::local_space);

          auto ndof = Shape::template get_ndof<nmode>();
          long nthd = idx.get_local_range(1);
          const auto cell_dof = &data.dofs[data.dof_offsets[cellx]];
          auto count = sycl::min(
              nthd, sycl::max(long{0}, npart - idx_global + idx_local));
          auto mode0 = &local_mem0[0];
          auto mode1 = &local_mem1[0];

          while (idx_local < ndof) {
            auto temp = Shape::template reduce_dof<nmode, T>(
                idx_local, count, mode0, mode1, stride);
            sycl::atomic_ref<T, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                coeff_atomic_ref(cell_dof[idx_local]);
            coeff_atomic_ref.fetch_add(temp);
            idx_local += nthd;
          }
        });
      });
    } else {
      return queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<T> local_mem0{
            Shape::template local_mem_size<nmode, 0>(stride), cgh};
        sycl::local_accessor<T> local_mem1{
            Shape::template local_mem_size<nmode, 1>(stride), cgh};
        sycl::local_accessor<T> local_mem2{
            Shape::template local_mem_size<nmode, 2>(stride), cgh};

        cgh.parallel_for<>(range, [=](sycl::nd_item<2> idx) {
          int const cellx = data.cell_ids[idx.get_global_id(0)];
          long const npart = data.par_per_cell[cellx];

          if (npart == 0)
            return;

          int idx_local = idx.get_local_id(1);
          int const idx_global = idx.get_global_id(1);
          int const layerx =
              Private::get_par_idx(data, cellx, idx_global, npart);

          if (idx_global < npart) {
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

          idx.barrier(sycl::access::fence_space::local_space);

          auto ndof = Shape::template get_ndof<nmode>();
          long nthd = idx.get_local_range(1);
          const auto cell_dof = &data.dofs[data.dof_offsets[cellx]];
          auto count = sycl::min(
              nthd, sycl::max(long{0}, npart - idx_global + idx_local));
          auto mode0 = &local_mem0[0];
          auto mode1 = &local_mem1[0];
          auto mode2 = &local_mem2[0];

          while (idx_local < ndof) {
            auto temp = Shape::template reduce_dof<nmode, T>(
                idx_local, count, mode0, mode1, mode2, stride);
            sycl::atomic_ref<T, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
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
#endif
