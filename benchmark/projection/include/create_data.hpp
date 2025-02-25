#ifndef _NESO_BENCHMARK_PROJECTION_CREATE_DATA_HPP
#define _NESO_BENCHMARK_PROJECTION_CREATE_DATA_HPP
#include <nektar_interface/projection/device_data.hpp>
#include <neso_particles/sycl_typedefs.hpp>
#include <random>
#include <utility>
#include <vector>

#define CREATE_DATA_SEED 32423

// Fill a device array with random numbers on device
template <typename T>
static inline void random_fill_array(T *d_ptr, int size, sycl::queue &q) {
  assert(size > 0);
  std::mt19937 random(CREATE_DATA_SEED * CREATE_DATA_SEED);
  std::uniform_real_distribution<T> dist(-20.0, 20.0);
  std::vector<T> vec(size, 0.0);
  std::generate(vec.begin(), vec.end(), [&]() { return dist(random); });
  q.copy<T>(vec.data(), d_ptr, size).wait_and_throw();
}

// assign <active_cells> random "cells" with random number of "particles"
// between min_per_cell and max_per_cells
static inline std::vector<int> distribute_particles(int ncell, int active_cells,
                                                    int min_per_cell,
                                                    int max_per_cell) {
  if (ncell < active_cells) {
    fprintf(stderr, "Active cells > total cells\n");
    exit(1);
  }
  // std::random_device device;
  assert(min_per_cell > 0);
  assert(max_per_cell >= min_per_cell);
  std::mt19937 random(CREATE_DATA_SEED);
  std::uniform_int_distribution<> pdist(min_per_cell, max_per_cell);
  std::vector<int> npar(ncell, 0);
  // fill first active_cells  cells

  for (int i = 0; i < active_cells; ++i) {
    npar[i] = pdist(random);
  }
  // then shuffle
  if (ncell != active_cells) {
    std::shuffle(npar.begin(), npar.end(), random);
  }
  return npar;
}

// Allocate data in a way the projection algorithms expect i.e. in a DeviceData
// struct
// returns that DeviceData struct and a vector of all the pointers so can be
// freed
constexpr int NO_ACTIVE_CELL_COUNT = -1;
template <int nmode, typename T, typename Shape>
static inline auto create_data(sycl::queue &Q, int ncell, int min_per_cell,
                               int max_per_cell,
                               int active_cells = NO_ACTIVE_CELL_COUNT) {
  // Shove all the pointers in a vector so we can free them later
  std::vector<void *> all_pointers;

  // Par per cell
  int *par_per_cell = sycl::malloc_device<int>(ncell, Q);
  assert(par_per_cell);
  all_pointers.push_back((void *)par_per_cell);
  int max_row = 0;
  std::vector<int> host_par_per_cell;
  if (active_cells == NO_ACTIVE_CELL_COUNT)
    active_cells = ncell;
  host_par_per_cell =
      distribute_particles(ncell, active_cells, min_per_cell, max_per_cell);
  max_row =
      *std::max_element(host_par_per_cell.begin(), host_par_per_cell.end());
  Q.copy<int>(host_par_per_cell.data(), par_per_cell, ncell).wait();
  // DOFS
  T *dofs =
      sycl::malloc_device<T>(ncell * Shape::template get_ndof<nmode>(), Q);
  assert(dofs);
  all_pointers.push_back((void *)dofs);
  Q.fill(dofs, T{0.0}, ncell * Shape::template get_ndof<nmode>())
      .wait_and_throw();

  // DOF offsets ndof per cell but scanned
  auto host_offsets =
      std::vector<int>(ncell, Shape::template get_ndof<nmode>());
  std::exclusive_scan(host_offsets.begin(), host_offsets.end(),
                      host_offsets.begin(), 0);
  int *dof_offsets = sycl::malloc_device<int>(ncell, Q);
  assert(dof_offsets);
  all_pointers.push_back((void *)dof_offsets);
  Q.copy<int>(host_offsets.data(), dof_offsets, ncell).wait_and_throw();

  // Cell ids
  int *cell_ids = sycl::malloc_device<int>(ncell, Q);
  assert(cell_ids);
  all_pointers.push_back((void *)cell_ids);
  Q.parallel_for<>(sycl::range<1>(ncell), [=](sycl::id<1> id) {
     cell_ids[id] = id;
   }).wait();
  // Particle postion pointers
  auto host_data_ptrs = std::vector<T **>(ncell, nullptr);
  for (int i = 0; i < ncell; ++i) {
    // allocate data arrays for positions
    T *par_data[Shape::dim] = {nullptr};
    for (int j = 0; j < Shape::dim; ++j) {
      par_data[j] = sycl::malloc_device<T>(host_par_per_cell[i], Q);
      assert(par_data[j]);
      random_fill_array(par_data[j], host_par_per_cell[i], Q);
      all_pointers.push_back((void *)par_data[j]);
    }
    // allocate array for cells
    host_data_ptrs[i] = sycl::malloc_device<T *>(Shape::dim, Q);
    assert(host_data_ptrs[i]);
    all_pointers.push_back((void *)host_data_ptrs[i]);
    Q.copy<T *>(par_data, host_data_ptrs[i], Shape::dim).wait_and_throw();
  }
  auto positions = sycl::malloc_device<T **>(ncell, Q);
  assert(positions);
  all_pointers.push_back((void *)positions);
  Q.copy<T **>(host_data_ptrs.data(), positions, ncell).wait_and_throw();

  // Particle value data

  auto host_value_ptrs = std::vector<T **>(ncell, nullptr);
  for (int i = 0; i < ncell; ++i) {
    // allocate value arrays for positions
    T *par_vals[1] = {nullptr};
    par_vals[0] = sycl::malloc_device<T>(host_par_per_cell[i], Q);
    assert(par_vals);
    all_pointers.push_back((void *)par_vals[0]);
    random_fill_array(par_vals[0], host_par_per_cell[i], Q);
    host_value_ptrs[i] = sycl::malloc_device<T *>(1, Q);
    assert(host_value_ptrs[i]);
    all_pointers.push_back((void *)host_value_ptrs[i]);
    Q.copy<T *>(par_vals, host_value_ptrs[i], 1).wait_and_throw();
  }
  auto input = sycl::malloc_device<T **>(ncell, Q);
  assert(input);
  all_pointers.push_back((void *)input);
  Q.copy<T **>(host_value_ptrs.data(), input, ncell).wait_and_throw();

  return std::pair(NESO::Project::DeviceData<T, NESO::Project::NoFilter>(
                       dofs, dof_offsets, ncell, max_row, cell_ids,
                       par_per_cell, positions, input),
                   all_pointers);
}

// Free all the data
static inline void free_data(sycl::queue &Q, std::vector<void *> &data) {
  for (auto &p : data) {
    sycl::free(p, Q);
  }
}
#endif
