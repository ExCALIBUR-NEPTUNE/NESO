#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <nektar_interface/projection/device_data.hpp>


static inline auto create_data(cl::sycl::queue &Q, int N, double val, double x,
                               double y, double z = double{0.0}) {

  double *dofs = cl::sycl::malloc_device<double>(N, Q);
  assert(dofs);
  Q.fill(dofs, double{0.0}, N).wait();

  int *dof_offsets = cl::sycl::malloc_device<int>(1, Q);
  assert(dof_offsets);
  Q.fill(dof_offsets, 0, 1).wait();

  int *cell_ids = cl::sycl::malloc_device<int>(1, Q);
  assert(cell_ids);
  Q.fill(cell_ids, 0, 1).wait();

  int *par_per_cell = cl::sycl::malloc_device<int>(1, Q);
  assert(par_per_cell);
  Q.fill(par_per_cell, 1, 1).wait();

  auto positions = cl::sycl::malloc_device<double **>(1, Q);
  assert(positions);
  auto temp0 = cl::sycl::malloc_device<double *>(3, Q);
  assert(temp0);
  Q.fill(positions, temp0, 1).wait();
  double P[3] = {x, y, z};
  double *pointP[3] = {cl::sycl::malloc_device<double>(1, Q),
                       cl::sycl::malloc_device<double>(1, Q),
                       cl::sycl::malloc_device<double>(1, Q)};
  assert(pointP[0] && pointP[1] && pointP[2]);
 //TODO: Something wrong here
  Q.parallel_for<>(3, [=](cl::sycl::id<1> id) {
     positions[0][id] = pointP[id];
     positions[0][id][0] = P[id];
   }).wait();

  auto input = cl::sycl::malloc_device<double **>(1, Q);
  assert(input);
  auto temp1 = cl::sycl::malloc_device<double *>(1, Q);
  assert(temp1);
  auto temp2 = cl::sycl::malloc_device<double>(1, Q);
  assert(temp2);
  Q.fill(input, temp1, 1).wait();
  Q.fill(temp1, temp2, 1).wait();
  Q.fill(temp2, val, 1).wait();

  return NESO::Project::DeviceData<double>(dofs, dof_offsets, 1, 1, cell_ids,
                                           par_per_cell, positions, input);
}

static inline void free_data(cl::sycl::queue &Q,
                             NESO::Project::DeviceData<double> &data) {
// LEAK IT!!!!
#if 0
  cl::sycl::free(data.dofs, Q);
  cl::sycl::free(data.dof_offsets, Q);
  cl::sycl::free(data.cell_ids, Q);
  cl::sycl::free(data.par_per_cell, Q);
  for (int i = 0; i < 3; ++i) {
    cl::sycl::free(data.positions[0][i], Q);
  }
  cl::sycl::free(data.positions[0], Q);
  cl::sycl::free(data.positions, Q);

  cl::sycl::free(data.input[0][0], Q);
  cl::sycl::free(data.input[0], Q);
  cl::sycl::free(data.input, Q);
#endif
}
