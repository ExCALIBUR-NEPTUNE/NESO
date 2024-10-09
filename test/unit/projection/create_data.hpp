#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <nektar_interface/projection/device_data.hpp>
#include <utility>
#include <vector>

struct TestData2D {
  int ndof;
  double val;
  double x, y;
  TestData2D(int ndof_, double val_, double x_, double y_)
      : ndof(ndof_), val(val_), x(x_), y(y_) {}
  friend void PrintTo(TestData2D const &data, std::ostream *os) {
    *os << data.ndof << ", " << data.val << ", (" << data.x << ", " << data.y
        << ")";
  }
};

struct TestData3D {
  int ndof;
  double val;
  double x, y, z;
  TestData3D(int ndof_, double val_, double x_, double y_, double z_ = 0.0)
      : ndof(ndof_), val(val_), x(x_), y(y_), z{z_} {}
  friend void PrintTo(TestData3D const &data, std::ostream *os) {
    *os << data.ndof << ", " << data.val << ", (" << data.x << ", " << data.y
        << ", " << data.z << ")";
  }
};

static inline auto create_data(cl::sycl::queue &Q, int N, double val, double x,
                               double y, double z = double{0.0}) {
  // Shove all the pointers in a vector so we can free them later
  std::vector<void *> all_pointers;
  double *dofs = cl::sycl::malloc_device<double>(N, Q);
  assert(dofs);
  all_pointers.push_back((void *)dofs);
  Q.fill(dofs, double{0.0}, N).wait_and_throw();

  int *dof_offsets = cl::sycl::malloc_device<int>(1, Q);
  all_pointers.push_back((void *)dof_offsets);
  assert(dof_offsets);
  Q.fill(dof_offsets, 0, 1).wait_and_throw();

  int *cell_ids = cl::sycl::malloc_device<int>(1, Q);
  all_pointers.push_back((void *)cell_ids);
  assert(cell_ids);
  Q.fill(cell_ids, 0, 1).wait_and_throw();

  int *par_per_cell = cl::sycl::malloc_device<int>(1, Q);
  all_pointers.push_back((void *)par_per_cell);
  assert(par_per_cell);
  Q.fill(par_per_cell, 1, 1).wait_and_throw();

  auto positions = cl::sycl::malloc_device<double **>(1, Q);
  all_pointers.push_back((void *)positions);
  assert(positions);
  auto temp0 = cl::sycl::malloc_device<double *>(3, Q);
  all_pointers.push_back((void *)temp0);
  assert(temp0);
  Q.fill(positions, temp0, 1).wait_and_throw();
  double P[3] = {x, y, z};
  double *pointP[3] = {cl::sycl::malloc_device<double>(1, Q),
                       cl::sycl::malloc_device<double>(1, Q),
                       cl::sycl::malloc_device<double>(1, Q)};
  assert(pointP[0] && pointP[1] && pointP[2]);
  all_pointers.push_back((void *)pointP[0]);
  all_pointers.push_back((void *)pointP[1]);
  all_pointers.push_back((void *)pointP[2]);
  Q.parallel_for<>(3, [=](cl::sycl::id<1> id) {
     positions[0][id] = pointP[id];
     positions[0][id][0] = P[id];
   }).wait_and_throw();

  auto input = cl::sycl::malloc_device<double **>(1, Q);
  all_pointers.push_back((void *)input);
  assert(input);
  auto temp1 = cl::sycl::malloc_device<double *>(1, Q);
  assert(temp1);
  all_pointers.push_back((void *)temp1);
  auto temp2 = cl::sycl::malloc_device<double>(1, Q);
  assert(temp2);
  all_pointers.push_back((void *)temp2);
  Q.fill(input, temp1, 1).wait_and_throw();
  Q.fill(temp1, temp2, 1).wait_and_throw();
  Q.fill(temp2, val, 1).wait_and_throw();

  return std::pair(NESO::Project::DeviceData<double>(dofs, dof_offsets, 1, 1,
                                                     cell_ids, par_per_cell,
                                                     positions, input),
                   all_pointers);
}

static inline void free_data(cl::sycl::queue &Q, std::vector<void *> &data) {
  for (auto &p : data) {
    cl::sycl::free(p, Q);
  }
}
