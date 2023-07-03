#include "one_dimensional_linear_interpolator.hpp"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <mpi.h>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO;
using namespace NESO::Particles;

TEST(InterpolatorTest, 1DLinear) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  std::vector<double> test = OneDimensionalLinearInterpolator({0, 1, 2, 3}, {0, 1, 2, 3}, sycl_target).get_y({1.5, 2.5});
  ASSERT_NEAR(test[0], 1.5, 1e-16);
  ASSERT_NEAR(test[1], 2.5, 1e-16);
}
