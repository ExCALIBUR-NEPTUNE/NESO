#include "linear_interpolator_1D.hpp"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <mpi.h>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO;
using namespace NESO::Particles;

TEST(InterpolatorTest, 1DLinear) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  std::vector<double> test_input = {1.5, 2.5};
  std::vector<double> &test_input_ref = test_input;
  std::vector<double> test_output(test_input.size());
  std::vector<double> &test_output_ref = test_output;
  LinearInterpolator1D({0, 1, 2, 4}, {0, 1, 3, 4}, sycl_target)
      .interpolate(test_input_ref, test_output_ref);
  ASSERT_NEAR(test_output[0], 2.0, 1e-16);
  ASSERT_NEAR(test_output[1], 3.25, 1e-16);
}
