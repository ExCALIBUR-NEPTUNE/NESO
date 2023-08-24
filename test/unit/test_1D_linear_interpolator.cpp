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
  const std::vector<double> &test_input_ref = test_input;
  std::vector<double> test_output(test_input.size());
  std::vector<double> &test_output_ref = test_output;

  std::vector<double> test_x_data = {0, 1, 2, 4};
  const std::vector<double> &test_x_data_ref = test_x_data;
  std::vector<double> test_y_data = {0, 1, 3, 4};
  const std::vector<double> &test_y_data_ref = test_y_data;

  LinearInterpolator1D(test_x_data_ref, test_y_data_ref, sycl_target)
      .interpolate(test_input_ref, test_output_ref);
  ASSERT_NEAR(test_output[0], 2.0, std::numeric_limits<double>::epsilon());
  ASSERT_NEAR(test_output[1], 3.25, std::numeric_limits<double>::epsilon());
}
