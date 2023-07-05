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
  std::vector<double> test_input = {1.5,2.5};
  std::vector<double> &test_input_ref = test_input;
  std::vector<double> test_output;
  std::vector<double> &test_output_ref = test_output;
  OneDimensionalLinearInterpolator({0, 1, 2, 3}, {0, 1, 2, 3}, sycl_target).get_y( test_input_ref , test_output_ref );
  ASSERT_NEAR(test_output[0], 1.5, 1e-16);
  ASSERT_NEAR(test_output[1], 2.5, 1e-16);
}
