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
  std::vector<double> test_input = {1.5, 2.5};
  std::vector<double> &test_input_ref = test_input;
  std::vector<double> test_output;
  std::vector<double> &test_output_ref = test_output;
  int error;
  int &error_ref = error;
  OneDimensionalLinearInterpolator({0, 1, 2, 3}, {0, 1, 2, 3}, sycl_target)
      .get_y(test_input_ref, test_output_ref, error_ref);
  ASSERT_NEAR(test_output[0], 1.5, 1e-16);
  ASSERT_NEAR(test_output[1], 2.5, 1e-16);

  // test error handling

  // Save current stderr buffer
  std::streambuf *saved_stderr = std::cerr.rdbuf();

  // Redirect stderr to a stringstream buffer
  std::stringstream ss;
  std::cerr.rdbuf(ss.rdbuf());

  test_input = {-0.5, 3.5};
  OneDimensionalLinearInterpolator({0, 1, 2, 3}, {0, 1, 2, 3}, sycl_target)
      .get_y(test_input_ref, test_output_ref, error_ref);
  // Should get 2 errors for test_input values being outside range
  ASSERT_EQ(error, 2);

  // Restore stderr buffer
  std::cerr.rdbuf(saved_stderr);
}
