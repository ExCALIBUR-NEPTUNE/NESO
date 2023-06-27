#include "One_Dimensional_Linear_Interpolator.hpp"
#include <gtest/gtest.h>
#include <vector>


using namespace NESO;

TEST(InterpolatorTest, 1DLinear) {

  OneDimensionalLinearInterpolator output =
      OneDimensionalLinearInterpolator({1,2,3},{1,2,3},{1.5,2.5});

  std::vector<double> output_test = output.get_y();

  ASSERT_NEAR(output_test[1], 1.5, 1e-16);
  ASSERT_NEAR(output_test[2], 2.5, 1e-16);

}
