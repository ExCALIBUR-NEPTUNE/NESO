#include "One_Dimensional_Linear_Interpolator.hpp"
#include <gtest/gtest.h>
#include <vector>


using namespace NESO;

TEST(InterpolatorTest, 1DLinear) {
	
  std::vector<double> output_test = OneDimensionalLinearInterpolator( {0,1,2,3} , {0,1,2,3} , {1.5,2.5} ).get_y();
  std::cout<<output_test[0]<<"  "<<output_test[1]<<std::endl;
  ASSERT_NEAR(output_test[0], 1.5, 1e-16);
  ASSERT_NEAR(output_test[1], 2.5, 1e-16);

}
