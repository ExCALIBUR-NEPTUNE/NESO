#ifndef __One_Dimensional_Linear_Interpolator_H__
#define __One_Dimensional_Linear_Interpolator_H__

#include "interpolator.hpp"

#include <map>
#include <memory>

#include <vector>
#include <CL/sycl.hpp>
#include <neso_particles.hpp>

using namespace cl;
using namespace NESO::Particles;
namespace NESO {

/**
 *  Class used to output a vector of y values, given some x values, based
 * on provided x,y values for input
 */

class OneDimensionalLinearInterpolator : public Interpolator {
public:
  OneDimensionalLinearInterpolator(std::vector<double> x_data , std::vector<double> y_data , std::vector<double> x_input) : Interpolator(x_data , y_data , x_input) {
    interpolate(x_data , y_data , x_input);
  };
  SYCLTargetSharedPtr sycl_target;
       
protected:
  virtual void interpolate(std::vector<double> x_data , std::vector<double> y_data , std::vector<double> x_input) final {
    
    //calculate change in y from between each vector array position
    std::vector<double> dy;  
    dy.push_back(0);
    for(int i=1;i<y_data.size();i++)
    {
	        dy.push_back((y_data[i]-y_data[i-1]));
    }
    dy[0]=dy[1];  
    
    //sycl code
    y_output.resize(x_input.size());
    sycl::buffer<double, 1> buffer_y_output(y_output.data(), sycl::range<1>{y_output.size()});
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          auto y_output_sycl_vector = buffer_y_output.get_access<sycl::access::mode::read_write>(cgh);
          cgh.parallel_for<>(
              sycl::range<1>(y_output.size()), [=](sycl::id<1> idx) { 			  
                 int k = lower_bound(x_data.begin(), x_data.end(), x_input[int(idx)])  - x_data.begin();
	             y_output_sycl_vector[int(idx)] = y_data[k-1] + dy[k]*(x_input[int(idx)]-x_data[k-1])/(x_data[k]-x_data[k-1]);	
              });
                 })
        .wait_and_throw();

  }
};

} // namespace NESO
#endif
