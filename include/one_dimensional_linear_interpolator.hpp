#ifndef __ONE_DIMENSIONAL_LINEAR_INTERPOLATOR_H__
#define __ONE_DIMENSIONAL_LINEAR_INTERPOLATOR_H__

#include "interpolator.hpp"
#include <CL/sycl.hpp>
#include <mpi.h>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
using namespace cl;
namespace NESO {

/**
 *  Class used to output a vector of y values, given some x values, based
 * on provided x,y values for input
 */

class OneDimensionalLinearInterpolator : public Interpolator {
public:
  OneDimensionalLinearInterpolator(std::vector<double> x_data,
                                   std::vector<double> y_data,
                                   SYCLTargetSharedPtr sycl_target)
      : Interpolator(x_data, y_data, sycl_target){

        };

  virtual void get_y(std::vector<double> &x_input,
                     std::vector<double> &y_output, int &error) {
    interpolate(x_input, y_output, error);
  }

protected:
  virtual void interpolate(std::vector<double> &x_input,
                           std::vector<double> &y_output, int &error) {
    // testing error handling of interpolator
    ErrorPropagate ep(m_sycl_target);
    // sycl code
    auto k_ep = ep.device_ptr();
    y_output = std::vector<double>(x_input.size());
    sycl::buffer<double, 1> buffer_x_data(m_x_data.data(),
                                          sycl::range<1>{m_x_data.size()});
    sycl::buffer<double, 1> buffer_y_data(m_y_data.data(),
                                          sycl::range<1>{m_y_data.size()});
    sycl::buffer<double, 1> buffer_x_input(x_input.data(),
                                           sycl::range<1>{x_input.size()});
    sycl::buffer<double, 1> buffer_y_output(y_output.data(),
                                            sycl::range<1>{y_output.size()});
    sycl::buffer<double, 1> buffer_dy(dy.data(), sycl::range<1>{dy.size()});
    auto event_interpolate =
        this->m_sycl_target->queue.submit([&](sycl::handler &cgh) {
          auto x_data_sycl =
              buffer_x_data.get_access<sycl::access::mode::read>(cgh);
          auto y_data_sycl =
              buffer_y_data.get_access<sycl::access::mode::read>(cgh);
          auto x_input_sycl =
              buffer_x_input.get_access<sycl::access::mode::read>(cgh);
          auto y_output_sycl =
              buffer_y_output.get_access<sycl::access::mode::read_write>(cgh);
          auto dy_sycl = buffer_dy.get_access<sycl::access::mode::read>(cgh);
          cgh.parallel_for<>(
              sycl::range<1>(y_output_sycl.size()), [=](sycl::id<1> idx) {
                int k;
                for (int i = 0; i < x_data_sycl.size(); i++) {
                  // error handling
                  if (x_input_sycl[int(idx)] < x_data_sycl[0] or
                      x_input_sycl[int(idx)] >
                          x_data_sycl[x_data_sycl.size() - 1]) {
                    std::cerr
                        << "Input value out of " << x_input_sycl[int(idx)]
                        << " outside of range  of x values provided by data"
                        << std::endl;
                    // throw an error
                    NESO_KERNEL_ASSERT(false, k_ep);
                    break;
                  }
                  // set value of index to first value for which x value
                  // of input is greater than x data values
                  if (x_input_sycl[int(idx)] - x_data_sycl[i] < 0.0) {
                    k = i;
                    break;
                  }
                }
                y_output_sycl[int(idx)] =
                    y_data_sycl[k - 1] +
                    dy_sycl[k] * (x_input_sycl[int(idx)] - x_data_sycl[k - 1]) /
                        (x_data_sycl[k] - x_data_sycl[k - 1]);
              });
        });
    event_interpolate.wait_and_throw();
    error = ep.get_flag();
  }
};

} // namespace NESO
#endif
