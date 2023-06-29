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
                                   std::vector<double> x_input,
                                   SYCLTargetSharedPtr sycl_target)
      : Interpolator(x_data, y_data, x_input, sycl_target) {
    interpolate(x_data, y_data, x_input);
  };

protected:
  virtual void interpolate(std::vector<double> x_data,
                           std::vector<double> y_data,
                           std::vector<double> x_input) {

    // calculate change in y from between each vector array position
    std::vector<double> dy;
    dy.push_back(0);
    for (int i = 1; i < y_data.size(); i++) {
      dy.push_back((y_data[i] - y_data[i - 1]));
    }
    dy[0] = dy[1];

    // sycl code
    y_output.resize(x_input.size());
    sycl::buffer<double, 1> buffer_x_data(x_data.data(),
                                          sycl::range<1>{x_data.size()});
    sycl::buffer<double, 1> buffer_y_data(y_data.data(),
                                          sycl::range<1>{y_data.size()});
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
                  if (y_data_sycl[int(idx)] - x_data_sycl[i] < 0.0) {
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
  }
};

} // namespace NESO
#endif
