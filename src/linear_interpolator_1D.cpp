#include "linear_interpolator_1D.hpp"
#include <CL/sycl.hpp>
#include <mpi.h>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
using namespace cl;
namespace NESO {

/**
 * This constructor provides the class with the x_input values the
 * interpolator will need. It creates a gradient vector based on the x,y data
 * provided by the Interpolator class from which this class is derived. It
 * then creates a sycl buffer for the x,y data and this gradient vector.
 *
 * @param[in] x_input x_input is reference to a vector of x values for which
 * you would like the y value returned.
 * @param[out] y_output y_output is reference to a vector of y values which
 * the interpolator calculated based on x_input.
 * @param[in] sycl_target The target that the sycl kernels will make use of.
 */
LinearInterpolator1D::LinearInterpolator1D(const std::vector<double> &x_data,
                                           const std::vector<double> &y_data,
                                           SYCLTargetSharedPtr sycl_target)
    : Interpolator(x_data, y_data, sycl_target), buffer_x_data(0),
      buffer_y_data(0), buffer_dydx(0) {

  dydx.reserve(m_y_data.size());
  for (int i = 1; i < m_y_data.size(); i++) {
    dydx.push_back((m_y_data[i] - m_y_data[i - 1]) /
                   ((m_x_data[i] - m_x_data[i - 1])));
  }

  // buffers for (x,y) data
  buffer_x_data =
      sycl::buffer<double, 1>(m_x_data.data(), sycl::range<1>{m_x_data.size()});
  buffer_y_data =
      sycl::buffer<double, 1>(m_y_data.data(), sycl::range<1>{m_y_data.size()});
  buffer_dydx =
      sycl::buffer<double, 1>(dydx.data(), sycl::range<1>{dydx.size()});
};

/**
 * This function returns a value of y (y_output) given a value of x (x_input)
 * given the (x,y) data provided by the interpolator class
 *
 * @param[in] x_input x_input is reference to a vector of x values for which
 * you would like the y value returned.
 * @param[out] y_output y_output is reference to a vector of y values which
 * the interpolator calculated, based on x_input.
 */
void LinearInterpolator1D::interpolate(const std::vector<double> &x_input,
                                       std::vector<double> &y_output) {

  // Used in error handling of interpolator
  ErrorPropagate ep(m_sycl_target);
  // sycl code
  auto k_ep = ep.device_ptr();
  NESOASSERT(x_input.size() == y_output.size(),
             "size of x_input vector doesn't equal y_output vector");
  sycl::buffer<double, 1> buffer_x_input(x_input.data(),
                                         sycl::range<1>{x_input.size()});
  sycl::buffer<double, 1> buffer_y_output(y_output.data(),
                                          sycl::range<1>{y_output.size()});
  const int m_x_data_size = m_x_data.size();
  const int x_input_size = x_input.size();
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
        auto dydx_sycl = buffer_dydx.get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for<>(sycl::range<1>(x_input_size), [=](sycl::id<1> idx) {
          int k;
          for (int i = 0; i < m_x_data_size; i++) {
            // error handling
            const double x = x_input_sycl[int(idx)];
            const double x_data_sycl_start = x_data_sycl[0];
            const double x_data_sycl_end = x_data_sycl[m_x_data_size - 1];
            if (x < x_data_sycl_start or x > x_data_sycl_end) {
              // throw an error
              NESO_KERNEL_ASSERT(false, k_ep);
              break;
            }
            // set value of index to first value for which x value
            // of input is greater than x data values
            if (x - x_data_sycl[i] < 0.0) {
              k = i;
              break;
            }
          }
          y_output_sycl[int(idx)] =
              y_data_sycl[k - 1] +
              dydx_sycl[k - 1] * (x_input_sycl[int(idx)] - x_data_sycl[k - 1]);
        });
      });
  event_interpolate.wait_and_throw();
  ep.check_and_throw("OneDimensionalLinearInterpolator: Input values are "
                     "outside the range provided by data");
}

} // namespace NESO
