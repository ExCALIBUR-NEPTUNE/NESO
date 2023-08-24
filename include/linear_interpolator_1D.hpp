#ifndef __LINEAR_INTERPOLATOR_1D_H__
#define __LINEAR_INTERPOLATOR_1D_H__

#include "interpolator.hpp"
#include <CL/sycl.hpp>
#include <mpi.h>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
using namespace cl;
namespace NESO {

/**
 * Class used to output a vector of y values, given some x values, based
 * on provided x values for input
 */

class LinearInterpolator1D : public Interpolator {
public:
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
  LinearInterpolator1D(const std::vector<double> &x_data,
                       const std::vector<double> &y_data,
                       SYCLTargetSharedPtr sycl_target);

  /**
   * This function returns a value of y (y_output) given a value of x (x_input)
   * given the (x,y) data provided by the interpolator class
   *
   * @param[in] x_input x_input is reference to a vector of x values for which
   * you would like the y value returned.
   * @param[out] y_output y_output is reference to a vector of y values which
   * the interpolator calculated, based on x_input.
   */
  virtual void interpolate(const std::vector<double> &x_input,
                           std::vector<double> &y_output);

protected:
  sycl::buffer<double, 1> buffer_x_data;
  sycl::buffer<double, 1> buffer_y_data;
  sycl::buffer<double, 1> buffer_dydx;
};

} // namespace NESO
#endif
