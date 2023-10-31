#ifndef __INTERPOLATOR_H__
#define __INTERPOLATOR_H__
#include <vector>

#include <neso_particles.hpp>

namespace NP = NESO::Particles;
namespace NESO {
/**
 * This class defines the interface through which derived classes gain access to
 * x,y data which will be interpolated
 *
 */
class Interpolator {
public:
  /**
   * This constructor initialises the x,y data derived classes will use
   * checking that that the size of both vectors is the same.
   * It also initialises the target that the sycl kernel will use
   * in the derived classes.
   *
   * @param[in] x_data The x data values which the interpolator will have access
   * to.
   * @param[in] y_data The y data values which the interpolator will have access
   * to.
   * @param[in] sycl_target The target that the sycl kernels will make use of.
   */
  Interpolator(const std::vector<double> &x_data,
               const std::vector<double> &y_data,
               NP::SYCLTargetSharedPtr sycl_target)
      : m_sycl_target(sycl_target), m_x_data(x_data), m_y_data(y_data) {

    NESOASSERT(m_x_data.size() == m_y_data.size(),
               "size of m_x_data vector doesn't equal m_y_data vector");
  };
  Interpolator() = delete;

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
                           std::vector<double> &y_output) = 0;

protected:
  NP::SYCLTargetSharedPtr m_sycl_target;
  const std::vector<double> &m_x_data;
  const std::vector<double> &m_y_data;
  std::vector<double> dydx;
};
} // namespace NESO
#endif
