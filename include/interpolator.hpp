#ifndef __INTERPOLATOR_H__
#define __INTERPOLATOR_H__
#include <vector>

#include <neso_particles.hpp>

namespace NP = NESO::Particles;
namespace NESO {
/**
 * This function returns a value of y (y_output) given a value of x (x_input)
 * given the (x,y) data provided by the interpolator class
 *
 */

class Interpolator {
public:
  /**
   * This function returns a value of y (y_output) given a value of x (x_input)
   * given the (x,y) data provided by the interpolator class
   *
   * @param[in] x_input x_input is a vector of x_vales for which you would like
   *            the y value returned
   * @param[in, out] y_output y_output is the
   */
  Interpolator(std::vector<double> x_data, std::vector<double> y_data,
               NP::SYCLTargetSharedPtr sycl_target)
      : m_sycl_target(sycl_target), m_x_data(x_data), m_y_data(y_data) {

    NP::NESOASSERT(m_x_data.size() == m_y_data.size(),
                   "size of m_x_data vector doesn't equal m_y_data vector");
  };
  Interpolator() = delete;

  /**
   * This function returns a value of y (y_output) given a value of x (x_input)
   * given the (x,y) data provided by the interpolator class
   *
   * @param[in] x_input x_input is a vector of x_vales for which you would like
   *            the y value returned
   * @param[in, out] y_output y_output is the
   */
  virtual void interpolate(std::vector<double> &x_input,
                           std::vector<double> &y_output) = 0;

protected:
  NP::SYCLTargetSharedPtr m_sycl_target;
  std::vector<double> m_x_data;
  std::vector<double> m_y_data;
  std::vector<double> dydx;
};
} // namespace NESO
#endif
