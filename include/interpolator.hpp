#ifndef __INTERPOLATOR_H__
#define __INTERPOLATOR_H__
#include <vector>

#include <neso_particles.hpp>

namespace NP = NESO::Particles;
namespace NESO {
/**
 *  Base class that defines functions and variables needed to read interpolate
 * data
 *
 * Derived classes must override interpolate() in order to populate y_output (1d
 * interpolator)
 */

class Interpolator {
public:
  Interpolator(std::vector<double> x_data, std::vector<double> y_data,
               NP::SYCLTargetSharedPtr sycl_target)
      : m_sycl_target(sycl_target), m_x_data(x_data), m_y_data(y_data) {
    dydx.reserve(m_y_data.size());
    dydx.push_back(0);
    for (int i = 1; i < m_y_data.size(); i++) {
      dydx.push_back((m_y_data[i] - m_y_data[i - 1]) /
                     ((m_x_data[i] - m_x_data[i - 1])));
    }
    dydx[0] = dydx[1];

    NP::NESOASSERT(m_x_data.size() == m_y_data.size(),
                   "size of m_x_data vector doesn't equal m_y_data vector");
  };
  Interpolator() = delete;

protected:
  NP::SYCLTargetSharedPtr m_sycl_target;
  std::vector<double> m_x_data;
  std::vector<double> m_y_data;
  std::vector<double> dydx;
  virtual void get_y(std::vector<double> &x_input,
                     std::vector<double> &y_output) = 0;
  virtual void interpolate(std::vector<double> &x_input,
                           std::vector<double> &y_output) = 0;
};
} // namespace NESO
#endif
