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
      : m_sycl_target(sycl_target) , m_x_data(x_data) ,  m_y_data(y_data) {};
  Interpolator() = delete;


protected:
  NP::SYCLTargetSharedPtr m_sycl_target;
  std::vector<double> m_x_data;
  std::vector<double> m_y_data;
  std::vector<double> x_input;
  std::vector<double> y_output;
  std::vector<double> get_y(std::vector<double> x_input);
  virtual void interpolate(std::vector<double> x_input) = 0;
};
} // namespace NESO
#endif
