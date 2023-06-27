#ifndef __INTERPOLATOR_H__
#define __INTERPOLATOR_H__
#include <vector>

#include <neso_particles.hpp>

namespace NP = NESO::Particles;
namespace NESO {
/**
 *  Base class that defines functions and variables needed to read interpolate data
 *
 * Derived classes must override interpolate() in order to populate y_output (1d interpolator)
 */
 
class Interpolator {
public:
  Interpolator(std::vector<double> x_data, std::vector<double> y_data, std::vector<double> x_input);
  Interpolator() = delete;
  std::vector<double> get_y() { return y_output; }

protected:

  virtual void interpolate() = 0;
  std::vector<double> x_data;
  std::vector<double> y_data;
  std::vector<double> x_input;
  std::vector<double> y_output;
};
} // namespace NESO
#endif
