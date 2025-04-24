#ifndef __NESOSOLVERS_SOLVERUTILS_HPP__
#define __NESOSOLVERS_SOLVERUTILS_HPP__
namespace NESO::Solvers {

/**
 * @brief Evaluate the Barry et al approximation to
 * the exponential integral function
 * https://en.wikipedia.org/wiki/Exponential_integral
 * E_1(x)
 *
 * @param x the argument of the exponential
 * @return double the integral approximation
 */
inline double expint_barry_approx(const double x) {
  constexpr double gamma_Euler_Mascheroni = 0.5772156649015329;
  const double G = sycl::exp(-gamma_Euler_Mascheroni);
  const double b = sycl::sqrt(2 * (1 - G) / G / (2 - G));
  const double h_inf =
      (1 - G) * ((G * G) - 6 * G + 12) / (3 * G * ((2 - G) * (2 - G)) * b);
  const double q = 20.0 / 47.0 * sycl::pow(x, sycl::sqrt(31.0 / 26.0));
  const double h = 1 / (1 + x * sycl::sqrt(x)) + h_inf * q / (1 + q);
  const double logfactor =
      sycl::log(1 + G / x - (1 - G) / ((h + b * x) * (h + b * x)));
  return sycl::exp(-x) / (G + (1 - G) * sycl::exp(-(x / (1 - G)))) * logfactor;
}
} // namespace NESO::Solvers

#endif // __NESOSOLVERS_SOLVERUTILS_HPP__
